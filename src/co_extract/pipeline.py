"""End to end: file -> ingest -> chunk -> extract -> validate -> confidence -> review routing."""
from __future__ import annotations

from .confidence import apply_confidence
from .extract import ExtractionError, _is_empty, extract_chunk, extract_missing, extract_one, merge_change_orders, split_chunks
from .ingest import Document, load_document
from .llm import BaseLLM, LLMError
from .schema import CRITICAL_FIELDS, FIELDS, ChangeOrder, DocumentResult
from .validate import build_line_items, normalize_field, validate_change_order

DEFAULT_THRESHOLD = 0.7


def _build_change_order(raw: dict, source_text: str, threshold: float) -> ChangeOrder:
    fields = {name: normalize_field(name, raw.get(name)) for name in FIELDS}
    line_items = build_line_items(raw.get("line_items"))
    reasons = validate_change_order(fields, line_items)
    apply_confidence(fields, source_text)

    needs_review = False
    for name in FIELDS:
        f = fields[name]
        if f.value is None:
            if name in CRITICAL_FIELDS:
                needs_review = True
                reasons.append(f"critical field '{name}' not found")
            continue
        low = f.confidence is not None and f.confidence < threshold
        err = any(x.startswith("error:") for x in f.flags)
        if low or err:
            needs_review = True
            why = "low confidence" if low else "validation error"
            reasons.append(f"{name}: {why} ({f.confidence})")
    return ChangeOrder(fields=fields, line_items=line_items, needs_review=needs_review, review_reasons=reasons)


def process_document(
    doc: Document, llm: BaseLLM, threshold: float = DEFAULT_THRESHOLD, max_chars: int = 12000
) -> DocumentResult:
    result = DocumentResult(source=doc.source, warnings=list(doc.warnings))
    raw_cos: list[dict] = []
    chunks = split_chunks(doc.pages, max_chars=max_chars)
    for i, chunk in enumerate(chunks, 1):
        try:
            found = extract_chunk(llm, chunk)
        except (ExtractionError, LLMError) as exc:
            # never silent: a skipped chunk is recorded so missing data is an explicit gap
            result.warnings.append(f"chunk {i}/{len(chunks)} failed: {exc}")
            continue
        raw_cos.extend(found)
        # models silently skip items (seen on a real PDF: a chunk returned nothing, or 6 of 8 change orders,
        # with no error). Always ask once more, listing everything found so far in the document.
        try:
            already = merge_change_orders(raw_cos)
            raw_cos.extend(extract_missing(llm, chunk, already) if already else extract_chunk(llm, chunk))
        except (ExtractionError, LLMError) as exc:
            result.warnings.append(
                f"chunk {i}/{len(chunks)}: completeness pass failed ({exc}); change orders may be missing"
            )
    merged = merge_change_orders(raw_cos)
    for idx, raw in enumerate(merged):
        if _is_empty(raw.get("cost_change")):
            try:  # a missing critical field gets one targeted retry before it goes to review
                merged[idx] = merge_change_orders([raw] + extract_one(llm, chunks, raw))[0]
            except (ExtractionError, LLMError) as exc:
                result.warnings.append(f"cost_change retry failed for CO {raw.get('change_order_number')}: {exc}")
    for raw in merged:
        result.change_orders.append(_build_change_order(raw, doc.text, threshold))
    if not result.change_orders and not any("failed" in w for w in result.warnings):
        result.warnings.append("no change orders found in document")
    return result


def run_pipeline(path, llm: BaseLLM, threshold: float = DEFAULT_THRESHOLD, max_chars: int = 12000) -> DocumentResult:
    return process_document(load_document(path), llm, threshold, max_chars)