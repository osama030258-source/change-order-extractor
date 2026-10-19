"""LLM extraction: chunking, prompting, robust JSON parsing, completeness pass, de-duplication."""
from __future__ import annotations

import json
import re
from typing import Any

from rapidfuzz import fuzz

from .llm import BaseLLM
from .schema import FIELDS, clean_co_number

FIELD_HELP = {
    "change_order_number": "The change order / contract modification number exactly as written (e.g. '5R', '3', '04').",
    "date": "Date of THIS change order, copied exactly as written (do not reformat).",
    "project_name": "Name of the project the change order belongs to.",
    "contractor": "Contractor the change order is issued to.",
    "description": "One-sentence summary of what changed. The ONLY field where you may paraphrase.",
    "cost_change": "Net cost impact of THIS change order, copied as written. Decreases/credits are negative.",
    "schedule_impact_days": "Days added (+) or removed (-) from contract time by THIS change order, integer. 0 only if the text states no change/0 days. null if not stated.",
    "requested_by": "Person or organisation that requested the change, only if stated.",
    "approved_by": "Name or body that approved/signed, ONLY if actually shown. Blank signature lines mean null.",
    "status": "One of: approved, pending, rejected, executed, void. Only if the document states it.",
    "original_contract_sum": "Original contract amount, only if stated.",
    "contract_sum_before_change": "Contract sum immediately before this change order, only if explicitly stated.",
    "new_contract_sum": "Contract sum after this change order, only if explicitly stated.",
}

SYSTEM_PROMPT = f"""You extract structured data from construction change orders.

Return ONE JSON object and nothing else:
{{"change_orders": [ {{ <field>: {{"value": ..., "evidence": "...", "self_confidence": 0.0-1.0}}, ..., "line_items": [{{"description": "...", "amount": "...", "evidence": "..."}}] }} ]}}

Rules:
1. Extract EVERY distinct change order in the text (a file can contain several). If there are none, return {{"change_orders": []}}.
2. "evidence" must be a VERBATIM copy (max ~200 chars) of the text that supports the value. Never invent or reword it.
3. If a field is not stated, use {{"value": null, "evidence": null, "self_confidence": null}}. Never guess, infer, or compute a value that is not written in the text.
4. Blank signature lines, form labels and boilerplate are not values.
5. Amounts: copy as written (e.g. "$2,310.00", "(12,136.90)", "<2,500.00>"). Credits/decreases are negative.
6. Use only the fields listed below. Include every field for every change order.
7. line_items: itemised lines that make up THIS change order's cost, if present; otherwise [].

Fields:
""" + "\n".join(f"- {name}: {FIELD_HELP[name]}" for name in FIELDS)


class ExtractionError(RuntimeError):
    pass


PAGE_SPLIT = re.compile(r"(?=\[\[PAGE \d+\]\])")


def split_chunks(pages: list[str], max_chars: int = 12000, overlap_pages: int = 1) -> list[str]:
    """Group pages into chunks <= max_chars, overlapping so a change order that
    straddles a boundary is seen whole at least once."""
    flat: list[str] = []
    for p in pages:
        if len(p) <= max_chars:
            flat.append(p)
            continue
        buf = ""
        for line in p.splitlines(keepends=True):
            if len(buf) + len(line) > max_chars and buf:
                flat.append(buf)
                buf = ""
            buf += line
        if buf:
            flat.append(buf)

    chunks: list[str] = []
    start = 0
    while start < len(flat):
        end, size = start, 0
        while end < len(flat) and (size + len(flat[end]) <= max_chars or end == start):
            size += len(flat[end])
            end += 1
        chunks.append("\n".join(flat[start:end]))
        if end >= len(flat):
            break
        start = max(end - overlap_pages, start + 1)
    return chunks


def parse_llm_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j <= i:
            raise ExtractionError("no JSON object in model output")
        try:
            obj = json.loads(t[i : j + 1])
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("change_orders"), list):
        raise ExtractionError("JSON missing 'change_orders' list")
    return obj


def _ask(llm: BaseLLM, user: str) -> list[dict]:
    last_err: Exception | None = None
    for attempt in range(2):
        prompt = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON in the required shape. Reply with the JSON object only."
        raw = llm.complete_json(SYSTEM_PROMPT, prompt)
        try:
            return [c for c in parse_llm_json(raw)["change_orders"] if isinstance(c, dict)]
        except ExtractionError as exc:
            last_err = exc
    raise ExtractionError(str(last_err))


def _doc_block(chunk: str) -> str:
    return f"Page markers look like [[PAGE n]].\n\n<document>\n{chunk}\n</document>\n\n"


def extract_chunk(llm: BaseLLM, chunk: str) -> list[dict]:
    return _ask(llm, _doc_block(chunk) + "Return the JSON now.")


def extract_missing(llm: BaseLLM, chunk: str, found: list[dict]) -> list[dict]:
    """Second pass: models silently skip items. Show what was found, ask only for the rest."""
    listed = "\n".join(
        f"- {_raw(co, 'change_order_number') or '?'} | {_raw(co, 'project_name') or '?'}" for co in found
    )
    user = (
        _doc_block(chunk)
        + f"A previous pass already extracted these change orders (number | project):\n{listed}\n\n"
        "Check the document for any change order that is NOT in that list (for example one that was skipped). "
        "Return ONLY change orders NOT in that list, using the same JSON format. "
        'If every change order in the document is already listed, return {"change_orders": []}. '
        "Never repeat a listed one."
    )
    return _ask(llm, user)


def extract_one(llm: BaseLLM, chunks: list[str], co: dict) -> list[dict]:
    """Third pass for one change order whose critical field came back empty: re-read only
    the chunks that mention its project and ask for that single change order again."""
    proj = _raw(co, "project_name")
    key = re.sub(r"\s+", " ", proj).lower()[:25]
    hits = [c for c in chunks if key and key in re.sub(r"\s+", " ", c).lower()][:2]
    if not hits:
        return []
    user = (
        _doc_block("\n".join(hits))
        + f"Extract ONLY change order {_raw(co, 'change_order_number') or '?'} for project \"{proj}\". "
        "Look for its total cost: the line 'TOTAL COST THIS CHANGE ORDER', a 'Current Contract Modification' "
        "row, or any total stated for this change order. Use the same JSON format with one change order."
    )
    return _ask(llm, user)


def _raw(co: dict, name: str) -> str:
    v = co.get(name)
    if isinstance(v, dict):
        v = v.get("value")
    return re.sub(r"\s+", " ", str(v)).strip() if v not in (None, "") else ""


def _val(co: dict, name: str) -> str:
    return _raw(co, name).lower()


def _num_key(co: dict) -> str:
    n = _raw(co, "change_order_number")
    if not n:
        return ""
    n = clean_co_number(n).lower().strip()
    return n.lstrip("0") or n


def _filled(co: dict) -> int:
    return sum(1 for n in FIELDS if _val(co, n))


SAME_PROJECT_MIN = 70
SAME_DESCRIPTION_MIN = 80


def merge_change_orders(cos: list[dict]) -> list[dict]:
    """Drop duplicates (overlapping chunks, completeness pass); keep the fuller record.

    Same number AND similar project name = same change order. Numbers alone are
    not enough: two projects can both have a "Change Order 4".
    """
    kept: list[dict] = []
    for co in cos:
        num, proj = _num_key(co), _val(co, "project_name")
        match = None
        for i, k in enumerate(kept):
            kn = _num_key(k)
            if num and kn == num:
                kp = _val(k, "project_name")
                if not proj or not kp or fuzz.token_set_ratio(proj, kp) >= SAME_PROJECT_MIN:
                    match = i
                    break
            elif not num and not kn:
                d, kd = _val(co, "description")[:80], _val(k, "description")[:80]
                if d and kd and fuzz.token_set_ratio(d, kd) >= SAME_DESCRIPTION_MIN:
                    match = i
                    break
        if match is None:
            kept.append(co)
        else:
            a, b = kept[match], co
            best, other = (b, a) if _filled(b) > _filled(a) else (a, b)
            kept[match] = _fill_gaps(dict(best), other)
    return kept


def _is_empty(v) -> bool:
    if isinstance(v, dict):
        v = v.get("value")
    return v is None or v == "" or v == []


def _fill_gaps(best: dict, other: dict) -> dict:
    """Same change order seen in two chunks (e.g. summary page and detail page): keep the
    fuller record, but take any field it lacks from the other one instead of dropping it."""
    for key, val in other.items():
        if _is_empty(best.get(key)) and not _is_empty(val):
            best[key] = val
    return best