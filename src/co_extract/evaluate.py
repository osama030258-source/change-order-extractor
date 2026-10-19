"""Score pipeline output against hand-made labels. Reports only what a run produced.

Label file (data/labels/<sample-stem>.json):
  {"source_url": "...", "change_orders": [{"change_order_number": "5R", "cost_change": 2310.0, ...}]}
A field that is ABSENT from a label is not scored. A field set to null means
"the document does not state this" and a non-null prediction counts as a hallucination.
Keys starting with "_" are notes and ignored.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from rapidfuzz import fuzz

from .llm import BaseLLM
from .pipeline import DEFAULT_THRESHOLD, run_pipeline
from .schema import AMOUNT_FIELDS, DATE_FIELDS, FIELDS, INT_FIELDS, DocumentResult, clean_co_number, parse_amount, parse_date

FUZZY_TEXT = {"project_name", "contractor", "description", "requested_by", "approved_by"}
FUZZY_MIN = 80
# Free-text paraphrases cannot be scored fairly with a string-similarity threshold
# (a correct one-sentence summary rarely matches a hand-written one). They are
# listed side by side in the report for a human to judge instead.
NOT_SCORED = {"description"}
PROJECT_MATCH_MIN = 50


def _norm_id(x: Any) -> str:
    s = clean_co_number(x or "").strip().lower()
    return s.lstrip("0") or s


def compare(field: str, pred: Any, label: Any) -> bool:
    if pred is None and label is None:
        return True
    if pred is None or label is None:
        return False
    if field in AMOUNT_FIELDS:
        lv, _ = parse_amount(label)
        return lv is not None and abs(float(pred) - lv) < 0.01
    if field in INT_FIELDS:
        return int(pred) == int(label)
    if field in DATE_FIELDS:
        lv, _ = parse_date(label)
        return pred == (lv or str(label))
    if field == "change_order_number":
        return _norm_id(pred) == _norm_id(label)
    if field in FUZZY_TEXT:
        return fuzz.token_set_ratio(str(pred).lower(), str(label).lower()) >= FUZZY_MIN
    return str(pred).strip().lower() == str(label).strip().lower()


def _match(lab: dict, preds: list, used: set) -> Optional[int]:
    """Index of the unused predicted change order that corresponds to this label.

    Numbers alone are not unique (two projects can both have a "Change Order 4"),
    so candidates with the same number are ranked by project-name similarity.
    """
    key = _norm_id(lab.get("change_order_number"))
    cands = [
        i for i, co in enumerate(preds)
        if i not in used and _norm_id(co.fields["change_order_number"].value) == key
    ]
    if not cands:
        return None
    want = lab.get("project_name")
    if want is None:
        return cands[0]

    def sim(i: int) -> float:
        got = preds[i].fields["project_name"].value
        return fuzz.token_set_ratio(str(got or "").lower(), str(want).lower())

    best = max(cands, key=sim)
    return best if sim(best) >= PROJECT_MATCH_MIN else None


def score_document(result: DocumentResult, labels: dict, doc_name: str, threshold: float) -> tuple[list[dict], dict]:
    entries: list[dict] = []
    meta: dict[str, Any] = {"label_cos": 0, "matched_cos": 0, "missed_cos": 0, "spurious_cos": 0, "missed_fields": 0, "descriptions": [], "unmatched": []}
    preds = list(result.change_orders)
    used: set[int] = set()
    for lab in labels["change_orders"]:
        meta["label_cos"] += 1
        idx = _match(lab, preds, used)
        if idx is None:
            meta["missed_cos"] += 1
            meta["missed_fields"] += sum(1 for f in FIELDS if f in lab and f not in NOT_SCORED)
            meta["unmatched"].append(
                {"doc": doc_name, "kind": "labeled, not found in output", "co": lab.get("change_order_number"), "project": lab.get("project_name")}
            )
            continue
        co = preds[idx]
        used.add(idx)
        meta["matched_cos"] += 1
        for field in FIELDS:
            if field not in lab:
                continue
            f = co.fields[field]
            if field in NOT_SCORED:
                meta["descriptions"].append(
                    {"doc": doc_name, "co": lab.get("change_order_number"), "label": lab[field], "pred": f.value}
                )
                continue
            correct = compare(field, f.value, lab[field])
            flagged = f.value is not None and (
                (f.confidence is not None and f.confidence < threshold)
                or any(x.startswith("error:") for x in f.flags)
            )
            entries.append(
                {
                    "doc": doc_name,
                    "co": lab.get("change_order_number"),
                    "field": field,
                    "label": lab[field],
                    "pred": f.value,
                    "correct": correct,
                    "confidence": f.confidence,
                    "flags": f.flags,
                    "flagged": flagged,
                }
            )
    meta["spurious_cos"] = len(preds) - len(used)
    for i, co in enumerate(preds):
        if i not in used:
            meta["unmatched"].append(
                {
                    "doc": doc_name,
                    "kind": "in output, not in labels",
                    "co": co.fields["change_order_number"].value,
                    "project": co.fields["project_name"].value,
                }
            )
    return entries, meta


def summarize(entries: list[dict]) -> dict:
    out: dict[str, Any] = {"n_scored": len(entries)}
    out["accuracy"] = round(mean(e["correct"] for e in entries), 4) if entries else None
    per_field = {}
    for field in FIELDS:
        es = [e for e in entries if e["field"] == field]
        if es:
            per_field[field] = {"n": len(es), "correct": sum(e["correct"] for e in es)}
    out["per_field"] = per_field
    null_labels = [e for e in entries if e["label"] is None]
    out["hallucinations"] = {"n_null_labels": len(null_labels), "predicted_value": sum(e["pred"] is not None for e in null_labels)}
    wrong = [e for e in entries if e["pred"] is not None and not e["correct"]]
    right = [e for e in entries if e["pred"] is not None and e["correct"]]
    stated = [e for e in entries if e["label"] is not None]
    out["left_blank"] = {"n_stated": len(stated), "returned_null": sum(e["pred"] is None for e in stated)}
    out["review_routing"] = {
        "wrong_values": len(wrong),
        "wrong_values_flagged": sum(e["flagged"] for e in wrong),
        "correct_values": len(right),
        "correct_values_flagged": sum(e["flagged"] for e in right),
        "mean_conf_correct": round(mean(e["confidence"] for e in right if e["confidence"] is not None), 3) if right else None,
        "mean_conf_wrong": round(mean(e["confidence"] for e in wrong if e["confidence"] is not None), 3) if wrong else None,
    }
    return out


def evaluate(samples_dir: Path, labels_dir: Path, llm: BaseLLM, threshold: float = DEFAULT_THRESHOLD) -> dict:
    all_entries: list[dict] = []
    totals = {"label_cos": 0, "matched_cos": 0, "missed_cos": 0, "spurious_cos": 0, "missed_fields": 0}
    warnings: list[str] = []
    docs = []
    descriptions: list[dict] = []
    unmatched: list[dict] = []
    for lab_path in sorted(Path(labels_dir).glob("*.json")):
        stem = lab_path.stem
        sample = next((p for ext in (".pdf", ".txt") if (p := Path(samples_dir) / f"{stem}{ext}").exists()), None)
        if sample is None:
            warnings.append(f"{stem}: label exists but no sample file found; skipped")
            continue
        labels = json.loads(lab_path.read_text(encoding="utf-8"))
        result = run_pipeline(sample, llm, threshold)
        entries, meta = score_document(result, labels, stem, threshold)
        all_entries.extend(entries)
        descriptions.extend(meta["descriptions"])
        unmatched.extend(meta["unmatched"])
        for k in totals:
            totals[k] += meta[k]
        warnings += [f"{stem}: {w}" for w in result.warnings]
        docs.append(stem)
    return {
        "model": llm.name,
        "threshold": threshold,
        "date": date.today().isoformat(),
        "documents": docs,
        "change_orders": totals,
        "summary": summarize(all_entries),
        "warnings": warnings,
        "descriptions": descriptions,
        "unmatched": unmatched,
        "entries": all_entries,
    }


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def _correct(rep: dict) -> int:
    return sum(v["correct"] for v in rep["summary"]["per_field"].values())


def _metrics(rep: dict) -> dict:
    s, co = rep["summary"], rep["change_orders"]
    return {
        "matched": co["matched_cos"],
        "labeled": co["label_cos"],
        "extra": co["spurious_cos"],
        "correct": _correct(rep),
        "scored": s["n_scored"],
        "e2e_total": s["n_scored"] + co.get("missed_fields", 0),
        "halluc": s["hallucinations"]["predicted_value"],
        "null_labels": s["hallucinations"]["n_null_labels"],
        "wrong": s["review_routing"]["wrong_values"],
        "wrong_flagged": s["review_routing"]["wrong_values_flagged"],
    }


def runs_summary(reps: list[dict]) -> str:
    """One row per run. Model output varies between runs, so never quote a single run."""
    ms = [_metrics(r) for r in reps]
    lines = [
        f"# Eval summary over {len(reps)} runs",
        "",
        f"Model: `{reps[0]['model']}` | threshold {reps[0]['threshold']} | date {reps[0]['date']}",
        "",
        "| run | change orders found | extra | fields correct (found COs only) | end to end | wrong values flagged | invented values (null labels) |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(ms, 1):
        lines.append(
            f"| {i} | {_pct(m['matched'], m['labeled'])} | {m['extra']} | {_pct(m['correct'], m['scored'])} "
            f"| {_pct(m['correct'], m['e2e_total'])} | {_pct(m['wrong_flagged'], m['wrong'])} | {_pct(m['halluc'], m['null_labels'])} |"
        )
    found = [m["matched"] for m in ms]
    pcts = [100 * m["correct"] / m["e2e_total"] for m in ms if m["e2e_total"]]
    lines += ["", f"- Change orders found per run: min {min(found)}, max {max(found)} (of {ms[0]['labeled']})"]
    if pcts:
        lines.append(f"- End-to-end accuracy (a missed change order counts all its labeled fields as wrong): {min(pcts):.0f}% to {max(pcts):.0f}%")
    lines.append("")
    return "\n".join(lines)


def to_markdown(rep: dict) -> str:
    s, rr, h = rep["summary"], rep["summary"]["review_routing"], rep["summary"]["hallucinations"]
    co = rep["change_orders"]
    lines = [
        "# Evaluation report",
        "",
        f"- Model: `{rep['model']}`  |  review threshold: {rep['threshold']}  |  run date: {rep['date']}",
        f"- Documents: {', '.join(rep['documents']) or 'none'}",
        f"- Labeled change orders: {co['label_cos']}; matched: {co['matched_cos']}; missed: {co['missed_cos']}; extra (unlabeled) predicted: {co['spurious_cos']}",
        "",
        "## Field accuracy",
        "",
        f"- Change orders found: {_pct(co['matched_cos'], co['label_cos'])}",
        f"- Fields correct, among the change orders that were found: {_pct(_correct(rep), s['n_scored'])}",
        f"- End to end (a missed change order counts all its labeled fields as wrong): {_pct(_correct(rep), s['n_scored'] + co.get('missed_fields', 0))}",
        "",
        "| field | correct |",
        "|---|---|",
    ]
    for f, v in s["per_field"].items():
        lines.append(f"| {f} | {_pct(v['correct'], v['n'])} |")
    lines += [
        "",
        "## Hallucination on fields the document does not state",
        "",
        f"{_pct(h['predicted_value'], h['n_null_labels'])} of null-labeled fields got a non-null value.",
        "",
        "## Fields left blank although the document states them",
        "",
        f"{_pct(s['left_blank']['returned_null'], s['left_blank']['n_stated'])} of labeled non-null fields came back null (safe failure: a human fills them in).",
        "",
        "## Does low confidence catch the mistakes?",
        "",
        f"- Wrong non-null values flagged for review: {_pct(rr['wrong_values_flagged'], rr['wrong_values'])}",
        f"- Correct values flagged for review (false alarms): {_pct(rr['correct_values_flagged'], rr['correct_values'])}",
        f"- Mean confidence, correct values: {rr['mean_conf_correct']}  |  wrong values: {rr['mean_conf_wrong']}",
        "",
    ]
    wrong = [e for e in rep["entries"] if not e["correct"]]
    if wrong:
        lines += ["## Every incorrect field", "", "| doc | CO | field | label | predicted | confidence | flags |", "|---|---|---|---|---|---|---|"]
        for e in wrong:
            lines.append(f"| {e['doc']} | {e['co']} | {e['field']} | {e['label']} | {e['pred']} | {e['confidence']} | {', '.join(e['flags'])} |")
        lines.append("")
    if rep.get("unmatched"):
        lines += ["## Change orders that did not match", "", "| doc | what | CO number | project |", "|---|---|---|---|"]
        for u in rep["unmatched"]:
            lines.append(f"| {u['doc']} | {u['kind']} | {u['co']} | {u['project']} |")
        lines.append("")
    if rep.get("descriptions"):
        lines += [
            "## Descriptions (not auto-scored; judge by eye)",
            "",
            "| doc | CO | label | predicted |",
            "|---|---|---|---|",
        ]
        for d in rep["descriptions"]:
            lines.append(f"| {d['doc']} | {d['co']} | {d['label']} | {d['pred']} |")
        lines.append("")
    lines += ["_Single run. Model output can vary between runs; repeat the eval before quoting a number._", ""]
    if rep["warnings"]:
        lines += ["## Warnings from the run", ""] + [f"- {w}" for w in rep["warnings"]] + [""]
    return "\n".join(lines)
