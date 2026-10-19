from co_extract.evaluate import compare, score_document, summarize, _norm_id
from co_extract.llm import StubLLM
from co_extract.pipeline import run_pipeline

from conftest import LABELS, glenwood_response
import json


def test_compare_rules():
    assert compare("cost_change", 2310.0, 2310.0)
    assert not compare("cost_change", 2310.0, 231.0)
    assert compare("cost_change", None, None)
    assert not compare("approved_by", "J. Miller", None)  # hallucination
    assert compare("contractor", "Cornerstone Commercial Contractors, Inc.", "Cornerstone Commercial Contractors")
    assert compare("date", "2025-05-13", "2025-05-13")
    assert compare("change_order_number", "5R", "5r")


def test_norm_id_ignores_leading_zero_and_hash():
    assert _norm_id("09") == _norm_id("9") == _norm_id("#9")
    assert _norm_id("5R") == _norm_id("5r")


def test_score_document_on_glenwood(glenwood_path):
    res = run_pipeline(glenwood_path, StubLLM([glenwood_response(hallucinate=True)]))
    labels = json.loads((LABELS / "glenwood_2025_cos.json").read_text(encoding="utf-8"))
    entries, meta = score_document(res, labels, "glenwood", 0.7)
    assert meta["label_cos"] == 2 and meta["matched_cos"] == 2 and meta["missed_cos"] == 0
    halluc = [e for e in entries if e["field"] == "approved_by" and not e["correct"]]
    assert len(halluc) == 1 and halluc[0]["flagged"]  # wrong AND routed to review
    s = summarize(entries)
    assert s["hallucinations"]["predicted_value"] == 1
    assert s["review_routing"]["wrong_values_flagged"] == s["review_routing"]["wrong_values"]


def test_missed_change_order_is_counted(glenwood_path):
    res = run_pipeline(glenwood_path, StubLLM(['{"change_orders": []}']))
    labels = json.loads((LABELS / "glenwood_2025_cos.json").read_text(encoding="utf-8"))
    entries, meta = score_document(res, labels, "glenwood", 0.7)
    assert meta["missed_cos"] == 2 and entries == []


def _co(number, project, cost):
    from co_extract.schema import ChangeOrder, FieldResult, FIELDS

    fields = {n: FieldResult() for n in FIELDS}
    fields["change_order_number"] = FieldResult(value=number, confidence=1.0)
    fields["project_name"] = FieldResult(value=project, confidence=1.0)
    fields["cost_change"] = FieldResult(value=cost, confidence=1.0)
    return ChangeOrder(fields=fields)


def test_two_change_orders_with_same_number_match_by_project():
    from co_extract.schema import DocumentResult

    res = DocumentResult(
        source="x",
        change_orders=[_co("4", "Birchwood ES Misc. Upgrades", 16690.91), _co("4", "Turnagain ES Restroom Remodel", -12136.9)],
    )
    labels = {
        "change_orders": [
            {"change_order_number": "4", "project_name": "Birchwood ES Misc. Upgrades", "cost_change": 16690.91},
            {"change_order_number": "4", "project_name": "Turnagain ES Restroom Remodel", "cost_change": -12136.9},
        ]
    }
    entries, meta = score_document(res, labels, "d", 0.7)
    assert meta["matched_cos"] == 2 and meta["spurious_cos"] == 0
    assert all(e["correct"] for e in entries)


def test_same_number_but_different_project_is_not_a_match():
    from co_extract.schema import DocumentResult

    res = DocumentResult(source="x", change_orders=[_co("4", "Totally Different Library", 1.0)])
    labels = {"change_orders": [{"change_order_number": "4", "project_name": "Birchwood ES Misc. Upgrades", "cost_change": 1.0}]}
    entries, meta = score_document(res, labels, "d", 0.7)
    assert meta["missed_cos"] == 1 and meta["spurious_cos"] == 1 and entries == []


def test_description_is_listed_not_scored(glenwood_path):
    res = run_pipeline(glenwood_path, StubLLM([glenwood_response()]))
    labels = json.loads((LABELS / "glenwood_2025_cos.json").read_text(encoding="utf-8"))
    entries, meta = score_document(res, labels, "glenwood", 0.7)
    assert not any(e["field"] == "description" for e in entries)
    assert len(meta["descriptions"]) == 2


def test_unmatched_change_orders_are_listed():
    from co_extract.schema import DocumentResult

    res = DocumentResult(source="x", change_orders=[_co("7", "Some Other Project", 1.0)])
    labels = {"change_orders": [{"change_order_number": "4", "project_name": "Turnagain ES Restroom Remodel", "cost_change": 1.0}]}
    entries, meta = score_document(res, labels, "d", 0.7)
    kinds = sorted(u["kind"] for u in meta["unmatched"])
    assert kinds == ["in output, not in labels", "labeled, not found in output"]


def test_norm_id_strips_co_prefix():
    assert _norm_id("CO #5R") == _norm_id("5R")
    assert _norm_id("CO #09") == _norm_id("9")


def test_end_to_end_counts_missed_fields():
    from co_extract.schema import DocumentResult

    res = DocumentResult(source="x", change_orders=[])
    labels = {"change_orders": [{"change_order_number": "4", "project_name": "P", "cost_change": 1.0, "description": "x"}]}
    entries, meta = score_document(res, labels, "d", 0.7)
    assert meta["missed_cos"] == 1 and meta["missed_fields"] == 3  # description is not scored


def test_runs_summary_lists_every_run():
    from co_extract.evaluate import runs_summary

    rep = {
        "model": "m", "threshold": 0.7, "date": "d",
        "summary": {
            "n_scored": 10,
            "per_field": {"cost_change": {"n": 10, "correct": 9}},
            "hallucinations": {"n_null_labels": 4, "predicted_value": 1},
            "review_routing": {"wrong_values": 1, "wrong_values_flagged": 0},
        },
        "change_orders": {"label_cos": 3, "matched_cos": 2, "missed_cos": 1, "spurious_cos": 0, "missed_fields": 5},
    }
    md = runs_summary([rep, rep])
    assert "| 1 |" in md and "| 2 |" in md and "9/15 (60%)" in md
