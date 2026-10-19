import json

from co_extract.extract import merge_change_orders, parse_llm_json, split_chunks
from co_extract.ingest import load_document
from co_extract.llm import StubLLM
from co_extract.pipeline import process_document, run_pipeline

from conftest import glenwood_response


def test_ingest_splits_pages(glenwood_path):
    doc = load_document(glenwood_path)
    assert len(doc.pages) == 2
    assert doc.pages[0].startswith("[[PAGE 1]]")
    assert "CO #09" in doc.pages[1]


def test_pipeline_end_to_end_with_stub(glenwood_path):
    llm = StubLLM([glenwood_response()])
    res = run_pipeline(glenwood_path, llm)
    assert [c.fields["change_order_number"].value for c in res.change_orders] == ["5R", "09"]
    co5r, co9 = res.change_orders
    assert co5r.fields["cost_change"].value == 2310.0
    assert co5r.fields["date"].value == "2025-05-13"
    assert co5r.fields["cost_change"].confidence >= 0.9
    assert not co5r.needs_review
    # CO #9: stated change is 0.00 but its only line item is -2,500 -> must go to review
    assert co9.needs_review
    assert "error:line_items_sum_mismatch" in co9.fields["cost_change"].flags
    assert "warn:ambiguous_date_format" in co9.fields["date"].flags


def test_hallucinated_field_is_caught(glenwood_path):
    llm = StubLLM([glenwood_response(hallucinate=True)])
    res = run_pipeline(glenwood_path, llm)
    co9 = res.change_orders[1]
    approved = co9.fields["approved_by"]
    assert approved.value == "J. Miller"
    assert approved.confidence < 0.5  # evidence quote is not in the document
    assert co9.needs_review


def test_bad_json_is_retried_then_recorded(glenwood_path):
    llm = StubLLM(["not json at all", "still not json"])
    res = run_pipeline(glenwood_path, llm)
    assert res.change_orders == []
    assert any("failed" in w for w in res.warnings)  # explicit gap, never silent
    assert len(llm.calls) == 2


def test_bad_json_then_good_json_recovers(glenwood_path):
    llm = StubLLM(["oops", glenwood_response()])
    res = run_pipeline(glenwood_path, llm)
    assert len(res.change_orders) == 2


def test_parse_llm_json_handles_fences_and_prose():
    obj = parse_llm_json('Sure!\n```json\n{"change_orders": []}\n```')
    assert obj == {"change_orders": []}


def test_split_chunks_covers_all_pages_with_overlap():
    pages = [f"[[PAGE {i}]]\n" + ("x" * 5000) for i in range(1, 6)]
    chunks = split_chunks(pages, max_chars=12000, overlap_pages=1)
    assert len(chunks) > 1
    for i in range(1, 6):
        assert any(f"[[PAGE {i}]]" in c for c in chunks)
    assert all(len(c) <= 12100 for c in chunks)


def test_split_chunks_breaks_oversized_page():
    page = "[[PAGE 1]]\n" + "\n".join("line %d" % i for i in range(5000))
    chunks = split_chunks([page], max_chars=2000)
    assert len(chunks) > 1 and all(len(c) <= 2100 for c in chunks)


def test_merge_drops_duplicates_keeps_fuller():
    a = {"change_order_number": {"value": "4"}, "project_name": {"value": "X"}}
    b = {"change_order_number": {"value": "4"}, "project_name": {"value": "X"}, "contractor": {"value": "Acme"}}
    c = {"change_order_number": {"value": "5"}, "project_name": {"value": "X"}}
    merged = merge_change_orders([a, b, c])
    assert len(merged) == 2 and merged[0]["contractor"]["value"] == "Acme"


def test_pdf_ingest_roundtrip(tmp_path):
    reportlab = __import__("pytest").importorskip("reportlab")
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "co.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 750, "CHANGE ORDER No. 7 - Project Alpha - Total $1,234.00")
    c.showPage()
    c.showPage()  # blank page: must be reported, not silently dropped
    c.save()
    doc = load_document(pdf)
    assert "CHANGE ORDER No. 7" in doc.text
    assert any("page 2" in w for w in doc.warnings)


def _full():
    return json.loads(glenwood_response())["change_orders"]


def test_completion_pass_recovers_skipped_change_order(glenwood_path):
    full = _full()
    first = json.dumps({"change_orders": [full[0]]})
    second = json.dumps({"change_orders": [full[1]]})
    llm = StubLLM(lambda s, u: second if "NOT in that list" in u else first)
    res = run_pipeline(glenwood_path, llm)
    assert [c.fields["change_order_number"].value for c in res.change_orders] == ["5R", "09"]
    assert len(llm.calls) == 2


def test_completion_pass_duplicate_with_co_prefix_is_merged(glenwood_path):
    full = _full()
    dup = dict(full[0])
    dup["change_order_number"] = {"value": "CO #5R", "evidence": "Change Order CO #5R Distribution List:", "self_confidence": 0.9}
    llm = StubLLM(
        lambda s, u: json.dumps({"change_orders": [dup]}) if "NOT in that list" in u else json.dumps({"change_orders": [full[0]]})
    )
    res = run_pipeline(glenwood_path, llm)
    assert len(res.change_orders) == 1
    assert res.change_orders[0].fields["change_order_number"].value == "5R"


def test_failed_completion_pass_keeps_first_pass_and_warns(glenwood_path):
    first = json.dumps({"change_orders": [_full()[0]]})
    llm = StubLLM(lambda s, u: "garbage" if "NOT in that list" in u else first)
    res = run_pipeline(glenwood_path, llm)
    assert len(res.change_orders) == 1
    assert any("completeness pass failed" in w for w in res.warnings)


def test_same_number_different_projects_are_not_merged():
    a = {"change_order_number": {"value": "4"}, "project_name": {"value": "Birchwood ES Misc. Upgrades"}}
    b = {"change_order_number": {"value": "Change Order No. 4"}, "project_name": {"value": "Turnagain ES Restroom Remodel"}}
    assert len(merge_change_orders([a, b])) == 2


def test_missing_cost_change_routes_change_order_to_review(glenwood_path):
    """Seen in a real run: the model returned an Airport Heights change order with cost_change = null."""
    full = _full()
    full[0]["cost_change"] = {"value": None, "evidence": None, "self_confidence": None}
    llm = StubLLM(lambda s, u: json.dumps({"change_orders": []}) if "NOT in that list" in u else json.dumps({"change_orders": full}))
    res = run_pipeline(glenwood_path, llm)
    co5r = res.change_orders[0]
    assert co5r.fields["cost_change"].value is None
    assert co5r.needs_review
    assert any("cost_change" in r and "not found" in r for r in co5r.review_reasons)
