from co_extract.confidence import apply_confidence, grounding_score, norm, value_check
from co_extract.schema import FIELDS, FieldResult, LineItem
from co_extract.validate import normalize_field, validate_change_order


def make_fields(**overrides):
    fields = {n: FieldResult() for n in FIELDS}
    for k, v in overrides.items():
        fields[k] = v
    return fields


def test_normalize_field_requires_evidence():
    f = normalize_field("cost_change", {"value": "$100.00", "evidence": None})
    assert f.value == 100.0 and "error:no_evidence" in f.flags


def test_line_item_mismatch_is_an_error():
    fields = make_fields(cost_change=FieldResult(value=0.0, evidence="amount of $ 0.00"))
    reasons = validate_change_order(fields, [LineItem(description="Electrical", amount=-2500.0)])
    assert "error:line_items_sum_mismatch" in fields["cost_change"].flags
    assert any("-2,500.00" in r or "2,500.00" in r for r in reasons)


def test_line_items_that_add_up_pass():
    fields = make_fields(cost_change=FieldResult(value=2310.0, evidence="TOTAL $ 2,310.00"))
    items = [LineItem(amount=2000.0), LineItem(amount=100.0), LineItem(amount=210.0)]
    validate_change_order(fields, items)
    assert fields["cost_change"].flags == []


def test_sign_suspect_when_credit_language_but_positive():
    fields = make_fields(cost_change=FieldResult(value=500.0, evidence="Credit for deleted casework $500.00"))
    validate_change_order(fields, [])
    assert "warn:sign_suspect" in fields["cost_change"].flags


def test_sum_reconciliation_warning():
    fields = make_fields(
        cost_change=FieldResult(value=100.0, evidence="x"),
        contract_sum_before_change=FieldResult(value=1000.0, evidence="x"),
        new_contract_sum=FieldResult(value=1500.0, evidence="x"),
    )
    validate_change_order(fields, [])
    assert "warn:sum_does_not_reconcile" in fields["new_contract_sum"].flags


def test_grounding_exact_fuzzy_and_missing():
    src = norm("TOTAL  $ 2,310.00 \n General   Conditions")
    assert grounding_score("TOTAL $ 2,310.00", src) == 1.0
    assert grounding_score("General Conditions", src) == 1.0
    assert grounding_score("Approved by J. Miller on 05/14/2025", src) < 0.5
    assert grounding_score(None, src) == 0.0


def test_value_check_amount_date_and_days():
    assert value_check("cost_change", 2310.0, "TOTAL $ 2,310.00") == 1.0
    assert value_check("cost_change", 2310.0, "TOTAL $ 9,999.00") == 0.0
    assert value_check("cost_change", -4000.0, "($4,000.00)") == 1.0
    assert value_check("date", "2025-05-13", "Date : 05/13/2025") == 1.0
    assert value_check("date", "2025-05-13", "Date : 05/12/2025") == 0.0
    assert value_check("schedule_impact_days", 0, "(unchanged) by Days: 0") == 1.0


def test_confidence_high_when_grounded_low_when_not():
    source = "TOTAL  $ 2,310.00"
    good = normalize_field("cost_change", {"value": "$ 2,310.00", "evidence": "TOTAL  $ 2,310.00", "self_confidence": 0.9})
    bad = normalize_field("cost_change", {"value": "$ 2,310.00", "evidence": "Total change: $2,310.00 approved", "self_confidence": 0.99})
    fields = make_fields(cost_change=good)
    apply_confidence(fields, source)
    fields2 = make_fields(cost_change=bad)
    apply_confidence(fields2, source)
    assert fields["cost_change"].confidence >= 0.9
    assert fields2["cost_change"].confidence < fields["cost_change"].confidence
    assert fields["date"].confidence is None  # null value has no confidence


def test_form_label_is_not_an_approval():
    fields = make_fields(
        approved_by=FieldResult(value="Mayor, City Clerk, City Attorney", evidence="Attest: Approved As to Form: Mayor City Clerk City Attorney"),
        status=FieldResult(value="approved", evidence="Approved As to Form"),
    )
    validate_change_order(fields, [])
    assert "error:approval_not_evidenced" in fields["approved_by"].flags
    assert "error:status_not_evidenced" in fields["status"].flags


def test_names_under_signature_lines_are_not_an_approval():
    fields = make_fields(
        approved_by=FieldResult(value="Rodney L. Curtis", evidence="Signature Signature Signature Rodney L. Curtis Jason Kentner Glenwood"),
    )
    validate_change_order(fields, [])
    assert "error:approval_not_evidenced" in fields["approved_by"].flags


def test_real_approval_statement_passes():
    fields = make_fields(
        approved_by=FieldResult(value="J. Miller", evidence="Approved by J. Miller on 05/14/2025"),
        status=FieldResult(value="approved", evidence="Status: Approved"),
    )
    validate_change_order(fields, [])
    assert fields["approved_by"].flags == [] and fields["status"].flags == []
