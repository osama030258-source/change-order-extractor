"""Normalise raw LLM output and run cross-field sanity checks.

Flags are strings: "warn:<code>" (soft) or "error:<code>" (forces human review).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from .schema import ALLOWED_STATUS, FIELDS, FieldResult, LineItem, parse_amount, normalize_value

TOL = 0.01
CREDIT_WORDS = re.compile(r"\b(credit|deduct\w*|decrease\w*|reduction|reduce\w*)\b", re.I)
# An approver needs an actual approval statement with a name after it. A form label
# ("Approved As to Form") or role titles under a blank signature line do not count.
APPROVAL_RE = re.compile(
    r"\b(?i:approved|signed|accepted|authorized|authorised|executed)(?!\s+(?i:as\s+to\s+form))"
    r"(?:\s+(?i:by)\b\s*[:\-]?|\s*[:\-])\s*[A-Z][\w.'\-]*\.?\s+[A-Z]"
)
STATUS_RE = re.compile(r"\b(?:approved|executed|rejected|void|pending)\b(?!\s+as\s+to\s+form)", re.I)


def normalize_field(name: str, raw: Any) -> FieldResult:
    """raw is the model's {"value","evidence","self_confidence"} object (or a bare value)."""
    if not isinstance(raw, dict):
        raw = {"value": raw}
    value, flags = normalize_value(name, raw.get("value"))
    evidence = raw.get("evidence")
    evidence = str(evidence).strip() if evidence not in (None, "") else None
    mc = raw.get("self_confidence")
    try:
        mc = float(mc) if mc is not None else None
        if mc is not None:
            mc = min(max(mc, 0.0), 1.0)
    except (TypeError, ValueError):
        mc = None
    if value is not None and evidence is None:
        flags.append("error:no_evidence")
    return FieldResult(value=value, evidence=evidence, model_confidence=mc, flags=flags)


def build_line_items(raw_items: Any) -> list[LineItem]:
    items: list[LineItem] = []
    if not isinstance(raw_items, list):
        return items
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        amount, _ = parse_amount(it.get("amount"))
        items.append(
            LineItem(
                description=str(it.get("description") or "").strip(),
                amount=amount,
                evidence=(str(it["evidence"]).strip() if it.get("evidence") else None),
            )
        )
    return items


def validate_change_order(fields: dict[str, FieldResult], line_items: list[LineItem]) -> list[str]:
    """Mutates field flags; returns human-readable reasons for the review queue."""
    reasons: list[str] = []

    # Date plausibility (corpus includes old documents, so only sanity-bound it)
    d = fields["date"].value
    if d:
        year = int(d[:4])
        if year < 1950 or year > date.today().year + 2:
            fields["date"].flags.append("warn:date_out_of_range")
            reasons.append(f"date {d} looks implausible")

    st = fields["status"].value
    if st and st not in ALLOWED_STATUS:
        fields["status"].flags.append("warn:unknown_status")

    cost: Optional[float] = fields["cost_change"].value

    # Line items should add up to the stated change
    priced = [li.amount for li in line_items if li.amount is not None]
    if cost is not None and priced:
        total = round(sum(priced), 2)
        if abs(total - cost) > TOL:
            fields["cost_change"].flags.append("error:line_items_sum_mismatch")
            reasons.append(
                f"line items sum to {total:,.2f} but cost_change is {cost:,.2f} "
                "(document may be inconsistent, or an item was missed/misread)"
            )

    # Contract-sum arithmetic, only when the document states all three numbers
    before = fields["contract_sum_before_change"].value
    new = fields["new_contract_sum"].value
    if cost is not None and before is not None and new is not None:
        if abs(before + cost - new) > TOL:
            fields["new_contract_sum"].flags.append("warn:sum_does_not_reconcile")
            fields["cost_change"].flags.append("warn:sum_does_not_reconcile")
            reasons.append(
                f"contract sum before ({before:,.2f}) + change ({cost:,.2f}) != new sum ({new:,.2f})"
            )

    # Sign sanity: credit/deduct language but positive amount
    if cost is not None and cost > 0:
        context = " ".join(
            filter(None, [fields["cost_change"].evidence, fields["description"].value])
        )
        if CREDIT_WORDS.search(context):
            fields["cost_change"].flags.append("warn:sign_suspect")
            reasons.append("text mentions credit/deduct/decrease but cost_change is positive")

    ab = fields["approved_by"]
    if ab.value is not None and not APPROVAL_RE.search(ab.evidence or ""):
        ab.flags.append("error:approval_not_evidenced")
        reasons.append("approved_by: the quoted text is not an approval statement (a form label or a blank signature block is not an approval)")
    stf = fields["status"]
    if stf.value is not None and not STATUS_RE.search(stf.evidence or ""):
        stf.flags.append("error:status_not_evidenced")
        reasons.append("status: the quoted text does not state a status")

    for name in FIELDS:
        for f in fields[name].flags:
            if f.startswith("warn:ambiguous_date"):
                reasons.append("date written numerically; month-first (US) order was assumed")
    return reasons
