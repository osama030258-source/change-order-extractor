"""Field definitions, value normalisation and result models."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from dateutil import parser as dtparser
from pydantic import BaseModel, Field

FIELDS = [
    "change_order_number",
    "date",
    "project_name",
    "contractor",
    "description",
    "cost_change",
    "schedule_impact_days",
    "requested_by",
    "approved_by",
    "status",
    "original_contract_sum",
    "contract_sum_before_change",
    "new_contract_sum",
]
AMOUNT_FIELDS = {
    "cost_change",
    "original_contract_sum",
    "contract_sum_before_change",
    "new_contract_sum",
}
INT_FIELDS = {"schedule_impact_days"}
DATE_FIELDS = {"date"}
# Fields where a missing value means a human must look at the document.
CRITICAL_FIELDS = {"change_order_number", "cost_change"}
ALLOWED_STATUS = {"approved", "pending", "rejected", "executed", "void"}

_AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_INT_RE = re.compile(r"-?\d+")
_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$")

# Flags are strings "warn:<code>" or "error:<code>".


def parse_amount(raw: Any) -> tuple[Optional[float], list[str]]:
    """Parse money as written. Handles $, commas, (1,000), <1,000>, -$1,000."""
    if raw is None:
        return None, []
    if isinstance(raw, bool):
        return None, ["error:amount_unparseable"]
    if isinstance(raw, (int, float)):
        return round(float(raw), 2), []
    s = str(raw).strip().replace("−", "-").replace("–", "-")
    if not s:
        return None, []
    negative = bool(
        re.match(r"^\s*[\(<]", s) or re.match(r"^\s*-", s) or re.search(r"\$\s*-", s)
    )
    matches = _AMOUNT_RE.findall(s)
    if not matches:
        return None, ["error:amount_unparseable"]
    flags: list[str] = []
    if len(matches) > 1:
        flags.append("warn:amount_multiple_numbers")
    value = float(matches[0].replace(",", ""))
    return (round(-value if negative else value, 2)), flags


def parse_days(raw: Any) -> tuple[Optional[int], list[str]]:
    if raw is None:
        return None, []
    if isinstance(raw, bool):
        return None, ["error:days_unparseable"]
    if isinstance(raw, (int, float)):
        return int(raw), []
    s = str(raw).replace("−", "-")
    if not s.strip():
        return None, []
    m = _INT_RE.search(s)
    if not m:
        return None, ["error:days_unparseable"]
    return int(m.group()), []


def parse_date(raw: Any) -> tuple[Optional[str], list[str]]:
    """Return ISO date or None. Refuses incomplete dates; flags d/m ambiguity."""
    if raw is None:
        return None, []
    s = str(raw).strip()
    if not s:
        return None, []
    flags: list[str] = []
    m = _NUMERIC_DATE.match(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= 12 and b <= 12 and a != b:
            flags.append("warn:ambiguous_date_format")
    try:
        d1 = dtparser.parse(s, dayfirst=False, default=datetime(1900, 1, 1))
        d2 = dtparser.parse(s, dayfirst=False, default=datetime(1901, 2, 2))
    except (ValueError, OverflowError):
        return None, ["error:date_unparseable"]
    if d1 != d2:  # some component (day/month/year) came from the default
        return None, ["error:date_incomplete"]
    return d1.date().isoformat(), flags


_CO_WORD = re.compile(r"^(?:change\s+order(?:\s+request)?|contract\s+modification|modification|cor|co)\b[\s.]*", re.I)
_CO_LABEL = re.compile(r"^(?:no\b\.?|number\b|num\b\.?|#)\s*[:\-]?\s*", re.I)


def clean_co_number(raw: Any) -> str:
    """'CO #5R' / 'Change Order No. 5R' / 'No.: 5R' -> '5R'. Keeps the original if nothing is left."""
    s = str(raw).strip()
    t = _CO_WORD.sub("", s, count=1)
    t = _CO_LABEL.sub("", t, count=1)
    t = re.sub(r"^[#:\-\s]+", "", t).strip()
    return t or s


def normalize_value(field: str, raw: Any) -> tuple[Any, list[str]]:
    if raw is None:
        return None, []
    if field in AMOUNT_FIELDS:
        return parse_amount(raw)
    if field in INT_FIELDS:
        return parse_days(raw)
    if field in DATE_FIELDS:
        return parse_date(raw)
    s = str(raw).strip()
    if not s or s.lower() in {"null", "none", "n/a"}:
        return None, []
    if field == "change_order_number":
        return clean_co_number(s), []
    if field == "status":
        return s.lower(), []
    return s, []


class FieldResult(BaseModel):
    value: Any = None
    evidence: Optional[str] = None
    model_confidence: Optional[float] = None
    confidence: Optional[float] = None
    flags: list[str] = Field(default_factory=list)


class LineItem(BaseModel):
    description: str = ""
    amount: Optional[float] = None
    evidence: Optional[str] = None


class ChangeOrder(BaseModel):
    fields: dict[str, FieldResult]
    line_items: list[LineItem] = Field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    def values(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.fields.items()}


class DocumentResult(BaseModel):
    source: str
    change_orders: list[ChangeOrder] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
