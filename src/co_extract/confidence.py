"""Per-field confidence from evidence, not from the model's say-so.

score = weighted mean of
  grounding   (0.5)  is the model's evidence quote actually in the source text?
  value_check (0.3)  does the normalised value appear in that evidence?
  model       (0.2)  the model's own self-reported confidence (input only)
If grounding is below 0.5 the score is capped at the grounding score (a
fabricated quote can't be rescued by a confident model). Finally it is
multiplied by 0.8 per "warn:" flag and 0.5 per "error:" flag.

The weights are initial heuristics. They have NOT been tuned against data;
see WRITEUP.md for how to calibrate them with eval results.
"""
from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz

from .ingest import PAGE_MARK
from .schema import AMOUNT_FIELDS, DATE_FIELDS, FIELDS, INT_FIELDS, FieldResult, parse_date

W_GROUND, W_VALUE, W_MODEL = 0.5, 0.3, 0.2
WARN_FACTOR, ERROR_FACTOR = 0.8, 0.5
GROUND_GATE = 0.5  # below this grounding, confidence is capped at the grounding score

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DATE_CAND = re.compile(
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}"
)
# Text fields whose value should literally appear in the evidence quote
VERBATIM_TEXT = {"change_order_number", "project_name", "contractor", "requested_by", "approved_by", "status"}


def norm(s: str) -> str:
    s = PAGE_MARK.sub(" ", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def grounding_score(evidence: Optional[str], source_norm: str) -> float:
    if not evidence:
        return 0.0
    e = norm(evidence)
    if not e:
        return 0.0
    if e in source_norm:
        return 1.0
    ratio = fuzz.partial_ratio(e, source_norm) / 100.0
    # tolerate OCR/whitespace noise, but cap so fuzzy never equals exact
    return min(ratio, 0.9) if ratio >= 0.85 else ratio * 0.5


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def value_check(name: str, value, evidence: Optional[str]) -> Optional[float]:
    """1.0 consistent, 0.0 inconsistent, None = not applicable."""
    if value is None:
        return None
    ev = evidence or ""
    if name in AMOUNT_FIELDS:
        target = abs(float(value))
        return 1.0 if any(abs(n - target) < 0.005 for n in _numbers(ev)) else 0.0
    if name in INT_FIELDS:
        v = int(value)
        if any(int(n) == abs(v) for n in _numbers(ev) if n == int(n)):
            return 1.0
        if v == 0 and re.search(r"unchanged|no change|none|\b0\b", ev, re.I):
            return 1.0
        return 0.0
    if name in DATE_FIELDS:
        for cand in _DATE_CAND.findall(ev):
            iso, _ = parse_date(cand)
            if iso == value:
                return 1.0
        return 0.0
    if name in VERBATIM_TEXT:
        return 1.0 if fuzz.partial_ratio(norm(str(value)), norm(ev)) >= 85 else 0.0
    return None  # description: the model is allowed to paraphrase


def score_field(name: str, f: FieldResult, source_norm: str) -> Optional[float]:
    if f.value is None:
        return None
    ground = grounding_score(f.evidence, source_norm)
    parts = [(W_GROUND, ground)]
    vc = value_check(name, f.value, f.evidence)
    if vc is not None:
        parts.append((W_VALUE, vc))
    if f.model_confidence is not None:
        parts.append((W_MODEL, f.model_confidence))
    score = sum(w * s for w, s in parts) / sum(w for w, _ in parts)
    # Gate: a value whose supporting quote is not in the document cannot be
    # trusted more than the quote itself, however confident the model sounds.
    if ground < GROUND_GATE:
        score = min(score, ground)
    for flag in f.flags:
        if flag.startswith("error:"):
            score *= ERROR_FACTOR
        elif flag.startswith("warn:"):
            score *= WARN_FACTOR
    return round(max(0.0, min(1.0, score)), 3)


def apply_confidence(fields: dict[str, FieldResult], source_text: str) -> None:
    source_norm = norm(source_text)
    for name in FIELDS:
        fields[name].confidence = score_field(name, fields[name], source_norm)
