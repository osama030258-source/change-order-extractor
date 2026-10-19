# Change-order extractor

Turns messy construction change orders (PDF or text) into **validated JSON with a confidence score on every field**, and routes anything doubtful to a human review queue instead of guessing.

A change order is the document that records a mid-project change to the contract (extra scope, a price change, a schedule slip). Every owner and contractor formats them differently, and the key numbers are easy to misread by hand.

## How it works

```
PDF / .txt ──ingest──> page-tagged text ──chunk──> LLM (Groq or Claude)
   │  (OCR fallback for scanned pages,        │  value + verbatim evidence quote per field
   │   missing text is a warning, not silent) ▼
   │                                    normalise (money, dates, days)
   │                                           ▼
   │                                    cross-field validation
   │                                    (line items vs total, sign, sums, date format)
   │                                           ▼
   └──────────────────────────────────> confidence = evidence-based score
                                               ▼
                                   JSON + needs_review + review_reasons
```

Design choices that matter:

- **The model must quote its evidence.** Every non-null value comes with a verbatim snippet; we check the snippet really exists in the document. A value with an invented quote gets a low score no matter how sure the model sounds.
- **Missing is a valid answer.** The prompt forbids guessing; absent fields come back `null`, and the evaluator counts a non-null value for an absent field as a hallucination.
- **The code does the arithmetic, not the model.** Line items are summed and compared to the stated total in Python.
- **Failures are explicit.** A chunk the model could not answer (bad JSON after one retry) or a page with no text is recorded in `warnings`; it never silently disappears.
- **Completeness pass.** Models silently skip items, so after the first extraction the model is shown what it found and asked only for what is missing. Duplicates are merged by identifier + project name.
- **An approver must be an approval.** `approved_by` / `status` are flagged for review unless the quoted text contains an approval statement with a name. A form label ("Approved As to Form") or role titles under a blank signature line are not approvals.

Confidence formula and its limits are in `src/co_extract/confidence.py` and `WRITEUP.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # or: pip install -e .
cp .env.example .env                    # then put your key in .env
```

`.env` is git-ignored. **Never paste an API key into code, chats, issues or commits.**

Free option: a Groq key (`LLM_PROVIDER=groq`). To switch to Claude later, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`; nothing else changes. Model names are configurable in `.env` because providers rename models over time.

Scanned PDFs: also `pip install pytesseract pdf2image` and install the system packages `tesseract-ocr` and `poppler-utils`.

## Use

```bash
# one file -> out/<name>.json plus a readable summary
PYTHONPATH=src python -m co_extract.cli extract data/samples/glenwood_2025_cos.txt
# (after `pip install -e .` the command is simply: co-extract extract <file>)

# score against hand-made labels -> out/eval_report.md and out/eval_report.json
PYTHONPATH=src python -m co_extract.cli eval
# model output varies from run to run, so repeat it and compare (writes out/eval_summary.md):
PYTHONPATH=src python -m co_extract.cli eval --runs 3
```

Output shape (format illustration, not real values):

```json
{
  "source": "<path>",
  "warnings": [],
  "change_orders": [{
    "fields": {
      "cost_change": {"value": "<number>", "evidence": "<verbatim quote>",
                       "model_confidence": "<0-1>", "confidence": "<0-1>", "flags": []},
      "...": "one entry per field in src/co_extract/schema.py"
    },
    "line_items": [{"description": "<text>", "amount": "<number>", "evidence": "<quote>"}],
    "needs_review": true,
    "review_reasons": ["<why a human should look>"]
  }]
}
```

## Evaluate on your own documents

1. Download public change-order PDFs into `data/samples/` (see `data/SOURCES.md` for the ones used so far).
2. For each, create `data/labels/<same-name>.json` by reading the document yourself. Use `null` for fields the document does not state; omit a field to leave it unscored.
3. Run the `eval` command. The report lists accuracy per field, hallucination rate, and whether low confidence actually catches the wrong values.

## Tests

```bash
python -m pytest
```

The tests use a stub LLM with canned replies, so they check the plumbing (parsing, validation, scoring, routing). They say nothing about how accurate a real model is; only `eval` with a real provider does.

## Known limits

See "Known limitations" in `WRITEUP.md`. Short version: confidence weights are untuned, only three sample documents ship with the repo, and the OCR path needs a real scanned PDF to be exercised.
