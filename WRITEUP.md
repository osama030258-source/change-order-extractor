# Change-order extraction: approach and failure modes

> **Scope of the numbers.** Every number below comes from real runs of `co-extract eval` (model `groq:openai/gpt-oss-120b`, 2026-10-18) on **3 public documents / 6 hand-labeled change orders**. That is a small, regression-sized set, not a benchmark. One label (Glenwood `approved_by`) has not been checked against the original PDF; see section 6. Section 4c reports separate single runs on one real 13-page PDF (2026-10-18 and 2026-10-19), without hand-checked labels.

## 1. Problem

Change orders record mid-project changes to a construction contract. They arrive as PDFs (typed or scanned) in a different layout per owner, often several to a file, and the fields that matter (number, date, cost change, schedule impact, approver) sit in different places under different names. Hand-keying them is slow and a single misread digit has money attached. The goal is structured JSON where **every field says how much to trust it**, so people only review what is doubtful.

## 2. Approach

1. **Ingest.** `pdfplumber` text per page, tagged `[[PAGE n]]`. Pages with no text layer go to OCR if available; otherwise a warning is recorded so missing content is visible.
2. **Chunk.** Pages are grouped to a size limit with one page of overlap so a change order that straddles a boundary is seen whole; duplicates are merged afterwards.
3. **Extract with an LLM**, one JSON object per change order. For every field the model returns the value, a **verbatim evidence quote**, and a self-reported confidence. The prompt forbids guessing and says blank signature lines and form labels are not values.
4. **Normalise** in code: money (`$2,310.00`, `(12,136.90)`, `<2,500.00>`, `-$9,214.80`), dates (incomplete dates are rejected; numeric dates where day and month are both <= 12 are flagged as ambiguous), integers.
5. **Validate** across fields in code: line items must sum to the stated total; contract-sum arithmetic when all three numbers are stated; credit language with a positive amount; implausible dates.
6. **Score** each field and **route**: anything below the threshold, with an error flag, or missing a critical field (`change_order_number`, `cost_change`) goes to review with human-readable reasons.

## 3. Confidence scoring

Per field: `0.5 x grounding + 0.3 x value-check + 0.2 x model self-report` (weights renormalised when a component does not apply), capped at the grounding score if grounding < 0.5, then x0.8 per warning flag and x0.5 per error flag.

- *Grounding*: is the evidence quote actually in the document (whitespace/case-insensitive, with a capped fuzzy match for OCR noise)?
- *Value-check*: does the normalised value appear in that quote (the number in the amount, the date in the date text)?
- The model's self-reported confidence is only one input, because it is the least trustworthy signal.

**The weights (0.5 / 0.3 / 0.2) and the 0.7 threshold are the initial heuristics. They were not tuned; no run changed them.** What the runs show about the score is in section 4b: the generic score did not separate right from wrong values (wrong values scored 0.99-1.0 in the baseline), and what catches the wrong values in v2 is a specific rule (the approval-evidence check), not the weighted score. Calibrating the weights needs many more labeled documents than are available here.

## 4. Results

### 4a. Baseline: five real runs before the v2 fixes

Model `groq:openai/gpt-oss-120b`, 3 documents, 6 hand-labeled change orders (Anchorage x3, Everett x1, Glenwood x2), run on 2026-10-18. Same code, same documents, five separate runs. Source: terminal output of those runs.

| run | change orders found (of 6) | fields correct among the found ones | what happened |
|---|---|---|---|
| 1 | 5 | 56/60 (93%) | 1 change order missed (this report did not say which) |
| 2 | 6 | 66/72 (92%) | `original_contract_sum` null for one Anchorage CO |
| 3 | 4 | 45/48 (94%) | Anchorage CO 3 (Airport Heights) and CO 4 (Turnagain) not extracted at all, no warning |
| 4 | 6 | 65/72 (90%) | Everett: `approved_by` = "Mayor, City Clerk, City Attorney" and `status` = "approved" invented from a blank form block; `contract_sum_before_change` filled on both Glenwood change orders (label: null) |
| 5 | 3 | 34/36 (94%) | 2 of the 3 "misses" were my evaluator not understanding the model's "CO #5R"; Turnagain again not extracted |

Reading it correctly: accuracy among the change orders that were found is steady at 90-94%, but **how many were found swings from 3 to 6**, so the percentage alone hides the real problem. v2 reports an end-to-end figure where a missed change order counts all its fields as wrong.

Wrong non-null values that were flagged for review in the baseline: 0 of 2 (runs 1, 2 and 3), 0 of 4 (run 4), none to flag in run 5. Mean confidence of wrong values was 0.99-1.0, i.e. the baseline confidence did not separate right from wrong.

### 4b. After the v2 fixes: three runs

Same model, same 3 documents, same 6 labeled change orders; `co-extract eval --runs 3`, 2026-10-18. Source: the terminal output of that command (`out/eval_summary.md` and the per-run reports).

| run | change orders found (of 6) | fields correct (of 72) | wrong values flagged for review | invented values on null-labeled fields (of 30) | correct values flagged (false alarms) |
|---|---|---|---|---|---|
| 1 | 6 | 66 (92%) | 2 of 2 | 2 | 1 of 38 |
| 2 | 6 | 66 (92%) | no wrong non-null values | 0 | 1 of 36 |
| 3 | 6 | 66 (92%) | 3 of 3 | 3 | 1 of 39 |

Before vs after, same documents: change orders found per run went from 3, 4, 5, 6, 6 to 6, 6, 6; fields correct among the found ones stayed at 90-94% before and 92% after.

**The 18 wrong fields (6 per run), by kind:**

| kind | count | detail |
|---|---|---|
| `new_contract_sum` returned null | 9 | all three Anchorage change orders in every run; the "Total" row of the summary table is never taken as the new contract sum |
| other fields returned null | 4 | Turnagain `original_contract_sum` (run 1); Airport Heights `contractor`, `cost_change` and `original_contract_sum` (run 2: one change order extracted only partly) |
| `approved_by` filled where the label is null | 5 | typed names beside the signature lines on both Glenwood change orders in runs 1 and 3 (4), and Everett "Mayor" in run 3 (1); the Glenwood label is unverified (section 6) |

What this does and does not show:

- **Every value scored as wrong was flagged.** The 5 `approved_by` values (wrong only if the null labels are right; the Glenwood one is unverified) were all flagged (`error:approval_not_evidenced`, confidence about 0.5, versus about 0.98 for correct values). But all 5 are the same field, and the rule that catches them was written after seeing exactly this failure. The generic grounding/value/self-report score would not have caught them: in the baseline the same kind of value scored 0.98-1.0.
- **Nulls are not all routed to review.** A null `cost_change` or `change_order_number` sends the change order to review (unit-tested); a null `new_contract_sum` or `original_contract_sum` does not. Nothing false enters the JSON, but a downstream reader has to notice the blank.
- **False alarms: 1 per run (about 3% of correct values).** I did not identify which field from the summaries. The likely one is Glenwood CO #9 `cost_change` (0.00), flagged because its line item (-2,500.00) disagrees with the stated total, a real inconsistency in the source.
- **Run-to-run variance remains.** The invented-value count was 2, 0 and 3 on identical input, and run 2 extracted one change order only partly.
- **Optimistic by construction.** The v2 fixes were designed after seeing these same documents fail (section 6). Treat 4b as a regression check on 3 documents, not as accuracy on unseen documents.

### 4c. First run on a real multi-page PDF (Anchorage school board packet, 13 pages, 8 change orders)

Until this point only short `.txt` excerpts had been run. The full original PDF (`J02M273.pdf`, see `data/SOURCES.md`) was then downloaded and run through the real `pdfplumber` path with `groq:openai/gpt-oss-120b` (runs on 2026-10-18 and 2026-10-19). **Each row is a single run, not an average.** The reference values are the amounts I read from this PDF's text layer; they were not checked against a visual copy of the pages.

| Run | Chunk size | Change orders found (of 8) | `cost_change` correct | Notes |
|---|---|---|---|---|
| 1 | 12,000 chars | 3 | 3 of 3 found | Groq returned HTTP 400 `json_validate_failed` on the completeness pass; 5 change orders missing |
| 2 | 12,000 chars | 3 (a different 3) | 2 of 3 found | completeness pass returned no JSON; same 5 missing |
| 3 | 4,000 chars | 8 | 2 | `cost_change` null for 6, because summary page and detail page of one change order landed in different chunks and the merge discarded the detail record |
| 4 | 4,000 chars, merge fills empty fields | 8 | 7 | 1 null (Birchwood, 16,690.91), routed to review; 2 correct values (#7 Service HS, #4 Polaris) were also sent to review because the line items the model listed did not sum to the total |
| 5 | not recorded (CLI default on my side was 12,000) | 6 (Bartlett and Turnagain absent, no warning) | 6 of 6 found | all 6 values correct; Birchwood now found (targeted retry or luck, not separated); 3 of 6 sent to review |
| 6 | not recorded | 4 | 4 of 4 found | no warning for the 4 missing change orders; all 4 values correct but all 4 sent to review by the line-item check |
| 7 | probably 12,000 (the warning says "chunk 1/2") | 3 | 3 of 3 found | Groq HTTP 429 on the completeness pass: daily token limit (200,000 per organization) used up, so this run is not a fair measurement |

What this shows:
- Silent under-extraction on long inputs was a real failure: a bigger chunk gave 3 of 8, with no error besides the warning, and which 3 changed between runs. The fix was smaller chunks (now the `extract` default, `--max-chars 4000`).
- A merge bug (keeping one whole record and dropping the other) lost a critical field. It was found only because a real PDF split one change order across chunks; the short excerpts never did that.
- The review routing did its job on the 1 missing value. It also over-flagged 2 correct values (the model lists only some of the line items in long tables), which is the cost of the line-item check.
- The Groq JSON-mode 400 is now retried, then retried without JSON mode.
- Runs 5-7 are on the same PDF and the same fixes, yet found 6, 4 and 3 of 8. The chunk size was not recorded for them, so I cannot say how much of the swing is chunk size and how much is model variance; this is the same silent-omission failure as in 5a-2, only larger on a real 13-page file.
- The line-item sum check is a poor signal on long tables here: the model lists a different subset of line items each run (sums such as 45,993.98, 105,813.30, 105,448.81 for the same change order whose stated total is 142,370.12), so correct `cost_change` values are repeatedly sent to review. In runs 5-7 every `cost_change` that was found matched the reference value.
- Cost: one full run of this PDF used about 60,000 tokens by my estimate (8 or more calls of about 7,500 tokens). The free Groq tier allows 200,000 tokens per day per organization, i.e. about 3 runs, and run 7 hit that limit. The completeness pass runs on every chunk (it also catches chunks that return nothing), which roughly doubles the token use.
- These fixes were designed on this same PDF, so run 4 is not evidence of generalisation. The `eval` numbers in 4a/4b were produced at 12,000 characters per call on the short excerpts, and `eval` still uses that setting; the real PDF is not yet in `eval` because it has no hand-checked label file.

## 5. Failure modes

### 5a. Seen in the real sample documents (found by reading them)

| what the document does | why it is hard | what the pipeline does |
|---|---|---|
| Glenwood CO #09: one line item of `<2,500.00>` (a credit) but the form says the change is `$0.00` (the credit goes to a contingency fund) | line items and total disagree *in the source document* | line-item check raises an error flag, so the field goes to review with the numbers shown. It does not "fix" either number. |
| Glenwood CO #5R: contract-sum lines do not reconcile by simple addition (815,550.00 + 2,310.00 is not 826,669.50); contingency funds are mixed in | arithmetic checks cannot be trusted on forms like this | `contract_sum_before_change` is only filled when explicitly stated; reconciliation failure is a warning only. |
| Dates like `05/12/2025` | day/month order is not stated | month-first (US) assumed and flagged as ambiguous. |
| Anchorage packet: eight change orders in one file; a summary table on one page and the line items several pages later | a change order's fields are split across pages and chunks | chunk overlap plus de-duplication by (number, project); linking across distant pages is a known weak point. |
| "Contract Modification No. 4" instead of "Change Order No." | vocabulary varies by owner | the prompt names both; handled: that change order (Turnagain) was extracted and matched in all 3 v2 runs. |
| Negative amounts written as `($4,000.00)`, `-$4,000.00`, `<2,500.00>` | three notations for one idea | all normalised in code, unit-tested. |
| Everett page: heavy OCR noise (`Date: 213/69`), contractor and cost absent from that page | tempting to hallucinate a date or amount | absent fields must be `null`; invented values are counted as hallucinations in eval. |
| Signature lines that are blank in the text layer, with typed names beneath them (e.g. "Rodney L. Curtis") | a typed name looks like an approver, and a signature that is an image never appears in extracted text | prompt says blank signature lines are not values; labeled `approved_by: null`. The model did not reliably comply: in the 3 v2 runs it still returned the typed names for both Glenwood change orders in 2 of 3 runs. The approval-evidence check flagged all of them for review. |

### 5a-2. Observed in the five baseline runs (measured, not guessed)

- **Silent omissions.** The model sometimes returns fewer change orders than the text contains and nothing signals it. The Turnagain change order was absent in at least 2 of 5 baseline runs (a third run missed one change order without saying which); in run 3 two of the three Anchorage change orders were absent. v2 adds a completeness pass (second call: "here is what you found, return only what is missing"). With it, all 6 change orders were found in each of 3 runs (section 4b). Three runs on documents the fix was designed against show that it works there, not that it always will.
- **Form labels taken for values.** Everett's blank block "Approved As to Form: Mayor / City Clerk / City Attorney" produced `approved_by = "Mayor, City Clerk, City Attorney"` and `status = "approved"` with confidence 0.98. The evidence quote really is in the document, so the grounding check cannot catch it. v2 requires the quoted text to contain an approval statement with a name (e.g. "Approved by J. Miller") and otherwise flags the field for review. A check on the quote's wording is a heuristic; it will miss unusual phrasings.
- **Run-to-run variance.** Identical input gave different results across runs (see 4a). Single-run numbers are not reported; `co-extract eval --runs N` repeats the evaluation.
- **Number formats.** The model sometimes writes "CO #5R" and sometimes "5R". Left alone, this broke de-duplication and evaluation matching. v2 normalises the identifier.
- **Stated but left null.** Anchorage `new_contract_sum` (the "Total" row of the summary table) came back null in every run where that change order was found. The field comes back null rather than wrong, but a null in this field is not routed to review (only `change_order_number` and `cost_change` are critical). It was deliberately not "fixed" with a document-specific prompt hint, which would only tune the score to these three documents.

### 5b. Expected, not yet measured

- Scanned documents with handwriting or skew (the OCR path is implemented but needs a real scanned PDF to exercise).
- Tables that break across pages; two change orders on one page.
- The model confusing architect and contractor, or the original contract sum with a revised one.
- Free-tier rate limits and JSON-mode failures on long files: both occurred on the real PDF (HTTP 429 daily limit, HTTP 400 `json_validate_failed`). Retries and recorded warnings handle them, but a run that hits the daily limit returns an incomplete result with a warning, not a full one.
- Amounts in text form ("twelve thousand dollars") or with currency other than USD.

## 6. Known limitations of this submission

- The test suite uses a stub LLM; it verifies plumbing, not accuracy.
- Only one real multi-page PDF has been run (section 4c, single runs, no hand-checked label file). The OCR path for scanned PDFs has still not been run on a real scanned document.
- Labels were made by reading text extracted by a web-fetch tool, not the original PDFs; they should be re-checked against the originals (see `data/SOURCES.md`).
- Three sample documents (six change orders) is far too few for statistical claims; the eval report states counts so results are not over-read.
- Confidence weights and the 0.7 threshold are untuned heuristics (section 3).
- Only `change_order_number` and `cost_change` are treated as critical. A null in any other field (for example `new_contract_sum`) is visible in the JSON but does not send the change order to review.
- The model is a hosted LLM whose output varied between identical runs; results are reported per run, never as one number.
- **The v2 fixes were designed after seeing failures on these same three documents, so v2 numbers measured on them are optimistic.** A fair test needs documents that were never used while fixing. Until then, treat 4b as a regression check, not as proof of generalisation.
- The `approved_by` label for Glenwood is `null` (blank signature lines in the text layer), but the original PDF may carry signatures or stamps that the text layer does not show. The packet's own bookmark for CO #5R is titled "...Rod and cornerstone signed", which suggests that page may be signed (CO #9's bookmark has no such note). I could not open the PDF to confirm. Verify against the original before calling anything a hallucination.
- Only the fields in `schema.py` are extracted; attachments, markups and drawings are ignored.