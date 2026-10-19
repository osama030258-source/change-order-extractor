"""Command line: `co-extract extract FILE` and `co-extract eval`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate, runs_summary, to_markdown
from .llm import LLMError, get_llm
from .pipeline import DEFAULT_THRESHOLD, run_pipeline


def _print_result(res) -> None:
    for w in res.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for i, co in enumerate(res.change_orders, 1):
        tag = "NEEDS REVIEW" if co.needs_review else "ok"
        print(f"\n=== change order {i}: {co.fields['change_order_number'].value!r} [{tag}] ===")
        for name, f in co.fields.items():
            if f.value is None:
                continue
            flags = f"  {','.join(f.flags)}" if f.flags else ""
            print(f"  {name:28s} {str(f.value)[:60]:60s} conf={f.confidence}{flags}")
        for r in co.review_reasons:
            print(f"  ! {r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="co-extract")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract change orders from one PDF/text file")
    e.add_argument("file")
    e.add_argument("--provider", choices=["groq", "anthropic"])
    e.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    e.add_argument("--max-chars", type=int, default=12000, help="max characters per LLM call; smaller = fewer change orders per call")
    e.add_argument("--out", default=None, help="write JSON here (default: out/<name>.json)")

    v = sub.add_parser("eval", help="score against data/labels")
    v.add_argument("--samples", default="data/samples")
    v.add_argument("--labels", default="data/labels")
    v.add_argument("--provider", choices=["groq", "anthropic"])
    v.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    v.add_argument("--out", default="out")
    v.add_argument("--runs", type=int, default=1, help="repeat the whole eval N times (model output varies between runs)")

    args = p.parse_args(argv)
    try:
        llm = get_llm(args.provider)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "extract":
        res = run_pipeline(args.file, llm, args.threshold, args.max_chars)
        out = Path(args.out) if args.out else Path("out") / (Path(args.file).stem + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res.model_dump_json(indent=2), encoding="utf-8")
        _print_result(res)
        print(f"\nJSON written to {out}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_runs = max(1, args.runs)
    reps = []
    for k in range(n_runs):
        rep = evaluate(Path(args.samples), Path(args.labels), llm, args.threshold)
        reps.append(rep)
        if n_runs > 1:
            (out_dir / f"eval_run{k + 1}.md").write_text(to_markdown(rep), encoding="utf-8")
            (out_dir / f"eval_run{k + 1}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
            print(f"run {k + 1}/{n_runs} done", file=sys.stderr)
    rep = reps[-1]
    (out_dir / "eval_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = to_markdown(rep)
    (out_dir / "eval_report.md").write_text(md, encoding="utf-8")
    if n_runs > 1:
        summary = runs_summary(reps)
        (out_dir / "eval_summary.md").write_text(summary, encoding="utf-8")
        print(summary)
        print(f"\nPer-run reports: {out_dir}/eval_run1.md ... eval_run{n_runs}.md ; summary: {out_dir}/eval_summary.md")
    else:
        print(md)
        print(f"\nReports written to {out_dir}/eval_report.md and eval_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())