"""Command-line interface for EmotiSense.

Examples
--------
Analyse a single string::

    python -m emotisense "I am so happy today!"

Analyse a file (one entry per line) and export a CSV::

    python -m emotisense --file entries.txt --csv out.csv

Run the bundled demo with sample journal entries::

    python -m emotisense --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from .engine import EmotionResult, EmotiSenseEngine
from .sample_data import SAMPLE_ENTRIES
from .visualize import export_csv


def _read_entries(args: argparse.Namespace) -> List[str]:
    if args.demo:
        return list(SAMPLE_ENTRIES)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    if args.text:
        return [" ".join(args.text)]
    # Fall back to stdin if data is piped in.
    if not sys.stdin.isatty():
        return [line.strip() for line in sys.stdin if line.strip()]
    return []


def _print_result(index: int, result: EmotionResult) -> None:
    preview = result.text if len(result.text) <= 70 else result.text[:67] + "..."
    print(f"\n[{index}] {preview}")
    print(f"    Primary: {result.primary_emotion.upper()} "
          f"({result.confidence:.3f}) via {result.source}")
    tops = " | ".join(f"{k}: {v:.2f}" for k, v in result.top_emotions())
    if tops:
        print(f"    Top: {tops}")


def _summary(results: List[EmotionResult]) -> None:
    total = len(results)
    high = sum(1 for r in results if r.confidence > 0.3)
    very_high = sum(1 for r in results if r.confidence > 0.5)
    rate = (high / total * 100) if total else 0.0
    print("\n" + "=" * 40)
    print(f"Total entries:            {total}")
    print(f"Confidence > 0.3:         {high} ({rate:.1f}%)")
    print(f"Confidence > 0.5:         {very_high} ({(very_high/total*100) if total else 0:.1f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emotisense", description="Analyse the emotional tone of text."
    )
    parser.add_argument("text", nargs="*", help="Text to analyse (quote multi-word input).")
    parser.add_argument("--file", "-f", help="Path to a file with one entry per line.")
    parser.add_argument("--demo", action="store_true", help="Run with bundled sample entries.")
    parser.add_argument("--csv", help="Export results to this CSV path.")
    parser.add_argument("--no-api", action="store_true",
                        help="Force the keyword fallback (skip the Hugging Face API).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    entries = _read_entries(args)
    if not entries:
        build_parser().print_help()
        return 1

    engine = EmotiSenseEngine(use_api=not args.no_api)
    if engine.use_api:
        print("Using Hugging Face API for analysis.")
    else:
        print("Using keyword-based fallback (no HF_TOKEN set or --no-api given).")

    results = engine.batch_analyze(entries)
    for i, result in enumerate(results, 1):
        _print_result(i, result)
    _summary(results)

    if args.csv:
        path = export_csv(results, args.csv)
        print(f"\nSaved results to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
