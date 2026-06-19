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

from .datasets import KNOWN_DATASETS, label_distribution, load_labeled_examples
from .engine import DEFAULT_MODEL, EmotionResult, EmotiSenseEngine
from .evaluate import evaluate, format_report
from .sample_data import SAMPLE_ENTRIES
from .sample_labeled import LABELED_SAMPLES
from .visualize import export_csv

# Convenience aliases for the transformer to explore via --model.
KNOWN_MODELS = {
    "default": DEFAULT_MODEL,                              # Ekman-7 + neutral
    "goemotions": "SamLowe/roberta-base-go_emotions",     # 28 fine-grained labels
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",  # pos/neg
}


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
    parser.add_argument("--eval", dest="evaluate", action="store_true",
                        help="Measure the engine against the bundled labelled set "
                             "(precision/recall/F1 + confusion matrix).")
    parser.add_argument("--csv", help="Export results to this CSV path.")
    parser.add_argument("--no-api", action="store_true",
                        help="Force the keyword fallback (skip the Hugging Face API).")
    parser.add_argument("--model", default=None,
                        help="Transformer to use: a Hub id, or an alias "
                             f"({', '.join(KNOWN_MODELS)}).")
    # Data-bank exploration: pull real labelled data from the Hugging Face Hub.
    data = parser.add_argument_group("data bank (Hugging Face datasets)")
    data.add_argument("--dataset", help="Dataset to load: an alias "
                      f"({', '.join(KNOWN_DATASETS)}) or a raw 'owner/name' id.")
    data.add_argument("--limit", type=int, default=200,
                      help="Max examples to stream from --dataset (default 200).")
    data.add_argument("--offset", type=int, default=0,
                      help="Row offset to start streaming from (sample a different slice).")
    data.add_argument("--peek", type=int, metavar="N", default=0,
                      help="Print the first N loaded examples instead of evaluating.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    model = KNOWN_MODELS.get(args.model, args.model) if args.model else DEFAULT_MODEL
    engine = EmotiSenseEngine(use_api=not args.no_api, model=model)
    backend_note = ("Hugging Face API" if engine.use_api
                    else "keyword-based fallback (no HF_TOKEN set or --no-api given)")
    if args.model and engine.use_api:
        backend_note += f" [model: {model}]"

    # Data-bank mode: stream real labelled data from the Hugging Face Hub.
    if args.dataset:
        print(f"Loading up to {args.limit} examples from '{args.dataset}' "
              f"(offset {args.offset})...")
        examples = load_labeled_examples(
            args.dataset, limit=args.limit, offset=args.offset,
            token=engine.api_key,
        )
        if not examples:
            print("No usable examples returned (check dataset/config/columns).")
            return 1
        dist = label_distribution(examples)
        print(f"Loaded {len(examples)} examples. Label distribution:")
        for emotion, count in dist.most_common():
            print(f"    {emotion:<10} {count}")

        if args.peek:
            print(f"\nFirst {min(args.peek, len(examples))} examples:")
            for text, label in examples[:args.peek]:
                preview = text if len(text) <= 70 else text[:67] + "..."
                print(f"    [{label}] {preview}")
            return 0

        print(f"\nMeasuring engine ({backend_note}) on {len(examples)} real examples...\n")
        print(format_report(evaluate(engine, examples)))
        return 0

    # Measurement mode: observe what the engine does on the bundled labelled set.
    if args.evaluate:
        print(f"Measuring engine ({backend_note}) on {len(LABELED_SAMPLES)} labelled entries...\n")
        print(format_report(evaluate(engine, LABELED_SAMPLES)))
        return 0

    entries = _read_entries(args)
    if not entries:
        build_parser().print_help()
        return 1

    print(f"Using {backend_note}.")

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
