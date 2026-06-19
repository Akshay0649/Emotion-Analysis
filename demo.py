"""Standalone demo that reproduces the original notebook's test run.

Usage::

    python demo.py            # analyse sample entries, print summary, save CSV + plot

Set ``HF_TOKEN`` for API-backed analysis; otherwise the keyword fallback runs.
"""

from emotisense.engine import EmotiSenseEngine
from emotisense.sample_data import SAMPLE_ENTRIES
from emotisense.visualize import export_csv, visualize_results


def main() -> None:
    print("EmotiSense Engine demo")
    print("=" * 60)

    engine = EmotiSenseEngine(use_api=True)
    print("Mode:", "Hugging Face API" if engine.use_api else "keyword fallback")

    results = engine.batch_analyze(SAMPLE_ENTRIES)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.text[:60]}{'...' if len(r.text) > 60 else ''}")
        print(f"    {r.primary_emotion.upper()} ({r.confidence:.3f}) via {r.source}")

    high = sum(1 for r in results if r.confidence > 0.3)
    print("\n" + "=" * 40)
    print(f"Entries: {len(results)} | confidence>0.3: {high} "
          f"({high / len(results) * 100:.1f}%)")

    csv_path = export_csv(results)
    print(f"Saved CSV -> {csv_path}")

    try:
        visualize_results(results, save_path="emotisense_plots.png")
        print("Saved plot -> emotisense_plots.png")
    except Exception as exc:  # plotting is optional
        print(f"(Skipped plotting: {exc})")


if __name__ == "__main__":
    main()
