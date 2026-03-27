from __future__ import annotations

import argparse

from .search import _resolve_artifact_dirs, load_artifacts, search


def analyze_all(
    artifact_dir: str,
    top_k: int,
    horizons: list[int],
    plot_dir: str | None,
    only_past: bool,
    gap_candles: int,
    calendar_gap_candles: int,
    min_range_ratio: float,
    max_range_ratio: float,
    min_vol_ratio: float,
    max_vol_ratio: float,
    min_shape_score: float,
) -> None:
    for timeframe, resolved_artifact_dir in _resolve_artifact_dirs(artifact_dir):
        metadata, _ = load_artifacts(resolved_artifact_dir)
        entities = sorted({item["entity_id"] for item in metadata["windows"]})
        print(f"timeframe={timeframe} entities={len(entities)} artifact_dir={resolved_artifact_dir}")
        for idx, entity_id in enumerate(entities, start=1):
            print(f"timeframe={timeframe} progress={idx}/{len(entities)} entity={entity_id}")
            search(
                entity_id=entity_id,
                top_k=top_k,
                artifact_dir=resolved_artifact_dir,
                horizons=horizons,
                plot_dir=plot_dir,
                only_past=only_past,
                gap_candles=gap_candles,
                calendar_gap_candles=calendar_gap_candles,
                min_range_ratio=min_range_ratio,
                max_range_ratio=max_range_ratio,
                min_vol_ratio=min_vol_ratio,
                max_vol_ratio=max_vol_ratio,
                min_shape_score=min_shape_score,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--only-past", action="store_true")
    parser.add_argument("--gap-candles", type=int, default=0)
    parser.add_argument("--calendar-gap-candles", type=int, default=0)
    parser.add_argument("--min-range-ratio", type=float, default=0.35)
    parser.add_argument("--max-range-ratio", type=float, default=3.0)
    parser.add_argument("--min-vol-ratio", type=float, default=0.35)
    parser.add_argument("--max-vol-ratio", type=float, default=3.0)
    parser.add_argument("--min-shape-score", type=float, default=0.55)
    args = parser.parse_args()

    analyze_all(
        artifact_dir=args.artifact_dir,
        top_k=args.top_k,
        horizons=[int(part) for part in args.horizons.split(",") if part.strip()],
        plot_dir=args.plot_dir,
        only_past=args.only_past,
        gap_candles=args.gap_candles,
        calendar_gap_candles=args.calendar_gap_candles,
        min_range_ratio=args.min_range_ratio,
        max_range_ratio=args.max_range_ratio,
        min_vol_ratio=args.min_vol_ratio,
        max_vol_ratio=args.max_vol_ratio,
        min_shape_score=args.min_shape_score,
    )


if __name__ == "__main__":
    main()
