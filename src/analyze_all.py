from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import SearchConfig
from .search import _resolve_artifact_dirs, load_artifacts, search


def _run_entity_search(
    entity_id: str,
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
    min_final_score: float,
    exclude_empty_plots: bool,
    include_relative_plots: bool,
) -> tuple[str, str | None]:
    try:
        search(
            entity_id=entity_id,
            top_k=top_k,
            artifact_dir=artifact_dir,
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
            min_final_score=min_final_score,
            exclude_empty_plots=exclude_empty_plots,
            include_relative_plots=include_relative_plots,
        )
        return entity_id, None
    except Exception as exc:
        return entity_id, str(exc)


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
    min_final_score: float,
    workers: int,
    exclude_empty_plots: bool,
    include_relative_plots: bool,
) -> None:
    for timeframe, resolved_artifact_dir in _resolve_artifact_dirs(artifact_dir):
        metadata, _ = load_artifacts(resolved_artifact_dir)
        entities = sorted({item["entity_id"] for item in metadata["windows"]})
        print(
            f"timeframe={timeframe} entities={len(entities)} "
            f"artifact_dir={resolved_artifact_dir} workers={workers}"
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _run_entity_search,
                    entity_id,
                    resolved_artifact_dir,
                    top_k,
                    horizons,
                    plot_dir,
                    only_past,
                    gap_candles,
                    calendar_gap_candles,
                    min_range_ratio,
                    max_range_ratio,
                    min_vol_ratio,
                    max_vol_ratio,
                    min_shape_score,
                    min_final_score,
                    exclude_empty_plots,
                    include_relative_plots,
                ): entity_id
                for entity_id in entities
            }

            completed = 0
            for future in as_completed(future_map):
                completed += 1
                entity_id = future_map[future]
                _, error = future.result()
                if error:
                    print(
                        f"timeframe={timeframe} progress={completed}/{len(entities)} "
                        f"entity={entity_id} status=error error={error}"
                    )
                else:
                    print(
                        f"timeframe={timeframe} progress={completed}/{len(entities)} "
                        f"entity={entity_id} status=done"
                    )


def main() -> None:
    search_cfg = SearchConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=search_cfg.artifact_dir)
    parser.add_argument("--top-k", type=int, default=search_cfg.top_k)
    parser.add_argument("--horizons", default=search_cfg.horizons_raw)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--only-past", action="store_true", default=search_cfg.only_past)
    parser.add_argument("--gap-candles", type=int, default=search_cfg.gap_candles)
    parser.add_argument("--calendar-gap-candles", type=int, default=search_cfg.calendar_gap_candles)
    parser.add_argument("--min-range-ratio", type=float, default=search_cfg.min_range_ratio)
    parser.add_argument("--max-range-ratio", type=float, default=search_cfg.max_range_ratio)
    parser.add_argument("--min-vol-ratio", type=float, default=search_cfg.min_vol_ratio)
    parser.add_argument("--max-vol-ratio", type=float, default=search_cfg.max_vol_ratio)
    parser.add_argument("--min-shape-score", type=float, default=search_cfg.min_shape_score)
    parser.add_argument("--min-final-score", type=float, default=search_cfg.min_final_score)
    parser.add_argument("--workers", type=int, default=search_cfg.analyze_all_workers)
    parser.add_argument("--exclude-empty-plots", action="store_true")
    parser.add_argument("--include-relative-plots", action="store_true")
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
        min_final_score=args.min_final_score,
        workers=args.workers,
        exclude_empty_plots=args.exclude_empty_plots,
        include_relative_plots=args.include_relative_plots,
    )


if __name__ == "__main__":
    main()
