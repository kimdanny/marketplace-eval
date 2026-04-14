"""
CLI entry point for post-simulation analysis.

Usage:
    python -m post_simulation.cli <log_path> [options]

Examples:
    python -m post_simulation.cli results/simulation_log.csv --output-dir results
    python -m post_simulation.cli results/simulation_log.csv --window-interval 10
    python -m post_simulation.cli results/simulation_log.csv --window-interval 10 --config configs/simple_qa_simulation.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from marketplace_eval.post_simulation.market_share import (
    compute_market_share,
    compute_market_share_windows,
    get_windows_from_interval,
    load_generator_introduce_from,
    parse_windows,
)
from marketplace_eval.post_simulation.plot import (
    plot_market_share,
    plot_market_share_windows,
    plot_market_share_windows_stacked,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot market share from simulation logs"
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to the simulation log CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save outputs (default: results)",
    )
    parser.add_argument(
        "--window-interval",
        type=int,
        default=None,
        metavar="N",
        help="Window size in steps (e.g. 10 -> windows 1-10, 11-20, ...)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default=None,
        help='Comma-separated windows in 1-based display form, e.g. "1-100,101-200"',
    )
    parser.add_argument(
        "--t-start",
        type=int,
        default=None,
        help="Start timestep for cumulative market share (inclusive)",
    )
    parser.add_argument(
        "--t-end",
        type=int,
        default=None,
        help="End timestep for cumulative market share (inclusive)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Simulation config (YAML/JSON) to read generator introduce_from; "
        "if set, plot lines only from each generator's introduce_from timestep onward",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_list = parse_windows(args.windows) if args.windows else None
    if args.window_interval is not None:
        windows_list = get_windows_from_interval(args.log_path, args.window_interval)

    generator_introduce_from = None
    if args.config:
        generator_introduce_from = load_generator_introduce_from(args.config)

    market_share = compute_market_share(
        args.log_path, t_start=args.t_start, t_end=args.t_end
    )
    if not market_share.empty:
        csv_path = output_dir / "market_share.csv"
        market_share.to_csv(csv_path)
        print(f"Market share saved to: {csv_path}")
        plot_market_share(
            market_share,
            save_path=output_dir / "market_share.png",
            generator_introduce_from=generator_introduce_from,
        )

    if windows_list:
        ms_windows = compute_market_share_windows(args.log_path, windows_list)
        if not ms_windows.empty:
            ms_windows.to_csv(output_dir / "market_share_windows.csv")
            print(
                f"Windowed market share saved to: {output_dir / 'market_share_windows.csv'}"
            )
            plot_market_share_windows(
                ms_windows,
                save_path=output_dir / "market_share_windows.png",
                generator_introduce_from=generator_introduce_from,
            )
            plot_market_share_windows_stacked(
                ms_windows,
                save_path=output_dir / "market_share_windows_stacked.png",
                generator_introduce_from=generator_introduce_from,
            )


if __name__ == "__main__":
    main()
