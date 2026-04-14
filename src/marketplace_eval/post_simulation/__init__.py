"""
post_simulation — metrics and plotting utilities for marketplace simulation analysis.

Metric functions
----------------
Market share:
    from marketplace_eval.post_simulation.market_share import (
        compute_market_share,
        compute_market_share_windows,
        get_windows_from_interval,
        load_generator_introduce_from,
        parse_windows,
    )

Customer retention rate:
    from marketplace_eval.post_simulation.crr import (
        compute_crr_windowed,
        compute_crr_report,
    )

Plotting
--------
    from marketplace_eval.post_simulation.plot import (
        plot_market_share,
        plot_market_share_windows,
        plot_market_share_windows_stacked,
    )
"""

from marketplace_eval.post_simulation.market_share import (
    compute_market_share,
    compute_market_share_windows,
    get_windows_from_interval,
    load_generator_introduce_from,
    parse_windows,
)
from marketplace_eval.post_simulation.crr import (
    compute_crr_windowed,
    compute_crr_report,
)
from marketplace_eval.post_simulation.plot import (
    plot_market_share,
    plot_market_share_windows,
    plot_market_share_windows_stacked,
)

__all__ = [
    # market share metrics
    "compute_market_share",
    "compute_market_share_windows",
    "get_windows_from_interval",
    "load_generator_introduce_from",
    "parse_windows",
    # crr metrics
    "compute_crr_windowed",
    "compute_crr_report",
    # plotting
    "plot_market_share",
    "plot_market_share_windows",
    "plot_market_share_windows_stacked",
]
