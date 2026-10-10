from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 1
    lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    change_windows: tuple[int, ...] = (1, 3, 12)
    rolling_windows: tuple[int, ...] = (3, 6, 12)
    n_splits: int = 5
    random_state: int = 42
    min_train_rows: int = 36
    interval_quantiles: tuple[float, float] = (0.05, 0.95)
    # Columns where ordinary differences are preferable to log differences.
    difference_columns: tuple[str, ...] = (
        "policy_rate",
        "partner_policy_rate",
        "current_account",
        "budget_balance",
        "portfolio_flows",
        "fdi",
    )
    # Known source names that are not predictors.
    excluded_columns: tuple[str, ...] = ("date",)
    feature_prefixes_to_skip_lagging: tuple[str, ...] = ("date_",)
    metadata: dict[str, str] = field(default_factory=dict)
