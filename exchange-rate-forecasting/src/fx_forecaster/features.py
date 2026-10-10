from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ForecastConfig


def _safe_log_change(series: pd.Series, periods: int) -> pd.Series:
    s = series.astype(float)
    valid = (s > 0) & (s.shift(periods) > 0)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[valid] = np.log(s.loc[valid]) - np.log(s.shift(periods).loc[valid])
    return out


def _add_calendar_features(out: pd.DataFrame) -> None:
    out["date_month"] = out["date"].dt.month.astype(float)
    out["date_quarter"] = out["date"].dt.quarter.astype(float)
    out["date_year"] = out["date"].dt.year.astype(float)
    # Cyclic month encoding.
    out["date_month_sin"] = np.sin(2 * np.pi * out["date_month"] / 12.0)
    out["date_month_cos"] = np.cos(2 * np.pi * out["date_month"] / 12.0)


def build_feature_frame(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Create model features and an h-step future log-return target.

    The function assumes each row represents information available at that row's
    forecast date. Real-world publication lags must therefore be handled upstream.
    """
    out = df.copy()
    _add_calendar_features(out)

    numeric_cols = [
        c for c in df.columns
        if c not in config.excluded_columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Exchange-rate dynamics.
    for k in config.change_windows:
        out[f"exchange_rate_logchg_{k}"] = _safe_log_change(df["exchange_rate"], k)

    # Generic transformations for available predictors.
    for col in numeric_cols:
        if col == "exchange_rate":
            continue
        s = df[col].astype(float)
        for k in config.change_windows:
            if col in config.difference_columns or (s.dropna() <= 0).any():
                out[f"{col}_diff_{k}"] = s - s.shift(k)
            else:
                out[f"{col}_logchg_{k}"] = _safe_log_change(s, k)

    # Economically interpretable composite features.
    if {"exports", "imports"}.issubset(df.columns):
        out["trade_balance"] = df["exports"] - df["imports"]
        out["trade_ratio"] = df["exports"] / df["imports"].replace(0, np.nan)

    if {"cpi", "partner_cpi"}.issubset(df.columns):
        out["inflation_diff"] = _safe_log_change(df["cpi"], 1) - _safe_log_change(df["partner_cpi"], 1)

    if {"policy_rate", "partner_policy_rate"}.issubset(df.columns):
        out["policy_rate_diff"] = df["policy_rate"] - df["partner_policy_rate"]

    if {"fx_reserves", "imports"}.issubset(df.columns):
        out["reserve_import_cover_proxy"] = df["fx_reserves"] / df["imports"].replace(0, np.nan)

    # Rolling behavior of the 1-period exchange-rate return.
    base_ret = out["exchange_rate_logchg_1"]
    for w in config.rolling_windows:
        out[f"fx_return_rollmean_{w}"] = base_ret.rolling(w).mean()
        out[f"fx_return_rollstd_{w}"] = base_ret.rolling(w).std()

    # Lag all model-eligible numeric engineered features. We retain current values too,
    # because a row is defined as an as-of information set. Users must lag unreleased
    # macro series upstream according to publication dates.
    lag_sources = [
        c for c in out.columns
        if c != "date"
        and pd.api.types.is_numeric_dtype(out[c])
        and c != "exchange_rate"
        and not c.startswith(config.feature_prefixes_to_skip_lagging)
    ]
    lagged = {
        f"{col}_lag_{lag}": out[col].shift(lag)
        for col in lag_sources
        for lag in config.lags
    }
    if lagged:
        out = pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1)

    # Forecast target and realized future level for evaluation.
    future_rate = df["exchange_rate"].shift(-config.horizon)
    targets = pd.DataFrame({
        "target_log_return": np.log(future_rate) - np.log(df["exchange_rate"]),
        "future_exchange_rate": future_rate,
    }, index=out.index)
    out = pd.concat([out, targets], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def select_model_columns(feature_frame: pd.DataFrame) -> list[str]:
    blocked = {"date", "exchange_rate", "target_log_return", "future_exchange_rate"}
    return [
        c for c in feature_frame.columns
        if c not in blocked and pd.api.types.is_numeric_dtype(feature_frame[c])
    ]
