from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .config import ForecastConfig


@dataclass
class BacktestResult:
    metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    ensemble_weights: dict[str, float]
    residual_quantiles: tuple[float, float]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, base_rate: np.ndarray, future_rate: np.ndarray) -> dict[str, float]:
    pred_level = base_rate * np.exp(y_pred)
    valid_level = np.isfinite(future_rate) & np.isfinite(pred_level)
    directional = np.mean(np.sign(y_true) == np.sign(y_pred)) if len(y_true) else np.nan
    return {
        "return_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "return_mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_accuracy": float(directional),
        "level_rmse": float(np.sqrt(mean_squared_error(future_rate[valid_level], pred_level[valid_level]))) if valid_level.any() else np.nan,
        "level_mae": float(mean_absolute_error(future_rate[valid_level], pred_level[valid_level])) if valid_level.any() else np.nan,
    }


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    base_rate: pd.Series,
    future_rate: pd.Series,
    models: dict[str, object],
    config: ForecastConfig,
) -> BacktestResult:
    n = len(X)
    if n < max(config.min_train_rows + 8, config.n_splits + 2):
        raise ValueError(
            f"Not enough usable rows ({n}). Need at least about {max(config.min_train_rows + 8, config.n_splits + 2)}."
        )

    # TimeSeriesSplit automatically keeps test observations after train observations.
    n_splits = min(config.n_splits, max(2, n // 12))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=config.horizon)

    pred_store = {name: np.full(n, np.nan, dtype=float) for name in models}
    fold_store = np.full(n, np.nan)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        # Skip folds with too little training history.
        if len(train_idx) < config.min_train_rows:
            continue
        for name, model in models.items():
            fitted = clone(model) if hasattr(model, "get_params") else type(model)()
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred_store[name][test_idx] = fitted.predict(X.iloc[test_idx])
        fold_store[test_idx] = fold

    oof = pd.DataFrame({
        "date": dates.to_numpy(),
        "fold": fold_store,
        "actual_return": y.to_numpy(),
        "base_exchange_rate": base_rate.to_numpy(),
        "future_exchange_rate": future_rate.to_numpy(),
    })
    for name, pred in pred_store.items():
        oof[name] = pred

    metric_rows: list[dict[str, float | str]] = []
    valid_model_names: list[str] = []
    rmse_for_weights: dict[str, float] = {}

    for name in models:
        mask = np.isfinite(oof[name].to_numpy()) & np.isfinite(oof["actual_return"].to_numpy())
        if not mask.any():
            continue
        m = _metrics(
            oof.loc[mask, "actual_return"].to_numpy(),
            oof.loc[mask, name].to_numpy(),
            oof.loc[mask, "base_exchange_rate"].to_numpy(),
            oof.loc[mask, "future_exchange_rate"].to_numpy(),
        )
        metric_rows.append({"model": name, **m})
        if name != "random_walk":
            valid_model_names.append(name)
            rmse_for_weights[name] = m["return_rmse"]

    if not valid_model_names:
        raise ValueError("No non-baseline model produced out-of-fold predictions.")

    inv = {k: 1.0 / max(v, 1e-12) for k, v in rmse_for_weights.items()}
    denom = sum(inv.values())
    weights = {k: v / denom for k, v in inv.items()}

    ensemble = np.zeros(n, dtype=float)
    ensemble[:] = np.nan
    joint_mask = np.ones(n, dtype=bool)
    for name in valid_model_names:
        joint_mask &= np.isfinite(oof[name].to_numpy())
    if joint_mask.any():
        ensemble[joint_mask] = sum(weights[name] * oof.loc[joint_mask, name].to_numpy() for name in valid_model_names)
    oof["ensemble"] = ensemble

    mask = joint_mask & np.isfinite(oof["actual_return"].to_numpy())
    ens_metrics = _metrics(
        oof.loc[mask, "actual_return"].to_numpy(),
        oof.loc[mask, "ensemble"].to_numpy(),
        oof.loc[mask, "base_exchange_rate"].to_numpy(),
        oof.loc[mask, "future_exchange_rate"].to_numpy(),
    )
    metric_rows.append({"model": "ensemble", **ens_metrics})

    residuals = oof.loc[mask, "actual_return"].to_numpy() - oof.loc[mask, "ensemble"].to_numpy()
    ql, qu = config.interval_quantiles
    residual_quantiles = (float(np.quantile(residuals, ql)), float(np.quantile(residuals, qu)))

    metrics = pd.DataFrame(metric_rows).sort_values("return_rmse").reset_index(drop=True)
    return BacktestResult(metrics, oof, weights, residual_quantiles)
