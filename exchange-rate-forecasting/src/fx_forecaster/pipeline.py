from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .backtest import run_backtest
from .config import ForecastConfig
from .data import load_data
from .features import build_feature_frame, select_model_columns
from .models import make_models


def _extract_feature_importance(fitted_models: dict[str, object], feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    elastic = fitted_models.get("elastic_net")
    if elastic is not None:
        coef = elastic.named_steps["model"].coef_
        for feature, value in zip(feature_names, coef):
            rows.append({"model": "elastic_net", "feature": feature, "importance": float(abs(value)), "signed_value": float(value)})

    rf = fitted_models.get("random_forest")
    if rf is not None:
        imp = rf.named_steps["model"].feature_importances_
        for feature, value in zip(feature_names, imp):
            rows.append({"model": "random_forest", "feature": feature, "importance": float(value), "signed_value": np.nan})

    return pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False]) if rows else pd.DataFrame()


def train_project(data_path: str | Path, output_dir: str | Path, config: ForecastConfig) -> dict[str, object]:
    df = load_data(data_path)
    ff = build_feature_frame(df, config)
    feature_names = select_model_columns(ff)

    train_mask = ff["target_log_return"].notna()
    train_df = ff.loc[train_mask].copy().reset_index(drop=True)
    X = train_df[feature_names]
    y = train_df["target_log_return"]

    models = make_models(config)
    bt = run_backtest(
        X=X,
        y=y,
        dates=train_df["date"],
        base_rate=train_df["exchange_rate"],
        future_rate=train_df["future_exchange_rate"],
        models=models,
        config=config,
    )

    fitted_models: dict[str, object] = {}
    for name, model in models.items():
        if name == "random_walk":
            continue
        fitted = clone(model)
        fitted.fit(X, y)
        fitted_models[name] = fitted

    # Forecast at the most recent row for which predictor features exist.
    latest = ff.iloc[[-1]]
    latest_X = latest[feature_names]
    component_predictions = {
        name: float(model.predict(latest_X)[0]) for name, model in fitted_models.items()
    }
    ensemble_return = float(sum(bt.ensemble_weights[name] * component_predictions[name] for name in bt.ensemble_weights))
    current_rate = float(latest["exchange_rate"].iloc[0])
    point_level = current_rate * float(np.exp(ensemble_return))
    ql, qu = bt.residual_quantiles

    forecast = {
        "forecast_origin": str(pd.Timestamp(latest["date"].iloc[0]).date()),
        "horizon_periods": config.horizon,
        "current_exchange_rate": current_rate,
        "predicted_log_return": ensemble_return,
        "predicted_pct_change": float(np.expm1(ensemble_return) * 100.0),
        "predicted_exchange_rate": point_level,
        "prediction_interval_exchange_rate": [
            current_rate * float(np.exp(ensemble_return + ql)),
            current_rate * float(np.exp(ensemble_return + qu)),
        ],
        "component_predictions": component_predictions,
        "ensemble_weights": bt.ensemble_weights,
        "interval_method": "empirical out-of-fold residual quantiles",
    }

    bundle = {
        "config": config,
        "feature_names": feature_names,
        "models": fitted_models,
        "ensemble_weights": bt.ensemble_weights,
        "residual_quantiles": bt.residual_quantiles,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "model.joblib")
    bt.metrics.to_csv(output / "metrics.csv", index=False)
    bt.oof_predictions.to_csv(output / "oof_predictions.csv", index=False)
    importance = _extract_feature_importance(fitted_models, feature_names)
    if not importance.empty:
        importance.to_csv(output / "feature_importance.csv", index=False)
    with (output / "forecast.json").open("w", encoding="utf-8") as f:
        json.dump(forecast, f, indent=2)
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    return {"metrics": bt.metrics, "forecast": forecast, "bundle": bundle}


def predict_latest(data_path: str | Path, model_path: str | Path) -> dict[str, object]:
    bundle = joblib.load(model_path)
    config: ForecastConfig = bundle["config"]
    df = load_data(data_path)
    ff = build_feature_frame(df, config)
    latest = ff.iloc[[-1]]
    X = latest[bundle["feature_names"]]

    component_predictions = {
        name: float(model.predict(X)[0]) for name, model in bundle["models"].items()
    }
    pred_return = float(sum(bundle["ensemble_weights"][name] * component_predictions[name] for name in bundle["ensemble_weights"]))
    current_rate = float(latest["exchange_rate"].iloc[0])
    ql, qu = bundle["residual_quantiles"]

    return {
        "forecast_origin": str(pd.Timestamp(latest["date"].iloc[0]).date()),
        "horizon_periods": config.horizon,
        "current_exchange_rate": current_rate,
        "predicted_log_return": pred_return,
        "predicted_pct_change": float(np.expm1(pred_return) * 100.0),
        "predicted_exchange_rate": current_rate * float(np.exp(pred_return)),
        "prediction_interval_exchange_rate": [
            current_rate * float(np.exp(pred_return + ql)),
            current_rate * float(np.exp(pred_return + qu)),
        ],
        "component_predictions": component_predictions,
        "ensemble_weights": bundle["ensemble_weights"],
    }
