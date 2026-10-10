from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ForecastConfig


@dataclass
class RandomWalkRegressor:
    """A sklearn-like zero-return benchmark."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def make_models(config: ForecastConfig) -> dict[str, object]:
    common_imputer = SimpleImputer(strategy="median")

    elastic_net = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.003,
                    l1_ratio=0.25,
                    max_iter=20_000,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=5,
                    min_samples_leaf=4,
                    max_features=0.65,
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    hist_gb = Pipeline(
        steps=[
            ("imputer", common_imputer),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=12,
                    l2_regularization=1.0,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    return {
        "random_walk": RandomWalkRegressor(),
        "elastic_net": elastic_net,
        "random_forest": random_forest,
        "hist_gradient_boosting": hist_gb,
    }
