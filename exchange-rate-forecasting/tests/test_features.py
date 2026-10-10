import numpy as np
import pandas as pd

from fx_forecaster.config import ForecastConfig
from fx_forecaster.features import build_feature_frame, select_model_columns


def test_target_and_trade_features():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-31", periods=20, freq="ME"),
        "exchange_rate": np.linspace(2.0, 2.5, 20),
        "imports": np.linspace(100, 120, 20),
        "exports": np.linspace(110, 140, 20),
        "cpi": np.linspace(100, 110, 20),
        "partner_cpi": np.linspace(100, 105, 20),
    })
    ff = build_feature_frame(df, ForecastConfig(horizon=1))
    assert "trade_balance" in ff.columns
    assert "inflation_diff" in ff.columns
    assert ff["target_log_return"].notna().sum() == 19
    assert "exchange_rate" not in select_model_columns(ff)
