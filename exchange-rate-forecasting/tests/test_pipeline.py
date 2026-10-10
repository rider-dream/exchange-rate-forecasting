from pathlib import Path

from fx_forecaster.config import ForecastConfig
from fx_forecaster.pipeline import predict_latest, train_project


def test_train_and_predict(tmp_path: Path):
    data = Path(__file__).resolve().parents[1] / "data" / "sample" / "exchange_rate_sample.csv"
    output = tmp_path / "artifacts"
    result = train_project(data, output, ForecastConfig(horizon=1, n_splits=3, min_train_rows=30))
    assert (output / "model.joblib").exists()
    assert (output / "metrics.csv").exists()
    assert result["forecast"]["predicted_exchange_rate"] > 0

    pred = predict_latest(data, output / "model.joblib")
    assert pred["predicted_exchange_rate"] > 0
    assert len(pred["prediction_interval_exchange_rate"]) == 2
