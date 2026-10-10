from __future__ import annotations

import argparse
import json

from .config import ForecastConfig
from .pipeline import predict_latest, train_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exchange-rate forecasting CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Backtest, fit models, and save artifacts")
    train.add_argument("--data", required=True, help="CSV file")
    train.add_argument("--output", default="artifacts", help="Output directory")
    train.add_argument("--horizon", type=int, default=1, help="Forecast horizon in rows/periods")
    train.add_argument("--splits", type=int, default=5, help="Chronological CV splits")

    pred = sub.add_parser("predict", help="Forecast latest row using a saved model")
    pred.add_argument("--data", required=True, help="CSV file with latest observations")
    pred.add_argument("--model", required=True, help="Path to model.joblib")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        config = ForecastConfig(horizon=args.horizon, n_splits=args.splits)
        result = train_project(args.data, args.output, config)
        print("\nBacktest metrics:\n")
        print(result["metrics"].to_string(index=False))
        print("\nLatest forecast:\n")
        print(json.dumps(result["forecast"], indent=2))
    elif args.command == "predict":
        print(json.dumps(predict_latest(args.data, args.model), indent=2))


if __name__ == "__main__":
    main()
