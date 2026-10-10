# Exchange Rate Forecasting Model

A GitHub-ready Python project for forecasting an exchange rate from monetary, external-sector, commodity, and macroeconomic variables.

The project starts from your variables:

- `date`
- `exchange_rate` — your `Br`
- `issue` — newly printed/issued money or the closest measurable monetary issuance series
- `gold_price`
- `oil_price` — your `yu`
- `imports` — your `AoI`
- `exports` — your `AoX`

and supports additional predictors such as inflation, policy rates, FX reserves, broad money, GDP/activity, current account, debt, capital flows, real effective exchange rate, and global risk variables.

## What the model predicts

For forecast horizon `h`, the target is the future logarithmic exchange-rate change:

\[
y_{t,h} = \log\left(\frac{S_{t+h}}{S_t}\right)
\]

where `S_t` is the domestic-currency price of one unit of foreign currency. A positive prediction therefore means **domestic-currency depreciation** when your quote is, for example, `local currency / USD`.

The predicted future level is reconstructed as:

\[
\hat S_{t+h} = S_t \exp(\hat y_{t,h})
\]

This formulation is usually preferable to fitting the raw exchange-rate level because it reduces scale/trend problems and makes error comparisons more meaningful.

## Implemented algorithms

The repository trains three models and an ensemble:

1. **Random-walk baseline**: predicts zero future log-change. Any serious FX model should beat this simple benchmark out of sample.
2. **Elastic Net**: interpretable linear model with L1/L2 regularization.
3. **Random Forest**: nonlinear model capturing interactions and threshold effects.
4. **Histogram Gradient Boosting**: nonlinear boosted-tree model.
5. **Inverse-RMSE ensemble**: combines the three trained models using weights estimated from chronological out-of-fold predictions.

Evaluation uses chronological `TimeSeriesSplit`, never random shuffling, with a gap equal to the forecast horizon to prevent target overlap across train/test blocks.

## Repository structure

```text
exchange-rate-forecasting/
├── data/
│   ├── raw/                         # put your real CSV here
│   └── sample/exchange_rate_sample.csv
├── docs/
│   ├── DATA_DICTIONARY.md
│   └── MODEL_DOCUMENTATION.md
├── scripts/
│   └── generate_sample_data.py
├── src/fx_forecaster/
│   ├── backtest.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── pipeline.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Train on the included synthetic monthly sample
fx-forecast train \
  --data data/sample/exchange_rate_sample.csv \
  --horizon 1 \
  --output artifacts

# Forecast from the latest row
fx-forecast predict \
  --data data/sample/exchange_rate_sample.csv \
  --model artifacts/model.joblib

pytest
```

You can also run the package without installing the command:

```bash
PYTHONPATH=src python -m fx_forecaster.cli train --data data/sample/exchange_rate_sample.csv --output artifacts
```

## Minimum input format

Only these two columns are mandatory:

```csv
date,exchange_rate
2022-01-31,2.58
2022-02-28,2.61
```

All other numeric columns are optional. The feature pipeline automatically uses available numeric predictors, while ignoring unknown nonnumeric columns.

For a useful macro model, a monthly dataset should ideally contain at least 8–10 years of history. More history is preferable when regimes are reasonably comparable. For small samples, prefer the regularized linear model over highly flexible tree models.

## Recommended columns

High-priority additions to your current variables are:

| Variable | Why it matters |
|---|---|
| `cpi` / inflation | Relative inflation is a core driver of real currency value. |
| `partner_cpi` | Allows an inflation differential rather than domestic inflation alone. |
| `policy_rate` | Captures domestic monetary-policy stance. |
| `partner_policy_rate` | Enables interest-rate differential features. |
| `fx_reserves` | Measures central-bank capacity to smooth FX pressure. |
| `broad_money` | Better monetary-liquidity measure than issuance alone. |
| `gdp` or `industrial_production` | Captures domestic activity and import demand. |
| `current_account` | Broad external-balance measure beyond trade alone. |
| `external_debt` | Captures foreign-currency financing pressure. |
| `fdi` | Structural capital inflow. |
| `portfolio_flows` | Market-sensitive capital-flow pressure. |
| `remittances` | Important FX supply source in many economies. |
| `reer` | Measures competitiveness after relative-price adjustment. |
| `dxy` or foreign-currency index | Captures global strength of the reference currency. |
| `global_risk` | VIX-like or global-risk proxy; useful for risk-off episodes. |
| `terms_of_trade` | Especially important for commodity exporters/importers. |
| `budget_balance` | Fiscal pressure can affect inflation, issuance, and credibility. |

See `docs/DATA_DICTIONARY.md` for definitions and transformations.

## Important data rule: publication lag

A forecast must use only information that was actually available at the forecast date. Monthly CPI, GDP, trade, and balance-of-payments values are often published after the month they describe. If you place a later-released value into the earlier month and train on it, the backtest leaks future information.

The safest production dataset therefore contains **as-of values**: each row should represent the information available on that forecast date. Alternatively, shift each macro series by its real publication lag before training.

## Modeling workflow

```text
Raw chronological data
      ↓
Validation / sorting / missing-value handling
      ↓
Transform levels to log changes, differences, ratios
      ↓
Create lags and rolling volatility
      ↓
Construct h-step future log-return target
      ↓
Chronological cross-validation
      ↓
Baseline + ElasticNet + RF + Gradient Boosting
      ↓
Out-of-fold metrics and ensemble weights
      ↓
Refit on all historical observations
      ↓
Forecast + empirical residual interval
```

## Main derived features

The pipeline creates, when possible:

- exchange-rate returns over 1, 3, and 12 periods;
- log changes for positive price/quantity variables;
- first differences for rates or variables that may be negative;
- lags at 1, 2, 3, 6, and 12 periods;
- rolling mean and volatility of exchange-rate returns;
- trade balance: `exports - imports`;
- trade ratio: `exports / imports`;
- inflation differential: domestic CPI inflation minus partner CPI inflation;
- policy-rate differential: domestic policy rate minus partner policy rate;
- reserve coverage proxy: FX reserves divided by imports, when both exist.

## Metrics

The backtest reports:

- RMSE of future log-change;
- MAE of future log-change;
- directional accuracy;
- exchange-rate-level MAE;
- exchange-rate-level RMSE.

Do not judge the model from in-sample R² alone. FX forecasting should be evaluated against a random-walk benchmark and across multiple time regimes.

## Output files

Training creates:

```text
artifacts/
├── model.joblib
├── metrics.csv
├── oof_predictions.csv
├── forecast.json
└── feature_importance.csv
```

`forecast.json` contains the latest forecast, point estimate, interval, and ensemble weights.

## Data sources to consider

Use official or primary data when possible:

- Your central bank and national statistical office
- IMF Data: IFS-related datasets, exchange rates, CPI, reserves, balance of payments, EER
- BIS: bilateral and effective exchange rates, policy-rate/central-bank statistics
- World Bank WDI: macro and external-sector indicators
- Commodity exchanges or reputable market-data providers for gold/oil

When mixing sources, normalize frequency, units, currency denomination, time zone, release date, and revision policy.

## Caveats

Exchange rates are noisy, regime-dependent, and influenced by policy interventions and events not captured in monthly macro data. A model can estimate conditional expectations; it cannot guarantee future exchange rates. Structural breaks, pegs, devaluations, sanctions, capital controls, wars, and changes in market microstructure can invalidate historical relationships quickly.

For full mathematical and implementation detail, read [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md).
