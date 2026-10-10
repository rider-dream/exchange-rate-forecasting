# Exchange Rate Forecasting — Mathematical and Algorithmic Documentation

## 1. Objective

Let `S_t` be the observed exchange rate at forecast origin `t`, quoted as domestic currency units per one unit of the reference foreign currency. The model forecasts the `h`-period-ahead logarithmic change:

\[
y_{t,h}=\ln(S_{t+h})-\ln(S_t)=\ln\left(\frac{S_{t+h}}{S_t}\right).
\]

A predicted value `ŷ > 0` implies a rise in the quoted rate, i.e. depreciation of the domestic currency under this quotation convention.

The level forecast is:

\[
\widehat{S}_{t+h}=S_t e^{\widehat{y}_{t,h}}.
\]

## 2. Why forecast returns rather than levels?

Raw exchange-rate levels are commonly persistent and can contain stochastic trends. Regressing one trending level on other trending macro levels can produce misleading relationships. Using log differences converts multiplicative changes into approximately additive percentage changes:

\[
\Delta \ln S_t \approx \frac{S_t-S_{t-1}}{S_{t-1}}
\]

for small changes.

This does not guarantee stationarity, so diagnostics are still required, but it is a safer default than a naïve levels regression.

## 3. Economic feature families

The model organizes information into several channels.

### 3.1 Monetary/liquidity channel

Candidate variables:

- issued/printed money (`issue`)
- monetary base
- broad money (`M2/M3`)
- policy rate
- credit growth

A simple reduced-form relationship is:

\[
\Delta s_{t+h}=\alpha+\beta_m\Delta m_t+\beta_i(i_t-i_t^*)+\epsilon_{t+h},
\]

where `s = ln(S)`, `m` is a money measure, and `i-i*` is an interest-rate differential. The sign is empirical because policy rates may rise in response to depreciation rather than cause appreciation contemporaneously.

### 3.2 Relative-price / PPP channel

Relative purchasing-power-parity intuition suggests that, over sufficiently long horizons, currency depreciation is related to the inflation differential:

\[
\Delta s \approx \pi-\pi^*,
\]

where `π` and `π*` denote domestic and foreign/partner inflation.

The implemented approximation is based on CPI log changes:

\[
\text{inflation\_diff}_t
=\Delta\ln(CPI_t)-\Delta\ln(CPI_t^*).
\]

### 3.3 Interest-rate differential

A forecasting model may include:

\[
\text{rate\_diff}_t=i_t-i_t^*.
\]

This variable is related to uncovered-interest-parity ideas, but UIP should not be imposed as an exact law. Empirically, risk premia and policy reactions matter.

### 3.4 External/trade channel

From exports `X_t` and imports `M_t`:

\[
TB_t=X_t-M_t,
\]

\[
TR_t=\frac{X_t}{M_t}.
\]

The current account, terms of trade, remittances, FDI, and portfolio flows add information that the goods trade balance alone cannot capture.

### 3.5 Reserve adequacy / intervention capacity

A simple data-driven proxy is:

\[
RIC_t=\frac{Reserves_t}{Imports_t}.
\]

Interpretation requires consistent units. A proper months-of-import-cover measure should divide reserves by average monthly imports expressed in compatible currency units.

### 3.6 Commodity channel

Gold and oil are transformed primarily through log changes:

\[
g_t=\Delta\ln(Gold_t),\qquad o_t=\Delta\ln(Oil_t).
\]

The expected sign depends on whether the domestic economy is a net producer/exporter or importer and on the role of gold as a global safe-haven asset.

### 3.7 Competitiveness and global factors

Useful variables include real effective exchange rate (`REER`), nominal effective exchange rate (`NEER`), a global USD factor (`DXY`), global-risk indices, and global rates.

REER is especially useful because it combines nominal currency movements with relative prices and trading-partner weights.

## 4. Feature engineering

Suppose `x_t > 0` is a price/index/stock variable. The pipeline can create:

\[
g^{(k)}_{x,t}=\ln(x_t)-\ln(x_{t-k})
\]

for `k ∈ {1,3,12}` when enough observations exist.

For variables that can be zero/negative (policy rate, current account, budget balance), ordinary differences are safer:

\[
d^{(k)}_{x,t}=x_t-x_{t-k}.
\]

For each engineered base feature `z_t`, lagged features are generated:

\[
z_{t-j},\qquad j\in\{1,2,3,6,12\}.
\]

The exchange-rate return also receives rolling statistics. For window `w`:

\[
\bar r_{t,w}=\frac{1}{w}\sum_{j=0}^{w-1}r_{t-j},
\]

\[
\sigma_{t,w}=\sqrt{\frac{1}{w-1}\sum_{j=0}^{w-1}(r_{t-j}-\bar r_{t,w})^2}.
\]

These provide momentum/trend and volatility/regime information.

## 5. Model 0 — random-walk benchmark

For a driftless random walk in log exchange rates:

\[
E_t[\ln S_{t+h}-\ln S_t]=0.
\]

Therefore:

\[
\widehat y^{RW}_{t,h}=0,
\quad
\widehat S^{RW}_{t+h}=S_t.
\]

A model that cannot consistently improve upon this benchmark in genuine out-of-sample testing has limited practical forecasting value.

## 6. Model 1 — Elastic Net

The linear predictor is:

\[
\widehat y_t=\beta_0+\sum_{j=1}^{p}\beta_j x_{t,j}.
\]

Elastic Net estimates coefficients by minimizing:

\[
\frac{1}{2n}\|y-X\beta\|_2^2
+\alpha\rho\|\beta\|_1
+\frac{\alpha(1-\rho)}{2}\|\beta\|_2^2.
\]

`α` controls total regularization and `ρ` is the L1 ratio. L1 regularization can shrink irrelevant variables to zero; L2 stabilizes correlated macro predictors.

The project standardizes features before fitting and uses fixed conservative hyperparameters by default to reduce overfitting in small macro samples.

## 7. Model 2 — Random Forest regression

A random forest estimates:

\[
\widehat f(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x),
\]

where each tree `T_b` is fit to a bootstrap-style/randomized sample and considers random feature subsets at splits.

Advantages:

- nonlinear responses;
- interactions without manual specification;
- robustness to monotonic transformations.

Risks:

- poor extrapolation outside observed ranges;
- overfitting on small macro samples;
- feature importance can be unstable with correlated predictors.

## 8. Model 3 — Histogram Gradient Boosting

Boosting constructs an additive model:

\[
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta h_m(x),
\]

where each tree `h_m` is chosen to reduce the residual loss from the previous ensemble and `η` is the learning rate.

This model captures nonlinearities and interactions while regularization is controlled by tree depth/leaf size, iteration count, and learning rate.

## 9. Ensemble

Let the chronological cross-validation RMSE of model `m` be `RMSE_m`. The project uses inverse-error weights:

\[
\tilde w_m=\frac{1}{RMSE_m+\varepsilon},
\qquad
w_m=\frac{\tilde w_m}{\sum_j\tilde w_j}.
\]

The ensemble prediction is:

\[
\widehat y^{ENS}_t=\sum_m w_m\widehat y^{(m)}_t.
\]

The random-walk baseline is reported but is not included in the learned ensemble by default.

## 10. Chronological validation

Randomly shuffling observations would allow a model to train on the future and validate on the past. Instead, the project uses expanding-window splits:

```text
Fold 1: train [1 ........ a]       test [a+1 ... b]
Fold 2: train [1 ............ b]   test [b+1 ... c]
Fold 3: train [1 ................ c] test [c+1 ... d]
```

The implementation also sets a validation gap equal to the forecast horizon. This prevents the last training labels from using exchange-rate observations that sit inside the following validation block. For a real production system, the strongest design is a **vintage/as-of backtest**: every fold should reconstruct what data were known at the historical forecast date, including publication lags and revisions.

## 11. Evaluation metrics

### Return RMSE

\[
RMSE_y=\sqrt{\frac{1}{n}\sum_t(y_t-\widehat y_t)^2}.
\]

### Return MAE

\[
MAE_y=\frac{1}{n}\sum_t|y_t-\widehat y_t|.
\]

### Directional accuracy

\[
DA=\frac{1}{n}\sum_t I(\operatorname{sign}(y_t)=\operatorname{sign}(\widehat y_t)).
\]

### Level error

Given forecast-origin rate `S_t`:

\[
\widehat S_{t+h}=S_t e^{\widehat y_t}.
\]

Then level MAE/RMSE are calculated against observed `S_{t+h}`.

## 12. Prediction interval

The project estimates an empirical interval from chronological out-of-fold ensemble residuals:

\[
e_t=y_t-\widehat y_t.
\]

If `q_L` and `q_U` are residual quantiles (default 5% and 95% in the implementation), the return interval is:

\[
[\widehat y+q_L,\widehat y+q_U].
\]

The corresponding exchange-rate interval is:

\[
[S_t e^{\widehat y+q_L},\;S_t e^{\widehat y+q_U}].
\]

This is an empirical error band, not a guaranteed probabilistic confidence interval. Coverage can deteriorate during regime shifts.

## 13. Missing data and scaling

For each fitted model, imputation is learned only from the training fold. Elastic Net additionally standardizes each feature:

\[
z_{t,j}=\frac{x_{t,j}-\mu_j}{\sigma_j}.
\]

Tree models do not require standardization, but the shared preprocessing still replaces non-finite values consistently.

## 14. Feature importance

For the Elastic Net, standardized coefficient magnitude is used as a simple importance measure:

\[
I_j=|\beta_j|.
\]

Tree models expose impurity/split-based feature importance where supported. The repository combines available model importances into a table for exploration. Treat importance as descriptive rather than causal.

## 15. Causality warning

This is a forecasting system, not a causal-identification design. For example, a higher policy rate may statistically coincide with depreciation because the central bank raises rates in response to currency pressure. A predictive coefficient does not establish that changing the policy rate will cause the forecasted exchange-rate change.

Causal questions require different methods: policy shocks, instruments, natural experiments, structural VARs, local projections with identification, or other explicitly causal designs.

## 16. Structural breaks and regimes

FX processes can change when there is:

- a peg/de-peg or crawling band;
- a discrete devaluation;
- capital controls;
- sanctions or payment restrictions;
- war/political shock;
- central-bank regime change;
- major commodity shock;
- banking/currency crisis.

Recommended extensions:

1. add regime dummy variables when dates are objectively known;
2. use rolling training windows as a robustness check;
3. compare pre/post-break performance;
4. retrain more frequently after a verified structural change;
5. consider Markov-switching/state-space models only with sufficient data and careful validation.

## 17. Data frequency and mixed-frequency problem

A monthly model is the recommended starting point. GDP is usually quarterly while market prices can be daily. Never interpolate GDP in a way that reveals future quarterly information. Instead:

- use the latest released GDP value and carry it forward after its publication date;
- use monthly activity proxies such as industrial production where possible;
- aggregate daily gold/oil/market variables into monthly averages or end-of-period values consistent with the forecast timing.

For a more advanced system, consider MIDAS or state-space mixed-frequency models.

## 18. Publication lag and revision leakage

This is one of the most important implementation risks.

Suppose March CPI is published on April 10. A forecast made March 31 did not know the March CPI value. A historical row dated March 31 must therefore not contain the final March CPI unless the forecast is explicitly defined as occurring after publication.

For production-grade backtests, store:

- `reference_period`
- `release_date`
- `vintage_date`
- `value`

and construct each forecast-origin row from values whose `release_date <= forecast_origin`.

## 19. Suggested model-selection policy

Use the following hierarchy:

1. Random walk is the baseline.
2. Elastic Net is the default for small samples and interpretation.
3. Add nonlinear tree models only when there is enough data.
4. Keep a nonlinear model only if chronological validation improves robustly across several folds/regimes.
5. Prefer a stable ensemble over whichever model happens to win one split.

## 20. Suggested minimum robustness tests before deployment

- Different horizons: 1, 3, 6, 12 months.
- Expanding window vs rolling 5-year/10-year window.
- Remove one feature family at a time.
- Test against random walk and simple autoregressive return model.
- Check performance before/after major policy regimes.
- Recalculate using real publication dates.
- Test sensitivity to revised vs first-release macro data.
- Check feature drift and missingness in the latest period.
- Evaluate interval coverage, not only point RMSE.

## 21. Optional advanced models

These are not required for version 1, but are logical extensions:

### ARDL / ARIMAX

\[
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\sum_{j=1}^{k}\sum_{l=0}^{q_j}\beta_{j,l}x_{j,t-l}+\epsilon_t.
\]

Useful when interpretability and lag structure matter.

### Vector autoregression (VAR)

\[
Y_t=c+A_1Y_{t-1}+\cdots+A_pY_{t-p}+u_t.
\]

Useful for a small system of jointly endogenous stationary variables, but parameter count grows rapidly.

### Error-correction model (ECM/VECM)

Appropriate only if nonstationary levels are demonstrably cointegrated.

### State-space / Kalman filter

Useful for time-varying coefficients, latent equilibrium rates, or mixed-frequency nowcasting.

### Gradient-boosting libraries

XGBoost/LightGBM/CatBoost can be tested later, but the first repository intentionally uses scikit-learn-only models to keep installation and reproducibility simple.

## 22. Recommended Python libraries

Core repository:

- `pandas` — tabular/time-series manipulation
- `numpy` — numerical calculations
- `scikit-learn` — preprocessing, time-series splitting, models, metrics
- `joblib` — model persistence
- `matplotlib` — diagnostics/plots
- `pytest` — tests

Useful advanced additions:

- `statsmodels` — ARIMA/ARDL/VAR/VECM/econometric diagnostics
- `arch` — volatility models such as GARCH
- `xgboost`, `lightgbm`, `catboost` — optional boosted trees
- `shap` — local/global model explanation, with care around correlated features
- `optuna` — controlled hyperparameter search using chronological validation

## 23. Official data references

Prefer primary sources and document the exact series IDs used. Useful starting points include IMF Data (exchange rates, CPI, reserves, balance of payments, effective exchange rates), BIS exchange-rate/effective-exchange-rate statistics, the World Bank World Development Indicators, and your national central bank/statistical office.

Record source, unit, frequency, seasonal adjustment, release calendar, and revision policy for every series in a production data catalog.
