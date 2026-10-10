# Data Dictionary

## Required columns

| Column | Meaning | Typical unit | Treatment |
|---|---|---|---|
| `date` | Forecast observation date | ISO date | Sort ascending; no duplicates |
| `exchange_rate` | Domestic currency per 1 unit of foreign/reference currency | local/foreign | Target source; log returns and lags |

## User-provided variables

| Column | Meaning | Recommended interpretation | Treatment |
|---|---|---|---|
| `issue` | Newly issued/printed money | Prefer actual monetary-base change if available | log change if strictly positive; otherwise difference |
| `gold_price` | Gold price | Prefer USD/oz or consistent reference currency | log changes + lags |
| `oil_price` | Oil price (`yu`) | Brent/WTI or country-relevant benchmark | log changes + lags |
| `imports` | Amount/value of imports (`AoI`) | Same currency/unit through history | log changes, lags, trade features |
| `exports` | Amount/value of exports (`AoX`) | Same currency/unit through history | log changes, lags, trade features |

## Recommended additions

| Column | Meaning | Why useful |
|---|---|---|
| `cpi` | Domestic consumer price index | Domestic inflation pressure |
| `partner_cpi` | CPI of reference-currency economy or trade-weighted partners | Relative inflation / PPP channel |
| `policy_rate` | Domestic central-bank policy rate | Monetary stance |
| `partner_policy_rate` | Reference-country policy rate | Interest-rate differential |
| `fx_reserves` | Official reserve assets / usable FX reserves | Intervention capacity and external buffer |
| `broad_money` | M2/M3 or closest broad aggregate | Liquidity and monetary expansion |
| `monetary_base` | Base money | Better-defined alternative/complement to `issue` |
| `gdp` | Real or nominal GDP | Macro activity; quarterly values need careful alignment |
| `industrial_production` | Production/activity index | Higher-frequency activity proxy |
| `current_account` | Current-account balance | Broader FX inflow/outflow balance |
| `external_debt` | External debt stock | Refinancing and FX-liability pressure |
| `short_term_external_debt` | Short-term external debt | Near-term rollover pressure |
| `fdi` | Net foreign direct investment inflow | Persistent capital inflow |
| `portfolio_flows` | Net portfolio inflows | Market-sensitive capital pressure |
| `remittances` | Worker/personal remittance inflows | FX supply in remittance-dependent economies |
| `budget_balance` | Fiscal balance | Fiscal/monetary credibility channel |
| `government_debt` | Public debt | Fiscal risk and monetization pressure |
| `reer` | Real effective exchange-rate index | Relative competitiveness/misalignment signal |
| `neer` | Nominal effective exchange-rate index | Broad currency strength |
| `dxy` | U.S. Dollar Index or suitable foreign-currency factor | Global reference-currency movement |
| `global_risk` | VIX-like/global financial risk measure | Risk-on/risk-off capital flow channel |
| `terms_of_trade` | Export-price / import-price index | Commodity and external income shock |
| `commodity_index` | Country-relevant commodity basket | Better than one commodity when exports are diversified |
| `bank_credit` | Credit to private sector | Domestic demand and financial cycle |

## Derived features

The code automatically derives a subset of the following when source columns exist.

### Target

`target_log_return_h = log(exchange_rate[t+h] / exchange_rate[t])`

### Trade

- `trade_balance = exports - imports`
- `trade_ratio = exports / imports`
- percentage/log changes in exports and imports

### Relative prices and rates

- `inflation_diff = Δlog(cpi) - Δlog(partner_cpi)`
- `policy_rate_diff = policy_rate - partner_policy_rate`

### Reserves

- `reserve_import_cover_proxy = fx_reserves / imports`

This is only a proxy unless imports are converted to a compatible monthly amount and reserve units are consistent.

### Exchange-rate dynamics

- 1-, 3-, and 12-period log returns
- rolling 3/6/12-period mean return
- rolling 3/6/12-period return volatility

## Frequency

Monthly frequency is recommended for a first macro model because many important variables are monthly or slower. If you forecast daily FX, macro values must be carried forward only after their publication timestamp and combined with higher-frequency market variables.

## Missing data

The implementation uses median imputation inside each model pipeline. However, long missing blocks should be repaired at the data-engineering stage rather than relying on imputation. Do not backfill a value into dates before it was published.

## Unit consistency

Before training, verify that:

- the exchange-rate quotation convention never changes;
- oil/gold units are constant;
- trade values use a consistent currency and nominal/real definition;
- rate columns use the same scale throughout (e.g. `12.5` means 12.5%, not `0.125`);
- CPI index rebasing does not create artificial jumps in percentage changes;
- stock variables and flow variables are not mixed without understanding the distinction.
