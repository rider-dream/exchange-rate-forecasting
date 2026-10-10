# Suggested Data Sources

Use primary/official sources where possible and record the exact series ID, unit, frequency, seasonal adjustment, release lag, and revision policy.

## International Monetary Fund

IMF Data portal: https://data.imf.org/

Useful datasets include:

- Exchange Rates
- Consumer Price Index
- Balance of Payments
- International Reserves and Foreign Currency Liquidity
- Effective Exchange Rate
- International Financial Statistics topic datasets

IMF Effective Exchange Rate dataset: https://data.imf.org/en/datasets/EER

## Bank for International Settlements

BIS Data Portal: https://data.bis.org/

Useful topics include:

- Bilateral exchange rates
- Nominal and real effective exchange rates
- Central-bank policy rates
- Central-bank balance-sheet statistics
- Credit and global-liquidity indicators

Effective exchange rates: https://data.bis.org/topics/EER

## World Bank

World Development Indicators: https://data.worldbank.org/

Useful series include:

- official exchange rate
- inflation / CPI
- GDP growth
- current-account balance
- reserves
- external debt
- imports and exports
- remittances
- FDI

## National sources

Prefer the national central bank, ministry of finance, customs authority, and national statistical office for the most detailed domestic series and release calendars.

## Market data

For gold, oil, DXY/reference-currency indices, and global-risk measures, use a reputable provider with a stable methodology and sufficient history. Keep the quote currency and aggregation method consistent.

## Production data catalog

For every variable, store at least:

- `series_name`
- `source`
- `series_id`
- `unit`
- `frequency`
- `reference_period`
- `release_date`
- `vintage_date`
- `seasonal_adjustment`
- `transformation`

This makes the backtest reproducible and allows proper as-of reconstruction.
