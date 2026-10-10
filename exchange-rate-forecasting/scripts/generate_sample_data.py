from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-31", periods=n, freq="ME")

    # Synthetic macro drivers; values are illustrative only.
    oil = 75 * np.exp(np.cumsum(rng.normal(0.001, 0.055, n)))
    gold = 1400 * np.exp(np.cumsum(rng.normal(0.002, 0.025, n)))
    cpi = 100 * np.exp(np.cumsum(rng.normal(0.0045, 0.0015, n)))
    partner_cpi = 100 * np.exp(np.cumsum(rng.normal(0.0020, 0.0008, n)))
    policy_rate = np.clip(8 + np.cumsum(rng.normal(0, 0.20, n)), 3, 20)
    partner_policy_rate = np.clip(2 + np.cumsum(rng.normal(0, 0.10, n)), 0, 8)
    broad_money = 50_000 * np.exp(np.cumsum(rng.normal(0.007, 0.006, n)))
    issue = 900 * np.exp(np.cumsum(rng.normal(0.005, 0.04, n)))
    imports = 2_500 * np.exp(np.cumsum(rng.normal(0.003, 0.035, n)))
    exports = 2_600 * np.exp(np.cumsum(rng.normal(0.0035, 0.035, n)))
    fx_reserves = 12_000 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    industrial_production = 100 * np.exp(np.cumsum(rng.normal(0.0025, 0.012, n)))
    external_debt = 25_000 * np.exp(np.cumsum(rng.normal(0.0025, 0.012, n)))
    remittances = 300 * np.exp(np.cumsum(rng.normal(0.003, 0.025, n)))
    fdi = 150 + rng.normal(0, 35, n)
    current_account = (exports - imports) * 0.35 + remittances * 0.25 + rng.normal(0, 80, n)
    dxy = 95 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    global_risk = np.clip(18 + rng.normal(0, 5, n), 8, 50)
    terms_of_trade = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))

    inflation_diff = np.diff(np.log(cpi), prepend=np.log(cpi[0])) - np.diff(np.log(partner_cpi), prepend=np.log(partner_cpi[0]))
    money_growth = np.diff(np.log(broad_money), prepend=np.log(broad_money[0]))
    reserve_growth = np.diff(np.log(fx_reserves), prepend=np.log(fx_reserves[0]))
    oil_growth = np.diff(np.log(oil), prepend=np.log(oil[0]))
    dxy_growth = np.diff(np.log(dxy), prepend=np.log(dxy[0]))
    trade_signal = (exports - imports) / (imports + 1e-9)

    # Synthetic FX return process with realistic noise dominating signal.
    fx_ret = np.zeros(n)
    for t in range(1, n):
        fx_ret[t] = (
            0.10 * fx_ret[t - 1]
            + 0.30 * inflation_diff[t - 1]
            + 0.12 * money_growth[t - 1]
            - 0.08 * reserve_growth[t - 1]
            - 0.04 * oil_growth[t - 1]
            + 0.18 * dxy_growth[t - 1]
            - 0.015 * trade_signal[t - 1]
            + 0.0005 * (global_risk[t - 1] - 18)
            + rng.normal(0.0015, 0.018)
        )
    exchange_rate = 2.0 * np.exp(np.cumsum(fx_ret))
    reer = 100 * np.exp(-0.25 * np.cumsum(fx_ret) + np.cumsum(rng.normal(0, 0.006, n)))

    return pd.DataFrame({
        "date": dates,
        "exchange_rate": exchange_rate,
        "issue": issue,
        "gold_price": gold,
        "oil_price": oil,
        "imports": imports,
        "exports": exports,
        "cpi": cpi,
        "partner_cpi": partner_cpi,
        "policy_rate": policy_rate,
        "partner_policy_rate": partner_policy_rate,
        "fx_reserves": fx_reserves,
        "broad_money": broad_money,
        "industrial_production": industrial_production,
        "current_account": current_account,
        "external_debt": external_debt,
        "remittances": remittances,
        "fdi": fdi,
        "reer": reer,
        "dxy": dxy,
        "global_risk": global_risk,
        "terms_of_trade": terms_of_trade,
    })


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "sample" / "exchange_rate_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(out, index=False)
    print(out)
