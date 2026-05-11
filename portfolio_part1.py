from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class Part1Config:
    data_clean_dir: Path = Path("./data_clean")
    output_dir: Path = Path("./outputs_part1")
    start_rebal_year: int = 2013
    end_rebal_year: int = 2024
    estimation_months: int = 120
    min_monthly_obs: int = 36
    stale_zero_return_threshold: float = 0.5


def _load_panel_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ISIN" not in df.columns:
        raise ValueError(f"Missing ISIN in {path}")
    df = df.set_index("ISIN")
    return df


def _extract_monthly_matrix(df: pd.DataFrame) -> pd.DataFrame:
    parsed_dates = pd.to_datetime(df.columns, errors="coerce", format="mixed")
    keep_mask = parsed_dates.notna()
    date_cols = pd.Index(df.columns)[keep_mask]
    out = df[date_cols].copy()
    out.columns = pd.DatetimeIndex(parsed_dates[keep_mask])
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.sort_index(axis=1)
    return out


def _extract_annual_matrix(df: pd.DataFrame) -> pd.DataFrame:
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    out = df[year_cols].copy()
    out.columns = [int(c) for c in out.columns]
    out = out.reindex(sorted(out.columns), axis=1)
    return out


def _year_end_month(year: int, months: pd.DatetimeIndex) -> pd.Timestamp:
    candidates = months[(months.year == year) & (months.month == 12)]
    if len(candidates) == 0:
        raise ValueError(f"No December month found for year {year}")
    return candidates.max()


def _build_investment_set(
    returns_m: pd.DataFrame,
    prices_m: pd.DataFrame,
    carbon_y: pd.DataFrame,
    year: int,
    estimation_months: int,
    min_monthly_obs: int,
    stale_zero_return_threshold: float,
) -> List[str]:
    all_months = returns_m.columns
    dec_y = _year_end_month(year, all_months)
    end_pos = all_months.get_loc(dec_y)
    start_pos = end_pos - estimation_months + 1
    if start_pos < 0:
        raise ValueError(f"Not enough history for estimation at year {year}.")
    window = returns_m.iloc[:, start_pos : end_pos + 1]

    available_obs = window.notna().sum(axis=1)
    enough_obs = available_obs >= min_monthly_obs

    # Stale price proxy from return zeros in the estimation window.
    zero_share = window.eq(0).sum(axis=1) / available_obs.replace(0, np.nan)
    not_stale = zero_share <= stale_zero_return_threshold

    # Must have carbon info at end of year Y for allocation in Y+1.
    has_carbon = carbon_y[year].notna() if year in carbon_y.columns else pd.Series(False, index=returns_m.index)

    # Must be investable at year-end price (PDF rule):
    # we should check price/RI availability at the end of year Y,
    # not the availability of the December return.
    has_december_price = prices_m[dec_y].notna()

    eligible = enough_obs & not_stale & has_carbon & has_december_price
    return eligible[eligible].index.tolist()

def _returns_from_prices_with_delisting(prices_m: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple returns from RI (prices) with a delisting rule:
    - if price at t-1 is observed, price at t becomes NaN and stays NaN
      for the rest of the sample -> assume price drops to 0 at t => R_t = -1
    Otherwise, keep R_t as NaN when it cannot be computed reliably.
    """
    prices = prices_m.copy()
    # Ensure numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")

    arr = prices.to_numpy(dtype=float)
    n_assets, n_months = arr.shape
    rets = np.full((n_assets, n_months), np.nan, dtype=float)

    for i in range(n_assets):
        p = arr[i]
        # Precompute mask of observed prices
        obs = np.isfinite(p)
        for t in range(1, n_months):
            if obs[t] and obs[t - 1]:
                rets[i, t] = (p[t] / p[t - 1]) - 1.0
            elif (not obs[t]) and obs[t - 1]:
                # Candidate delisting month: if all future prices are NaN.
                if np.all(~obs[t:]):
                    rets[i, t] = -1.0
            # else: cannot compute => NaN

    out = pd.DataFrame(rets, index=prices_m.index, columns=prices_m.columns)
    return out


def _min_var_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    w0 = np.repeat(1.0 / n, n)

    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)

    # Small ridge for numerical stability when Σ is nearly singular.
    trace = float(np.trace(cov))
    ridge = (1e-10 * trace / n) if np.isfinite(trace) and trace != 0 else 1e-10
    cov = cov + ridge * np.eye(n)

    # Scale Σ to improve numerical conditioning (weights unaffected).
    scale = float(np.nanmax(np.abs(cov)))
    if np.isfinite(scale) and scale > 0:
        cov = cov / scale

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        # Gradient of w'Σw is 2Σw (since Σ is symmetric).
        return 2.0 * (cov @ w)

    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    res = minimize(
        objective,
        x0=w0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 250, "ftol": 1e-6, "disp": False},
    )
    if (not res.success) or (res.x is None) or (not np.all(np.isfinite(res.x))):
        return w0
    return np.asarray(res.x, dtype=float)


def _compute_year_monthly_returns(
    returns_m: pd.DataFrame,
    mv_m: pd.DataFrame,
    year: int,
    isins: List[str],
    w_mv_start: np.ndarray,
) -> Tuple[pd.Series, pd.Series]:
    months_next_year = returns_m.columns[returns_m.columns.year == (year + 1)]
    months_next_year = months_next_year.sort_values()
    if len(months_next_year) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    rp_mv = {}
    rp_vw = {}

    w_mv = pd.Series(w_mv_start, index=isins, dtype=float)

    for m in months_next_year:
        r = returns_m.loc[isins, m].copy()

        valid = r.notna() & w_mv.notna() & (w_mv != 0.0)
        if valid.any():
            w_eff = w_mv[valid]
            w_eff = w_eff / w_eff.sum()
            rp_m = float((w_eff * r[valid]).sum())
        else:
            rp_m = np.nan
        rp_mv[m] = rp_m

        # Drift MV portfolio weights within year.
        if np.isfinite(rp_m):
            portfolio_gross = 1.0 + rp_m
            if portfolio_gross == 0:
                w_mv.loc[:] = 0.0
            else:
                gross = 1.0 + r[valid]
                w_new_valid = w_eff * gross / portfolio_gross
                w_mv.loc[:] = 0.0
                w_mv.loc[w_new_valid.index] = w_new_valid

        prev_month = returns_m.columns[returns_m.columns < m].max()
        caps_prev = mv_m.loc[isins, prev_month]
        valid_caps = caps_prev.notna()
        if valid_caps.any() and float(caps_prev[valid_caps].sum()) > 0:
            w_vw = caps_prev[valid_caps] / float(caps_prev[valid_caps].sum())
            valid_vw = r[valid_caps].notna()
            if valid_vw.any():
                rp_vw[m] = float((w_vw[valid_vw] * r[valid_caps][valid_vw]).sum())
            else:
                rp_vw[m] = np.nan
        else:
            rp_vw[m] = np.nan

    return pd.Series(rp_mv), pd.Series(rp_vw)


def _annualized_stats(monthly_returns: pd.Series) -> Dict[str, float]:
    """
    Project Section 2.2 asks for the annualized average return mu_bar_p,
    which is the arithmetic annualized return = mean(R) * 12.
    We also report the geometric (compound) annualized return for completeness.
    """
    r = monthly_returns.dropna()
    if r.empty:
        return {
            "ann_return_arith": np.nan,
            "ann_return_geom": np.nan,
            "ann_vol": np.nan,
            "sharpe_arith": np.nan,
            "sharpe_geom": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    ann_return_arith = float(r.mean() * 12.0)
    ann_return_geom = float((1.0 + r).prod() ** (12.0 / len(r)) - 1.0)
    ann_vol = float(r.std(ddof=0) * np.sqrt(12.0))
    sharpe_arith = ann_return_arith / ann_vol if ann_vol > 0 else np.nan
    sharpe_geom = ann_return_geom / ann_vol if ann_vol > 0 else np.nan
    return {
        "ann_return_arith": ann_return_arith,
        "ann_return_geom": ann_return_geom,
        "ann_vol": ann_vol,
        "sharpe_arith": float(sharpe_arith) if np.isfinite(sharpe_arith) else np.nan,
        "sharpe_geom": float(sharpe_geom) if np.isfinite(sharpe_geom) else np.nan,
        "min": float(r.min()),
        "max": float(r.max()),
    }


def run_part1(config: Part1Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ri_raw = _load_panel_csv(config.data_clean_dir / "clean_ri_m.csv")
    mv_raw = _load_panel_csv(config.data_clean_dir / "clean_mv_m.csv")
    carbon_raw = _load_panel_csv(config.data_clean_dir / "clean_carbon_intensity_y.csv")

    prices_m = _extract_monthly_matrix(ri_raw)
    returns_m = _returns_from_prices_with_delisting(prices_m)
    mv_m = _extract_monthly_matrix(mv_raw)
    carbon_y = _extract_annual_matrix(carbon_raw)

    common_isins = sorted(set(returns_m.index) & set(prices_m.index) & set(mv_m.index) & set(carbon_y.index))
    returns_m = returns_m.loc[common_isins]
    prices_m = prices_m.loc[common_isins]
    mv_m = mv_m.loc[common_isins]
    carbon_y = carbon_y.loc[common_isins]

    monthly_mv_all = []
    monthly_vw_all = []
    alloc_rows = []

    for y in range(config.start_rebal_year, config.end_rebal_year + 1):
        investable = _build_investment_set(
            returns_m=returns_m,
            prices_m=prices_m,
            carbon_y=carbon_y,
            year=y,
            estimation_months=config.estimation_months,
            min_monthly_obs=config.min_monthly_obs,
            stale_zero_return_threshold=config.stale_zero_return_threshold,
        )
        if len(investable) == 0:
            continue

        dec_y = _year_end_month(y, returns_m.columns)
        end_pos = returns_m.columns.get_loc(dec_y)
        start_pos = end_pos - config.estimation_months + 1
        window = returns_m.loc[investable, returns_m.columns[start_pos : end_pos + 1]].T
        # Robustify covariance estimation against extreme Datastream artifacts.
        window = window.apply(pd.to_numeric, errors="coerce")
        window = window.replace([np.inf, -np.inf], np.nan)
        cov = window.cov(ddof=0).to_numpy()

        w_mv = _min_var_weights(cov)
        rp_mv, rp_vw = _compute_year_monthly_returns(
            returns_m=returns_m,
            mv_m=mv_m,
            year=y,
            isins=investable,
            w_mv_start=w_mv,
        )
        monthly_mv_all.append(rp_mv)
        monthly_vw_all.append(rp_vw)

        alloc_rows.append(
            {
                "rebal_year": y,
                "n_investable": len(investable),
                "largest_weight_mv": float(np.max(w_mv)),
            }
        )

    r_mv = pd.concat(monthly_mv_all).sort_index()
    r_vw = pd.concat(monthly_vw_all).sort_index()
    returns_out = pd.DataFrame({"R_mv": r_mv, "R_vw": r_vw})
    returns_out.index.name = "date"
    returns_out.to_csv(config.output_dir / "part1_monthly_returns_2014_2025.csv")

    stats = pd.DataFrame(
        [
            {"portfolio": "P_oos_mv", **_annualized_stats(r_mv)},
            {"portfolio": "P_oos_vw", **_annualized_stats(r_vw)},
        ]
    )
    stats.to_csv(config.output_dir / "part1_summary_stats.csv", index=False)
    pd.DataFrame(alloc_rows).to_csv(config.output_dir / "part1_allocation_diagnostics.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        cum = (1.0 + returns_out).cumprod()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cum.index, cum["R_mv"], label="Minimum Variance (oos)")
        ax.plot(cum.index, cum["R_vw"], label="Value Weighted")
        ax.set_title("Cumulative Performance VW vs. Minimum-Variance Portfolio "
            "(Europe, Scope 1 & 2, 2014-2025)")
        ax.set_ylabel("Cumulative Return (Growth of 1$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.output_dir / "part1_cumulative_returns.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped due to environment error: {exc}")

    print("Part I done. Files saved in:", config.output_dir)


if __name__ == "__main__":
    run_part1(Part1Config())
