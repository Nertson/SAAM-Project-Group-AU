"""
Robustness checks and active-weight decomposition for Part II.

Runs the Part II backtest under five sensitivity scenarios and computes
firm-level / country-level active weights for the carbon-constrained
value-weighted portfolio. Outputs:
  outputs_part2/robustness.csv             -- annualised geom return, vol,
                                              sharpe of each variant
  outputs_part2/active_weights_2013.csv    -- top over/underweights vs VW
                                              in 2013
  outputs_part2/active_weights_2024.csv    -- same for 2024
  outputs_part2/active_weights_country.csv -- country-level active tilts
                                              (averaged over the sample)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", over="ignore", invalid="ignore")

from data_cleaning import clean_saam_data, SAAMConfig
from portfolio_part1 import _build_investment_set, _year_end_month


# --------------------------------------------------------------------------
def condition_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    n = cov.shape[0]
    trace = float(np.trace(cov))
    ridge = (1e-10 * trace / n) if np.isfinite(trace) and trace != 0 else 1e-10
    cov = cov + ridge * np.eye(n)
    scale = float(np.nanmax(np.abs(cov)))
    if np.isfinite(scale) and scale > 0:
        cov = cov / scale
    return cov


def sample_cov(window: pd.DataFrame) -> np.ndarray:
    w = window.replace([np.inf, -np.inf], np.nan)
    return condition_cov(w.cov(ddof=0).to_numpy())


def lw_cov(window: pd.DataFrame) -> np.ndarray:
    """Ledoit--Wolf shrinkage on a complete-case data block."""
    w = window.replace([np.inf, -np.inf], np.nan)
    w = w.dropna(axis=0, how="any")  # complete-case rows (months)
    if w.empty or w.shape[0] < 12:
        return condition_cov(np.eye(window.shape[1]))
    lw = LedoitWolf().fit(w.values)
    return condition_cov(lw.covariance_)


def solve_te_min(
    cov: np.ndarray, w_b: np.ndarray, cf: np.ndarray, cap: float
) -> np.ndarray:
    n = cov.shape[0]
    res = minimize(
        lambda x: float((x - w_b) @ cov @ (x - w_b)),
        x0=w_b.copy(),
        jac=lambda x: 2.0 * (cov @ (x - w_b)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[
            {"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},
            {"type": "ineq", "fun": lambda x: float(cap - (x @ cf))},
        ],
        options={"maxiter": 250, "ftol": 1e-6, "disp": False},
    )
    return np.asarray(res.x, dtype=float)


def parse_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Coerce annual int columns or monthly datetime columns."""
    df = df.copy()
    if kind == "annual":
        df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
    elif kind == "monthly":
        parsed = pd.to_datetime(df.columns, errors="coerce")
        keep = parsed.notna()
        df = df.loc[:, df.columns[keep]]
        df.columns = parsed[keep]
        df = df.sort_index(axis=1)
    return df


def annual_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return dict(arith=np.nan, geom=np.nan, vol=np.nan, sharpe_geom=np.nan)
    arith = float(r.mean() * 12.0)
    geom = float((1.0 + r).prod() ** (12.0 / len(r)) - 1.0)
    vol = float(r.std(ddof=0) * np.sqrt(12.0))
    return dict(
        arith=arith, geom=geom, vol=vol,
        sharpe_geom=geom / vol if vol > 0 else np.nan,
    )


# --------------------------------------------------------------------------
def run_vw_cf50(
    prices_m: pd.DataFrame,
    returns_m: pd.DataFrame,
    mv_m: pd.DataFrame,
    emissions_y: pd.DataFrame,
    mv_y: pd.DataFrame,
    ci_y: pd.DataFrame,
    *,
    min_monthly_obs: int,
    stale_threshold: float,
    cov_fn=sample_cov,
    estimation_months: int = 120,
    start_year: int = 2013,
    end_year: int = 2024,
    track_weights: bool = False,
):
    """Build the P_vw(0.5) monthly return series under the given config."""
    monthly = []
    weights_dict: dict[int, pd.Series] = {}
    benchmarks_dict: dict[int, pd.Series] = {}

    for y in range(start_year, end_year + 1):
        inv = _build_investment_set(
            returns_m, prices_m, ci_y, y,
            estimation_months, min_monthly_obs, stale_threshold,
        )
        ok = (emissions_y[y].notna() & mv_y[y].notna() & (mv_y[y] > 0) & ci_y[y].notna())
        inv = [i for i in inv if ok.get(i, False)]
        if not inv:
            continue

        dec = _year_end_month(y, returns_m.columns)
        end_pos = returns_m.columns.get_loc(dec)
        window = returns_m.loc[inv, returns_m.columns[end_pos - estimation_months + 1 : end_pos + 1]].T
        cov = cov_fn(window)

        em_vec = emissions_y.loc[inv, y].to_numpy(float)
        cap_vec = mv_y.loc[inv, y].to_numpy(float)
        cf_coeff = em_vec / cap_vec
        cap_tot = float(np.nansum(cap_vec))
        w_vw = np.where(np.isfinite(cap_vec) & (cap_vec > 0), cap_vec / cap_tot, 0.0)
        cf_vw = float(np.nansum(em_vec)) / cap_tot
        cap_y = 0.5 * cf_vw

        w_p = solve_te_min(cov, w_vw, cf_coeff, cap_y)

        if track_weights:
            weights_dict[y] = pd.Series(w_p, index=inv)
            benchmarks_dict[y] = pd.Series(w_vw, index=inv)

        # OOS monthly returns with end-of-month cap drift.
        months_next = returns_m.columns[returns_m.columns.year == y + 1].sort_values()
        w_cur = pd.Series(w_p, index=inv, dtype=float).copy()
        for m in months_next:
            r_assets = returns_m.loc[inv, m]
            rp = float((w_cur * r_assets.fillna(0)).sum())
            monthly.append((m, rp))
            w_cur = w_cur * (1.0 + r_assets.fillna(0))
            s = w_cur.sum()
            if s > 0:
                w_cur = w_cur / s

    series = pd.Series(dict(monthly)).sort_index()
    return series, weights_dict, benchmarks_dict


def main():
    cfg = SAAMConfig(
        data_dir=Path("./data_raw"),
        output_dir=Path("./data_clean"),
        region_code="EUR",
    )
    cleaned = clean_saam_data(cfg)

    prices_m = parse_columns(cleaned["ri_m_clean"], "monthly")
    returns_m = parse_columns(cleaned["returns_m"], "monthly")
    mv_m = parse_columns(cleaned["mv_m"], "monthly")

    emissions_y = parse_columns(cleaned["emissions_s1s2_y"], "annual")
    mv_y = parse_columns(cleaned["mv_y"], "annual")
    ci_y = parse_columns(cleaned["carbon_intensity_y"], "annual")
    static = pd.read_csv("data_clean/clean_static_eur.csv", index_col="ISIN")

    # ----------------------------------------------------------------------
    # 1. ROBUSTNESS GRID for P_vw(0.5)
    # ----------------------------------------------------------------------
    scenarios = [
        ("Baseline (stale 50%, min 36, sample cov)",
            dict(min_monthly_obs=36, stale_threshold=0.5, cov_fn=sample_cov)),
        ("Stale threshold 40%",
            dict(min_monthly_obs=36, stale_threshold=0.4, cov_fn=sample_cov)),
        ("Stale threshold 60%",
            dict(min_monthly_obs=36, stale_threshold=0.6, cov_fn=sample_cov)),
        ("Min history 24 months",
            dict(min_monthly_obs=24, stale_threshold=0.5, cov_fn=sample_cov)),
        ("Min history 48 months",
            dict(min_monthly_obs=48, stale_threshold=0.5, cov_fn=sample_cov)),
        ("Ledoit-Wolf shrinkage covariance",
            dict(min_monthly_obs=36, stale_threshold=0.5, cov_fn=lw_cov)),
    ]
    rows = []
    for label, kw in scenarios:
        print(f"  running: {label}")
        r, _, _ = run_vw_cf50(
            prices_m, returns_m, mv_m, emissions_y, mv_y, ci_y, **kw
        )
        s = annual_stats(r)
        s["scenario"] = label
        rows.append(s)
    df_rob = pd.DataFrame(rows)[["scenario", "arith", "geom", "vol", "sharpe_geom"]]
    Path("outputs_part2/robustness.csv").write_text(df_rob.to_csv(index=False))
    print("\n=== Robustness ===")
    print(df_rob.round(4).to_string(index=False))

    # ----------------------------------------------------------------------
    # 2. ACTIVE WEIGHTS in 2013 and 2024 (P_vw(0.5) - VW)
    # ----------------------------------------------------------------------
    print("\n=== Computing active weights ===")
    _, weights, bmks = run_vw_cf50(
        prices_m, returns_m, mv_m, emissions_y, mv_y, ci_y,
        min_monthly_obs=36, stale_threshold=0.5, cov_fn=sample_cov,
        track_weights=True,
    )

    for year in (2013, 2024):
        w = weights[year]
        wb = bmks[year]
        delta = (w - wb) * 100.0  # percentage points
        names = static["NAME"].reindex(delta.index)
        country = static["Country"].reindex(delta.index)
        em = emissions_y.loc[delta.index, year]
        cap = mv_y.loc[delta.index, year]
        ci = ci_y.loc[delta.index, year]
        cf_coeff = em / cap

        out = pd.DataFrame({
            "ISIN": delta.index,
            "NAME": names.values,
            "Country": country.values,
            "w_vw_pct": wb.values * 100,
            "w_constr_pct": w.values * 100,
            "active_pct": delta.values,
            "carbon_intensity": ci.values,
            "cf_coeff": cf_coeff.values,
        })

        top_under = out.sort_values("active_pct").head(10)
        top_over = out.sort_values("active_pct", ascending=False).head(10)
        result = pd.concat([
            top_over.assign(side="overweight"),
            top_under.assign(side="underweight"),
        ])
        result.to_csv(f"outputs_part2/active_weights_{year}.csv", index=False)
        print(f"\n  Year {year} - top 5 underweights vs VW (constraint cuts these):")
        print(top_under.head(5)[["NAME", "Country", "w_vw_pct", "w_constr_pct", "active_pct", "carbon_intensity"]].to_string(index=False))
        print(f"\n  Year {year} - top 5 overweights vs VW (constraint shifts here):")
        print(top_over.head(5)[["NAME", "Country", "w_vw_pct", "w_constr_pct", "active_pct", "carbon_intensity"]].to_string(index=False))

    # ----------------------------------------------------------------------
    # 3. COUNTRY-LEVEL active tilt (avg over years)
    # ----------------------------------------------------------------------
    print("\n=== Country-level active tilt (averaged over years) ===")
    all_deltas = []
    for y, w in weights.items():
        wb = bmks[y]
        d = (w - wb).to_frame("active").assign(year=y, ISIN=lambda x: x.index)
        all_deltas.append(d)
    df = pd.concat(all_deltas)
    df["Country"] = static["Country"].reindex(df["ISIN"]).values
    by_country = df.groupby(["year", "Country"])["active"].sum().unstack("year").mean(axis=1) * 100
    by_country = by_country.sort_values()
    out = pd.DataFrame({
        "Country": by_country.index,
        "avg_active_pct": by_country.values,
    })
    out.to_csv("outputs_part2/active_weights_country.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
