from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_part1 import (
    _build_investment_set,
    _compute_year_monthly_returns,
    _extract_annual_matrix,
    _extract_monthly_matrix,
    _load_panel_csv,
    _returns_from_prices_with_delisting,
    _year_end_month,
)


@dataclass(frozen=True)
class Part2Config:
    data_clean_dir: Path = Path("./data_clean")
    output_dir: Path = Path("./outputs_part2")
    start_rebal_year: int = 2013
    end_rebal_year: int = 2024
    estimation_months: int = 120
    min_monthly_obs: int = 36
    stale_zero_return_threshold: float = 0.5
    cf_reduction_ratio: float = 0.5  # Section 3.2 target
    nz_theta: float = 0.10  # Section 4.1: 10% yearly reduction


def _min_var_with_optional_cf_constraint(
    cov: np.ndarray,
    cf_coeff: np.ndarray | None = None,
    cf_limit: float | None = None,
) -> np.ndarray:
    n = cov.shape[0]
    w0 = np.repeat(1.0 / n, n)

    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)

    trace = float(np.trace(cov))
    ridge = (1e-10 * trace / n) if np.isfinite(trace) and trace != 0 else 1e-10
    cov = cov + ridge * np.eye(n)

    scale = float(np.nanmax(np.abs(cov)))
    if np.isfinite(scale) and scale > 0:
        cov = cov / scale

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    if cf_coeff is not None and cf_limit is not None and np.isfinite(cf_limit):
        cf_coeff = np.asarray(cf_coeff, dtype=float)
        constraints.append({"type": "ineq", "fun": lambda w: float(cf_limit - (w @ cf_coeff))})

        # Try to start from a feasible point when possible.
        if float(w0 @ cf_coeff) > cf_limit:
            idx_min_cf = int(np.argmin(cf_coeff))
            if cf_coeff[idx_min_cf] <= cf_limit:
                w0 = np.zeros(n)
                w0[idx_min_cf] = 1.0

    bounds = [(0.0, 1.0)] * n
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


def _te_min_with_cf_constraint(
    cov: np.ndarray,
    w_bench: np.ndarray,
    cf_coeff: np.ndarray,
    cf_limit: float,
) -> np.ndarray:
    n = cov.shape[0]
    w0 = np.asarray(w_bench, dtype=float).copy()
    if w0.shape[0] != n or not np.all(np.isfinite(w0)):
        w0 = np.repeat(1.0 / n, n)

    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    ridge = (1e-10 * trace / n) if np.isfinite(trace) and trace != 0 else 1e-10
    cov = cov + ridge * np.eye(n)

    scale = float(np.nanmax(np.abs(cov)))
    if np.isfinite(scale) and scale > 0:
        cov = cov / scale

    cf_coeff = np.asarray(cf_coeff, dtype=float)

    def objective(w: np.ndarray) -> float:
        d = w - w_bench
        return float(d @ cov @ d)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ (w - w_bench))

    constraints = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "ineq", "fun": lambda w: float(cf_limit - (w @ cf_coeff))},
    ]
    bounds = [(0.0, 1.0)] * n
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


def _annualized_stats(monthly_returns: pd.Series) -> Dict[str, float]:
    r = monthly_returns.dropna()
    if r.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "min": np.nan, "max": np.nan}
    ann_return = (1.0 + r).prod() ** (12.0 / len(r)) - 1.0
    ann_vol = r.std(ddof=0) * np.sqrt(12.0)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "min": float(r.min()),
        "max": float(r.max()),
    }


def run_part2(config: Part2Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ri_raw = _load_panel_csv(config.data_clean_dir / "clean_ri_m.csv")
    mv_m_raw = _load_panel_csv(config.data_clean_dir / "clean_mv_m.csv")
    mv_y_raw = _load_panel_csv(config.data_clean_dir / "clean_mv_y.csv")
    emissions_raw = _load_panel_csv(config.data_clean_dir / "clean_emissions_scope1_scope2_y.csv")
    carbon_raw = _load_panel_csv(config.data_clean_dir / "clean_carbon_intensity_y.csv")
    static_raw = _load_panel_csv(config.data_clean_dir / "clean_static_eur.csv")

    prices_m = _extract_monthly_matrix(ri_raw)
    returns_m = _returns_from_prices_with_delisting(prices_m)
    mv_m = _extract_monthly_matrix(mv_m_raw)
    mv_y = _extract_annual_matrix(mv_y_raw)
    emissions_y = _extract_annual_matrix(emissions_raw)
    carbon_y = _extract_annual_matrix(carbon_raw)

    common_isins = sorted(
        set(returns_m.index)
        & set(prices_m.index)
        & set(mv_m.index)
        & set(mv_y.index)
        & set(emissions_y.index)
        & set(carbon_y.index)
        & set(static_raw.index)
    )
    returns_m = returns_m.loc[common_isins]
    prices_m = prices_m.loc[common_isins]
    mv_m = mv_m.loc[common_isins]
    mv_y = mv_y.loc[common_isins]
    emissions_y = emissions_y.loc[common_isins]
    carbon_y = carbon_y.loc[common_isins]
    static_raw = static_raw.loc[common_isins]

    monthly_mv_all: List[pd.Series] = []
    monthly_mv_cf_all: List[pd.Series] = []
    monthly_vw_all: List[pd.Series] = []
    monthly_vw_cf_all: List[pd.Series] = []
    monthly_vw_nz_all: List[pd.Series] = []
    alloc_rows = []
    waci_cf_rows = []
    top_emitters_rows = []
    cf_vw_baseline_y0: float | None = None

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

        # Keep firms with required annual data for carbon metrics at year y.
        annual_ok = (
            emissions_y[y].notna()
            & mv_y[y].notna()
            & (mv_y[y] > 0)
            & carbon_y[y].notna()
        )
        investable = [i for i in investable if annual_ok.get(i, False)]
        if len(investable) == 0:
            continue

        dec_y = _year_end_month(y, returns_m.columns)
        end_pos = returns_m.columns.get_loc(dec_y)
        start_pos = end_pos - config.estimation_months + 1
        window = returns_m.loc[investable, returns_m.columns[start_pos : end_pos + 1]].T
        window = window.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        cov = window.cov(ddof=0).to_numpy()

        w_mv = _min_var_with_optional_cf_constraint(cov)

        emissions_vec = emissions_y.loc[investable, y].to_numpy(dtype=float)
        cap_vec = mv_y.loc[investable, y].to_numpy(dtype=float)
        ci_vec = carbon_y.loc[investable, y].to_numpy(dtype=float)

        cf_coeff = emissions_vec / cap_vec
        cf_mv = float(w_mv @ cf_coeff)
        waci_mv = float(w_mv @ ci_vec)

        cf_limit = config.cf_reduction_ratio * cf_mv
        w_mv_cf = _min_var_with_optional_cf_constraint(cov, cf_coeff=cf_coeff, cf_limit=cf_limit)
        cf_mv_cf = float(w_mv_cf @ cf_coeff)
        waci_mv_cf = float(w_mv_cf @ ci_vec)

        caps_dec = mv_m.loc[investable, dec_y]
        valid_caps_dec = caps_dec.notna()
        if valid_caps_dec.any() and float(caps_dec[valid_caps_dec].sum()) > 0:
            w_vw = (caps_dec[valid_caps_dec] / float(caps_dec[valid_caps_dec].sum())).reindex(investable).fillna(0.0)
            cf_vw = float(w_vw.to_numpy() @ cf_coeff)
            waci_vw = float(w_vw.to_numpy() @ ci_vec)
        else:
            cf_vw = np.nan
            waci_vw = np.nan
            w_vw = pd.Series(np.repeat(1.0 / len(investable), len(investable)), index=investable)

        if y == config.start_rebal_year and np.isfinite(cf_vw):
            cf_vw_baseline_y0 = float(cf_vw)

        # Section 3.3: TE minimization around VW with CF <= 50% of VW CF.
        cf_limit_vw50 = config.cf_reduction_ratio * cf_vw if np.isfinite(cf_vw) else np.nan
        if np.isfinite(cf_limit_vw50):
            w_vw_cf = _te_min_with_cf_constraint(
                cov=cov,
                w_bench=w_vw.to_numpy(dtype=float),
                cf_coeff=cf_coeff,
                cf_limit=cf_limit_vw50,
            )
        else:
            w_vw_cf = w_vw.to_numpy(dtype=float)
        cf_vw_cf = float(w_vw_cf @ cf_coeff)
        waci_vw_cf = float(w_vw_cf @ ci_vec)

        # Section 4.1: net-zero path with 10% yearly reduction from baseline CF at Y0=2013.
        if cf_vw_baseline_y0 is not None:
            years_since_y0 = y - config.start_rebal_year + 1
            cf_limit_nz = ((1.0 - config.nz_theta) ** years_since_y0) * cf_vw_baseline_y0
            w_vw_nz = _te_min_with_cf_constraint(
                cov=cov,
                w_bench=w_vw.to_numpy(dtype=float),
                cf_coeff=cf_coeff,
                cf_limit=cf_limit_nz,
            )
            cf_vw_nz = float(w_vw_nz @ cf_coeff)
            waci_vw_nz = float(w_vw_nz @ ci_vec)
        else:
            cf_limit_nz = np.nan
            w_vw_nz = w_vw.to_numpy(dtype=float)
            cf_vw_nz = float(w_vw_nz @ cf_coeff)
            waci_vw_nz = float(w_vw_nz @ ci_vec)

        rp_mv, rp_vw = _compute_year_monthly_returns(
            returns_m=returns_m,
            mv_m=mv_m,
            year=y,
            isins=investable,
            w_mv_start=w_mv,
        )
        rp_mv_cf, _ = _compute_year_monthly_returns(
            returns_m=returns_m,
            mv_m=mv_m,
            year=y,
            isins=investable,
            w_mv_start=w_mv_cf,
        )
        rp_vw_cf, _ = _compute_year_monthly_returns(
            returns_m=returns_m,
            mv_m=mv_m,
            year=y,
            isins=investable,
            w_mv_start=w_vw_cf,
        )
        rp_vw_nz, _ = _compute_year_monthly_returns(
            returns_m=returns_m,
            mv_m=mv_m,
            year=y,
            isins=investable,
            w_mv_start=w_vw_nz,
        )

        monthly_mv_all.append(rp_mv)
        monthly_vw_all.append(rp_vw)
        monthly_mv_cf_all.append(rp_mv_cf)
        monthly_vw_cf_all.append(rp_vw_cf)
        monthly_vw_nz_all.append(rp_vw_nz)

        alloc_rows.append(
            {
                "rebal_year": y,
                "n_investable": len(investable),
                "largest_weight_mv": float(np.max(w_mv)),
                "largest_weight_mv_cf50": float(np.max(w_mv_cf)),
                "largest_weight_vw_cf50": float(np.max(w_vw_cf)),
                "largest_weight_vw_nz": float(np.max(w_vw_nz)),
                "cf_limit_mv_cf50": float(cf_limit),
                "cf_limit_vw_cf50": float(cf_limit_vw50) if np.isfinite(cf_limit_vw50) else np.nan,
                "cf_limit_vw_nz": float(cf_limit_nz) if np.isfinite(cf_limit_nz) else np.nan,
            }
        )
        waci_cf_rows.append(
            {
                "year": y,
                "WACI_mv": waci_mv,
                "CF_mv": cf_mv,
                "WACI_vw": waci_vw,
                "CF_vw": cf_vw,
                "WACI_mv_cf50": waci_mv_cf,
                "CF_mv_cf50": cf_mv_cf,
                "WACI_vw_cf50": waci_vw_cf,
                "CF_vw_cf50": cf_vw_cf,
                "WACI_vw_nz": waci_vw_nz,
                "CF_vw_nz": cf_vw_nz,
            }
        )

        # Top 10 carbon-intensity contributors in MV portfolio (by alpha * CI).
        contrib = pd.Series(w_mv * ci_vec, index=investable).sort_values(ascending=False).head(10)
        for rank, (isin, val) in enumerate(contrib.items(), start=1):
            top_emitters_rows.append(
                {
                    "year": y,
                    "rank": rank,
                    "ISIN": isin,
                    "NAME": static_raw.loc[isin, "NAME"] if "NAME" in static_raw.columns else np.nan,
                    "contribution_waci_mv": float(val),
                }
            )

    r_mv = pd.concat(monthly_mv_all).sort_index()
    r_vw = pd.concat(monthly_vw_all).sort_index()
    r_mv_cf = pd.concat(monthly_mv_cf_all).sort_index()
    r_vw_cf = pd.concat(monthly_vw_cf_all).sort_index()
    r_vw_nz = pd.concat(monthly_vw_nz_all).sort_index()
    monthly_returns = pd.DataFrame(
        {"R_mv": r_mv, "R_vw": r_vw, "R_mv_cf50": r_mv_cf, "R_vw_cf50": r_vw_cf, "R_vw_nz": r_vw_nz}
    )
    monthly_returns.index.name = "date"
    monthly_returns.to_csv(config.output_dir / "part2_monthly_returns_2014_2025.csv")

    stats = pd.DataFrame(
        [
            {"portfolio": "P_oos_mv", **_annualized_stats(r_mv)},
            {"portfolio": "P_oos_vw", **_annualized_stats(r_vw)},
            {"portfolio": "P_oos_mv_cf50", **_annualized_stats(r_mv_cf)},
            {"portfolio": "P_oos_vw_cf50", **_annualized_stats(r_vw_cf)},
            {"portfolio": "P_oos_vw_nz", **_annualized_stats(r_vw_nz)},
        ]
    )
    stats.to_csv(config.output_dir / "part2_summary_stats.csv", index=False)
    pd.DataFrame(alloc_rows).to_csv(config.output_dir / "part2_allocation_diagnostics.csv", index=False)
    pd.DataFrame(waci_cf_rows).to_csv(config.output_dir / "part2_waci_cf_by_year.csv", index=False)
    pd.DataFrame(top_emitters_rows).to_csv(config.output_dir / "part2_top10_waci_contributors_mv.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        cum = (1.0 + monthly_returns).cumprod()
        waci_cf_df = pd.DataFrame(waci_cf_rows)

        # Carbon footprint - MV family (own scale to avoid hiding spikes).
        fig_cf_mv, ax_cf_mv = plt.subplots(figsize=(11, 5))
        ax_cf_mv.plot(waci_cf_df["year"], waci_cf_df["CF_mv"], label="CF P_oos_mv", color="tab:orange")
        ax_cf_mv.plot(
            waci_cf_df["year"], waci_cf_df["CF_mv_cf50"], label="CF P_oos_mv(0.5)", color="tab:green"
        )
        ax_cf_mv.set_title("Part II (3.1-3.2) - Carbon Footprint, MV family")
        ax_cf_mv.set_xlabel("Rebalancing year (Y)")
        ax_cf_mv.set_ylabel("Carbon Footprint (tCO2e per $M invested)")
        ax_cf_mv.grid(True, alpha=0.3)
        ax_cf_mv.legend()
        fig_cf_mv.tight_layout()
        fig_cf_mv.savefig(config.output_dir / "part2_carbon_footprint_mv_family.png", dpi=150)
        plt.close(fig_cf_mv)

        # Carbon footprint - VW family.
        fig_cf_vw, ax_cf_vw = plt.subplots(figsize=(11, 5))
        ax_cf_vw.plot(waci_cf_df["year"], waci_cf_df["CF_vw"], label="CF P_oos_vw", color="tab:blue")
        ax_cf_vw.plot(
            waci_cf_df["year"], waci_cf_df["CF_vw_cf50"], label="CF P_oos_vw(0.5)", color="tab:cyan"
        )
        ax_cf_vw.plot(
            waci_cf_df["year"], waci_cf_df["CF_vw_nz"], label="CF P_oos_vw(NZ)", color="tab:red"
        )
        ax_cf_vw.set_title("Part II (3.3-4.1) - Carbon Footprint, VW family")
        ax_cf_vw.set_xlabel("Rebalancing year (Y)")
        ax_cf_vw.set_ylabel("Carbon Footprint (tCO2e per $M invested)")
        ax_cf_vw.grid(True, alpha=0.3)
        ax_cf_vw.legend()
        fig_cf_vw.tight_layout()
        fig_cf_vw.savefig(config.output_dir / "part2_carbon_footprint_vw_family.png", dpi=150)
        plt.close(fig_cf_vw)

        # Cumulative returns - MV family.
        fig_mv, ax_mv = plt.subplots(figsize=(11, 5))
        ax_mv.plot(cum.index, cum["R_mv"], label="P_oos_mv", color="tab:orange")
        ax_mv.plot(cum.index, cum["R_mv_cf50"], label="P_oos_mv(0.5)", color="tab:green")
        ax_mv.set_title("Part II (3.4) - Cumulative Returns, MV vs MV(0.5)")
        ax_mv.set_ylabel("Growth of $1")
        ax_mv.grid(True, alpha=0.3)
        ax_mv.legend()
        fig_mv.tight_layout()
        fig_mv.savefig(config.output_dir / "part2_compare_mv_vs_mv_cf50.png", dpi=150)
        plt.close(fig_mv)

        # Cumulative returns - VW family (3.4 + 4.2 main comparison).
        fig_vw, ax_vw = plt.subplots(figsize=(11, 5))
        ax_vw.plot(cum.index, cum["R_vw"], label="P_oos_vw", color="tab:blue")
        ax_vw.plot(cum.index, cum["R_vw_cf50"], label="P_oos_vw(0.5)", color="tab:cyan")
        ax_vw.plot(cum.index, cum["R_vw_nz"], label="P_oos_vw(NZ)", color="tab:red")
        ax_vw.set_title("Part II (4.2) - Cumulative Returns, VW family")
        ax_vw.set_ylabel("Growth of $1")
        ax_vw.grid(True, alpha=0.3)
        ax_vw.legend()
        fig_vw.tight_layout()
        fig_vw.savefig(config.output_dir / "part2_compare_vw_vw_cf50_vw_nz.png", dpi=150)
        plt.close(fig_vw)

        # Excess return vs VW (highlights small financial cost of decarbonization).
        excess = pd.DataFrame(
            {
                "VW(0.5) - VW": (1.0 + monthly_returns["R_vw_cf50"]).cumprod()
                / (1.0 + monthly_returns["R_vw"]).cumprod()
                - 1.0,
                "VW(NZ) - VW": (1.0 + monthly_returns["R_vw_nz"]).cumprod()
                / (1.0 + monthly_returns["R_vw"]).cumprod()
                - 1.0,
            }
        )
        fig_x, ax_x = plt.subplots(figsize=(11, 5))
        ax_x.plot(excess.index, excess["VW(0.5) - VW"], label="P_oos_vw(0.5) - P_oos_vw", color="tab:cyan")
        ax_x.plot(excess.index, excess["VW(NZ) - VW"], label="P_oos_vw(NZ) - P_oos_vw", color="tab:red")
        ax_x.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax_x.set_title("Part II - Cumulative Excess Performance vs VW")
        ax_x.set_ylabel("Cumulative excess return")
        ax_x.grid(True, alpha=0.3)
        ax_x.legend()
        fig_x.tight_layout()
        fig_x.savefig(config.output_dir / "part2_excess_return_vs_vw.png", dpi=150)
        plt.close(fig_x)

        # All cumulative returns (kept for completeness, separated VW family vs MV family).
        fig_all, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        axes[0].plot(cum.index, cum["R_mv"], label="P_oos_mv", color="tab:orange")
        axes[0].plot(cum.index, cum["R_mv_cf50"], label="P_oos_mv(0.5)", color="tab:green")
        axes[0].set_title("MV family")
        axes[0].set_ylabel("Growth of $1")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(cum.index, cum["R_vw"], label="P_oos_vw", color="tab:blue")
        axes[1].plot(cum.index, cum["R_vw_cf50"], label="P_oos_vw(0.5)", color="tab:cyan")
        axes[1].plot(cum.index, cum["R_vw_nz"], label="P_oos_vw(NZ)", color="tab:red")
        axes[1].set_title("VW family")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig_all.suptitle("Part II - Cumulative Returns by Family (2014-2025)")
        fig_all.tight_layout()
        fig_all.savefig(config.output_dir / "part2_cumulative_returns.png", dpi=150)
        plt.close(fig_all)
    except Exception as exc:
        print(f"Plot skipped due to environment error: {exc}")

    print("Part II (3.1 + 4.2) done. Files saved in:", config.output_dir)


if __name__ == "__main__":
    run_part2(Part2Config())
