"""Re-render the six PNG figures used in the report with a cleaner style.

Loads existing CSV outputs and the cleaned data to redraw all figures
with consistent typography, palettes, and grid styling. Outputs replace
the existing PNGs in-place in outputs_part1/ and outputs_part2/.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------- Style ------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.color": "#cccccc",
    "grid.alpha": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "lines.linewidth": 1.7,
    "axes.titlepad": 10,
})

C = {
    "mv":    "#1f6f9c",   # deep blue
    "vw":    "#2c8f3d",   # deep green
    "mv05":  "#1f6f9c",   # same blue, dashed
    "vw05":  "#a23636",   # deep red
    "vwnz":  "#d49b1f",   # gold
    "limit": "#888888",
    "vw_bg": "#9bd1a9",
}

OUT1 = Path("outputs_part1")
OUT2 = Path("outputs_part2")


# ----------------- Helpers ----------------------------------------------
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _pct(x, _):
    return f"{x:.0%}" if abs(x) < 10 else f"{x:.1f}"


def _load_monthly(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ----------------- Figure 1: Part I cumulative returns ------------------
def fig_part1_cum():
    df = _load_monthly(OUT1 / "part1_monthly_returns_2014_2025.csv")
    cum = (1.0 + df).cumprod()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(cum.index, cum["R_mv"], color=C["mv"], label=r"$P^{mv}$")
    ax.plot(cum.index, cum["R_vw"], color=C["vw"], label=r"$P^{vw}$")
    ax.set_title("Cumulative wealth of \u00241 invested, Jan 2014 - Dec 2025")
    ax.set_ylabel("Wealth ($)")
    ax.set_xlabel("")
    ax.set_ylim(bottom=0.9)
    ax.legend(loc="upper left")
    _save(fig, OUT1 / "part1_cumulative_returns.png")


# ----------------- Figure 2 & 5: Carbon footprint split / VW / MV --------
def _load_cf() -> pd.DataFrame:
    df = pd.read_csv(OUT2 / "part2_waci_cf_by_year.csv")
    df = df.set_index("year")
    return df


def fig_carbon_split():
    df = _load_cf()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True,
                             gridspec_kw={"wspace": 0.32})

    # left: MV family
    ax = axes[0]
    ax.plot(df.index, df["CF_mv"],       color=C["mv"],   label=r"CF $P^{mv}$")
    ax.plot(df.index, df["CF_mv_cf50"],  color=C["mv05"], label=r"CF $P^{mv}(0.5)$", linestyle="--")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["WACI_mv"],      color=C["mv"],   label=r"WACI $P^{mv}$",     alpha=0.45)
    ax2.plot(df.index, df["WACI_mv_cf50"], color=C["mv05"], label=r"WACI $P^{mv}(0.5)$",alpha=0.45, linestyle="--")
    ax.set_title("MV family")
    ax.set_ylabel("CF (tCO\u2082/M\u0024)")
    ax2.set_ylabel("WACI (tCO\u2082/M\u0024)", color="#666")
    ax.legend(loc="upper right", fontsize=8.5)
    ax2.grid(False)

    # right: VW family
    ax = axes[1]
    ax.plot(df.index, df["CF_vw"],        color=C["vw"],   label=r"CF $P^{vw}$")
    ax.plot(df.index, df["CF_vw_cf50"],   color=C["vw05"], label=r"CF $P^{vw}(0.5)$",  linestyle="--")
    ax.plot(df.index, df["CF_vw_nz"],     color=C["vwnz"], label=r"CF $P^{vw}(NZ)$",   linestyle="-")
    base = df.loc[2013, "CF_vw"]
    glide = [base * 0.9 ** (y - 2012) for y in df.index]
    ax.plot(df.index, glide, color=C["limit"], linestyle=":", label="NZ glide path", linewidth=1.2)
    ax.set_title("VW family")
    ax.set_ylabel("CF (tCO\u2082/M\u0024)")
    ax.legend(loc="upper right", fontsize=8.5)

    for ax_ in axes:
        ax_.set_xlabel("")

    fig.suptitle("Carbon footprint and WACI by rebalance year",
                 y=1.02, fontsize=12.5, fontweight="bold")
    _save(fig, OUT2 / "part2_carbon_footprint_split.png")


def fig_carbon_mv_family():
    df = _load_cf()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(df.index, df["CF_mv"],       color=C["mv"],   label=r"CF $P^{mv}$",     linewidth=2.1)
    ax.plot(df.index, df["CF_mv_cf50"],  color=C["mv05"], label=r"CF $P^{mv}(0.5)$", linewidth=2.1, linestyle="--")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["WACI_mv"],      color=C["mv"],   label=r"WACI $P^{mv}$",     alpha=0.4)
    ax2.plot(df.index, df["WACI_mv_cf50"], color=C["mv05"], label=r"WACI $P^{mv}(0.5)$",alpha=0.4, linestyle="--")
    ax.set_title("MV family - carbon footprint and WACI by year")
    ax.set_ylabel("CF (tCO\u2082/M\u0024)")
    ax2.set_ylabel("WACI (tCO\u2082/M\u0024)", color="#666")
    ax2.grid(False)
    ax.legend(loc="upper right", fontsize=9.5)
    _save(fig, OUT2 / "part2_carbon_footprint_mv_family.png")


def fig_carbon_vw_family():
    df = _load_cf()
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    base = df.loc[2013, "CF_vw"]
    glide = pd.Series([base * 0.9 ** (y - 2012) for y in df.index], index=df.index)
    ax.fill_between(df.index, 0, glide, color=C["limit"], alpha=0.10, label="below NZ budget")
    ax.plot(df.index, df["CF_vw"],      color=C["vw"],   label=r"$P^{vw}$",      linewidth=2.2)
    ax.plot(df.index, df["CF_vw_cf50"], color=C["vw05"], label=r"$P^{vw}(0.5)$", linewidth=2.2, linestyle="--")
    ax.plot(df.index, df["CF_vw_nz"],   color=C["vwnz"], label=r"$P^{vw}(NZ)$",  linewidth=2.2)
    ax.plot(df.index, glide,            color=C["limit"], linestyle=":", label="NZ glide path", linewidth=1.4)
    ax.set_title("VW family - carbon footprint by year")
    ax.set_ylabel("CF (tCO\u2082/M\u0024)")
    ax.legend(loc="upper right", fontsize=9.5)
    _save(fig, OUT2 / "part2_carbon_footprint_vw_family.png")


# ----------------- Figure 3: Section 3.4 cumulative ---------------------
def fig_section34_cum():
    df = _load_monthly(OUT2 / "part2_monthly_returns_2014_2025.csv")
    cum = (1.0 + df).cumprod()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)

    ax = axes[0]
    ax.plot(cum.index, cum["R_mv"],      color=C["mv"],   label=r"$P^{mv}$")
    ax.plot(cum.index, cum["R_mv_cf50"], color=C["mv05"], label=r"$P^{mv}(0.5)$", linestyle="--")
    ax.set_title("Minimum-variance family")
    ax.set_ylabel("Wealth ($)")
    ax.legend(loc="upper left", fontsize=9.5)

    ax = axes[1]
    ax.plot(cum.index, cum["R_vw"],      color=C["vw"],   label=r"$P^{vw}$")
    ax.plot(cum.index, cum["R_vw_cf50"], color=C["vw05"], label=r"$P^{vw}(0.5)$", linestyle="--")
    ax.plot(cum.index, cum["R_vw_nz"],   color=C["vwnz"], label=r"$P^{vw}(NZ)$")
    ax.set_title("Value-weighted family")
    ax.legend(loc="upper left", fontsize=9.5)

    fig.suptitle("Cumulative wealth of \u00241, Jan 2014 - Dec 2025",
                 y=1.02, fontsize=12.5, fontweight="bold")
    _save(fig, OUT2 / "part2_cumulative_returns.png")


# ----------------- Figure 6: Excess return vs VW ------------------------
def fig_excess_return():
    df = _load_monthly(OUT2 / "part2_monthly_returns_2014_2025.csv")
    cum = (1.0 + df).cumprod()
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    excess_05 = np.log(cum["R_vw_cf50"] / cum["R_vw"]) * 100
    excess_nz = np.log(cum["R_vw_nz"]   / cum["R_vw"]) * 100
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.plot(excess_05.index, excess_05, color=C["vw05"], label=r"$P^{vw}(0.5)$",  linestyle="--", linewidth=2.0)
    ax.plot(excess_nz.index, excess_nz, color=C["vwnz"], label=r"$P^{vw}(NZ)$",   linewidth=2.0)
    ax.set_title(r"Cumulative excess return vs. $P^{vw}$ (log scale, %)")
    ax.set_ylabel("Cumulative log-excess return (%)")
    ax.legend(loc="upper right", fontsize=9.5)
    _save(fig, OUT2 / "part2_excess_return_vs_vw.png")


# ----------------- Run everything ---------------------------------------
def main():
    fig_part1_cum()
    print("  part1_cumulative_returns.png")
    fig_carbon_split()
    print("  part2_carbon_footprint_split.png")
    fig_carbon_mv_family()
    print("  part2_carbon_footprint_mv_family.png")
    fig_carbon_vw_family()
    print("  part2_carbon_footprint_vw_family.png")
    fig_section34_cum()
    print("  part2_cumulative_returns.png")
    fig_excess_return()
    print("  part2_excess_return_vs_vw.png")
    print("All figures regenerated.")


if __name__ == "__main__":
    main()
