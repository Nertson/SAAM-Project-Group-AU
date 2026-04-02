from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SAAMConfig:
    data_dir: Path
    output_dir: Path
    region_code: str = "EUR"
    stale_zero_return_threshold: float = 0.5
    min_monthly_obs: int = 36
    price_floor: float = 0.5


def _load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_excel_from_candidates(data_dir: Path, candidates: list[str]) -> pd.DataFrame:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return _load_excel(p)
    raise FileNotFoundError(
        "None of the expected files were found: "
        + ", ".join(str(data_dir / c) for c in candidates)
    )


def _detect_id_column(df: pd.DataFrame) -> str:
    candidates = ["ISIN", "Isin", "isin", "Code", "code"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No ISIN-like identifier column found.")


def _detect_region_column(df: pd.DataFrame) -> str:
    candidates = ["Region", "REGION", "region"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No region column found in Static file.")


def _normalize_index(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    out[id_col] = out[id_col].astype(str).str.strip()
    out = out.dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    return out.set_index(id_col)


def _coerce_timeseries_numeric(ts: pd.DataFrame) -> pd.DataFrame:
    out = ts.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _year_columns(columns: Iterable[str]) -> list[str]:
    years = []
    for c in columns:
        s = str(c)
        if s.isdigit() and len(s) == 4:
            years.append(s)
    return years


def _sort_year_columns(cols: list[str]) -> list[str]:
    return sorted(cols, key=lambda x: int(x))


def _fill_annual_with_project_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    SAAM rule:
    - Missing in the middle or at end: fill with previous year
    - Missing at beginning: keep missing
    """
    out = df.copy()
    return out.ffill(axis=1)


def _clean_monthly_prices(price_df: pd.DataFrame, price_floor: float) -> pd.DataFrame:
    cleaned = price_df.copy()
    cleaned = cleaned.mask(cleaned < price_floor, np.nan)
    return cleaned


def _monthly_returns_from_ri(ri_df: pd.DataFrame) -> pd.DataFrame:
    returns = ri_df.pct_change(axis=1, fill_method=None)
    return returns


def _keep_common_isins(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common = None
    for df in frames.values():
        idx = set(df.index)
        common = idx if common is None else (common & idx)
    if common is None:
        return frames
    common_sorted = sorted(common)
    return {k: v.loc[common_sorted].copy() for k, v in frames.items()}


def clean_saam_data(config: SAAMConfig) -> Dict[str, pd.DataFrame]:
    data_dir = config.data_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    static = _load_excel_from_candidates(
        data_dir,
        ["Static_2025.xlsx", "Static.xlsx"],
    )
    co2_s1 = _load_excel_from_candidates(
        data_dir,
        ["DS_CO2_SCOPE_1_Y_2025.xlsx", "DS CO2 SCOPE 1.xlsx"],
    )
    co2_s2 = _load_excel_from_candidates(
        data_dir,
        ["DS_CO2_SCOPE_2_Y_2025.xlsx", "DS CO2 SCOPE 2.xlsx"],
    )
    rev_y = _load_excel_from_candidates(
        data_dir,
        ["DS_REV_Y_2025.xlsx", "DS REV USD Y.xlsx"],
    )
    mv_y = _load_excel_from_candidates(
        data_dir,
        ["DS_MV_T_USD_Y_2025.xlsx", "DS MV T USD Y.xlsx"],
    )
    mv_m = _load_excel_from_candidates(
        data_dir,
        ["DS_MV_T_USD_M_2025.xlsx", "DS MV T USD M.xlsx"],
    )
    ri_y = _load_excel_from_candidates(
        data_dir,
        ["DS_RI_T_USD_Y_2025.xlsx", "DS RI T USD Y.xlsx"],
    )
    ri_m = _load_excel_from_candidates(
        data_dir,
        ["DS_RI_T_USD_M_2025.xlsx", "DS RI T USD M.xlsx"],
    )

    id_col = _detect_id_column(static)
    region_col = _detect_region_column(static)

    static = _normalize_index(static, id_col)
    target_isins = static.index[static[region_col].astype(str).str.upper() == config.region_code].tolist()
    static_eur = static.loc[target_isins].copy()

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        local_id_col = _detect_id_column(df)
        out = _normalize_index(df, local_id_col)
        out = out.loc[out.index.intersection(target_isins)]
        out = _coerce_timeseries_numeric(out)
        return out

    co2_s1 = prep(co2_s1)
    co2_s2 = prep(co2_s2)
    rev_y = prep(rev_y)
    mv_y = prep(mv_y)
    mv_m = prep(mv_m)
    ri_y = prep(ri_y)
    ri_m = prep(ri_m)

    # Remove firms with no market data at all (full row missing).
    has_any_price = ri_m.notna().any(axis=1)
    valid_price_isins = ri_m.index[has_any_price]
    static_eur = static_eur.loc[static_eur.index.intersection(valid_price_isins)]

    frames = {
        "co2_s1": co2_s1,
        "co2_s2": co2_s2,
        "rev_y": rev_y,
        "mv_y": mv_y,
        "mv_m": mv_m,
        "ri_y": ri_y,
        "ri_m": ri_m,
    }
    frames = {k: v.loc[v.index.intersection(static_eur.index)] for k, v in frames.items()}
    frames = _keep_common_isins(frames)
    static_eur = static_eur.loc[frames["ri_m"].index]

    # Annual fill rules (forward fill only).
    # Per project instructions, we apply the "fill missing annual values with previous year"
    # rule to CO2 emissions and revenues. For market variables (mv_y, ri_y), we keep raw values.
    for key in ["co2_s1", "co2_s2", "rev_y"]:
        year_cols = _sort_year_columns(_year_columns(frames[key].columns))
        if year_cols:
            frames[key][year_cols] = _fill_annual_with_project_rules(frames[key][year_cols])

    # Combined emissions (Scope 1 + Scope 2).
    common_year_cols = sorted(
        set(_year_columns(frames["co2_s1"].columns)) & set(_year_columns(frames["co2_s2"].columns)),
        key=lambda x: int(x),
    )
    emissions_s1s2 = pd.DataFrame(index=frames["co2_s1"].index)
    emissions_s1s2[common_year_cols] = (
        frames["co2_s1"][common_year_cols] + frames["co2_s2"][common_year_cols]
    )

    # Carbon intensity = emissions / revenues_in_million_usd ; revenues given in thousands USD.
    rev_common_cols = sorted(
        set(common_year_cols) & set(_year_columns(frames["rev_y"].columns)),
        key=lambda x: int(x),
    )
    carbon_intensity = pd.DataFrame(index=emissions_s1s2.index)
    rev_in_million = frames["rev_y"][rev_common_cols] / 1000.0
    carbon_intensity[rev_common_cols] = emissions_s1s2[rev_common_cols] / rev_in_million.replace(0, np.nan)

    # Monthly prices cleanup and returns.
    cleaned_ri_m = _clean_monthly_prices(frames["ri_m"], config.price_floor)
    returns_m = _monthly_returns_from_ri(cleaned_ri_m)

    # Stale-price diagnostic over full monthly sample (you can re-compute by rolling window later).
    zero_share = returns_m.eq(0).sum(axis=1) / returns_m.notna().sum(axis=1).replace(0, np.nan)
    stale_flag = zero_share > config.stale_zero_return_threshold

    # Minimum data availability diagnostic.
    monthly_obs = returns_m.notna().sum(axis=1)
    enough_obs_flag = monthly_obs >= config.min_monthly_obs

    diagnostics = pd.DataFrame(
        {
            "stale_zero_return_share": zero_share,
            "is_stale": stale_flag,
            "monthly_obs_count": monthly_obs,
            "enough_monthly_obs": enough_obs_flag,
        }
    )

    # Save cleaned outputs.
    static_eur.to_csv(output_dir / "clean_static_eur.csv")
    emissions_s1s2.to_csv(output_dir / "clean_emissions_scope1_scope2_y.csv")
    frames["rev_y"].to_csv(output_dir / "clean_revenue_y.csv")
    frames["mv_y"].to_csv(output_dir / "clean_mv_y.csv")
    frames["mv_m"].to_csv(output_dir / "clean_mv_m.csv")
    cleaned_ri_m.to_csv(output_dir / "clean_ri_m.csv")
    returns_m.to_csv(output_dir / "clean_returns_m.csv")
    carbon_intensity.to_csv(output_dir / "clean_carbon_intensity_y.csv")
    diagnostics.to_csv(output_dir / "clean_diagnostics.csv")

    return {
        "static_eur": static_eur,
        "emissions_s1s2_y": emissions_s1s2,
        "revenue_y": frames["rev_y"],
        "mv_y": frames["mv_y"],
        "mv_m": frames["mv_m"],
        "ri_m_clean": cleaned_ri_m,
        "returns_m": returns_m,
        "carbon_intensity_y": carbon_intensity,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    # Edit these two paths, then run: python data_cleaning.py
    cfg = SAAMConfig(
        data_dir=Path("./data_raw"),
        output_dir=Path("./data_clean"),
        region_code="EUR",
    )
    outputs = clean_saam_data(cfg)
    print("Cleaning done. Tables generated:")
    for key, value in outputs.items():
        print(f"- {key}: {value.shape}")
