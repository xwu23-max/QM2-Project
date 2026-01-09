"""
Clean MetaSUB complete_metadata.csv for London Underground analysis.

What it does:
1) Load complete_metadata.csv
2) Filter city == London (case-insensitive)
3) Create a stable sample_id (sl_name -> metasub_name -> uuid)
4) Merge duplicate rows for the same sample (often multiple flowcells)
   - num_reads: sum
   - other fields: first non-null
5) Create station_name (station if present else line; London data often stores station in 'line')
6) Split into air vs surface subsets
7) Export 3 CSVs + print key stats
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path


def first_nonnull(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else pd.NA


def clean_london_metadata(
    in_csv: str | Path,
    out_dir: str | Path = ".",
    london_name: str = "london",
) -> dict[str, Path]:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = pd.read_csv(in_csv, low_memory=False)

    # 2) Filter London
    lon = df[df["city"].astype(str).str.lower() == london_name].copy()

    # 3) Build stable sample_id
    lon["sample_id"] = lon.get("sl_name")
    m1 = lon["sample_id"].isna() | (lon["sample_id"].astype(str).str.strip() == "")
    lon.loc[m1, "sample_id"] = lon.loc[m1, "metasub_name"]
    m2 = lon["sample_id"].isna() | (lon["sample_id"].astype(str).str.strip() == "")
    lon.loc[m2, "sample_id"] = lon.loc[m2, "uuid"]

    # Ensure num_reads numeric
    lon["num_reads"] = pd.to_numeric(lon.get("num_reads"), errors="coerce")

    # 4) Keep only analysis-relevant columns (if they exist)
    keep_cols = [
        "sample_id", "sl_name", "metasub_name", "uuid",
        "project", "core_project",
        "city", "city_code",
        "latitude", "longitude",
        "location_type", "setting",
        "station", "line",
        "sample_type", "surface", "surface_material",
        "surface_ontology_coarse", "surface_ontology_fine",
        "temperature", "elevation",
        "coastal", "coastal_city",
        "num_reads",
        "hudson_alpha_flowcell", "hudson_alpha_uid", "hudson_alpha_project", "ha_id",
        "index_sequence", "barcode",
    ]
    keep_cols = [c for c in keep_cols if c in lon.columns]
    lon_kept = lon[keep_cols].copy()

    # 5) Merge duplicate flowcell rows per sample_id
    agg = {c: first_nonnull for c in lon_kept.columns if c != "num_reads"}
    agg["num_reads"] = "sum"

    lon_clean = (
        lon_kept.sort_values(["sample_id", "num_reads"], ascending=[True, False])
        .groupby("sample_id", as_index=False)
        .agg(agg)
    )

    # 6) Create station_name (station preferred; else line)
    def pick_station(row):
        st = row.get("station")
        ln = row.get("line")
        if pd.notna(st) and str(st).strip() not in ["", "nan", "None"]:
            return st
        if pd.notna(ln) and str(ln).strip() not in ["", "nan", "None"]:
            return ln
        return pd.NA

    lon_clean["station_name"] = lon_clean.apply(pick_station, axis=1)

    # 7) Split: air vs surface
    project_norm = lon_clean["project"].astype(str).str.upper()
    stype_norm = lon_clean["sample_type"].astype(str).str.lower()

    lon_air = lon_clean[(project_norm == "CSD17_AIR") | (stype_norm.str.contains("air", na=False))].copy()
    lon_surface = lon_clean.drop(lon_air.index).copy()

    # 8) Export
    out_clean = out_dir / "london_metadata_cleaned.csv"
    out_air = out_dir / "london_air_metadata_cleaned.csv"
    out_surface = out_dir / "london_surface_metadata_cleaned.csv"

    lon_clean.to_csv(out_clean, index=False)
    lon_air.to_csv(out_air, index=False)
    lon_surface.to_csv(out_surface, index=False)

    # 9) Print quick stats
    print("=== CLEANING SUMMARY ===")
    print(f"Input file: {in_csv}")
    print(f"Raw rows (all): {len(df)}")
    print(f"Raw rows (London): {len(lon)}")
    print(f"Unique samples after merge (London): {len(lon_clean)}")
    print(f"Air samples (London): {len(lon_air)}")
    print(f"Surface samples (London): {len(lon_surface)}")
    print(f"Columns in cleaned output: {lon_clean.shape[1]}")
    print("\nTop projects (n_samples):")
    print(lon_clean["project"].value_counts().head(10))
    print("\nTop station_name (derived) (n_samples):")
    print(lon_clean["station_name"].dropna().astype(str).str.lower().value_counts().head(10))

    return {"clean_all": out_clean, "clean_air": out_air, "clean_surface": out_surface}


if __name__ == "__main__":
    # Example usage:
    # python clean_metasub_london.py
    outputs = clean_london_metadata(
        in_csv="complete_metadata.csv",
        out_dir="complete_metadata_revise.csv"
    )
    print("\nSaved files:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")
