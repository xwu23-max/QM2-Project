from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Helpers
# ----------------------------
def canon_surface(s):
    """Canonicalize surface strings like 'platform;bench' and 'bench;platform'."""
    if pd.isna(s):
        return np.nan
    parts = [p.strip().lower() for p in str(s).split(";") if p.strip()]
    parts = sorted(parts)
    return ";".join(parts) if parts else np.nan

def norm_station(s):
    """Light normalization for station name matching."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower().strip()
    for ch in [".", ",", "'", "’", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s

def describe_reads(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return x.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99])

# ----------------------------
# Load cleaned data
# ----------------------------
clean_all = pd.read_csv("complete_metadata_revise/london_metadata_cleaned.csv")
clean_air = pd.read_csv("complete_metadata_revise/london_air_metadata_cleaned.csv")
clean_surface = pd.read_csv("complete_metadata_revise/london_surface_metadata_cleaned.csv")

print("=== BASIC COUNTS ===")
print("All London samples:", len(clean_all))
print("Surface samples:", len(clean_surface))
print("Air samples:", len(clean_air))
print()

# ----------------------------
# Identify & remove likely controls (surface subset)
# (common pattern: station_name and surface both missing)
# ----------------------------
controls = clean_surface[(clean_surface["station_name"].isna()) & (clean_surface["surface"].isna())].copy()
surf_use = clean_surface.drop(controls.index).copy()

print("=== CONTROL-LIKE SAMPLES (surface subset) ===")
print("n_controls_like:", len(controls))
if len(controls):
    print("Top 5 controls by num_reads:")
    print(controls.sort_values("num_reads", ascending=False)[["sample_id","metasub_name","num_reads"]].head(5))
print()

# ----------------------------
# Feature engineering for station-level analysis
# ----------------------------
surf_use["surface_canon"] = surf_use["surface"].apply(canon_surface)
surf_use["station_norm"] = surf_use["station_name"].apply(norm_station)

print("=== COVERAGE (surface station samples) ===")
print("usable surface samples:", len(surf_use))
print("unique stations:", surf_use["station_norm"].dropna().nunique())
print("unique surface types (canon):", surf_use["surface_canon"].dropna().nunique())
print("unique materials:", surf_use["surface_material"].dropna().astype(str).str.lower().nunique())
print()

print("=== TOP SURFACES ===")
print(surf_use["surface_canon"].value_counts().head(10))
print()

print("=== TOP MATERIALS ===")
print(surf_use["surface_material"].astype(str).str.lower().value_counts().head(10))
print()

print("=== READ DEPTH (usable surface samples) ===")
print(describe_reads(surf_use["num_reads"]))
print()

# ----------------------------
# Station-level summary table (for merging with air quality later)
# ----------------------------
station_summary = (
    surf_use.groupby("station_norm", dropna=False)
    .agg(
        n_samples=("sample_id","count"),
        n_surface_types=("surface_canon", lambda x: x.dropna().nunique()),
        top_surface=("surface_canon", lambda x: x.dropna().value_counts().index[0] if x.dropna().size else pd.NA),
        material_mode=("surface_material", lambda x: x.dropna().astype(str).str.lower().value_counts().index[0] if x.dropna().size else pd.NA),
        reads_median=("num_reads","median"),
        reads_p90=("num_reads", lambda x: np.nanpercentile(pd.to_numeric(x, errors="coerce"), 90)),
        lat=("latitude","median"),
        lon=("longitude","median"),
    )
    .reset_index()
    .sort_values(["n_samples","reads_median"], ascending=[False, False])
)

surface_summary = (
    surf_use.groupby("surface_canon")
    .agg(
        n_samples=("sample_id","count"),
        n_stations=("station_norm", lambda x: x.dropna().nunique()),
        reads_median=("num_reads","median"),
        reads_mean=("num_reads","mean"),
    )
    .reset_index()
    .sort_values("n_samples", ascending=False)
)

material_summary = (
    surf_use.assign(material=surf_use["surface_material"].astype(str).str.lower())
    .groupby("material")
    .agg(
        n_samples=("sample_id","count"),
        n_stations=("station_norm", lambda x: x.dropna().nunique()),
        reads_median=("num_reads","median"),
    )
    .reset_index()
    .sort_values("n_samples", ascending=False)
)

station_summary.to_csv("london_station_summary.csv", index=False)
surface_summary.to_csv("london_surface_summary.csv", index=False)
material_summary.to_csv("london_material_summary.csv", index=False)
controls.to_csv("london_controls_like_samples.csv", index=False)

print("Saved:")
print("- london_station_summary.csv")
print("- london_surface_summary.csv")
print("- london_material_summary.csv")
print("- london_controls_like_samples.csv")
