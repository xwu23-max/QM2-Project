# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 0) Load (use your exported files)
# -----------------------------
station_fp  = Path("london_station_summary.csv")
surface_fp  = Path("london_surface_summary.csv")
material_fp = Path("london_material_summary.csv")
controls_fp = Path("london_controls_like_samples.csv")

station  = pd.read_csv(station_fp)
surface  = pd.read_csv(surface_fp)
material = pd.read_csv(material_fp)
controls = pd.read_csv(controls_fp)

print("Loaded:",
      "\n - station:", station.shape,
      "\n - surface:", surface.shape,
      "\n - material:", material.shape,
      "\n - controls:", controls.shape)

# -----------------------------
# Helpers
# -----------------------------
def savefig(name: str, out_dir="figs", dpi=160):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / name
    plt.tight_layout()
    plt.savefig(fp, dpi=dpi)
    print("Saved:", fp)
    plt.close()

# -----------------------------
# 1) Top stations by sample coverage
# -----------------------------
def plot_top_stations(station_df: pd.DataFrame, top_n=20):
    df = station_df.dropna(subset=["station_norm"]).copy()
    df = df.sort_values("n_samples", ascending=False).head(top_n)
    # barh looks nicer for station names
    plt.figure(figsize=(10, 6))
    plt.barh(df["station_norm"][::-1], df["n_samples"][::-1])
    plt.xlabel("Number of surface samples (per station)")
    plt.ylabel("Station")
    plt.title(f"London stations with most MetaSUB surface samples (Top {top_n})")
    savefig(f"01_top_stations_top{top_n}.png")

plot_top_stations(station, top_n=25)

# -----------------------------
# 2) Surface type distribution
# -----------------------------
def plot_top_surfaces(surface_df: pd.DataFrame, top_n=15):
    df = surface_df.dropna(subset=["surface_canon"]).copy()
    df = df.sort_values("n_samples", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(df["surface_canon"][::-1], df["n_samples"][::-1])
    plt.xlabel("Number of samples")
    plt.ylabel("Surface type (canonicalized)")
    plt.title(f"Most sampled surface types in London (Top {top_n})")
    savefig(f"02_top_surfaces_top{top_n}.png")

plot_top_surfaces(surface, top_n=15)

# -----------------------------
# 3) Material distribution
# -----------------------------
def plot_top_materials(material_df: pd.DataFrame, top_n=15):
    df = material_df.dropna(subset=["material"]).copy()
    df = df.sort_values("n_samples", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(df["material"][::-1], df["n_samples"][::-1])
    plt.xlabel("Number of samples")
    plt.ylabel("Surface material")
    plt.title(f"Most common surface materials in London samples (Top {top_n})")
    savefig(f"03_top_materials_top{top_n}.png")

plot_top_materials(material, top_n=15)

# -----------------------------
# 4) Read depth summaries (station-level)
# -----------------------------
def plot_station_reads_distribution(station_df: pd.DataFrame):
    df = station_df.copy()
    x = pd.to_numeric(df["reads_median"], errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=40)
    plt.xlabel("Median reads per station (merged)")
    plt.ylabel("Count of stations")
    plt.title("Distribution of sequencing depth (median reads) across stations")
    savefig("04_station_reads_median_hist.png")

plot_station_reads_distribution(station)

def plot_station_samples_vs_reads(station_df: pd.DataFrame):
    df = station_df.dropna(subset=["n_samples","reads_median"]).copy()
    df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")
    df["reads_median"] = pd.to_numeric(df["reads_median"], errors="coerce")
    df = df.dropna(subset=["n_samples","reads_median"])
    plt.figure(figsize=(7, 6))
    plt.scatter(df["n_samples"], df["reads_median"])
    plt.xlabel("Number of samples at station")
    plt.ylabel("Median reads at station")
    plt.title("Sampling coverage vs sequencing depth (station-level)")
    savefig("05_station_samples_vs_reads.png")

plot_station_samples_vs_reads(station)

# -----------------------------
# 5) Simple "map" (lon/lat scatter) without basemap
#    (works if station lat/lon exists; will ignore missing)
# -----------------------------
def plot_station_map(station_df: pd.DataFrame):
    df = station_df.dropna(subset=["lat","lon"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])

    if len(df) == 0:
        print("No lat/lon available in station_summary; skip map.")
        return

    # size by samples (scaled)
    sizes = np.clip(df["n_samples"].values, 1, None)
    sizes = 10 + 8 * np.sqrt(sizes)

    plt.figure(figsize=(7, 7))
    # color by reads_median (optional), using default colormap (do not specify cmap)
    cvals = pd.to_numeric(df["reads_median"], errors="coerce").fillna(0).values
    sc = plt.scatter(df["lon"], df["lat"], s=sizes, c=cvals)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("London stations (points) sized by sample count; colored by median reads")
    plt.colorbar(sc, label="Median reads")
    savefig("06_station_map_samples_reads.png")

plot_station_map(station)

# -----------------------------
# 6) OPTIONAL: If you have station-level air quality & traffic data
#    Provide a CSV, e.g. aq_station.csv with columns:
#    - station_norm (same normalization as in station_summary)
#    - pm25_indoor (or pm25)
#    - pm25_outdoor (optional)
#    - no2_outdoor (optional)
#    - traffic (optional)
# -----------------------------
def optional_plots_with_aq(station_df: pd.DataFrame, aq_csv="aq_station.csv"):
    aq_path = Path(aq_csv)
    if not aq_path.exists():
        print(f"[Optional] {aq_csv} not found; skip AQ plots.")
        return

    aq = pd.read_csv(aq_path)
    # merge
    df = station_df.merge(aq, on="station_norm", how="inner")

    # (a) Traffic vs PM (scatter)
    if "traffic" in df.columns and "pm25_indoor" in df.columns:
        x = pd.to_numeric(df["traffic"], errors="coerce")
        y = pd.to_numeric(df["pm25_indoor"], errors="coerce")
        ok = x.notna() & y.notna()
        plt.figure(figsize=(7, 6))
        plt.scatter(x[ok], y[ok])
        plt.xlabel("Traffic (e.g., entries/exits)")
        plt.ylabel("Indoor PM2.5")
        plt.title("Traffic vs indoor PM2.5 (station-level)")
        savefig("07_traffic_vs_pm25_indoor.png")

    # (b) In vs out (box / distribution)
    if "pm25_indoor" in df.columns and "pm25_outdoor" in df.columns:
        ind = pd.to_numeric(df["pm25_indoor"], errors="coerce")
        out = pd.to_numeric(df["pm25_outdoor"], errors="coerce")
        ok = ind.notna() & out.notna()
        plt.figure(figsize=(7, 6))
        plt.boxplot([ind[ok].values, out[ok].values], labels=["Indoor", "Outdoor"])
        plt.ylabel("PM2.5")
        plt.title("Indoor vs Outdoor PM2.5 (station-level)")
        savefig("08_in_vs_out_pm25_boxplot.png")

        # ratio plot
        ratio = (ind[ok] / out[ok]).replace([np.inf, -np.inf], np.nan).dropna()
        plt.figure(figsize=(10, 4))
        plt.hist(ratio, bins=40)
        plt.xlabel("Indoor / Outdoor PM2.5 ratio")
        plt.ylabel("Count of stations")
        plt.title("Distribution of indoor/outdoor PM2.5 ratio")
        savefig("09_in_out_ratio_hist.png")

    # (c) PM vs microbe proxy (here: station reads_median as placeholder proxy)
    # Replace 'reads_median' with real microbial metric later (e.g., Shannon, pathogen_fraction)
    if "pm25_indoor" in df.columns:
        x = pd.to_numeric(df["pm25_indoor"], errors="coerce")
        y = pd.to_numeric(df["reads_median"], errors="coerce")
        ok = x.notna() & y.notna()
        plt.figure(figsize=(7, 6))
        plt.scatter(x[ok], y[ok])
        plt.xlabel("Indoor PM2.5")
        plt.ylabel("Microbe proxy (median reads)  # replace with real microbial metric")
        plt.title("Indoor PM2.5 vs microbe proxy (station-level)")
        savefig("10_pm25_vs_microbe_proxy.png")

    print("[Optional] AQ plots done. (Check figs/ folder)")

optional_plots_with_aq(station, aq_csv="aq_station.csv")
print("All done. Check ./figs for exported PNGs.")
