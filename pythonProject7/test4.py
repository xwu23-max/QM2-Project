import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Re-load and clean data (to ensure we have the dataframe ready)
df = pd.read_csv('complete_metadata.csv')
target_cities = ['london', 'islington', 'kensington']
london_df = df[df['city'].isin(target_cities)].copy()
london_df['line'] = london_df['line'].fillna('Control/Unknown')
london_df = london_df.rename(columns={'line': 'station_name'})
london_df['surface_material'] = london_df['surface_material'].fillna('Unknown')
london_df['surface_material'] = london_df['surface_material'].replace('-', 'Unknown')
# Convert num_reads to numeric, coerce errors just in case
london_df['num_reads'] = pd.to_numeric(london_df['num_reads'], errors='coerce')

# Filter out controls for the main station analysis to focus on actual stations
station_data = london_df[london_df['station_name'] != 'Control/Unknown'].copy()

# Set up the visualization style
sns.set(style="whitegrid")

# Visualization 1: Top 20 Stations by Median Microbial Load (num_reads)
# Using median is more robust to outliers than mean
top_stations = station_data.groupby('station_name')['num_reads'].median().sort_values(ascending=False).head(20).index
plt.figure(figsize=(12, 8))
sns.barplot(x='num_reads', y='station_name', data=station_data[station_data['station_name'].isin(top_stations)],
            order=top_stations, palette='viridis', errorbar=None)
plt.title('Top 20 London Tube Stations by Median Microbial Read Count', fontsize=16)
plt.xlabel('Median Number of Reads (Proxy for Microbial Load)', fontsize=12)
plt.ylabel('Station Name', fontsize=12)
plt.tight_layout()
plt.savefig('top_20_stations_microbial_load.png')


material_data = london_df[~london_df['surface_material'].isin(['Unknown'])]
plt.figure(figsize=(12, 6))
sns.boxplot(x='surface_material', y='num_reads', data=material_data, palette='Set2')
plt.yscale('log') # Log scale is often better for sequencing data ranges
plt.title('Microbial Load Distribution by Surface Material (Log Scale)', fontsize=16)
plt.xlabel('Surface Material', fontsize=12)
plt.ylabel('Number of Reads (Log Scale)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('microbial_load_by_material.png')


geo_data = london_df.dropna(subset=['latitude', 'longitude', 'num_reads'])
plt.figure(figsize=(10, 8))
sc = plt.scatter(geo_data['longitude'], geo_data['latitude'],
                 c=geo_data['num_reads'], cmap='plasma',
                 alpha=0.7, s=50, edgecolors='w', norm=plt.Normalize(vmin=geo_data['num_reads'].min(), vmax=geo_data['num_reads'].max()))
plt.colorbar(sc, label='Number of Reads')
plt.title('Spatial Distribution of Microbial Load in London', fontsize=16)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.tight_layout()
plt.savefig('spatial_distribution_london.png')

print("Visualizations generated.")