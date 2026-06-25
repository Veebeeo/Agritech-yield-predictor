import pandas as pd
import geopandas as gpd
import imdlib as imd
import os

def process_and_fuse_weather():
    print("Loading Partial Master Dataset...")
    master_df = pd.read_csv('./data/processed/master_dataset_partial.csv')
    
    # Identify the exact years we need weather data for based on our yield data
    years = master_df['Year'].unique()
    start_yr, end_yr = int(min(years)), int(max(years))

    print(f"Fetching IMD Rainfall data from {start_yr} to {end_yr}...")
    # get_data automatically skips downloading if the .grd files are already in the folder
    data = imd.get_data('rain', start_yr, end_yr, fn_format='yearwise', file_dir='./data/raw/')
    ds = data.get_xarray()

    print("Converting spatial grids to tabular format and isolating Kharif season...")
    rain_df = ds.to_dataframe().reset_index()

    # Extract month and year for filtering
    rain_df['month'] = rain_df['time'].dt.month
    rain_df['year'] = rain_df['time'].dt.year

    # Filter for Kharif season (June to October)
    kharif_rain = rain_df[(rain_df['month'] >= 6) & (rain_df['month'] <= 10)]

    # IMD 'rain' is daily precipitation in mm. We want the total sum for the season per grid point.
    yearly_grid_rain = kharif_rain.groupby(['lat', 'lon', 'year'])['rain'].sum().reset_index()
    yearly_grid_rain.rename(columns={'rain': 'Kharif_Rainfall_mm'}, inplace=True)

    print("Loading District Boundaries...")
    # IMPORTANT: Update this path if your extracted shapefile has a different name
    shp_path = './data/shapefiles/indian_districts.shp' 
    
    if not os.path.exists(shp_path):
        print(f"ERROR: Shapefile not found at {shp_path}. Please check the filename.")
        return
        
    india_map = gpd.read_file(shp_path)

    # Standardize the district column name. 
    shapefile_district_col = 'District' 
    if shapefile_district_col not in india_map.columns:
        print(f"Available shapefile columns: {india_map.columns.tolist()}")
        print(f"Please update 'shapefile_district_col' in the code to match the district name column.")
        return
        
    india_map['District'] = india_map[shapefile_district_col].astype(str).str.strip().str.lower()

    print("Performing Spatial Joins (This is computationally heavy, please wait)...")
    # Convert the IMD lat/lon coordinates into geographic point objects
    gdf_rain = gpd.GeoDataFrame(
        yearly_grid_rain,
        geometry=gpd.points_from_xy(yearly_grid_rain.lon, yearly_grid_rain.lat),
        crs="EPSG:4326" # Standard WGS84 coordinate system
    )
    
    # Ensure our shapefile uses the exact same coordinate system
    india_map = india_map.to_crs("EPSG:4326")

    # The Spatial Join: Map each point to the polygon it falls within
    joined = gpd.sjoin(gdf_rain, india_map, how="inner", predicate="within")

    # Average the rainfall of all points that fall within a specific district for each year
    district_rain = joined.groupby(['District', 'year'])['Kharif_Rainfall_mm'].mean().reset_index()
    district_rain.rename(columns={'year': 'Year'}, inplace=True)

    print("Merging Weather with the Master Dataset...")
    final_df = pd.merge(master_df, district_rain, on=['District', 'Year'], how='inner')

    # Drop rows that might have NaN values after the merges
    final_df = final_df.dropna(subset=['Yield_Tonnes_Hectare', 'NDVI', 'Kharif_Rainfall_mm'])

    output_path = './data/processed/master_dataset_final.csv'
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(f"Saved to: {output_path}")
    print(f"Final Dataset Shape: {final_df.shape}")
    print("\nPreview of Model-Ready Data:")
    print(final_df[['Year', 'District', 'Yield_Tonnes_Hectare', 'NDVI', 'Kharif_Rainfall_mm']].head())

if __name__ == "__main__":
    process_and_fuse_weather()