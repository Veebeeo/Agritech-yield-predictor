import ee
import pandas as pd

# Initialize GEE
ee.Initialize(project='gen-lang-client-0010804108')

def extract_ndvi_for_districts(start_yr, end_yr):
    print(f"Requesting NDVI district aggregates from GEE ({start_yr}-{end_yr})...")
    
    # Load India District boundaries (FAO GAUL dataset)
    india_districts = ee.FeatureCollection("FAO/GAUL/2015/level2") \
                        .filter(ee.Filter.eq('ADM0_NAME', 'India'))

    results = []
    
    for year in range(start_yr, end_yr + 1):
        # Define Kharif season dates for the year
        start_date = f'{year}-06-01'
        end_date = f'{year}-10-31'
        
        # Get mean NDVI for this season
        modis = ee.ImageCollection("MODIS/061/MOD13Q1") \
                  .filterBounds(india_districts) \
                  .filterDate(start_date, end_date) \
                  .select('NDVI')
        
        season_mean = modis.mean()
        
        # Calculate the mean NDVI for each district polygon
        # Scale is 250m (MODIS resolution)
        district_means = season_mean.reduceRegions(
            collection=india_districts,
            reducer=ee.Reducer.mean(),
            scale=250 
        )
        
        features = district_means.getInfo()['features']
        
        for feat in features:
            props = feat['properties']
            # Divide by 10000 because MODIS NDVI is scaled by 10000 in GEE
            ndvi_val = props.get('mean', None)
            if ndvi_val is not None:
                results.append({
                    'Year': year,
                    'District': str(props.get('ADM2_NAME', '')).strip().lower(),
                    'State': str(props.get('ADM1_NAME', '')).strip().lower(),
                    'NDVI': ndvi_val / 10000.0 
                })
        print(f"Processed NDVI for {year}")

    df_ndvi = pd.DataFrame(results)
    df_ndvi.to_csv('./data/processed/district_ndvi.csv', index=False)
    print("NDVI extraction complete. Saved to ./data/processed/district_ndvi.csv")

extract_ndvi_for_districts(2016, 2016)