import ee
import geemap 
import pandas as pd

# Initialize
ee.Initialize(project='gen-lang-client-0010804108')

def get_ndvi_for_india(start_date, end_date):
    # Load India boundary (using FAO GAUL dataset natively in GEE)
    india = ee.FeatureCollection("FAO/GAUL/2015/level2") \
              .filter(ee.Filter.eq('ADM0_NAME', 'India'))
    
    # Load MODIS NDVI collection
    modis_ndvi = ee.ImageCollection("MODIS/061/MOD13Q1") \
                   .filterBounds(india) \
                   .filterDate(start_date, end_date) \
                   .select('NDVI')
    
    # Get the mean NDVI across the time period
    mean_ndvi = modis_ndvi.mean().clip(india)
    
    print("Successfully created MODIS NDVI composite for India.")
    return mean_ndvi

# Test the function for the 2020 Kharif season
ndvi_image = get_ndvi_for_india('2020-06-01', '2020-10-31')