import imdlib as imd

def download_imd_data(start_yr, end_yr, variable='rain'):
    # variables can be 'rain', 'tmax', or 'tmin'
    print(f"Downloading {variable} data from {start_yr} to {end_yr}...")
    
    # This downloads the raw .grd files to your working directory
    data = imd.get_data(variable, start_yr, end_yr, fn_format='yearwise', file_dir='./data/raw/')
    print(f"Download complete for {variable}.")

# Test by downloading 1 year of rainfall data (2020)
download_imd_data(2020, 2020, 'rain')