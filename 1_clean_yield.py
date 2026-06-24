import pandas as pd
import os

def clean_yield_data(file_path):
    print(f"Loading crop yield data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Filter for Kharif season using the correct lowercase column name
    df_kharif = df[df['season'].str.strip().str.lower() == 'kharif'].copy()
    
    cols_to_keep = {
        'year': 'Year',
        'state_name': 'State',
        'district_name': 'District',
        'crop_name': 'Crop',
        'area': 'Area_Hectares',
        'production': 'Production_Tonnes',
        'yield': 'Yield_Tonnes_Hectare'
    }
    
    # Filter and rename
    df_kharif = df_kharif[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    
    # Clean the Year column (handles both "2016-2017" and "2016" formats safely)
    df_kharif['Year'] = df_kharif['Year'].astype(str).str.split('-').str[0].astype(int)
    
    # Standardize string columns (lowercase, strip whitespace) to prevent merge errors later
    df_kharif['District'] = df_kharif['District'].astype(str).str.strip().str.lower()
    df_kharif['State'] = df_kharif['State'].astype(str).str.strip().str.lower()
    
    # Drop rows where our target variable (Yield) is missing
    df_kharif = df_kharif.dropna(subset=['Yield_Tonnes_Hectare'])
    
    # Ensure the output directory exists
    os.makedirs('./data/processed', exist_ok=True)
    
    # Save processed tabular data
    output_path = './data/processed/clean_yield_kharif.csv'
    df_kharif.to_csv(output_path, index=False)
    print(f"Cleaned yield data saved to {output_path} | Shape: {df_kharif.shape}")

# Run the function
clean_yield_data('./data/raw/crop-wise-area-production-yield.csv')