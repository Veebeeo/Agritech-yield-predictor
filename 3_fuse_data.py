import pandas as pd

def fuse_datasets():
    print("Fusing Yield and NDVI datasets...")
    
    # Load the cleaned datasets
    yield_df = pd.read_csv('./data/processed/clean_yield_kharif.csv')
    ndvi_df = pd.read_csv('./data/processed/district_ndvi.csv')
    
    # Merge on District and Year
    # We use an 'inner' join to only keep rows where we have BOTH Yield and NDVI data
    merged_df = pd.merge(yield_df, ndvi_df, on=['District', 'Year'], how='inner', suffixes=('', '_ndvi'))
    
    # Drop redundant state column from the merge if it exists
    if 'State_ndvi' in merged_df.columns:
        merged_df = merged_df.drop(columns=['State_ndvi'])
        
    merged_df.to_csv('./data/processed/master_dataset_partial.csv', index=False)
    
    print(f"Fusion complete!")
    print(f"Rows with matching Yield and NDVI data: {len(merged_df)}")
    print("\nSample Data:")
    print(merged_df[['Year', 'District', 'Crop', 'Yield_Tonnes_Hectare', 'NDVI']].head())

fuse_datasets()