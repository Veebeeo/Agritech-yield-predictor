import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def train_optimized_predictor():
    print("Loading master dataset...")
    df = pd.read_csv('./data/processed/master_dataset_final.csv')

    print("Engineering new features...")
    # 1. Advanced Feature Engineering: Interaction Term
    # Crops need water, but vegetation health (NDVI) combined with water tells a richer story.
    df['NDVI_Rain_Interaction'] = df['NDVI'] * df['Kharif_Rainfall_mm']

    # Define Features (X), Target (y), and Groups (for Spatial CV)
    features = ['NDVI', 'Kharif_Rainfall_mm', 'NDVI_Rain_Interaction', 'Crop', 'State']
    target = 'Yield_Tonnes_Hectare'

    X = df[features]
    y = df[target]
    
    # We use District as our grouping variable. This ensures that all data for a 
    # specific district is kept entirely in the test fold during cross-validation,
    # proving the model can generalize to unseen geographical areas.
    groups = df['District'] 

    print("Building the ML Pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop', 'State'])
        ],
        remainder='passthrough' 
    )

    # Base XGBoost model 
    model = xgb.XGBRegressor(random_state=42, n_jobs=1)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 2. Hyperparameter Optimization Space
    # We define a grid of potential parameters for XGBoost to test
    param_distributions = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [4, 6, 8, 10],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    # 3. Spatial Cross-Validation Setup
    gkf = GroupKFold(n_splits=5)

    print("\nCommencing Hyperparameter Tuning with Spatial Cross-Validation...")
    print("(This will spin up your CPU as it tests dozens of model combinations. Please wait...)")

    # RandomizedSearchCV tests random combinations from our parameter grid
    # It is much faster than GridSearchCV but yields near-identical results
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,          # Tests 25 random parameter combinations
        cv=gkf,             # Uses our Spatial Folds
        scoring='r2',
        verbose=1,
        n_jobs=-1,          # Run jobs in parallel
        random_state=42
    )

    # Fit the random search (passing our groups to ensure strict spatial splitting)
    random_search.fit(X, y, groups=groups)

    print("\n" + "="*50)
    print("Optimization Complete!")
    
    # Extract the winning parameters
    best_params = random_search.best_params_
    print("\nBest Parameters Found:")
    for key, value in best_params.items():
        # Clean up the print output by removing the 'model__' prefix
        clean_key = key.replace('model__', '')
        print(f"  - {clean_key}: {value}")

    # The best pipeline is automatically saved within random_search
    best_pipeline = random_search.best_estimator_

    # Evaluate the final optimized model
    y_pred = best_pipeline.predict(X)
    final_r2 = r2_score(y, y_pred)
    final_mae = mean_absolute_error(y, y_pred)

    print("\nFinal Optimized Performance Metrics:")
    print(f"R² Score: {final_r2:.4f}")
    print(f"Mean Absolute Error: {final_mae:.4f} Tonnes/Hectare")
    print("="*50 + "\n")

    # Save the optimized pipeline for the Streamlit app
    os.makedirs('./app/models', exist_ok=True)
    
    print("Saving the heavily optimized pipeline for deployment...")
    joblib.dump(best_pipeline, './app/models/xgboost_yield_model.joblib')

    # Save UI config
    ui_config = {
        'crops': sorted(df['Crop'].unique().tolist()),
        'states': sorted(df['State'].unique().tolist())
    }
    joblib.dump(ui_config, './app/models/ui_config.joblib')

    print("Complete!")

if __name__ == "__main__":
    train_optimized_predictor()