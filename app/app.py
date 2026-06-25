import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title="Agritech Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Load Models & Config 
@st.cache_resource
def load_assets():
    model = joblib.load('./app/models/xgboost_yield_model.joblib')
    config = joblib.load('./app/models/ui_config.joblib')
    return model, config

model, config = load_assets()

# State Coordinate Dictionary for the Interactive Map
# (Approximated center points to make the map dynamic)
state_coords = {
    'andhra pradesh': [15.9129, 79.7400], 'arunachal pradesh': [28.2180, 94.7278],
    'assam': [26.2006, 92.9376], 'bihar': [25.0961, 85.3131],
    'chhattisgarh': [21.2787, 81.8661], 'gujarat': [22.2587, 71.1924],
    'haryana': [29.0588, 76.0856], 'himachal pradesh': [31.1048, 77.1665],
    'jammu and kashmir': [33.7782, 76.5762], 'jharkhand': [23.6102, 85.2799],
    'karnataka': [15.3173, 75.7139], 'kerala': [10.8505, 76.2711],
    'madhya pradesh': [22.9734, 78.6569], 'maharashtra': [19.7515, 75.7139],
    'manipur': [24.6637, 93.9063], 'meghalaya': [25.4670, 91.3662],
    'mizoram': [23.1645, 92.9376], 'nagaland': [26.1584, 94.5624],
    'odisha': [20.9517, 85.0985], 'punjab': [31.1471, 75.3412],
    'rajasthan': [27.0238, 74.2179], 'sikkim': [27.5330, 88.5122],
    'tamil nadu': [11.1271, 78.6569], 'telangana': [18.1124, 79.0193],
    'tripura': [23.9408, 91.9882], 'uttar pradesh': [26.8467, 80.9462],
    'uttarakhand': [30.0668, 79.0193], 'west bengal': [22.9868, 87.8550]
}

# App Header 
st.title("Agritech Spatial Yield Predictor")
st.markdown("""
Predict district-level Kharif crop yields across India by fusing **MODIS Satellite NDVI** data with **IMD Meteorological Rainfall Grids**.
""")
st.divider()

# Sidebar Controls
with st.sidebar:
    st.header("Simulation Parameters")
    
    selected_state = st.selectbox("Select State", config['states'])
    selected_crop = st.selectbox("Select Crop", config['crops'])
    
    st.subheader("Environmental Variables")
    # NDVI typically ranges from 0 (barren) to 1 (dense vegetation)
    ndvi_input = st.slider("Mean NDVI (Vegetation Health)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    
    # Rainfall in mm for the Kharif season
    rain_input = st.slider("Kharif Rainfall (mm)", min_value=0, max_value=4000, value=800, step=10)
    
    predict_btn = st.button("Predict Yield ", use_container_width=True)

# Main Layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Prediction Results")
    
    if predict_btn:
       
        interaction_term = ndvi_input * rain_input
        
        input_data = pd.DataFrame({
            'NDVI': [ndvi_input],
            'Kharif_Rainfall_mm': [rain_input],
            'NDVI_Rain_Interaction': [interaction_term],
            'Crop': [selected_crop],
            'State': [selected_state]
        })
        
        # 2. Run Inference
        prediction = model.predict(input_data)[0]
        
        # 3. Display Result
        st.success("Analysis Complete")
        st.metric(
            label=f"Estimated {selected_crop.title()} Yield", 
            value=f"{prediction:.2f} Tonnes / Hectare"
        )
        
        st.info(f"**Insight:** An NDVI of {ndvi_input} combined with {rain_input}mm of rain yields a localized interaction factor of {interaction_term:.1f}, which drove this prediction.")
    else:
        st.write("Adjust the parameters in the sidebar and click **Predict Yield** to see the machine learning model in action.")

with col2:
    st.subheader("Spatial Overview")
    
    # Render the Folium Map dynamically centered on the selected state
    center_coords = state_coords.get(selected_state, [20.5937, 78.9629]) # Default to India center
    
    m = folium.Map(location=center_coords, zoom_start=6, tiles="CartoDB positron")
    
    # Add a marker for the selected state
    folium.Marker(
        location=center_coords,
        popup=selected_state.title(),
        icon=folium.Icon(color="green", icon="leaf")
    ).add_to(m)
    
    st_folium(m, height=400, width=700)