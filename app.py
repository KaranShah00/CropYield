import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# --- Helper Function to Clean Names ---
def clean_name(name):
    """Cleans up raw column names for display."""
    return name.replace('_', ' ').replace('Yield', '').strip().title()

# --- Load Pre-computed Assets ---
@st.cache_data
def load_assets():
    """Loads all necessary data and the list of trained models."""
    try:
        # We only need the list of crops that have a trained model.
        with open('saved_features/trained_crop_models.json', 'r') as f:
            trained_crops = json.load(f)
        
        # Create a mapping from clean names to original names for the UI
        cleaned_crops_map = {clean_name(crop): crop for crop in trained_crops}
        return cleaned_crops_map
    except FileNotFoundError:
        st.error("Model files not found. Please run the `train_models.py` script first.")
        return None

cleaned_crops_map = load_assets()

# --- App UI ---
if cleaned_crops_map:
    st.title("🌾 Crop Yield Prediction Engine")
    st.markdown("Select a season and crop, then input the relevant features to get a yield prediction.")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Prediction Inputs")
    
    # --- FIX: Add radio buttons for season selection ---
    selected_season = st.sidebar.radio(
        "Select Crop Season",
        ('Kharif', 'Rabi'),
        horizontal=True
    )

    # Filter the crop dictionary based on the selected season
    filtered_crops = {
        clean: original for clean, original in cleaned_crops_map.items() 
        if selected_season.lower().replace(' ', '_') in original.lower()
    }

    if not filtered_crops:
        st.sidebar.warning(f"No models found for the '{selected_season}' season.")
        st.stop()

    # The selectbox now uses the filtered list of crops
    selected_clean_name = st.sidebar.selectbox(
        f"Select {selected_season} Crop to Predict", 
        list(filtered_crops.keys())
    )
    
    # Get the original filename from the selected clean name
    selected_crop_original = filtered_crops[selected_clean_name]

    sanitized_selected_crop = selected_crop_original.replace('/', '_')

    try:
        # Load the specific list of features the selected model needs
        with open(f'saved_features/{sanitized_selected_crop}_features.json', 'r') as f:
            feature_info = json.load(f)
        model_specific_features = feature_info['features']
        state_columns = feature_info['state_columns']
        
        # This list contains only the non-state features for user input
        user_input_features = [f for f in model_specific_features if not f.startswith('State_')]
        
    except FileNotFoundError:
        st.error(f"Feature information for '{selected_clean_name}' not found.")
        st.stop()
        
    all_states = sorted([s.replace('State_', '') for s in state_columns])
    selected_state = st.sidebar.selectbox("Select State", all_states, key=f"state_select_{sanitized_selected_crop}")

    st.sidebar.markdown("---")
    
    # --- Dummy Irrigation Inputs ---
    st.sidebar.markdown("**Enter Irrigation Area (Dummy Inputs):**")
    canal_area = st.sidebar.number_input("Canal Irrigation Area", value=0.0, format="%.2f")
    well_area = st.sidebar.number_input("Well Irrigation Area", value=0.0, format="%.2f")
    other_irrigation_area = st.sidebar.number_input("Other Irrigation Area", value=0.0, format="%.2f")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Enter Relevant Feature Values:**")
    
    input_data = {}
    # The UI now displays only the relevant features for the selected crop
    for feature in user_input_features:
        # Use the clean_name function for the input label
        input_data[feature] = st.sidebar.number_input(
            clean_name(feature), 
            value=0.0, 
            format="%.2f", 
            key=f"{sanitized_selected_crop}_{feature}"
        )

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Yield", type="primary"):
        # Create a DataFrame from the user's relevant inputs
        prediction_df = pd.DataFrame([input_data])
        
        # Add the one-hot encoded state columns, initializing all to 0
        for state_col in state_columns:
            prediction_df[state_col] = 0
        
        # Set the selected state's column to 1
        selected_state_col_name = f'State_{selected_state}'
        if selected_state_col_name in prediction_df.columns:
            prediction_df[selected_state_col_name] = 1
            
        # Ensure the final DataFrame's column order matches the model's training order
        prediction_df_final = prediction_df[model_specific_features]

        # --- Display Only Ensemble Model Prediction ---
        st.subheader(f"Ensemble Model Prediction for {selected_clean_name}")
        
        model_name = 'Ensemble (Voting)'
        
        try:
            model = joblib.load(f'saved_models/{sanitized_selected_crop}_{model_name}.joblib')
            prediction = model.predict(prediction_df_final)
            
            # --- FIX: Check if the prediction is negative ---
            if prediction[0] < 0:
                st.warning("Prediction is not possible due to less data or unusual input values.")
            else:
                # Display the result prominently in the main area
                st.metric(label="Predicted Yield", value=f"{prediction[0]/1.7:.2f} kg/Ha")
                st.info("This prediction is an average from multiple models for improved accuracy and stability.")
        
        except FileNotFoundError:
             st.error(f"The '{model_name}' model for the selected crop could not be found.")
