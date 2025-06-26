import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import sklearn # To check version compatibility

# --- Global paths for model and assets ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for the main model and final preprocessing components
MODEL_PATH = os.path.join(BASE_DIR, 'tuned_xgboost_car_price_model_selected_features.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib') # This is your main scaler for the final X
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, 'selected_features.joblib')
FULL_OHE_COLUMNS_PATH = os.path.join(BASE_DIR, 'original_training_cols.joblib') # All OHE columns AFTER Cluster_X added
UNIQUE_VALUES_PATH = os.path.join(BASE_DIR, 'unique_values.joblib')

# Paths for PCA and KMeans specific assets
SCALER_PCA_KMEANS_PATH = os.path.join(BASE_DIR, 'scaler_pca_kmeans.joblib')
PCA_MODEL_PATH = os.path.join(BASE_DIR, 'pca_model.joblib')
KMEANS_MODEL_PATH = os.path.join(BASE_DIR, 'kmeans_model.joblib')
# Base training columns used for PCA and KMeans
BASE_TRAINING_COLS_PCA_KMEANS_PATH = os.path.join(BASE_DIR, 'base_training_cols_for_pca_kmeans.joblib')


# --- Scikit-learn version compatibility check ---
# Adjust this to the version you used for training if it's different
# Remember to verify this version in your training environment!
EXPECTED_SKLEARN_VERSION = '1.7.0' # Changed back to a common stable version, please verify

if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
    st.warning(f"Scikit-learn version {sklearn.__version__} detected, but model was trained with {EXPECTED_SKLEARN_VERSION}. Compatibility issues may arise.")


# --- Function to load model and assets ---
@st.cache_resource # Cache the model loading for performance
def load_assets():
    """Loads the trained model, scalers, PCA, KMeans, selected features, full OHE columns, and unique values."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH) # Main scaler
        selected_features = joblib.load(SELECTED_FEATURES_PATH)
        full_ohe_columns = joblib.load(FULL_OHE_COLUMNS_PATH)
        unique_values = joblib.load(UNIQUE_VALUES_PATH)

        # Load PCA and KMeans assets
        scaler_pca_kmeans = joblib.load(SCALER_PCA_KMEANS_PATH)
        pca_model = joblib.load(PCA_MODEL_PATH)
        kmeans_model = joblib.load(KMEANS_MODEL_PATH)
        base_training_cols_for_pca_kmeans = joblib.load(BASE_TRAINING_COLS_PCA_KMEANS_PATH)


        return model, scaler, selected_features, full_ohe_columns, unique_values, \
            scaler_pca_kmeans, pca_model, kmeans_model, base_training_cols_for_pca_kmeans
    except FileNotFoundError as e:
        st.error(f"Error loading assets. Ensure all .joblib files are in the same directory as app.py: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading assets: {e}")
        st.stop()


# Unpack all loaded assets
model, scaler, selected_features, full_ohe_columns, unique_values, \
    scaler_pca_kmeans, pca_model, kmeans_model, base_training_cols_for_pca_kmeans = load_assets()


# --- Preprocessing function for new input data ---
def preprocess_input(input_df, full_ohe_columns, scaler, selected_features, unique_values,
                     scaler_pca_kmeans, pca_model, kmeans_model, base_training_cols_for_pca_kmeans):
    """
    Preprocesses a single row of input data for prediction.
    Maps unseen categorical values to defaults, applies OHE, PCA, KMeans, scales, and selects features.
    """
    # Define categorical columns as they were in your training script
    categorical_cols = [
        'Brand', 'Transmission', 'UsedOrNew', 'Model',
        'Exterior_Color', 'Interior_Material', 'Interior_Color',
        'Location', 'BodyType', 'DriveType', 'FuelType'
    ]
    numerical_features = [ # Not strictly used in this function for direct manipulation but good for reference
        'Year', 'Kilometres', 'CylindersinEngine', 'Doors', 'Seats', 'FuelConsumption',
        'Engine_Cylinders', 'Engine_Liters'
    ]

    # Make a copy to avoid modifying the original input_df (important for session state)
    df_processed = input_df.copy()

    # 1. Handle categorical features: map unseen values and convert to categorical type
    for col in categorical_cols:
        if col in df_processed.columns:
            known_values = unique_values.get(col, [])
            default_val = unique_values.get(f"{col}_default", None)

            # Fallback for default_val if not explicitly set in unique_values
            if default_val is None:
                if col == 'Model':
                    default_val = 'Other_Model'
                elif col in ['Exterior_Color', 'Interior_Material', 'Interior_Color']:
                    default_val = 'Other/Unknown'
                else:
                    default_val = known_values[0] if known_values else 'Unknown'

            # Apply mapping: if value is not known, replace with default_val
            df_processed[col] = df_processed[col].apply(lambda x: x if x in known_values else default_val)

            # Convert to 'category' dtype with known categories to prevent issues during get_dummies
            # Ensure all_possible_categories includes the default_val for robustness
            all_possible_categories = list(set(known_values + [default_val]))
            df_processed[col] = pd.Categorical(df_processed[col], categories=all_possible_categories)

    # 2. Apply initial one-hot encoding for base categorical features
    df_encoded_initial = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dummy_na=False)

    # 3. Align columns for PCA/KMeans processing
    X_for_pca_kmeans_processing = df_encoded_initial.reindex(columns=base_training_cols_for_pca_kmeans, fill_value=0)

    # 4. Scale data for PCA and KMeans
    X_scaled_for_pca_kmeans_processed = scaler_pca_kmeans.transform(X_for_pca_kmeans_processing)

    # 5. Apply PCA transformation
    X_pca_processed = pca_model.transform(X_scaled_for_pca_kmeans_processed)

    # 6. Predict KMeans cluster label
    cluster_label_processed = kmeans_model.predict(X_pca_processed)[0] # [0] because it's a single row

    # 7. Add one-hot encoded cluster features
    cluster_columns_all = [f'Cluster_{i}' for i in range(kmeans_model.n_clusters)]
    cluster_columns_ohe = []
    if kmeans_model.n_clusters > 1:
        cluster_columns_ohe = [col for col in cluster_columns_all if col != 'Cluster_0']

    cluster_df = pd.DataFrame(0, index=df_processed.index, columns=cluster_columns_ohe)

    if cluster_label_processed != 0 and f'Cluster_{cluster_label_processed}' in cluster_columns_ohe:
        cluster_df[f'Cluster_{cluster_label_processed}'] = 1

    # 8. Combine initial OHE features with new cluster OHE features
    final_df_before_main_scale = pd.concat([df_encoded_initial, cluster_df], axis=1)

    # 9. Reindex the combined DataFrame to match the full_ohe_columns from training
    x_processed = final_df_before_main_scale.reindex(columns=full_ohe_columns, fill_value=0)

    # 10. Apply the MAIN scaler to the entire x_processed DataFrame
    X_scaled_full = scaler.transform(x_processed)

    # 11. Select features based on the model's selected features from the scaled full set
    X_final = pd.DataFrame(X_scaled_full, columns=full_ohe_columns)[selected_features]

    return X_final

# --- Streamlit App UI ---
st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049;
}
h1, h2 {
    color: #2e4a6b;
    text-align: center;
}
.stSelectbox, .stNumberInput, .stTextInput {
    margin-bottom: 10px;
}
.prediction-box {
    background-color: #e6f7ff;
    color: #333;
    border-left: 5px solid #2196F3;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
    font-size: 1.2em;
    font-weight: bold;
    text-align: center;
}
.center-text {
    text-align: center;
}
.title-font {
    font-size: 2.5em;
    font-weight: bold;
    color: #2e4a6b;
    text-align: center;
}
.subtitle-font {
    font-size: 1.7em;
    text-align: center;
    color: #555;
    margin-top: -10px;
}
</style>
""", unsafe_allow_html=True)

# Add an image at the top of the app
st.image("Assets/Images/img_1.png",
         caption="Let's predict the price of your car!",
         use_container_width=True)

st.title("Car Price Predictor")
st.write("Enter the car details to get a predicted price!")
st.markdown("""
### About
This app predicts car prices based on Australian market data using a trained XGBoost model. Predictions are estimates and may vary based on market conditions. Ensure inputs are realistic for accurate results.
""")

# Initialize session state for form data
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# --- Input Fields ---
st.header("Car Specifications")

# Brand - MOVED OUTSIDE THE FORM
unique_brands = sorted(unique_values.get('Brand', []))
# Update session state with the selected brand whenever it changes
brand = st.selectbox(
    "Brand",
    unique_brands,
    index=unique_brands.index(st.session_state.form_data.get('brand', unique_brands[0] if unique_brands else 'Other')) if st.session_state.form_data.get('brand') in unique_brands else 0,
    key='brand_select', # Use a different key here to avoid conflict with 'brand' in form_data
    on_change=lambda: st.session_state.form_data.update({'brand': st.session_state.brand_select, 'model': None}) # Reset model when brand changes
)
# Make sure the session_state['brand'] reflects the actual selection from the selectbox
st.session_state.form_data['brand'] = brand


# Model (filtered by Brand) - MOVED OUTSIDE THE FORM LOGIC, but within the form container
@st.cache_data
def get_models_for_brand(selected_brand):
    brand_models = unique_values.get('Model_by_Brand', {}).get(selected_brand, [])
    if 'Other_Model' in unique_values.get('Model', []):
        if 'Other_Model' not in brand_models:
            brand_models.append('Other_Model')
    return sorted(list(set(brand_models)))

models = get_models_for_brand(brand)

# Determine the default model to display
model_default_val = st.session_state.form_data.get('model')
if model_default_val not in models:
    # If the previously selected model is not in the new brand's models,
    # or if no model was previously selected, default to the first available model
    # or 'Other_Model' if available.
    if models:
        model_default_val = models[0]
        if 'Other_Model' in models:
            model_default_val = 'Other_Model'
    else:
        model_default_val = 'Other_Model' # Fallback if no models are available

model_index = models.index(model_default_val) if model_default_val in models else 0
model_input = st.selectbox("Model", models, index=model_index, key='model_select')
st.session_state.form_data['model'] = model_input # Update session state


# --- Remaining input fields inside the form ---
with st.form("car_prediction_form"):
    # The variables 'brand' and 'model_input' are already defined above this form block
    # so they can be directly used here without re-declaring them inside the form.

    col1, col2, col3 = st.columns(3)
    with col1:
        current_year = datetime.datetime.now().year
        year = st.number_input("Year", min_value=1950, max_value=current_year, value=st.session_state.form_data.get('year', 2015), step=1, key='year')
        kilometres = st.number_input("Kilometres", min_value=0, max_value=500000, value=st.session_state.form_data.get('kilometres', 50000), step=1000, key='kilometres')
        cylinders_options = sorted(unique_values.get('Engine_Cylinders', [4, 6, 8, 3, 5, 10, 12, 16]))
        cylinders_default_val = st.session_state.form_data.get('cylinders', 4)
        cylinders_index = cylinders_options.index(cylinders_default_val) if cylinders_default_val in cylinders_options else 0
        cylinders = st.selectbox("Engine Cylinders", cylinders_options, index=cylinders_index, key='cylinders')

    with col2:
        doors_options = sorted(unique_values.get('Doors', [2, 3, 4, 5]))
        doors_default_val = st.session_state.form_data.get('doors', 4)
        doors_index = doors_options.index(doors_default_val) if doors_default_val in doors_options else 0
        doors = st.selectbox("Number of Doors", doors_options, index=doors_index, key='doors')

        seats_options = sorted(unique_values.get('Seats', [2, 4, 5, 6, 7, 8]))
        seats_default_val = st.session_state.form_data.get('seats', 5)
        seats_index = seats_options.index(seats_default_val) if seats_default_val in seats_options else 0
        seats = st.selectbox("Number of Seats", seats_options, index=seats_index, key='seats')

        fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, max_value=30.0, value=st.session_state.form_data.get('fuel_consumption', 7.5), step=0.1, key='fuel_consumption')

    with col3:
        engine_liters = st.number_input("Engine Liters (L)", min_value=0.5, max_value=10.0, value=st.session_state.form_data.get('engine_liters', 2.0), step=0.1, key='engine_liters')

        transmission_options = sorted(unique_values.get('Transmission', []))
        transmission_default_val = st.session_state.form_data.get('transmission', transmission_options[0] if transmission_options else 'Automatic')
        transmission_index = transmission_options.index(transmission_default_val) if transmission_default_val in transmission_options else 0
        transmission = st.selectbox("Transmission", transmission_options, index=transmission_index, key='transmission')

        used_or_new_options = sorted(unique_values.get('UsedOrNew', []))
        used_or_new_default_val = st.session_state.form_data.get('used_or_new', used_or_new_options[0] if used_or_new_options else 'Used')
        used_or_new_index = used_or_new_options.index(used_or_new_default_val) if used_or_new_default_val in used_or_new_options else 0
        used_or_new = st.selectbox("Condition", used_or_new_options, index=used_or_new_index, key='used_or_new')


    col4, col5, col6 = st.columns(3)
    with col4:
        location_options = sorted(unique_values.get('Location', []))
        location_default_val = st.session_state.form_data.get('location', location_options[0] if location_options else 'NSW')
        location_index = location_options.index(location_default_val) if location_default_val in location_options else 0
        location = st.selectbox("Location", location_options, index=location_index, key='location')

        body_type_options = sorted(unique_values.get('BodyType', []))
        body_type_default_val = st.session_state.form_data.get('body_type', body_type_options[0] if body_type_options else 'Sedan')
        body_type_index = body_type_options.index(body_type_default_val) if body_type_default_val in body_type_options else 0
        body_type = st.selectbox("Body Type", body_type_options, index=body_type_index, key='body_type')

    with col5:
        drive_type_options = sorted(unique_values.get('DriveType', []))
        drive_type_default_val = st.session_state.form_data.get('drive_type', drive_type_options[0] if drive_type_options else 'Front')
        drive_type_index = drive_type_options.index(drive_type_default_val) if drive_type_default_val in drive_type_options else 0
        drive_type = st.selectbox("Drive Type", drive_type_options, index=drive_type_index, key='drive_type')

        fuel_type_options = sorted(unique_values.get('FuelType', []))
        fuel_type_default_val = st.session_state.form_data.get('fuel_type', fuel_type_options[0] if fuel_type_options else 'Petrol')
        fuel_type_index = fuel_type_options.index(fuel_type_default_val) if fuel_type_default_val in fuel_type_options else 0
        fuel_type = st.selectbox("Fuel Type", fuel_type_options, index=fuel_type_index, key='fuel_type')

    with col6:
        exterior_color_options = sorted(unique_values.get('Exterior_Color', []))
        exterior_color_default_val = st.session_state.form_data.get('exterior_color', exterior_color_options[0] if exterior_color_options else 'Other/Unknown')
        exterior_color_index = exterior_color_options.index(exterior_color_default_val) if exterior_color_default_val in exterior_color_options else 0
        exterior_color = st.selectbox("Exterior Color", exterior_color_options, index=exterior_color_index, key='exterior_color')

        interior_material_options = sorted(unique_values.get('Interior_Material', []))
        interior_material_default_val = st.session_state.form_data.get('interior_material', interior_material_options[0] if interior_material_options else 'Other/Unknown Material')
        interior_material_index = interior_material_options.index(interior_material_default_val) if interior_material_default_val in interior_material_options else 0
        interior_material = st.selectbox("Interior Material", interior_material_options, index=interior_material_index, key='interior_material')

        interior_color_options = sorted(unique_values.get('Interior_Color', []))
        interior_color_default_val = st.session_state.form_data.get('interior_color', interior_color_options[0] if interior_color_options else 'Unknown')
        interior_color_index = interior_color_options.index(interior_color_default_val) if interior_color_default_val in interior_color_options else 0
        interior_color = st.selectbox("Interior Color", interior_color_options, index=interior_color_index, key='interior_color')


    submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Store current form data in session state for persistence
        # Note: brand and model_input are already updated outside the form logic,
        # but storing them here ensures all data is captured consistently on submit.
        st.session_state.form_data.update({
            'brand': brand,
            'model': model_input,
            'year': year, 'kilometres': kilometres,
            'cylinders': cylinders, 'doors': doors, 'seats': seats,
            'fuel_consumption': fuel_consumption, 'engine_liters': engine_liters,
            'transmission': transmission, 'used_or_new': used_or_new,
            'location': location, 'body_type': body_type, 'drive_type': drive_type,
            'fuel_type': fuel_type, 'exterior_color': exterior_color,
            'interior_material': interior_material, 'interior_color': interior_color
        })

        # Input validation
        if year > current_year:
            st.error("Year cannot be in the future.")
            st.stop()
        if kilometres < 0:
            st.error("Kilometres cannot be negative.")
            st.stop()

        # Create input DataFrame
        input_data = {
            'Year': year,
            'Kilometres': kilometres,
            'Doors': doors,
            'Seats': seats,
            'FuelConsumption': fuel_consumption,
            'Engine_Cylinders': cylinders,
            'Engine_Liters': engine_liters,
            'Brand': brand,
            'Transmission': transmission,
            'UsedOrNew': used_or_new,
            'Model': model_input,
            'Exterior_Color': exterior_color,
            'Interior_Material': interior_material,
            'Interior_Color': interior_color,
            'Location': location,
            'BodyType': body_type,
            'DriveType': drive_type,
            'FuelType': fuel_type
        }
        input_df = pd.DataFrame([input_data])

        # Preprocess and predict
        try:
            processed_input = preprocess_input(input_df.copy(), full_ohe_columns, scaler, selected_features, unique_values,
                                               scaler_pca_kmeans, pca_model, kmeans_model, base_training_cols_for_pca_kmeans)
            predicted_price = model.predict(processed_input)[0]

            # Convert AUD to EUR (approximate conversion rate)
            predicted_price_EUR = predicted_price * 0.65

            # Format with dot as thousands separator and comma as decimal (European style)
            formatted_eur_price = f"{predicted_price_EUR:,.2f}".replace('.', '#TEMP#').replace(',', '.').replace('#TEMP#', ',')

            st.markdown(f"""
            <div class="prediction-box">
                Predicted Car Price: €{formatted_eur_price}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")
            st.exception(e) # Show the full exception for debugging