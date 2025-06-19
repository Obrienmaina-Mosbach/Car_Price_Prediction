import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import sklearn

# --- Global paths for model and assets ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'tuned_xgboost_car_price_model_selected_features.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, 'selected_features.joblib')
FULL_OHE_COLUMNS_PATH = os.path.join(BASE_DIR, 'original_training_cols.joblib')
UNIQUE_VALUES_PATH = os.path.join(BASE_DIR, 'unique_values.joblib') # New path for unique values


EXPECTED_SKLEARN_VERSION = '1.7.0' # Updated to 1.7.0
if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
    st.warning(f"Scikit-learn version {sklearn.__version__} detected, but model was trained with {EXPECTED_SKLEARN_VERSION}. Compatibility issues may arise.")

# --- Function to load model and assets ---
@st.cache_resource # Cache the model loading for performance
def load_assets():
    """Loads the trained model, scaler, selected features, full OHE columns, and unique values for dropdowns."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selected_features = joblib.load(SELECTED_FEATURES_PATH)
        full_ohe_columns = joblib.load(FULL_OHE_COLUMNS_PATH)
        unique_values = joblib.load(UNIQUE_VALUES_PATH) # Load the unique values dictionary
        return model, scaler, selected_features, full_ohe_columns, unique_values
    except FileNotFoundError as e:
        st.error(f"Error loading assets. Ensure all .joblib files are in the same directory as app.py: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading assets: {e}")
        st.stop()


model, scaler, selected_features, full_ohe_columns, unique_values = load_assets()

# --- Preprocessing function for new input data ---
def preprocess_input(input_df, full_ohe_columns, scaler, selected_features, unique_values):
    """
    Preprocesses a single row of input data for prediction.
    Maps unseen categorical values to defaults, applies OHE, scales, and selects features.
    """
    # Define categorical columns as they were in your training script
    categorical_cols = [
        'Brand', 'Transmission', 'UsedOrNew', 'Model',
        'Exterior_Color', 'Interior_Material', 'Interior_Color',
        'Location', 'BodyType', 'DriveType', 'FuelType'
    ]

    # Map unseen categorical values to defaults from unique_values
    for col in categorical_cols:
        if col in input_df.columns:
            # Get the list of known unique values for this column from the training data
            known_values = unique_values.get(col, [])
            # Get the default value for unseen categories, falling back to 'Other' or 'Unknown'
            default_val = unique_values.get(f"{col}_default", None)

            if default_val is None: # Fallback for default_val if not explicitly set in unique_values
                if col == 'Model':
                    default_val = 'Other_Model'
                elif col in ['Exterior_Color', 'Interior_Material', 'Interior_Color']:
                    default_val = 'Other/Unknown'
                else:
                    default_val = known_values[0] if known_values else 'Unknown' # Fallback to first known or 'Unknown'

            # Apply mapping: if value is not known, replace with default_val
            input_df[col] = input_df[col].apply(lambda x: x if x in known_values else default_val)

            # Ensure categories are unique by converting to a set and back to list
            # This is crucial for pd.Categorical to avoid "Categorical categories must be unique" error
            all_possible_categories = list(set(known_values + [default_val]))


            # Convert to 'category' dtype with known categories to prevent issues during get_dummies
            input_df[col] = pd.Categorical(input_df[col], categories=all_possible_categories)


    # Apply one-hot encoding
    # `handle_unknown='ignore'` is vital here to prevent errors for unseen categories in production
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dummy_na=False)


    # Reindex the input_encoded DataFrame to match the full_ohe_columns from training
    # This ensures all columns are present (with 0 if not in input) and in the correct order
    X_processed = input_encoded.reindex(columns=full_ohe_columns, fill_value=0)

    # Apply scaling to the entire X_processed DataFrame
    # The scaler was fitted on X_train (which contained all OHE features and numerical features)
    X_scaled_full = scaler.transform(X_processed)

    # Select features based on the model's selected features from the scaled full set
    # Ensure the output of scaler.transform is converted back to DataFrame for column selection
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
    font-size: 2.5em; /* Default for h1, can adjust */
    font-weight: bold;
    color: #2e4a6b; /* Ensure color consistency */
    text-align: center;
}
.subtitle-font {
    font-size: 1.7em; /* As requested by user */
    text-align: center;
    color: #555; /* A slightly lighter color for subtitle */
    margin-top: -10px; /* Pull it closer to the title */
}
</style>
""", unsafe_allow_html=True)

# Add an image at the top of the app
# Updated use_column_width to use_container_width
st.image("Assets/Images/img_1.png",
         caption="Let's predict the price of your car!",
         use_container_width=True) # Changed from use_column_width

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
with st.form("car_prediction_form"):
    st.header("Car Specifications")

    # Brand
    unique_brands = sorted(unique_values.get('Brand', []))
    # Ensure a default is always available for index if unique_brands is empty or default is not in it
    brand_default_val = st.session_state.form_data.get('brand', unique_brands[0] if unique_brands else 'Other')
    brand_index = unique_brands.index(brand_default_val) if brand_default_val in unique_brands else 0
    brand = st.selectbox("Brand", unique_brands, index=brand_index, key='brand')

    # Model (filtered by Brand)
    @st.cache_data
    def get_models_for_brand(selected_brand):
        # Fetch models specific to the selected brand, falling back to 'Other_Model' if none found
        brand_models = unique_values.get('Model_by_Brand', {}).get(selected_brand, [])
        # Ensure 'Other_Model' is always an option if it was used in training for low cardinality models
        if 'Other_Model' in unique_values.get('Model', []): # Check if 'Other_Model' was a category in training
            if 'Other_Model' not in brand_models:
                brand_models.append('Other_Model')

        return sorted(list(set(brand_models))) # Ensure uniqueness and sort

    models = get_models_for_brand(brand)
    # Ensure model_default is a valid choice in `models`
    model_default_val = st.session_state.form_data.get('model', models[0] if models else 'Other_Model')
    model_index = models.index(model_default_val) if model_default_val in models else (models.index('Other_Model') if 'Other_Model' in models else 0)
    model_input = st.selectbox("Model", models, index=model_index, key='model')


    # Changed to 3 columns
    col1, col2, col3 = st.columns(3)
    with col1:
        current_year = datetime.datetime.now().year
        year = st.number_input("Year", min_value=1950, max_value=current_year, value=st.session_state.form_data.get('year', 2015), step=1, key='year')
        kilometres = st.number_input("Kilometres", min_value=0, max_value=500000, value=st.session_state.form_data.get('kilometres', 50000), step=1000, key='kilometres')
        cylinders_options = sorted(unique_values.get('Engine_Cylinders', [4, 6, 8, 3, 5, 10, 12, 16])) # Fallback options
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


    col4, col5, col6 = st.columns(3) # New set of columns for the rest of the inputs
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
        st.session_state.form_data = {
            'brand': brand, 'model': model_input, 'year': year, 'kilometres': kilometres,
            'cylinders': cylinders, 'doors': doors, 'seats': seats,
            'fuel_consumption': fuel_consumption, 'engine_liters': engine_liters,
            'transmission': transmission, 'used_or_new': used_or_new,
            'location': location, 'body_type': body_type, 'drive_type': drive_type,
            'fuel_type': fuel_type, 'exterior_color': exterior_color,
            'interior_material': interior_material, 'interior_color': interior_color
        }

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
            # Pass unique_values to preprocess_input
            processed_input = preprocess_input(input_df.copy(), full_ohe_columns, scaler, selected_features, unique_values)
            predicted_price = model.predict(processed_input)[0]

            # Convert AUD to EUR (approximate conversion rate)
            predicted_price_EUR = predicted_price * 0.65

            # Format with dot as thousands separator and comma as decimal (European style)
            formatted_eur_price = f"{predicted_price_EUR:,.2f}".replace('.', '#TEMP#').replace(',', '.').replace('#TEMP#', ',')

            st.markdown(f"""
            <div class="prediction-box">
                Predicted Car Price: **€{formatted_eur_price}**
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")

