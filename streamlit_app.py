import streamlit as st
import numpy as np
import joblib
import traceback

# ---- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ----
st.set_page_config(page_title="Rossmann Sales Predictor", layout="wide", page_icon="💸")

# ---- Load Model ----
MODEL_PATH = 'rossmann_model.pkl'

@st.cache_resource # It's good practice to cache resource loading
def load_model_resource(): # Renamed to avoid confusion with global 'model'
    try:
        loaded_model = joblib.load(MODEL_PATH)
        return loaded_model
    except FileNotFoundError:
        # We can't use st.error here if set_page_config hasn't run.
        # Instead, we'll handle the error display after set_page_config.
        # For now, just print to console (won't show in app UI yet if this fails before set_page_config)
        # and return a specific error indicator or raise an exception.
        print(f"CRITICAL ERROR (before Streamlit UI init): Model file not found: '{MODEL_PATH}'.")
        # To show error in UI later, we need to call st.error *after* set_page_config
        return "FileNotFound" # Return an indicator
    except Exception as e:
        print(f"CRITICAL ERROR (before Streamlit UI init): Error loading model: {e}")
        print(traceback.format_exc())
        return "LoadError" # Return an indicator

# Attempt to load model AFTER st.set_page_config
model = load_model_resource() # Call the function to get the model object or error indicator

# ---- Now display errors if model loading failed, or proceed ----
st.title("📊 Rossmann Store Sales Prediction") # This can come after set_page_config

if isinstance(model, str) and model == "FileNotFound":
    st.error(f"Error: Model file not found: '{MODEL_PATH}'. Ensure it's in the repository and Git LFS was used correctly.")
    st.warning("Application cannot proceed without the model.")
elif isinstance(model, str) and model == "LoadError":
    st.error("A critical error occurred while loading the model. Check the application logs for details.")
    st.warning("Application cannot proceed without the model.")
elif model is None: # General None case, though custom indicators are better
    st.error("Model could not be loaded for an unknown reason.")
    st.warning("Application cannot proceed.")
else:
    # Model loaded successfully, proceed with the rest of the app
    st.success("Model loaded successfully!")

    # ---- Input Columns ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📅 Date & Holiday")
        dayofweek = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)), index=4)
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=31, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=7, step=1)
        year = st.number_input("Year", min_value=2013, max_value=2025, value=2015, step=1)
        weekofyear_input = st.number_input("Week of Year", min_value=1, max_value=53, value=31, step=1)
        state_holiday_cat_input = st.selectbox("State Holiday Category", ['0', 'a', 'b', 'c'], index=0,
                                         help="'0': None, 'a': Public, 'b': Easter, 'c': Christmas")
        school_holiday = st.selectbox("School Holiday?", [0, 1], index=1)

    with col2:
        st.subheader("🏪 Store & Competition")
        open_store = st.selectbox("Is Store Open?", [1, 0], index=0)
        promo = st.selectbox("Promo Active?", [1, 0], index=0)
        competition_distance_raw = st.number_input("Competition Distance (meters)", value=1270.0, min_value=0.0, step=10.0)
        competition_open_since_month = st.number_input("Competition Open Since Month", min_value=1, max_value=12, value=9, step=1)
        competition_open_since_year = st.number_input("Competition Open Since Year", min_value=1900, max_value=2025, value=2008, step=1)

    with col3:
        st.subheader("🚀 Extended Promotions (Promo2)")
        promo2 = st.selectbox("Promo2 Active?", [1, 0], index=0)
        promo2_since_week_input = st.number_input("Promo2 Since Week (if Promo2 active)", min_value=0, max_value=53, value=0, step=1,
                                             help="Enter 0 if Promo2 is not active or if week is unknown/not applicable")
        promo2_since_year_input = st.number_input("Promo2 Since Year (if Promo2 active)", min_value=0, max_value=2025, value=0, step=1,
                                              help="Enter 0 if Promo2 is not active or if year is unknown/not applicable")

        promo_interval_header_order = ['Jan', 'Apr', 'Jul', 'Oct', 'Feb', 'May', 'Aug', 'Nov', 'Mar', 'Jun', 'Sept', 'Dec']
        st.markdown("**Promo2 Interval Months (if Promo2 active):**")
        promo_interval_active_months = []
        form_col1, form_col2 = st.columns(2)
        current_form_col = form_col1
        for i, m_name in enumerate(promo_interval_header_order):
            if i == len(promo_interval_header_order) // 2:
                current_form_col = form_col2
            if current_form_col.checkbox(m_name, key=f"promo_month_{m_name}"):
                promo_interval_active_months.append(m_name)


    # ---- Feature Engineering Function ----
    def create_feature_vector(inputs):
        features = []
        # ... (your existing feature engineering logic - keep it as is) ...
        # 1. DayOfWeek (i64)
        features.append(inputs['dayofweek'])
        # 2. Open (i64)
        features.append(inputs['open_store'])
        # 3. Promo (i64)
        features.append(inputs['promo'])
        # 4. StateHoliday (i64) - YOU MUST VERIFY THIS LOGIC
        sh_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
        features.append(sh_map.get(inputs['state_holiday_cat_input'], 0))
        # 5. SchoolHoliday (i64)
        features.append(inputs['school_holiday'])
        # 6. day (i8)
        features.append(inputs['day'])
        # 7. month (i8)
        features.append(inputs['month'])
        # 8. year (i32)
        features.append(inputs['year'])
        # 9. Yearofweek (i8) -> weekofyear_input
        features.append(inputs['weekofyear_input'])
        # 10. CompetitionDistance (f64) - Assumed to be log1p transformed
        features.append(np.log1p(inputs['competition_distance_raw']))
        # 11. CompetitionOpenSinceMonth (i64)
        features.append(inputs['competition_open_since_month'])
        # 12. CompetitionOpenSinceYear (i64)
        features.append(inputs['competition_open_since_year'])
        # 13. Promo2 (i64)
        features.append(inputs['promo2'])
        # 14. Promo2SinceWeek (i64)
        p2sw = inputs['promo2_since_week_input']
        features.append(p2sw if inputs['promo2'] == 1 and p2sw > 0 else 0)
        # 15. Promo2SinceYear (i64)
        p2sy = inputs['promo2_since_year_input']
        features.append(p2sy if inputs['promo2'] == 1 and p2sy > 0 else 0)
        # 16-27. PromoInterval Months
        for month_name in inputs['promo_interval_header_order']:
            features.append(1 if month_name in inputs['promo_interval_active_months'] else 0)
        # 28-31. StateHoliday One-Hot Encoding
        features.append(1 if inputs['state_holiday_cat_input'] == 'c' else 0)
        features.append(1 if inputs['state_holiday_cat_input'] == 'a' else 0)
        features.append(0)
        features.append(1 if inputs['state_holiday_cat_input'] == 'b' else 0)
        
        return np.array(features, dtype=np.float32).reshape(1, -1)

    # ---- Prediction ----
    if st.button("💥 Predict Sales", type="primary"):
        # ... (your existing prediction logic - keep it as is) ...
        current_inputs = {
            'dayofweek': dayofweek, 'open_store': open_store, 'promo': promo,
            'state_holiday_cat_input': state_holiday_cat_input, 'school_holiday': school_holiday,
            'day': day, 'month': month, 'year': year, 'weekofyear_input': weekofyear_input,
            'competition_distance_raw': competition_distance_raw,
            'competition_open_since_month': competition_open_since_month,
            'competition_open_since_year': competition_open_since_year,
            'promo2': promo2, 'promo2_since_week_input': promo2_since_week_input,
            'promo2_since_year_input': promo2_since_year_input,
            'promo_interval_header_order': promo_interval_header_order,
            'promo_interval_active_months': promo_interval_active_months
        }
        st.markdown("---")
        st.subheader("🧪 Diagnostic Information")
        try:
            feature_vector = create_feature_vector(current_inputs)
            st.write(f"Number of features generated: {feature_vector.shape[1]}")
            st.text("Feature vector (first 15 values):")
            st.write(feature_vector[0, :15])
            st.text("Feature vector (last 16 values):")
            st.write(feature_vector[0, 15:])
            st.write(f"Feature vector data type: {feature_vector.dtype}")

            EXPECTED_NUM_FEATURES = 31
            if feature_vector.shape[1] != EXPECTED_NUM_FEATURES:
                st.error(f"CRITICAL ERROR: Generated {feature_vector.shape[1]} features, "
                         f"but model expects {EXPECTED_NUM_FEATURES} features. "
                         "Review `create_feature_vector` and the 'StateHoliday' (feature #4) interpretation.")
            else:
                st.info("Feature vector shape matches. Predicting...")
                prediction = model.predict(feature_vector)
                predicted_sales = np.expm1(prediction[0])
                st.success(f"💰 **Predicted Sales: ₹{predicted_sales:,.2f}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.text(traceback.format_exc())
