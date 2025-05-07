import streamlit as st
import numpy as np
import joblib
import traceback

# ---- Load Model ----
MODEL_PATH = 'rossmann_model.pkl'

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found: '{MODEL_PATH}'. Ensure it's in the repository.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.text(traceback.format_exc())
        return None

model = load_model()

# ---- Page Configuration ----
st.set_page_config(page_title="Rossmann Sales Predictor", layout="wide", page_icon="💸")

st.title("📊 Rossmann Store Sales Prediction")

if model is None:
    st.warning("Model could not be loaded. Application cannot proceed.")
else:
    st.success("Model loaded successfully!")

    # ---- Input Columns ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📅 Date & Holiday")
        dayofweek = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)), index=4) # Default to Fri (5)
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=31, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=7, step=1)
        year = st.number_input("Year", min_value=2013, max_value=2025, value=2015, step=1)
        weekofyear_input = st.number_input("Week of Year", min_value=1, max_value=53, value=31, step=1) # Matches 'Yearofweek'
        # For StateHoliday, we will construct c,a,d,b from a single input
        state_holiday_cat_input = st.selectbox("State Holiday Category", ['0', 'a', 'b', 'c'], index=0,
                                         help="'0': None, 'a': Public, 'b': Easter, 'c': Christmas")
        school_holiday = st.selectbox("School Holiday?", [0, 1], index=1)

    with col2:
        st.subheader("🏪 Store & Competition")
        open_store = st.selectbox("Is Store Open?", [1, 0], index=0) # Default to Open
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

        # PromoInterval Months - Order based on your header
        promo_interval_header_order = ['Jan', 'Apr', 'Jul', 'Oct', 'Feb', 'May', 'Aug', 'Nov', 'Mar', 'Jun', 'Sept', 'Dec']
        st.markdown("**Promo2 Interval Months (if Promo2 active):**")
        # Create checkboxes for these in the specified order
        promo_interval_active_months = []
        # Display in a more readable format, perhaps two columns of checkboxes
        form_col1, form_col2 = st.columns(2)
        current_form_col = form_col1
        for i, m_name in enumerate(promo_interval_header_order):
            if i == len(promo_interval_header_order) // 2: # Switch column
                current_form_col = form_col2
            if current_form_col.checkbox(m_name, key=f"promo_month_{m_name}"):
                promo_interval_active_months.append(m_name)


    # ---- Feature Engineering Function ----
    def create_feature_vector(inputs):
        features = []

        # Order based on your header:
        # 1. DayOfWeek (i64)
        features.append(inputs['dayofweek'])
        # 2. Open (i64)
        features.append(inputs['open_store'])
        # 3. Promo (i64)
        features.append(inputs['promo'])

        # 4. StateHoliday (i64) - THIS IS THE BIG QUESTION
        # Based on your sample data (0) and later OHE columns (c,a,d,b),
        # it seems this single 'StateHoliday' column might be a simplified numeric
        # version (e.g., 0 for '0', 1 for 'a', 2 for 'b', 3 for 'c').
        # OR it's always 0 if the OHE columns are used.
        # The sample data has 'StateHoliday' as 0, and then 'c'=1, 'a'=1. This is confusing.
        # LET'S ASSUME for now, based on its position and the sample '0', it's a numerical mapping of state_holiday_cat_input
        # If your OHE (c,a,d,b) are the *only* representation of state holiday, then remove this and adjust EXPECTED_NUM_FEATURES
        # Or if this column means "is there any state holiday" (binary)
        # For now, I'll map it: 0->0, a->1, b->2, c->3. YOU MUST VERIFY THIS.
        sh_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
        features.append(sh_map.get(inputs['state_holiday_cat_input'], 0))
        # Alternative if it's just a binary "is holiday" flag:
        # features.append(1 if inputs['state_holiday_cat_input'] != '0' else 0)

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

        # 14. Promo2SinceWeek (i64) - Handle nulls (impute with 0 if Promo2 is 0 or if input is 0)
        p2sw = inputs['promo2_since_week_input']
        features.append(p2sw if inputs['promo2'] == 1 and p2sw > 0 else 0)

        # 15. Promo2SinceYear (i64) - Handle nulls (impute with 0 if Promo2 is 0 or if input is 0)
        p2sy = inputs['promo2_since_year_input']
        features.append(p2sy if inputs['promo2'] == 1 and p2sy > 0 else 0)

        # 16-27. PromoInterval Months (Order: Jan, Apr, Jul, Oct, Feb, May, Aug, Nov, Mar, Jun, Sept, Dec)
        for month_name in inputs['promo_interval_header_order']:
            features.append(1 if month_name in inputs['promo_interval_active_months'] else 0)

        # 28-31. StateHoliday One-Hot Encoding (Order: c, a, d, b)
        # This assumes these ARE the state holiday features and feature #4 is something else or redundant
        features.append(1 if inputs['state_holiday_cat_input'] == 'c' else 0)  # c
        features.append(1 if inputs['state_holiday_cat_input'] == 'a' else 0)  # a
        features.append(0)  # d - Always 0 as it's not an input option and your header implies a column for it
        features.append(1 if inputs['state_holiday_cat_input'] == 'b' else 0)  # b
        
        # Convert to numpy array with appropriate types
        # The dtypes from your header are mixed. We'll cast to float32 for the model,
        # as most sklearn models handle this fine.
        return np.array(features, dtype=np.float32).reshape(1, -1)

    # ---- Prediction ----
    if st.button("💥 Predict Sales", type="primary"):
        current_inputs = {
            'dayofweek': dayofweek,
            'open_store': open_store,
            'promo': promo,
            'state_holiday_cat_input': state_holiday_cat_input, # This will be used for OHE c,a,d,b and potentially feature #4
            'school_holiday': school_holiday,
            'day': day,
            'month': month,
            'year': year,
            'weekofyear_input': weekofyear_input,
            'competition_distance_raw': competition_distance_raw,
            'competition_open_since_month': competition_open_since_month,
            'competition_open_since_year': competition_open_since_year,
            'promo2': promo2,
            'promo2_since_week_input': promo2_since_week_input,
            'promo2_since_year_input': promo2_since_year_input,
            'promo_interval_header_order': promo_interval_header_order, # Pass the order
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

            # **VALIDATION STEP (CRITICAL):**
            EXPECTED_NUM_FEATURES = 31 # Based on your header
            if feature_vector.shape[1] != EXPECTED_NUM_FEATURES:
                st.error(f"CRITICAL ERROR: Generated {feature_vector.shape[1]} features, "
                         f"but model expects {EXPECTED_NUM_FEATURES} features. "
                         "Review `create_feature_vector` and the 'StateHoliday' (feature #4) interpretation.")
            else:
                st.info("Feature vector shape matches. Predicting...")
                prediction = model.predict(feature_vector)
                predicted_sales = np.expm1(prediction[0]) # Assuming sales were log1p transformed
                st.success(f"💰 **Predicted Sales: ₹{predicted_sales:,.2f}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.text(traceback.format_exc())
