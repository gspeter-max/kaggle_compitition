import streamlit as st
import numpy as np
import joblib
import traceback

# ---- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ----
# Inspired by pip.org's dark theme colors and feel
st.set_page_config(
    page_title="Rossmann Sales Forecaster 📈",
    page_icon="🛒", # Changed icon slightly
    layout="centered", # Centered layout can look more focused
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark Mode & Enhanced Prediction Display ---
st.markdown("""
    <style>
        /* Base Dark Theme - Inspired by pip.org */
        body {
            color: #e8e8e8; /* Light grey text */
            background-color: #1e1e2f; /* Dark blue/purple background */
        }
        .stApp {
             background-color: #1e1e2f;
        }
        /* Sidebar styling */
        .css-1d391kg { /* Sidebar main background */
            background-color: #2b303a; /* Slightly lighter dark */
        }
        /* Input widgets */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div,
        .stMultiSelect > div > div > div > div { /* Adjusted for multiselect */
            background-color: #2b303a;
            color: #e8e8e8;
            border-radius: 0.3rem;
        }
        .stButton>button {
            border-radius: 0.3rem;
            background-color: #0073B7; /* A nice blue for buttons */
            color: white;
            border: none;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #005A8E;
        }
        .stAlert { /* Alert boxes styling */
            border-radius: 0.3rem;
        }
        h1, h2, h3 {
            color: #c5c8c6; /* Lighter grey for headers */
        }

        /* Enhanced Prediction Result Display */
        .prediction-container {
            text-align: center;
            padding: 2em;
            background-color: #2b303a; /* Card background */
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            margin-top: 2em;
        }
        .prediction-header {
            font-size: 1.5em;
            color: #81A2BE; /* Light blue */
            margin-bottom: 0.5em;
        }
        .prediction-value {
            font-size: 3.5em;
            font-weight: bold;
            color: #56C8A9; /* A nice green for success/value */
            margin-bottom: 0.5em;
        }
        .prediction-currency {
            font-size: 2em;
            color: #56C8A9;
            vertical-align: super;
        }
    </style>
""", unsafe_allow_html=True)


# ---- Load Model ----
MODEL_PATH = 'rossmann_model.pkl'

@st.cache_resource
def load_model_resource():
    try:
        loaded_model = joblib.load(MODEL_PATH)
        return loaded_model
    except FileNotFoundError:
        print(f"CRITICAL ERROR (before Streamlit UI init): Model file not found: '{MODEL_PATH}'.")
        return "FileNotFound"
    except Exception as e:
        print(f"CRITICAL ERROR (before Streamlit UI init): Error loading model: {e}")
        print(traceback.format_exc())
        return "LoadError"

model_load_status = load_model_resource() # Store status/model

# --- Initialize Session State ---
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'predicted_sales_value' not in st.session_state:
    st.session_state.predicted_sales_value = None


# ---- Main App Logic ----
st.title("🛒 Rossmann Sales Forecaster")

# Handle model loading status first
if isinstance(model_load_status, str):
    if model_load_status == "FileNotFound":
        st.error(f"Error: Model file ('{MODEL_PATH}') not found. Please ensure it's in the repository and Git LFS was used correctly if it's a large file.")
    elif model_load_status == "LoadError":
        st.error("A critical error occurred while loading the prediction model. Check the application logs for details.")
    st.warning("Application cannot proceed without the model.", icon="🚫")
    st.stop() # Stop execution if model isn't loaded
else:
    model = model_load_status # Assign the loaded model
    if not st.session_state.show_results:
        st.success("🤖 Model loaded successfully! Please provide the details below.", icon="✅")


# ---- Conditional Display: Input Form OR Results Page ----

if not st.session_state.show_results and model is not None: # Show form if model is loaded and not showing results
    with st.form(key="sales_input_form"):
        st.header("Enter Store & Promotion Details")
        
        # ---- Input Columns ----
        # Using columns for better layout on wider screens
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        row3_col1, row3_col2 = st.columns(2)

        with row1_col1:
            st.subheader("📅 Date & Holiday")
            dayofweek = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)), index=4)
            day = st.number_input("Day of Month", min_value=1, max_value=31, value=15, step=1) # Default to mid-month
            month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
            year = st.number_input("Year", min_value=2013, max_value=2025, value=2015, step=1)
        
        with row1_col2:
            st.write("") # Spacer for alignment
            st.write("") # Spacer
            weekofyear_input = st.number_input("Week of Year", min_value=1, max_value=53, value=22, step=1)
            state_holiday_cat_input = st.selectbox("State Holiday", ['0', 'a', 'b', 'c'], index=0,
                                             help="'0': None, 'a': Public, 'b': Easter, 'c': Christmas")
            school_holiday = st.selectbox("School Holiday?", [0, 1], index=0) # Default to No

        with row2_col1:
            st.subheader("🏪 Store & Competition")
            open_store = st.selectbox("Is Store Open?", [1, 0], index=0)
            promo = st.selectbox("Promotion Active?", [1, 0], index=0)
        
        with row2_col2:
            st.write("")
            st.write("")
            competition_distance_raw = st.number_input("Competition Distance (meters)", value=1270.0, min_value=20.0, step=10.0, format="%.1f")
            competition_open_since_month = st.number_input("Comp. Open Month", min_value=1, max_value=12, value=9, step=1)
            competition_open_since_year = st.number_input("Comp. Open Year", min_value=1900, max_value=2025, value=2008, step=1)

        with row3_col1:
            st.subheader("🚀 Extended Promotions (Promo2)")
            promo2 = st.selectbox("Promo2 Active?", [1, 0], index=0)
            promo2_since_week_input = st.number_input("Promo2 Since Week", min_value=0, max_value=53, value=14, step=1,
                                                 help="If Promo2 active; else can be 0")
            promo2_since_year_input = st.number_input("Promo2 Since Year", min_value=0, max_value=2025, value=2011, step=1,
                                                  help="If Promo2 active; else can be 0")
        with row3_col2:
            st.markdown("**Promo2 Interval Months** (if Promo2 is active):")
            promo_interval_header_order = ['Jan', 'Apr', 'Jul', 'Oct', 'Feb', 'May', 'Aug', 'Nov', 'Mar', 'Jun', 'Sept', 'Dec']
            promo_interval_active_months = st.multiselect("", promo_interval_header_order,
                                                     help="Select relevant months for Promo2")


        # ---- Feature Engineering Function (Keep this inside where it's used or make global) ----
        def create_feature_vector(inputs_dict):
            features = []
            features.append(inputs_dict['dayofweek'])
            features.append(inputs_dict['open_store'])
            features.append(inputs_dict['promo'])
            sh_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3} # YOU MUST VERIFY THIS MAPPING
            features.append(sh_map.get(inputs_dict['state_holiday_cat_input'], 0))
            features.append(inputs_dict['school_holiday'])
            features.append(inputs_dict['day'])
            features.append(inputs_dict['month'])
            features.append(inputs_dict['year'])
            features.append(inputs_dict['weekofyear_input'])
            features.append(np.log1p(inputs_dict['competition_distance_raw']))
            features.append(inputs_dict['competition_open_since_month'])
            features.append(inputs_dict['competition_open_since_year'])
            features.append(inputs_dict['promo2'])
            p2sw = inputs_dict['promo2_since_week_input']
            features.append(p2sw if inputs_dict['promo2'] == 1 and p2sw > 0 else 0)
            p2sy = inputs_dict['promo2_since_year_input']
            features.append(p2sy if inputs_dict['promo2'] == 1 and p2sy > 0 else 0)
            for month_name in inputs_dict['promo_interval_header_order']:
                features.append(1 if month_name in inputs_dict['promo_interval_active_months'] else 0)
            features.append(1 if inputs_dict['state_holiday_cat_input'] == 'c' else 0)
            features.append(1 if inputs_dict['state_holiday_cat_input'] == 'a' else 0)
            features.append(0) # 'd'
            features.append(1 if inputs_dict['state_holiday_cat_input'] == 'b' else 0)
            return np.array(features, dtype=np.float32).reshape(1, -1)

        # ---- Submit Button ----
        submitted = st.form_submit_button(" forecasting Predict Sales", use_container_width=True)

        if submitted:
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
            
            with st.spinner("🧠 Calculating forecast..."):
                try:
                    feature_vector = create_feature_vector(current_inputs)
                    EXPECTED_NUM_FEATURES = 31 # Based on your header
                    
                    if feature_vector.shape[1] != EXPECTED_NUM_FEATURES:
                        st.error(f"Feature Mismatch: Expected {EXPECTED_NUM_FEATURES}, got {feature_vector.shape[1]}. Review `create_feature_vector` and StateHoliday logic.")
                    else:
                        prediction = model.predict(feature_vector)
                        predicted_sales = np.expm1(prediction[0])
                        st.session_state.predicted_sales_value = predicted_sales
                        st.session_state.show_results = True
                        st.rerun() # Rerun to show the results "page"

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.text(traceback.format_exc())

elif st.session_state.show_results and st.session_state.predicted_sales_value is not None:
    # ---- Results "Page" ----
    st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-header">Forecasted Sales</div>
            <div class="prediction-value">
                <span class="prediction-currency">€</span>
                {st.session_state.predicted_sales_value:,.2f}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("‹ Back to Input Form", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.predicted_sales_value = None
        st.rerun()

# ---- Sidebar for extra info (Optional) ----
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts Rossmann store sales based on historical data patterns. "
    "Provide the necessary details and click 'Predict Sales' to get a forecast."
)
st.sidebar.markdown("---")
st.sidebar.caption("Model Version: 1.0") # Example
