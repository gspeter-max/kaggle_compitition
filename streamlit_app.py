import streamlit as st
import numpy as np
import joblib
import traceback

# ---- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ----
st.set_page_config(
    page_title="Rossmann Sales Forecaster 📈",
    page_icon="🛒",
    layout="wide", # Wide layout is often good for dashboards
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Simple, High-Contrast Dark Mode ---
# This aims for clarity and readability, less "designed"
st.markdown("""
    <style>
        /* Base Dark Theme - High Contrast */
        body {
            color: #FAFAFA; /* Very light grey / off-white for text */
            background-color: #0E1117; /* Standard Streamlit dark bg */
        }
        .stApp {
             background-color: #0E1117;
        }
        /* Ensure headers are also very light */
        h1, h2, h3, h4, h5, h6 {
            color: #ECECEC;
        }
        /* Sidebar styling for consistency */
        .css-1d391kg { /* Sidebar main background */
            background-color: #1A1C22; /* Slightly different dark for sidebar */
        }
        /* Input widgets - keep them distinguishable but dark */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div,
        .stMultiSelect > div > div > div > div {
            background-color: #262730; /* Streamlit's dark input bg */
            color: #FAFAFA;
            border-radius: 0.25rem;
            border: 1px solid #3A3F4A; /* Subtle border */
        }
        .stButton>button {
            border-radius: 0.25rem;
            background-color: #0068C9; /* A clear blue for buttons */
            color: white;
            border: none;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #0052A2;
        }
        .stAlert {
            border-radius: 0.25rem;
        }

        /* Prediction Result Styling - Prominent but Clean */
        .prediction-result-area {
            text-align: center;
            padding: 1.5em;
            margin-top: 2em;
            border: 1px solid #3A3F4A;
            border-radius: 0.5rem;
            background-color: #1A1C22; /* Slightly offset dark background */
        }
        .prediction-result-header {
            font-size: 1.8em; /* Larger than standard h3 */
            color: #A0A0A5; /* Muted but readable header color */
            margin-bottom: 0.5em;
        }
        .prediction-result-value {
            font-size: 3em; /* Big sales value */
            font-weight: bold;
            color: #28A745; /* Green for positive/value */
            margin-bottom: 0.3em;
        }
        .prediction-result-currency {
            font-size: 1.5em; /* Smaller currency symbol */
            color: #28A745; /* Match value color */
            vertical-align: baseline; /* Align with bottom of number */
            margin-right: 0.2em;
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

model_load_status = load_model_resource()

# --- Initialize Session State ---
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'predicted_sales_value' not in st.session_state:
    st.session_state.predicted_sales_value = None


# ---- Main App Logic ----
st.title("🛒 Rossmann Sales Forecaster")

if isinstance(model_load_status, str):
    if model_load_status == "FileNotFound":
        st.error(f"Error: Model file ('{MODEL_PATH}') not found. Please ensure it's in the repository and Git LFS was used correctly if it's a large file.")
    elif model_load_status == "LoadError":
        st.error("A critical error occurred while loading the prediction model. Check the application logs for details.")
    st.warning("Application cannot proceed without the model.", icon="🚫")
    st.stop()
else:
    model = model_load_status
    if not st.session_state.show_results: # Only show success if not on results page
        st.success("🤖 Model loaded successfully! Please provide the details below.", icon="✅")


# ---- Conditional Display: Input Form OR Results Page ----

if not st.session_state.show_results and model is not None:
    with st.form(key="sales_input_form"):
        st.header("Enter Store & Promotion Details")
        
        row1_col1, row1_col2, row1_col3 = st.columns(3) # Adjusted to 3 columns for inputs
        
        with row1_col1:
            st.markdown("##### 📅 Date & Time")
            dayofweek = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)), index=4)
            day = st.number_input("Day of Month", min_value=1, max_value=31, value=15, step=1)
            month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
            year = st.number_input("Year", min_value=2013, max_value=2025, value=2015, step=1)
            weekofyear_input = st.number_input("Week of Year", min_value=1, max_value=53, value=22, step=1)

        with row1_col2:
            st.markdown("##### 🏪 Store Status & Holidays")
            open_store = st.selectbox("Is Store Open?", [1, 0], index=0)
            promo = st.selectbox("Promotion Active?", [1, 0], index=0)
            state_holiday_cat_input = st.selectbox("State Holiday", ['0', 'a', 'b', 'c'], index=0,
                                             help="'0': None, 'a': Public, 'b': Easter, 'c': Christmas")
            school_holiday = st.selectbox("School Holiday?", [0, 1], index=0)
            competition_distance_raw = st.number_input("Competition Distance (m)", value=1270.0, min_value=20.0, step=10.0, format="%.1f")


        with row1_col3:
            st.markdown("##### 🏆 Competition & Promo2")
            competition_open_since_month = st.number_input("Comp. Open Month", min_value=1, max_value=12, value=9, step=1)
            competition_open_since_year = st.number_input("Comp. Open Year", min_value=1900, max_value=2025, value=2008, step=1)
            promo2 = st.selectbox("Promo2 Active?", [1, 0], index=0)
            promo2_since_week_input = st.number_input("Promo2 Since Week", min_value=0, max_value=53, value=14, step=1)
            promo2_since_year_input = st.number_input("Promo2 Since Year", min_value=0, max_value=2025, value=2011, step=1)
            
        st.markdown("---")
        st.markdown("##### 🚀 Promo2 Interval Months (if Promo2 is active):")
        promo_interval_header_order = ['Jan', 'Apr', 'Jul', 'Oct', 'Feb', 'May', 'Aug', 'Nov', 'Mar', 'Jun', 'Sept', 'Dec']
        promo_interval_active_months = st.multiselect("", promo_interval_header_order,
                                                     help="Select relevant months for Promo2 if Promo2 is active")


        def create_feature_vector(inputs_dict): # Same feature engineering as before
            features = []
            features.append(inputs_dict['dayofweek'])
            features.append(inputs_dict['open_store'])
            features.append(inputs_dict['promo'])
            sh_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
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
            features.append(0) 
            features.append(1 if inputs_dict['state_holiday_cat_input'] == 'b' else 0)
            return np.array(features, dtype=np.float32).reshape(1, -1)

        submitted = st.form_submit_button("📈 Predict Sales", use_container_width=True)

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
                    EXPECTED_NUM_FEATURES = 31
                    if feature_vector.shape[1] != EXPECTED_NUM_FEATURES:
                        st.error(f"Feature Mismatch Error. Please check inputs or model expectations.")
                    else:
                        prediction = model.predict(feature_vector)
                        predicted_sales = np.expm1(prediction[0])
                        st.session_state.predicted_sales_value = predicted_sales
                        st.session_state.show_results = True
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.text(traceback.format_exc())

elif st.session_state.show_results and st.session_state.predicted_sales_value is not None:
    st.markdown(f"""
        <div class="prediction-result-area">
            <div class="prediction-result-header">Forecasted Sales</div>
            <div class="prediction-result-value">
                <span class="prediction-result-currency">€</span>{st.session_state.predicted_sales_value:,.2f}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("‹ New Forecast", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.predicted_sales_value = None
        st.rerun()

# Sidebar
st.sidebar.title("About This App")
st.sidebar.info(
    "This Rossmann Sales Forecaster uses a machine learning model to predict store sales "
    "based on various factors like promotions, holidays, and competition."
)
st.sidebar.markdown("---")
st.sidebar.caption("Model v1.0 | Developed with Streamlit")
