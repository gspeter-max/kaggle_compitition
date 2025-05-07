import streamlit as st
import numpy as np
import joblib
import traceback
import random # For potential "Surprise Me" or varied presets

# ---- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ----
st.set_page_config(
    page_title="Rossmann Sales Forecaster Pro 🚀", # Added "Pro" and emoji
    page_icon="✨", # Changed icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Refined Dark Mode & Enhanced UI Elements ---
st.markdown("""
    <style>
        /* Base Dark Theme - High Contrast & Clean */
        body, .stApp {
            color: #EAEAEA; /* Main text color */
            background-color: #0E1117; /* Main dark background */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF; /* Brighter white for headers */
            font-weight: 600; /* Slightly bolder headers */
        }
        /* Sidebar */
        .css-1d391kg { /* Sidebar main background */
            background-color: #161A1F; /* Slightly different dark for sidebar */
            border-right: 1px solid #30363F;
        }
        .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown li { /* Sidebar text */
            color: #C0C0C0;
        }
        /* Input widgets */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div,
        .stMultiSelect > div > div > div > div {
            background-color: #20242A; /* Darker input background */
            color: #EAEAEA;
            border-radius: 0.3rem;
            border: 1px solid #3A3F4A;
        }
        /* Buttons */
        .stButton>button {
            border-radius: 0.3rem;
            background-color: #0078D4; /* A good, accessible blue */
            color: white;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: 500;
            transition: background-color 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #005A9E;
        }
        .stButton>button:active {
            background-color: #004C82;
        }
        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 1.1em;
            font-weight: 500;
            color: #B0B0B0;
        }
        /* Prediction Result Area */
        .prediction-result-area {
            text-align: center;
            padding: 2em;
            margin: 2em auto; /* Center it more */
            border: 1px solid #30363F;
            border-radius: 0.75rem;
            background: linear-gradient(145deg, #1A1C22, #20242A); /* Subtle gradient */
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.3);
            max-width: 600px; /* Constrain width for better focus */
        }
        .prediction-result-header {
            font-size: 1.5em;
            color: #A0A0A5;
            margin-bottom: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .prediction-result-value {
            font-size: 3.8em;
            font-weight: 700; /* Bolder */
            color: #4CAF50; /* Vibrant Green */
            margin-bottom: 0.3em;
            line-height: 1.1;
        }
        .prediction-result-currency {
            font-size: 1.8em;
            color: #4CAF50;
            vertical-align: middle; /* Better alignment */
            margin-right: 0.15em;
        }
        /* Key Factors Styling */
        .key-factors-container {
            margin-top: 1.5em;
            padding-top: 1em;
            border-top: 1px dashed #3A3F4A;
        }
        .key-factors-title {
            font-size: 1.1em;
            color: #A0A0A5;
            text-align: center;
            margin-bottom: 0.75em;
        }
        .factor-badge {
            display: inline-block;
            background-color: #30363F;
            color: #B0B0B0;
            padding: 0.3em 0.6em;
            border-radius: 1em;
            font-size: 0.9em;
            margin: 0.2em;
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
        return "FileNotFound"
    except Exception:
        return "LoadError"

model_load_status = load_model_resource()

# --- Initialize Session State ---
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'predicted_sales_value' not in st.session_state:
    st.session_state.predicted_sales_value = None
if 'key_factors_for_display' not in st.session_state:
    st.session_state.key_factors_for_display = []
if 'current_inputs' not in st.session_state: # To store inputs for presets
    st.session_state.current_inputs = {}


# --- Input Presets / Scenarios ---
PRESETS = {
    "Default Values": {
        'dayofweek': 4, 'day': 15, 'month': 6, 'year': 2015, 'weekofyear_input': 22,
        'state_holiday_cat_input': '0', 'school_holiday': 0, 'open_store': 1, 'promo': 0,
        'competition_distance_raw': 1270.0, 'competition_open_since_month': 9, 'competition_open_since_year': 2008,
        'promo2': 0, 'promo2_since_week_input': 14, 'promo2_since_year_input': 2011,
        'promo_interval_active_months': []
    },
    "Busy Weekend (Promo)": {
        'dayofweek': 5, 'day': 16, 'month': 7, 'year': 2014, 'weekofyear_input': 29, # Friday
        'state_holiday_cat_input': '0', 'school_holiday': 1, 'open_store': 1, 'promo': 1, # Promo active
        'competition_distance_raw': 500.0, 'competition_open_since_month': 3, 'competition_open_since_year': 2010,
        'promo2': 1, 'promo2_since_week_input': 22, 'promo2_since_year_input': 2012,
        'promo_interval_active_months': ['Jan', 'Apr', 'Jul', 'Oct']
    },
    "Quiet Midweek (No Promo)": {
        'dayofweek': 2, 'day': 10, 'month': 2, 'year': 2013, 'weekofyear_input': 7, # Tuesday
        'state_holiday_cat_input': '0', 'school_holiday': 0, 'open_store': 1, 'promo': 0,
        'competition_distance_raw': 2500.0, 'competition_open_since_month': 1, 'competition_open_since_year': 2015,
        'promo2': 0, 'promo2_since_week_input': 0, 'promo2_since_year_input': 0,
        'promo_interval_active_months': []
    },
    "Public Holiday (Store Closed)": {
        'dayofweek': 0, 'day': 25, 'month': 12, 'year': 2015, 'weekofyear_input': 52, # Christmas
        'state_holiday_cat_input': 'c', 'school_holiday': 1, 'open_store': 0, 'promo': 0, # Store Closed
        'competition_distance_raw': 1000.0, 'competition_open_since_month': 6, 'competition_open_since_year': 2005,
        'promo2': 1, 'promo2_since_week_input': 40, 'promo2_since_year_input': 2014,
        'promo_interval_active_months': ['Mar', 'Jun', 'Sept', 'Dec']
    }
}

def apply_preset(preset_name):
    st.session_state.current_inputs = PRESETS[preset_name]
    # We don't directly set widget values here to avoid complex state management if widgets are already rendered.
    # Instead, the form will use these default values when it's created or if reset.
    # For an immediate update, you'd need to use widget keys and st.session_state for each.
    # For simplicity, this will apply when the form is next drawn fresh or a button forces rerun.

# ---- Sidebar ----
st.sidebar.title("⚙️ Controls & Options")
selected_preset = st.sidebar.selectbox(
    "Load Scenario Preset:",
    options=list(PRESETS.keys()),
    index=0, # Default to "Default Values"
    on_change=apply_preset, # This might not immediately update rendered widgets without keys
    args=(st.session_state.get('selected_preset_name_for_change', list(PRESETS.keys())[0]),) # Hack to pass current selection for on_change
)
# Store the selectbox value to make on_change work more reliably with args
st.session_state.selected_preset_name_for_change = selected_preset

if st.sidebar.button("Apply Selected Scenario", use_container_width=True, key="apply_preset_button"):
    apply_preset(selected_preset)
    st.experimental_rerun() # Force rerun to update form with preset values if they are keyed

st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    "This Rossmann Sales Forecaster uses a machine learning model to predict store sales "
    "based on various factors like promotions, holidays, and competition."
)
st.sidebar.markdown("---")
st.sidebar.caption("Model v1.1 | UI Enhanced")


# ---- Main App Logic ----
st.title("✨ Rossmann Sales Forecaster Pro") # Title with emoji

if isinstance(model_load_status, str): # Handle model loading issues
    if model_load_status == "FileNotFound":
        st.error(f"Critical Error: Model file ('{MODEL_PATH}') not found. Please ensure it's in the repository and Git LFS was used correctly if it's a large file.")
    elif model_load_status == "LoadError":
        st.error("Critical Error: A problem occurred while loading the prediction model. Check the application logs (if deployed) or console output for details.")
    st.warning("Application cannot proceed without the model.", icon="🚫")
    st.stop()
else:
    model = model_load_status
    if not st.session_state.show_results:
         st.markdown("🤖 **Model loaded successfully!** Fill in the details below or select a scenario from the sidebar to get a sales forecast.", unsafe_allow_html=True)

# ---- Conditional Display: Input Form OR Results Page ----
if not st.session_state.show_results and model is not None:
    # Get default values from session state (updated by presets) or use initial defaults
    defaults = st.session_state.current_inputs if st.session_state.current_inputs else PRESETS["Default Values"]

    with st.form(key="sales_input_form"):
        st.header("📝 Input Data for Forecast")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.expander("📅 Date & Time Details", expanded=True):
                dayofweek = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)), index=defaults['dayofweek']-1, key="dayofweek_in") # Index is 0-based
                day = st.number_input("Day of Month", min_value=1, max_value=31, value=defaults['day'], step=1, key="day_in")
                month = st.number_input("Month", min_value=1, max_value=12, value=defaults['month'], step=1, key="month_in")
                year = st.number_input("Year", min_value=2013, max_value=2025, value=defaults['year'], step=1, key="year_in")
                weekofyear_input = st.number_input("Week of Year", min_value=1, max_value=53, value=defaults['weekofyear_input'], step=1, key="weekofyear_in")

        with col2:
            with st.expander("🏪 Store & Holiday Status", expanded=True):
                open_store = st.selectbox("Store Open?", [1, 0], index=1-defaults['open_store'], key="open_in") # index for 0/1
                promo = st.selectbox("Promotion Active?", [1, 0], index=1-defaults['promo'], key="promo_in")
                state_holiday_options = ['0', 'a', 'b', 'c']
                state_holiday_cat_input = st.selectbox("State Holiday", state_holiday_options, index=state_holiday_options.index(defaults['state_holiday_cat_input']),
                                                 help="'0': None, 'a': Public, 'b': Easter, 'c': Christmas", key="stateholiday_in")
                school_holiday = st.selectbox("School Holiday?", [0, 1], index=1-defaults['school_holiday'], key="schoolholiday_in")
        
        with col3:
            with st.expander("🏆 Competition Factors", expanded=True):
                competition_distance_raw = st.number_input("Competition Distance (m)", value=float(defaults['competition_distance_raw']), min_value=20.0, step=10.0, format="%.1f", key="compdist_in")
                competition_open_since_month = st.number_input("Comp. Open Month", min_value=1, max_value=12, value=defaults['competition_open_since_month'], step=1, key="compmonth_in")
                competition_open_since_year = st.number_input("Comp. Open Year", min_value=1900, max_value=2025, value=defaults['competition_open_since_year'], step=1, key="compyear_in")

        st.markdown("---")
        with st.expander("🚀 Extended Promotions (Promo2)", expanded=False): # Start collapsed
            promo2_col1, promo2_col2 = st.columns([1,2])
            with promo2_col1:
                promo2 = st.selectbox("Promo2 Active?", [1, 0], index=1-defaults['promo2'], key="promo2_in")
                promo2_since_week_input = st.number_input("Promo2 Since Week", min_value=0, max_value=53, value=defaults['promo2_since_week_input'], step=1, key="promo2week_in")
                promo2_since_year_input = st.number_input("Promo2 Since Year", min_value=0, max_value=2025, value=defaults['promo2_since_year_input'], step=1, key="promo2year_in")
            with promo2_col2:
                st.markdown("**Promo2 Interval Months** (if Promo2 active):")
                promo_interval_header_order = ['Jan', 'Apr', 'Jul', 'Oct', 'Feb', 'May', 'Aug', 'Nov', 'Mar', 'Jun', 'Sept', 'Dec']
                promo_interval_active_months = st.multiselect("", promo_interval_header_order, default=defaults['promo_interval_active_months'],
                                                         help="Select relevant months", key="promointerval_in")

        def create_feature_vector(inputs_dict_form): # Same feature engineering
            features = []
            features.append(inputs_dict_form['dayofweek_in'])
            features.append(inputs_dict_form['open_in'])
            features.append(inputs_dict_form['promo_in'])
            sh_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3} # CRITICAL: VERIFY THIS MAPPING
            features.append(sh_map.get(inputs_dict_form['stateholiday_in'], 0))
            features.append(inputs_dict_form['schoolholiday_in'])
            features.append(inputs_dict_form['day_in'])
            features.append(inputs_dict_form['month_in'])
            features.append(inputs_dict_form['year_in'])
            features.append(inputs_dict_form['weekofyear_in'])
            features.append(np.log1p(inputs_dict_form['compdist_in']))
            features.append(inputs_dict_form['compmonth_in'])
            features.append(inputs_dict_form['compyear_in'])
            features.append(inputs_dict_form['promo2_in'])
            p2sw = inputs_dict_form['promo2week_in']
            features.append(p2sw if inputs_dict_form['promo2_in'] == 1 and p2sw > 0 else 0)
            p2sy = inputs_dict_form['promo2year_in']
            features.append(p2sy if inputs_dict_form['promo2_in'] == 1 and p2sy > 0 else 0)
            for month_name in promo_interval_header_order: # Use the locally defined order
                features.append(1 if month_name in inputs_dict_form['promointerval_in'] else 0)
            features.append(1 if inputs_dict_form['stateholiday_in'] == 'c' else 0)
            features.append(1 if inputs_dict_form['stateholiday_in'] == 'a' else 0)
            features.append(0)
            features.append(1 if inputs_dict_form['stateholiday_in'] == 'b' else 0)
            return np.array(features, dtype=np.float32).reshape(1, -1)

        submitted = st.form_submit_button("🔮 Generate Forecast", use_container_width=True)

        if submitted:
            # Collect inputs directly from their keys in st.session_state because they are now keyed
            form_inputs = {key.replace('_in', ''): st.session_state[key] for key in st.session_state if key.endswith('_in')}
            
            # Manually add multiselect as it's not directly in st.session_state with _in suffix the same way
            form_inputs['promo_interval_active_months'] = st.session_state.promointerval_in
            # And the header order used in the form
            form_inputs['promo_interval_header_order'] = promo_interval_header_order


            with st.spinner("⏳ Crunching the numbers... one moment!"):
                try:
                    feature_vector = create_feature_vector(st.session_state) # Pass session_state which holds keyed inputs
                    EXPECTED_NUM_FEATURES = 31
                    
                    if feature_vector.shape[1] != EXPECTED_NUM_FEATURES:
                        st.error(f"Feature Mismatch Error (Expected {EXPECTED_NUM_FEATURES}, Got {feature_vector.shape[1]}). Please check inputs or model feature expectations. The 'StateHoliday Feature #4 vs OHE' is a common area to re-verify from your training data.")
                    else:
                        prediction = model.predict(feature_vector)
                        predicted_sales = np.expm1(prediction[0])
                        st.session_state.predicted_sales_value = predicted_sales
                        
                        # Simple Key Factors Teaser
                        key_factors = []
                        if st.session_state.promo_in == 1: key_factors.append("🎉 Active Promotion")
                        if st.session_state.open_in == 0: key_factors.append("Store Closed")
                        if st.session_state.dayofweek_in in [5, 6, 7]: key_factors.append("🗓️ Weekend") # Corrected index
                        if st.session_state.stateholiday_in != '0': key_factors.append(f"State Holiday '{st.session_state.stateholiday_in}'")
                        if st.session_state.compdist_in < 1000: key_factors.append("Low Competition")
                        st.session_state.key_factors_for_display = key_factors

                        st.session_state.show_results = True
                        st.experimental_rerun() # Use experimental_rerun for cleaner state transition
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
            <div class="key-factors-container">
                <div class="key-factors-title">Potential Key Influencers:</div>
                {''.join([f'<span class="factor-badge">{factor}</span>' for factor in st.session_state.key_factors_for_display]) if st.session_state.key_factors_for_display else "<span class='factor-badge'>Standard Conditions</span>"}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("‹ Make Another Forecast", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.predicted_sales_value = None
        st.session_state.key_factors_for_display = []
        st.session_state.current_inputs = {} # Reset current inputs to allow form to use defaults again
        st.experimental_rerun()
