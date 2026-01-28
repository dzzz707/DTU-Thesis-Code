import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="EU Grid Emission Factor Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EUGridEmissionSimulator:
    """European Union Grid Emission Factor Simulator"""
    
    def __init__(self):
        self.load_eu_data()
        self.setup_styles()
    
    def setup_styles(self):
        """Add global blue theme styles"""
        st.markdown("""
        <style>
        /* Global blue theme */
        .main .block-container {
            padding-top: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Primary button styles */
        .stButton > button {
            background-color: #1f4e79 !important;
            border-color: #4a90e2 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #2d5aa0 !important;
            border-color: #5ba3f5 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
        }
        
        /* Metric card styles */
        .metric-container {
            background-color: #f8f9fa;
            border: 1px solid #4a90e2;
            border-radius: 15px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.15);
            height: 200px !important;
            min-height: 200px !important;
            max-height: 200px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            align-items: center !important;
            overflow: hidden;
            width: 100%;
            box-sizing: border-box;
        }
                
        .metric-value {
            font-size: clamp(1rem, 2.5vw, 2.2rem);
            font-weight: 700;
            color: #1f4e79;
            margin: 0;
            line-height: 1.1;
            flex: 0 0 auto;
            word-break: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            max-height: 70px;
            text-align: center;
            padding: 0.2rem;
        }
                
        .metric-label {
            color: #4a90e2;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.3;
            flex: 1 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .metric-change {
            font-size: 0.9rem;
            margin: 0;
            font-weight: 500;
            flex: 0 0 auto;
        }
        
        .metric-change.positive {
            color: #e74c3c;
        }

        /* Selectbox styling */
        .stSelectbox > div > div > div {
            background-color: #2c3e50 !important;
            color: #ffffff !important;
            border-color: #4a90e2 !important;
            border-radius: 8px !important;
        }
        
        .stSelectbox > div > div > div > div {
            color: #ffffff !important;
        }
        
        /* Slider styles */
        .stSlider > div > div > div {
            background-color: #4a90e2 !important;
        }
        
        /* Header styles */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .sub-header {
            color: #4a90e2;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        /* Info box styles */
        .stInfo {
            background-color: rgba(74, 144, 226, 0.1) !important;
            border-color: #4a90e2 !important;
        }
        
        .stSuccess {
            background-color: rgba(39, 174, 96, 0.1) !important;
            border-color: #27ae60 !important;
        }
        
        .stWarning {
            background-color: rgba(243, 156, 18, 0.1) !important;
            border-color: #f39c12 !important;
        }
        
        /* Chart container styles */
        .chart-container {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_eu_data(self):
        """Load European Union country grid emission factor data"""
        # Based on real CoM database EU country data
        self.years = list(range(1990, 2022))
        
        # EU country data (tonnes CO2/MWh) - built from real data
        self.eu_countries = {
            "Austria": {
                "code": "AT",
                "factors": [0.354, 0.365, 0.322, 0.290, 0.302, 0.307, 0.340, 0.327, 0.313, 0.308, 0.313, 0.333, 0.341, 0.390, 0.362, 0.377, 0.372, 0.371, 0.330, 0.314, 0.332, 0.374, 0.324, 0.319, 0.315, 0.337, 0.301, 0.314, 0.288, 0.252, 0.220, 0.243]
            },
            "Belgium": {
                "code": "BE",
                "factors": [0.408, 0.395, 0.389, 0.393, 0.413, 0.401, 0.382, 0.343, 0.364, 0.318, 0.317, 0.298, 0.294, 0.299, 0.303, 0.307, 0.297, 0.291, 0.292, 0.261, 0.260, 0.226, 0.244, 0.222, 0.234, 0.270, 0.216, 0.216, 0.226, 0.188, 0.188, 0.169]
            },
            "Bulgaria": {
                "code": "BG",
                "factors": [0.640, 0.781, 0.903, 0.936, 0.845, 0.779, 0.733, 0.818, 0.830, 0.798, 0.732, 0.791, 0.731, 0.798, 0.770, 0.718, 0.697, 0.847, 0.773, 0.745, 0.749, 0.824, 0.737, 0.655, 0.638, 0.648, 0.609, 0.657, 0.568, 0.542, 0.475, 0.505]
            },
            "Croatia": {
                "code": "HR",
                "factors": [0.235, 0.161, 0.308, 0.305, 0.171, 0.207, 0.163, 0.243, 0.309, 0.299, 0.264, 0.305, 0.349, 0.374, 0.255, 0.193, 0.209, 0.286, 0.146, 0.118, 0.092, 0.093, 0.100, 0.097, 0.362, 0.480, 0.511, 0.417, 0.482, 0.345, 0.318, 0.376]
            },
            "Cyprus": {
                "code": "CY",
                "factors": [0.944, 0.936, 0.970, 0.949, 0.946, 0.944, 0.966, 0.985, 0.986, 0.993, 0.963, 0.907, 0.864, 0.946, 0.885, 0.890, 0.865, 0.864, 0.855, 0.844, 0.789, 0.787, 0.799, 0.716, 0.734, 0.735, 0.742, 0.720, 0.709, 0.686, 0.684, 0.660]
            },
            "Czechia": {
                "code": "CZ",
                "factors": [1.064, 1.069, 1.056, 1.068, 1.050, 1.040, 1.046, 0.985, 0.982, 0.925, 0.929, 0.928, 0.897, 0.833, 0.848, 0.871, 0.864, 0.869, 0.832, 0.794, 0.786, 0.780, 0.740, 0.707, 0.701, 0.717, 0.708, 0.657, 0.634, 0.580, 0.513, 0.544]
            },
            "Denmark": {
                "code": "DK",
                "factors": [0.508, 0.753, 0.583, 0.606, 0.695, 0.626, 0.714, 0.616, 0.563, 0.500, 0.414, 0.462, 0.431, 0.513, 0.392, 0.270, 0.473, 0.361, 0.310, 0.363, 0.375, 0.272, 0.175, 0.297, 0.219, 0.134, 0.193, 0.138, 0.165, 0.131, 0.085, 0.103]
            },
            "Estonia": {
                "code": "EE",
                "factors": [1.336, 1.187, 1.466, 1.605, 1.929, 2.081, 1.989, 1.842, 1.822, 1.826, 1.585, 1.519, 1.435, 1.496, 1.429, 1.414, 1.353, 1.459, 1.233, 0.912, 1.256, 1.185, 0.991, 1.070, 0.910, 0.595, 0.859, 1.026, 0.889, 0.510, 0.228, 0.249]
            },
            "Finland": {
                "code": "FI",
                "factors": [0.181, 0.195, 0.157, 0.196, 0.257, 0.228, 0.304, 0.262, 0.190, 0.186, 0.166, 0.211, 0.228, 0.297, 0.252, 0.147, 0.252, 0.247, 0.192, 0.197, 0.236, 0.191, 0.124, 0.159, 0.124, 0.089, 0.101, 0.097, 0.108, 0.078, 0.055, 0.057]
            },
            "France": {
                "code": "FR",
                "factors": [0.142, 0.159, 0.124, 0.089, 0.085, 0.096, 0.098, 0.090, 0.122, 0.107, 0.091, 0.073, 0.081, 0.088, 0.084, 0.103, 0.095, 0.101, 0.089, 0.102, 0.099, 0.085, 0.092, 0.086, 0.064, 0.068, 0.077, 0.087, 0.069, 0.068, 0.066, 0.068]
            },
            "Germany": {
                "code": "DE",
                "factors": [0.745, 0.743, 0.703, 0.700, 0.701, 0.682, 0.673, 0.651, 0.645, 0.613, 0.617, 0.634, 0.622, 0.588, 0.577, 0.568, 0.566, 0.589, 0.553, 0.543, 0.530, 0.532, 0.545, 0.549, 0.528, 0.507, 0.508, 0.473, 0.452, 0.385, 0.342, 0.382]
            },
            "Greece": {
                "code": "EL",
                "factors": [1.245, 1.172, 1.185, 1.175, 1.113, 1.089, 1.035, 1.080, 1.066, 1.014, 1.022, 1.045, 0.989, 0.960, 0.962, 0.957, 0.890, 0.913, 0.880, 0.865, 0.833, 0.848, 0.839, 0.759, 0.793, 0.688, 0.572, 0.614, 0.636, 0.560, 0.453, 0.411]
            },
            "Hungary": {
                "code": "HU",
                "factors": [0.796, 0.779, 0.797, 0.763, 0.706, 0.729, 0.696, 0.703, 0.728, 0.734, 0.635, 0.606, 0.554, 0.618, 0.582, 0.490, 0.498, 0.478, 0.470, 0.428, 0.434, 0.415, 0.384, 0.383, 0.354, 0.345, 0.343, 0.281, 0.294, 0.245, 0.224, 0.220]
            },
            "Iceland": {
                "code": "IS",
                "factors": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
            },
            "Ireland": {
                "code": "IE",
                "factors": [0.906, 0.909, 0.914, 0.891, 0.886, 0.878, 0.855, 0.845, 0.847, 0.826, 0.768, 0.792, 0.752, 0.682, 0.675, 0.668, 0.607, 0.584, 0.544, 0.520, 0.527, 0.483, 0.519, 0.496, 0.484, 0.463, 0.459, 0.422, 0.363, 0.318, 0.291, 0.347]
            },
            "Italy": {
                "code": "IT",
                "factors": [0.583, 0.564, 0.545, 0.524, 0.518, 0.551, 0.532, 0.520, 0.524, 0.507, 0.511, 0.489, 0.514, 0.522, 0.517, 0.497, 0.495, 0.481, 0.465, 0.424, 0.421, 0.410, 0.401, 0.351, 0.328, 0.334, 0.329, 0.326, 0.303, 0.283, 0.269, 0.284]
            },
            "Latvia": {
                "code": "LV",
                "factors": [0.043, 0.187, 0.086, 0.439, 0.387, 0.283, 0.377, 0.246, 0.152, 0.309, 0.365, 0.321, 0.263, 0.368, 0.356, 0.368, 0.240, 0.361, 0.371, 0.299, 0.506, 0.510, 0.511, 0.581, 0.602, 0.586, 0.471, 0.367, 0.465, 0.332, 0.216, 0.301]
            },
            "Lithuania": {
                "code": "LT",
                "factors": [0.242, 0.268, 0.201, 0.209, 0.221, 0.210, 0.253, 0.203, 0.239, 0.233, 0.197, 0.198, 0.173, 0.169, 0.160, 0.159, 0.155, 0.132, 0.131, 0.141, 0.166, 0.137, 0.130, 0.124, 0.097, 0.123, 0.096, 0.070, 0.102, 0.068, 0.061, 0.078]
            },
            "Luxembourg": {
                "code": "LU",
                "factors": [0.851, 0.907, 0.831, 0.829, 0.756, 0.688, 0.662, 0.622, 0.595, 0.531, 0.523, 0.533, 0.560, 0.525, 0.503, 0.515, 0.501, 0.514, 0.492, 0.490, 0.529, 0.510, 0.531, 0.507, 0.481, 0.489, 0.472, 0.435, 0.396, 0.314, 0.283, 0.285]
            },
            "Malta": {
                "code": "MT",
                "factors": [1.935, 1.330, 1.225, 1.657, 1.491, 1.265, 1.232, 1.187, 1.169, 1.129, 1.024, 1.275, 1.182, 1.197, 1.155, 1.063, 1.078, 1.097, 1.073, 1.094, 1.022, 1.024, 1.039, 0.850, 0.807, 0.569, 0.496, 0.423, 0.372, 0.373, 0.393, 0.356]
            },
            "Netherlands": {
                "code": "NL",
                "factors": [0.640, 0.624, 0.616, 0.612, 0.608, 0.600, 0.581, 0.574, 0.566, 0.536, 0.539, 0.567, 0.535, 0.547, 0.556, 0.535, 0.525, 0.529, 0.500, 0.468, 0.467, 0.449, 0.483, 0.486, 0.505, 0.521, 0.499, 0.464, 0.447, 0.390, 0.317, 0.329]
            },
            "Norway": {
                "code": "NO",
                "factors": [0.001, 0.003, 0.001, 0.003, 0.016, 0.005, 0.035, 0.013, 0.005, 0.006, 0.002, 0.015, 0.012, 0.026, 0.019, 0.005, 0.014, 0.009, 0.006, 0.023, 0.038, 0.027, 0.012, 0.019, 0.015, 0.014, 0.015, 0.015, 0.016, 0.018, 0.008, 0.012]
            },
            "Poland": {
                "code": "PL",
                "factors": [1.280, 1.405, 1.434, 1.423, 1.438, 1.357, 1.311, 1.279, 1.240, 1.232, 1.206, 1.183, 1.174, 1.169, 1.150, 1.132, 1.129, 1.100, 1.063, 1.056, 1.035, 1.017, 0.972, 0.974, 0.929, 0.902, 0.876, 0.864, 0.843, 0.773, 0.725, 0.776]
            },
            "Portugal": {
                "code": "PT",
                "factors": [0.633, 0.635, 0.757, 0.662, 0.602, 0.684, 0.508, 0.544, 0.542, 0.637, 0.565, 0.525, 0.601, 0.485, 0.526, 0.576, 0.482, 0.442, 0.438, 0.424, 0.291, 0.347, 0.416, 0.329, 0.320, 0.402, 0.335, 0.422, 0.346, 0.271, 0.213, 0.179]
            },
            "Romania": {
                "code": "RO",
                "factors": [0.940, 1.034, 0.879, 1.009, 1.064, 1.011, 0.969, 0.754, 0.653, 0.704, 0.761, 0.709, 0.723, 0.837, 0.727, 0.643, 0.730, 0.754, 0.719, 0.675, 0.561, 0.682, 0.659, 0.497, 0.444, 0.465, 0.444, 0.470, 0.452, 0.457, 0.379, 0.377]
            },
            "Slovakia": {
                "code": "SK",
                "factors": [0.546, 0.647, 0.600, 0.582, 0.482, 0.511, 0.530, 0.568, 0.545, 0.513, 0.458, 0.423, 0.416, 0.479, 0.447, 0.443, 0.443, 0.518, 0.425, 0.419, 0.359, 0.428, 0.458, 0.368, 0.407, 0.457, 0.411, 0.427, 0.369, 0.358, 0.340, 0.352]
            },
            "Slovenia": {
                "code": "SI",
                "factors": [0.526, 0.407, 0.511, 0.503, 0.432, 0.448, 0.416, 0.410, 0.432, 0.380, 0.336, 0.368, 0.386, 0.387, 0.364, 0.372, 0.389, 0.428, 0.381, 0.323, 0.312, 0.364, 0.318, 0.301, 0.227, 0.250, 0.252, 0.234, 0.218, 0.218, 0.213, 0.203]
            },
            "Spain": {
                "code": "ES",
                "factors": [0.522, 0.517, 0.584, 0.508, 0.496, 0.548, 0.430, 0.467, 0.458, 0.531, 0.512, 0.450, 0.520, 0.453, 0.455, 0.471, 0.446, 0.449, 0.383, 0.349, 0.279, 0.336, 0.357, 0.292, 0.305, 0.348, 0.294, 0.337, 0.302, 0.232, 0.183, 0.174]
            },
            "Sweden": {
                "code": "SE",
                "factors": [0.012, 0.024, 0.026, 0.025, 0.032, 0.021, 0.092, 0.048, 0.033, 0.027, 0.024, 0.034, 0.047, 0.099, 0.065, 0.033, 0.061, 0.033, 0.028, 0.037, 0.055, 0.032, 0.017, 0.030, 0.020, 0.013, 0.021, 0.016, 0.020, 0.016, 0.012, 0.014]
            },
            "United Kingdom": {
                "code": "UK",
                "factors": [0.808, 0.782, 0.759, 0.663, 0.638, 0.638, 0.610, 0.560, 0.561, 0.520, 0.558, 0.576, 0.573, 0.607, 0.580, 0.569, 0.599, 0.594, 0.575, 0.531, 0.529, 0.521, 0.577, 0.536, 0.482, 0.411, 0.326, 0.289, 0.264, 0.241, np.nan, np.nan]
            }
        }

        # Calculate statistical information
        for country_name, data in self.eu_countries.items():
            factors = data["factors"]
            # Handle NaN values in the data
            factors_clean = [f for f in factors if not pd.isna(f)]
            
            if len(factors_clean) > 0:
                data["latest"] = factors_clean[-1]
                data["earliest"] = factors_clean[0]
                data["total_change"] = ((factors_clean[-1] - factors_clean[0]) / factors_clean[0] * 100)
                data["avg_reduction"] = -data["total_change"] / len(factors_clean)
            else:
                # If all data is NaN, set default values
                data["latest"] = 0
                data["earliest"] = 0
                data["total_change"] = 0
                data["avg_reduction"] = 0
    
    def display_header(self):
        """Display application header"""
        st.markdown('<h1 class="main-header">🇪🇺 European Union Grid Emission Factor Simulator</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
            Simulate the impact of grid decarbonization on corporate Scope 2 emissions across EU member states
        </div>
        """, unsafe_allow_html=True)
    
    def display_sidebar_controls(self):
        """Display sidebar control panel"""
        st.sidebar.markdown('<h2 class="sub-header">Simulation Controls</h2>', unsafe_allow_html=True)
        
        # Country selection
        st.sidebar.markdown("### Country Selection")
        country_names = list(self.eu_countries.keys())
        selected_country = st.sidebar.selectbox(
            "Select EU Member State",
            country_names,
            index=country_names.index("Denmark"),
            key="country_selector"
        )
        
        # Corporate electricity usage settings
        st.sidebar.markdown("### Corporate Electricity Usage")
        electricity_usage = st.sidebar.number_input(
            "Annual Electricity Consumption (MWh)",
            min_value=100,
            max_value=10000000,
            value=10000,
            step=500,
            help="Enter your organization's annual electricity consumption"
        )
        
        # Advanced parameters
        st.sidebar.markdown("### Prediction Parameters")
        custom_reduction_rate = st.sidebar.slider(
            "Annual Reduction Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
            help="Controls steepness of decarbonization curve (higher = faster transition)"
        )
        
        target_year = st.sidebar.slider(
            "Carbon Neutrality Target",
            min_value=2030,
            max_value=2060,
            value=2050,
            step=5,
            help="Year to achieve near-zero emissions"
        )
        
        carbon_price = st.sidebar.slider(
            "Carbon Price (€/tonne CO2)",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            help="Carbon tax or ETS price for cost calculations"
        )
        
        return {
            "country": selected_country,
            "electricity_usage": electricity_usage,
            "reduction_rate": custom_reduction_rate,
            "target_year": target_year,
            "carbon_price": carbon_price
        }
    
    def sigmoid_emission_factor(self, year, baseline_year, baseline_factor, target_year, target_factor=0.01, steepness=1.0):
        """
        S-curve (sigmoid) model for grid emission factor prediction
        
        Args:
            year: prediction year
            baseline_year: reference year (2021)
            baseline_factor: emission factor at baseline year
            target_year: carbon neutrality target year
            target_factor: minimum achievable emission factor
            steepness: controls the steepness of the S-curve (higher = steeper)
        """
        if year <= baseline_year:
            return baseline_factor
        
        # Calculate S-curve parameters
        total_years = target_year - baseline_year
        midpoint = baseline_year + total_years * 0.6  # Inflection point at 60% of timeline
        k = steepness * 6.0 / total_years  # Steepness parameter
        
        # Sigmoid function (0 to 1)
        progress = 1 / (1 + np.exp(-k * (year - midpoint)))
        
        # Map to emission factor range
        max_reduction = 0.95  # Maximum 95% reduction from baseline
        reduction_ratio = progress * max_reduction
        
        predicted_factor = baseline_factor * (1 - reduction_ratio) + target_factor * progress
        
        return max(target_factor, predicted_factor)
    
    def generate_prediction(self, country_data, params):
        """Generate future emission factor predictions using S-curve model"""
        historical_factors = country_data["factors"]
        last_factor = [f for f in historical_factors if not pd.isna(f)][-1]
        
        # Generate predictions up to target year for better visualization
        end_year = max(2040, params["target_year"])
        future_years = list(range(2022, end_year + 1))
        future_factors = []
        
        baseline_year = 2021
        target_year = params["target_year"]
        
        # Convert reduction_rate to steepness parameter
        # Higher reduction rate = steeper curve
        steepness = params["reduction_rate"] / 8.0  # Normalize to reasonable range
        
        for year in future_years:
            if year <= target_year:
                # Use S-curve model
                predicted_factor = self.sigmoid_emission_factor(
                    year=year,
                    baseline_year=baseline_year,
                    baseline_factor=last_factor,
                    target_year=target_year,
                    target_factor=0.01,
                    steepness=steepness
                )
            else:
                # Post-target: very slow improvement
                years_after_target = year - target_year
                target_factor = self.sigmoid_emission_factor(
                    year=target_year,
                    baseline_year=baseline_year,
                    baseline_factor=last_factor,
                    target_year=target_year,
                    steepness=steepness
                )
                predicted_factor = max(0.005, target_factor * np.power(0.98, years_after_target))
            
            future_factors.append(predicted_factor)
        
        return future_years, future_factors

    def create_emission_chart(self, country_name, params):
        """Create emission factor trend chart"""
        country_data = self.eu_countries[country_name]
        future_years, future_factors = self.generate_prediction(country_data, params)
        
        # Create chart with increased spacing
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Grid Emission Factor Trend", "Annual Emissions Projection"),
            vertical_spacing=0.2,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Historical emission factors
        fig.add_trace(
            go.Scatter(
                x=self.years,
                y=country_data["factors"],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='#3498db', width=3),
                marker=dict(size=6),
                hovertemplate='Year: %{x}<br>Factor: %{y:.4f} tonnes CO2/MWh<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Predicted emission factors
        fig.add_trace(
            go.Scatter(
                x=future_years,
                y=future_factors,
                mode='lines+markers',
                name='Predicted Trend',
                line=dict(color='#e74c3c', width=3, dash='dash'),
                marker=dict(size=6),
                hovertemplate='Year: %{x}<br>Factor: %{y:.4f} tonnes CO2/MWh<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Calculate annual emissions
        electricity_usage = params["electricity_usage"]
        historical_emissions = [f * electricity_usage for f in country_data["factors"]]
        future_emissions = [f * electricity_usage for f in future_factors]
        
        # Historical emissions
        fig.add_trace(
            go.Bar(
                x=self.years[-10:],  # Only show last 10 years
                y=historical_emissions[-10:],
                name='Historical Emissions',
                marker_color='#3498db',
                opacity=0.7,
                hovertemplate='Year: %{x}<br>Emissions: %{y:.0f} tonnes CO2<extra></extra>'
            ),
            row=2, col=1
        )

        # Predicted emissions - show up to target year + 5 years
        years_to_show = min(len(future_years), params["target_year"] - 2022 + 6)
        fig.add_trace(
            go.Bar(
                x=future_years[:years_to_show],
                y=future_emissions[:years_to_show],
                name='Predicted Emissions',
                marker_color='#e74c3c',
                opacity=0.7,
                hovertemplate='Year: %{x}<br>Emissions: %{y:.0f} tonnes CO2<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text=f"{country_name} Grid Decarbonization Impact (Target: {params['target_year']})",
            title_x=0.5,
            title_font_size=20,
            template="plotly_white",
            uirevision=f"{params['target_year']}_{params['reduction_rate']}_{params['carbon_price']}"  # Update when key parameters change
        )
            
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(
            title_text="Emission Factor (tonnes CO2/MWh)", 
            range=[0, None],  # Fix y-axis to start from 0
            row=1, col=1
        )
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Annual Emissions (tonnes CO2)", row=2, col=1)
        
        return fig, future_emissions

    def display_metrics(self, country_name, params, future_emissions):
        """Display key metrics"""
        st.markdown('<h3 class="sub-header">Impact Analysis</h3>', unsafe_allow_html=True)
        
        country_data = self.eu_countries[country_name]
        electricity_usage = params["electricity_usage"]
        current_factor = country_data["latest"]
        future_years, future_factors = self.generate_prediction(country_data, params)
        
        # Calculate key metrics
        current_emissions = current_factor * electricity_usage
        emissions_2030 = future_factors[8] * electricity_usage
        
        # Cumulative reduction (2025-2035)
        cumulative_reduction = 0
        for i in range(3, 14):  # 2025-2035
            if i < len(future_factors):
                yearly_reduction = max(0, current_emissions - (future_factors[i] * electricity_usage))
                cumulative_reduction += yearly_reduction
        
        # Cost savings
        cost_savings = cumulative_reduction * params["carbon_price"]
        
        # Display metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{current_emissions:,.0f}</div>
                <div class="metric-label">Current Emissions</div>
                <div class="metric-change">tonnes CO2/year</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{emissions_2030:,.0f}</div>
                <div class="metric-label">2030 Projected Emissions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{cumulative_reduction:,.0f}</div>
                <div class="metric-label">Cumulative Reduction</div>
                <div class="metric-change">2025-2035 period</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">€{cost_savings:,.0f}</div>
                <div class="metric-label">Potential Cost Savings</div>
                <div class="metric-change">Carbon tax avoidance</div>
            </div>
            """, unsafe_allow_html=True)
    
    def display_country_comparison(self):
        """Display country comparison analysis"""
        st.markdown('<h3 class="sub-header">EU Country Comparison</h3>', unsafe_allow_html=True)
        
        # Prepare comparison data
        comparison_data = []
        for country_name, data in self.eu_countries.items():
            comparison_data.append({
                "Country": country_name,
                "Code": data["code"],
                "Current Factor": data["latest"],
                "1990 Factor": data["earliest"],
                "Total Change (%)": data["total_change"],
                "Decarbonization Level": "High" if data["latest"] < 0.2 else ("Medium" if data["latest"] < 0.5 else "Low")
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Handle NaN values in Total Change (%) by replacing with 0
        df["Total Change (%)"] = df["Total Change (%)"].fillna(0)
        
        # Sort by current factor
        df = df.sort_values("Current Factor")
        
        # Create size array, handling any remaining issues
        size_values = [max(5, abs(x) if not pd.isna(x) else 5) for x in df["Total Change (%)"]]
        
        # Create comparison chart
        fig = px.scatter(
            df,
            x="Current Factor",
            y="Total Change (%)",
            size=size_values,
            color="Decarbonization Level",
            hover_name="Country",
            color_discrete_map={
                "High": "#27ae60",
                "Medium": "#f39c12", 
                "Low": "#e74c3c"
            },
            title="EU Countries: Current Emission Factor vs Historical Change",
            labels={
                "Current Factor": "2021 Emission Factor (tonnes CO2/MWh)",
                "Total Change (%)": "Change Since 1990 (%)"
            }
        )
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run main application"""
        self.display_header()
        
        # Get user parameters
        params = self.display_sidebar_controls()
        
        # Main content area
        country_data = self.eu_countries[params["country"]]
        
        # Create and display chart
        chart, future_emissions = self.create_emission_chart(params["country"], params)
        st.plotly_chart(chart, use_container_width=True)
        
        # Display key metrics
        self.display_metrics(params["country"], params, future_emissions)
        
        # Display country comparison
        st.markdown("---")
        self.display_country_comparison()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; padding: 1rem;">
            Data Source: CoM European Database | Based on IPCC methodology | 
            For demonstration purposes
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    simulator = EUGridEmissionSimulator()
    simulator.run()