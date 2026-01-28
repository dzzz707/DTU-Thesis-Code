# monte_carlo_simulation_demo.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Monte Carlo Uncertainty Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class ParameterUncertainty:
    """Class to store parameter uncertainty information"""
    name: str
    base_value: float
    distribution_type: str  # 'normal', 'lognormal', 'uniform'
    uncertainty_params: Dict  # Parameters for the distribution
    unit: str
    category: str  # 'emission_factor', 'activity_data', 'gwp'

class MonteCarloSimulator:
    """Monte Carlo uncertainty analysis simulator"""
    
    def __init__(self):
        self.parameters = {}
        self.results = None
        self.simulation_count = 1000
        
    def add_parameter(self, param_id: str, parameter: ParameterUncertainty):
        """Add parameter with uncertainty"""
        self.parameters[param_id] = parameter
    
    def update_parameter(self, param_id: str, base_value: float, uncertainty_percent: float, distribution_type: str = None):
        """Update parameter value, uncertainty, and distribution type"""
        if param_id in self.parameters:
            param = self.parameters[param_id]
            param.base_value = base_value
            
            # Update distribution type if provided
            if distribution_type:
                param.distribution_type = distribution_type
            
            # Update uncertainty parameters based on distribution type
            if param.distribution_type == 'normal':
                param.uncertainty_params = {
                    'mean': base_value,
                    'std': base_value * uncertainty_percent / 100
                }
            elif param.distribution_type == 'lognormal':
                param.uncertainty_params = {
                    'mean_log': np.log(base_value),
                    'std_log': uncertainty_percent / 100
                }
            elif param.distribution_type == 'uniform':
                # For uniform distribution, use uncertainty as percentage range around base value
                range_value = base_value * uncertainty_percent / 100
                param.uncertainty_params = {
                    'low': max(0, base_value - range_value),  # Ensure non-negative
                    'high': base_value + range_value
                }
    
    def generate_samples(self, n_simulations: int) -> Dict[str, np.ndarray]:
        """Generate random samples for all parameters"""
        samples = {}
        
        for param_id, param in self.parameters.items():
            if param.distribution_type == 'normal':
                mean = param.uncertainty_params['mean']
                std = param.uncertainty_params['std']
                samples[param_id] = np.random.normal(mean, std, n_simulations)
                
            elif param.distribution_type == 'lognormal':
                mean_log = param.uncertainty_params['mean_log']
                std_log = param.uncertainty_params['std_log']
                samples[param_id] = np.random.lognormal(mean_log, std_log, n_simulations)
                
            elif param.distribution_type == 'uniform':
                low = param.uncertainty_params['low']
                high = param.uncertainty_params['high']
                samples[param_id] = np.random.uniform(low, high, n_simulations)
            
            # Ensure non-negative values for emission factors
            if param.category in ['emission_factor', 'gwp']:
                samples[param_id] = np.abs(samples[param_id])
        
        return samples
    
    def calculate_emissions(self, samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate emissions for each simulation"""
        n_simulations = len(next(iter(samples.values())))
        
        # Extract samples
        natural_gas = samples['natural_gas_consumption']
        ng_co2_factor = samples['ng_co2_factor']
        ng_ch4_factor = samples['ng_ch4_factor']
        ng_n2o_factor = samples['ng_n2o_factor']
        
        electricity = samples['electricity_consumption']
        elec_factor = samples['electricity_factor']
        
        diesel = samples['diesel_consumption']
        diesel_co2_factor = samples['diesel_co2_factor']
        diesel_ch4_factor = samples['diesel_ch4_factor']
        diesel_n2o_factor = samples['diesel_n2o_factor']
        
        # GWP values
        ch4_gwp = samples['ch4_gwp']
        n2o_gwp = samples['n2o_gwp']
        
        # Calculate emissions for each simulation
        results = {}
        
        # Natural Gas Emissions (Scope 1)
        ng_co2 = natural_gas * ng_co2_factor
        ng_ch4_co2e = natural_gas * ng_ch4_factor * ch4_gwp
        ng_n2o_co2e = natural_gas * ng_n2o_factor * n2o_gwp
        results['scope1_natural_gas'] = ng_co2 + ng_ch4_co2e + ng_n2o_co2e
        
        # Diesel Emissions (Scope 1)
        diesel_co2 = diesel * diesel_co2_factor
        diesel_ch4_co2e = diesel * diesel_ch4_factor * ch4_gwp
        diesel_n2o_co2e = diesel * diesel_n2o_factor * n2o_gwp
        results['scope1_diesel'] = diesel_co2 + diesel_ch4_co2e + diesel_n2o_co2e
        
        # Total Scope 1
        results['scope1_total'] = results['scope1_natural_gas'] + results['scope1_diesel']
        
        # Electricity Emissions (Scope 2)
        results['scope2_electricity'] = electricity * elec_factor
        
        # Total Emissions
        results['total_emissions'] = results['scope1_total'] + results['scope2_electricity']
        
        return results
    
    def run_simulation(self, n_simulations: int) -> Dict:
        """Run Monte Carlo simulation"""
        # Generate samples
        samples = self.generate_samples(n_simulations)
        
        # Calculate emissions
        emission_results = self.calculate_emissions(samples)
        
        # Calculate statistics
        statistics = {}
        for key, values in emission_results.items():
            statistics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p5': np.percentile(values, 5),
                'p25': np.percentile(values, 25),
                'p50': np.percentile(values, 50),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95),
                'cv': np.std(values) / np.mean(values) * 100  # Coefficient of variation
            }
        
        return {
            'samples': samples,
            'emissions': emission_results,
            'statistics': statistics,
            'n_simulations': n_simulations
        }

def add_global_styles():
    """Add consistent styling matching the original application"""
    st.markdown("""
    <style>
    /* Global blue theme styling */
    .main .block-container {
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        padding-bottom: 2rem !important;
        max-width: none !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 0.6rem 0.8rem !important;
        font-size: 0.9rem !important;
        line-height: 1.3 !important;
        height: auto !important;
        min-height: 2.8rem;
        white-space: normal !important;
        word-wrap: break-word !important;
        border-radius: 10px !important;
        background-color: transparent !important;
        border: 1px solid #3b4252 !important;
        color: #8e8ea0 !important;
        transition: all 0.3s ease !important;
        margin: 0.25rem 0 !important;
    }
    
    .stButton > button:hover {
        background-color: #2d3748 !important;
        border-color: #4a90e2 !important;
        color: #e2e8f0 !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.25) !important;
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #1f4e79 !important;
        border-color: #4a90e2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #2d5aa0 !important;
        border-color: #5ba3f5 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
    }
    
    /* Form styling */
    .stForm {
        border: 1px solid #4a5568 !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        background-color: rgba(45, 55, 72, 0.3) !important;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        color: #e2e8f0 !important;
        margin: 0.5rem 0 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(26, 32, 44, 0.8) !important;
        border-color: #4a5568 !important;
        padding: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        color: #e2e8f0 !important;
        margin: 0.5rem 0 !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        color: #e2e8f0 !important;
        margin: 0.25rem 0 !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 1px #4a90e2 !important;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: rgba(72, 187, 120, 0.1) !important;
        border-color: #48bb78 !important;
    }
    
    .stError {
        background-color: rgba(245, 101, 101, 0.1) !important;
        border-color: #f56565 !important;
    }
    
    .stWarning {
        background-color: rgba(236, 201, 75, 0.1) !important;
        border-color: #ecc94b !important;
    }
    
    .stInfo {
        background-color: rgba(74, 144, 226, 0.1) !important;
        border-color: #4a90e2 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_default_parameters() -> MonteCarloSimulator:
    """Initialize simulator with default parameter uncertainties"""
    simulator = MonteCarloSimulator()
    
    # Activity Data Parameters - all default to normal distribution
    simulator.add_parameter('natural_gas_consumption', ParameterUncertainty(
        name='Natural Gas Consumption',
        base_value=151091.0,
        distribution_type='normal',
        uncertainty_params={'mean': 151091.0, 'std': 7554.55},  # ±5%
        unit='m³',
        category='activity_data'
    ))
    
    simulator.add_parameter('electricity_consumption', ParameterUncertainty(
        name='Electricity Consumption',
        base_value=25461600.0,
        distribution_type='normal',
        uncertainty_params={'mean': 25461600.0, 'std': 1273080.0},  # ±5%
        unit='kWh',
        category='activity_data'
    ))
    
    simulator.add_parameter('diesel_consumption', ParameterUncertainty(
        name='Diesel Consumption',
        base_value=8831.991,
        distribution_type='normal',
        uncertainty_params={'mean': 8831.991, 'std': 441.59955},  # ±5%
        unit='L',
        category='activity_data'
    ))
    
    # Emission Factor Parameters - all default to normal distribution
    simulator.add_parameter('ng_co2_factor', ParameterUncertainty(
        name='Natural Gas CO2 Emission Factor',
        base_value=1.87904,
        distribution_type='normal',
        uncertainty_params={'mean': 1.87904, 'std': 0.08952},  # ±5%
        unit='kg CO2/m³',
        category='emission_factor'
    ))
    
    simulator.add_parameter('ng_ch4_factor', ParameterUncertainty(
        name='Natural Gas CH4 Emission Factor',
        base_value=0.00093,
        distribution_type='normal',
        uncertainty_params={'mean': 0.00093, 'std': 0.0000465},  # ±5%
        unit='kg CH4/m³',
        category='emission_factor'
    ))
    
    simulator.add_parameter('ng_n2o_factor', ParameterUncertainty(
        name='Natural Gas N2O Emission Factor',
        base_value=0.00091,
        distribution_type='normal',
        uncertainty_params={'mean': 0.00091, 'std': 0.0000455},  # ±5%
        unit='kg N2O/m³',
        category='emission_factor'
    ))
    
    simulator.add_parameter('electricity_factor', ParameterUncertainty(
        name='Electricity Grid Emission Factor',
        base_value=0.495,
        distribution_type='normal',
        uncertainty_params={'mean': 0.495, 'std': 0.02475},  # ±5%
        unit='kg CO2/kWh',
        category='emission_factor'
    ))
    
    simulator.add_parameter('diesel_co2_factor', ParameterUncertainty(
        name='Diesel CO2 Emission Factor',
        base_value=2.60603,
        distribution_type='normal',
        uncertainty_params={'mean': 2.60603, 'std': 0.1303015},  # ±5%
        unit='kg CO2/L',
        category='emission_factor'
    ))
    
    simulator.add_parameter('diesel_ch4_factor', ParameterUncertainty(
        name='Diesel CH4 Emission Factor',
        base_value=0.00383,
        distribution_type='normal',
        uncertainty_params={'mean': 0.00383, 'std': 0.0001915},  # ±5%
        unit='kg CH4/L',
        category='emission_factor'
    ))
    
    simulator.add_parameter('diesel_n2o_factor', ParameterUncertainty(
        name='Diesel N2O Emission Factor',
        base_value=0.03744,
        distribution_type='normal',
        uncertainty_params={'mean': 0.03744, 'std': 0.001872},  # ±5%
        unit='kg N2O/L',
        category='emission_factor'
    ))
    
    # GWP Parameters - all default to normal distribution
    simulator.add_parameter('ch4_gwp', ParameterUncertainty(
        name='CH4 Global Warming Potential',
        base_value=28.0,
        distribution_type='normal',
        uncertainty_params={'mean': 28.0, 'std': 1.4},  # ±5%
        unit='kg CO2-eq/kg CH4',
        category='gwp'
    ))
    
    simulator.add_parameter('n2o_gwp', ParameterUncertainty(
        name='N2O Global Warming Potential',
        base_value=265.0,
        distribution_type='normal',
        uncertainty_params={'mean': 265.0, 'std': 13.25},  # ±5%
        unit='kg CO2-eq/kg N2O',
        category='gwp'
    ))
    
    return simulator

def display_header():
    """Display application header"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem; color: #4a90e2;">
            Monte Carlo Uncertainty Analysis
        </h1>
    </div>
    """, unsafe_allow_html=True)

def get_current_uncertainty_percentage(param: ParameterUncertainty) -> float:
    """Calculate current uncertainty percentage based on distribution type"""
    if param.distribution_type == 'normal':
        return (param.uncertainty_params['std'] / param.uncertainty_params['mean'] * 100)
    elif param.distribution_type == 'lognormal':
        return param.uncertainty_params['std_log'] * 100
    elif param.distribution_type == 'uniform':
        range_value = (param.uncertainty_params['high'] - param.uncertainty_params['low']) / 2
        return (range_value / param.base_value * 100)
    else:
        return 10.0

def display_parameter_configuration(simulator: MonteCarloSimulator):
    
    # Group parameters by category
    categories = {
        'activity_data': 'Activity Data',
        'emission_factor': 'Emission Factors',
        'gwp': 'Global Warming Potentials'
    }
    
    config_changed = False
    
    for category_id, category_name in categories.items():
        with st.expander(f"{category_name}", expanded=True):
            category_params = {k: v for k, v in simulator.parameters.items() 
                             if v.category == category_id}
            
            for param_id, param in category_params.items():
                st.markdown(f"**{param.name}**")
                st.caption(f"Unit: {param.unit}")
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    # Distribution type selection
                    distribution_options = {
                        'normal': 'Normal',
                        'lognormal': 'Lognormal', 
                        'uniform': 'Uniform'
                    }
                    
                    current_dist_index = list(distribution_options.keys()).index(param.distribution_type)
                    new_distribution = st.selectbox(
                        "Distribution Type",
                        options=list(distribution_options.keys()),
                        format_func=lambda x: distribution_options[x],
                        index=current_dist_index,
                        key=f"dist_{param_id}"
                    )
                
                with col2:
                    # Base value input
                    new_value = st.number_input(
                        "Base Value",
                        value=float(param.base_value),
                        min_value=0.0,
                        key=f"base_{param_id}",
                        format="%.6f"
                    )
                    
                with col3:
                    # Uncertainty percentage input
                    current_uncertainty = get_current_uncertainty_percentage(param)
                    
                    new_uncertainty = st.number_input(
                        "Uncertainty (%)",
                        value=float(current_uncertainty),
                        min_value=0.1,
                        max_value=100.0,
                        key=f"uncert_{param_id}",
                        format="%.1f"
                    )
                
                with col4:
                    # Display current distribution parameters based on type
                    if param.distribution_type == 'normal':
                        std_value = param.uncertainty_params.get('std', 0)
                        st.metric("Std Dev", f"{std_value:.4f}")
                    elif param.distribution_type == 'lognormal':
                        std_log = param.uncertainty_params.get('std_log', 0)
                        st.metric("Log Std", f"{std_log:.4f}")
                    elif param.distribution_type == 'uniform':
                        low = param.uncertainty_params.get('low', 0)
                        high = param.uncertainty_params.get('high', 0)
                        st.metric("Range", f"{low:.2f} - {high:.2f}")
                
                # Update parameter if changed
                if (new_value != param.base_value or 
                    abs(new_uncertainty - current_uncertainty) > 0.1 or
                    new_distribution != param.distribution_type):
                    simulator.update_parameter(param_id, new_value, new_uncertainty, new_distribution)
                    config_changed = True
                
                st.markdown("---")
    
    if config_changed:
        st.success("Configuration updated! Run simulation to see results with new parameters.")
    
    # Reset to defaults button
    if st.button("Reset to Default Values", use_container_width=True):
        st.session_state.simulator = initialize_default_parameters()
        st.rerun()

def run_simulation_interface():
    """Simulation interface"""
    st.subheader("Run Monte Carlo Simulation")
    
    if 'simulator' not in st.session_state:
        st.session_state.simulator = initialize_default_parameters()
    
    simulator = st.session_state.simulator
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_simulations = st.selectbox(
            "Number of Simulations",
            [1000, 5000, 10000, 50000],
            index=0
        )
    
    with col2:
        if st.button("Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running Monte Carlo simulation..."):
                results = simulator.run_simulation(n_simulations)
                st.session_state.simulation_results = results
                st.success(f"Simulation completed with {n_simulations:,} iterations!")
                st.info("View detailed results in the 'Results' tab.")
    
    # Show simulation status
    if 'simulation_results' in st.session_state:
        st.markdown("---")
        results = st.session_state.simulation_results
        total_stats = results['statistics']['total_emissions']
        
        st.markdown("#### Last Simulation Summary")
        
        # First row - two columns  
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Simulations", f"{results['n_simulations']:,}")
        with col2:
            st.metric("Mean Emissions", f"{total_stats['mean']:,.0f} kg CO2e")
        
        # Second row - two columns
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("Uncertainty (CV)", f"{total_stats['cv']:.1f}%")
        with col4:
            confidence_range = total_stats['p95'] - total_stats['p5']
            st.metric("90% CI Range", f"{confidence_range:,.0f}")

def display_simulation_results(results: Dict):
    """Display comprehensive simulation results and analysis"""
    statistics = results['statistics']
    emissions = results['emissions']
    
    # Summary metrics
    st.markdown("### Emission Summary Statistics")
    
    total_stats = statistics['total_emissions']
    confidence_range = total_stats['p95'] - total_stats['p5']
    
    # First row - two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Total Emissions", f"{total_stats['mean']:,.0f} kg CO2e")
    
    with col2:
        st.metric("Standard Deviation", f"{total_stats['std']:,.0f} kg CO2e")
    
    # Second row - two columns
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric("Coefficient of Variation", f"{total_stats['cv']:.1f}%")
    
    with col4:
        st.metric("90% Confidence Range", f"{confidence_range:,.0f} kg CO2e")
    
    # Detailed statistics table
    st.markdown("### Detailed Statistics by Scope")
    
    stats_data = []
    for scope, stats in statistics.items():
        stats_data.append({
            'Emission Source': scope.replace('_', ' ').title(),
            'Mean (kg CO2e)': f"{stats['mean']:,.0f}",
            'Std Dev (kg CO2e)': f"{stats['std']:,.0f}",
            'CV (%)': f"{stats['cv']:.1f}",
            '5th Percentile': f"{stats['p5']:,.0f}",
            '95th Percentile': f"{stats['p95']:,.0f}",
            'Min': f"{stats['min']:,.0f}",
            'Max': f"{stats['max']:,.0f}"
        })
    
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    # Show current parameter distribution summary
    display_parameter_distribution_summary()
    
    # Visualization
    display_uncertainty_visualizations(emissions, statistics)
    
    # Sensitivity analysis
    display_sensitivity_analysis(results)
    
    # Export options
    display_export_options(results)

def display_parameter_distribution_summary():
    """Display summary of current parameter distributions"""
    st.markdown("### Current Parameter Distribution Configuration")
    
    if 'simulator' in st.session_state:
        simulator = st.session_state.simulator
        
        # Count distributions by type
        dist_counts = {'normal': 0, 'lognormal': 0, 'uniform': 0}
        
        param_summary = []
        for param_id, param in simulator.parameters.items():
            dist_counts[param.distribution_type] += 1
            
            # Get distribution info
            if param.distribution_type == 'normal':
                dist_info = f"μ={param.uncertainty_params['mean']:.4f}, σ={param.uncertainty_params['std']:.4f}"
            elif param.distribution_type == 'lognormal':
                dist_info = f"μ_log={param.uncertainty_params['mean_log']:.4f}, σ_log={param.uncertainty_params['std_log']:.4f}"
            elif param.distribution_type == 'uniform':
                dist_info = f"[{param.uncertainty_params['low']:.4f}, {param.uncertainty_params['high']:.4f}]"
            
            param_summary.append({
                'Parameter': param.name,
                'Distribution': param.distribution_type.title(),
                'Base Value': f"{param.base_value:.6f}",
                'Distribution Info': dist_info
            })
        
        # Show distribution counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parameters", len(simulator.parameters))
        with col2:
            st.metric("Normal", dist_counts['normal'])
        with col3:
            st.metric("Lognormal", dist_counts['lognormal'])
        with col4:
            st.metric("Uniform", dist_counts['uniform'])
        
        # Show detailed parameter table
        df_params = pd.DataFrame(param_summary)
        st.dataframe(df_params, use_container_width=True, hide_index=True)

def display_uncertainty_visualizations(emissions: Dict, statistics: Dict):
    """Create uncertainty visualization charts"""
    st.markdown("### Uncertainty Visualization")
    
    # Distribution plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Emissions Distribution', 'Scope 1 vs Scope 2'),
        specs=[[{"colspan": 1}, {"colspan": 1}]]
    )
    
    # Total emissions histogram
    fig.add_trace(
        go.Histogram(x=emissions['total_emissions'], nbinsx=50, name='Total Emissions',
                    marker_color='rgba(74, 144, 226, 0.7)'),
        row=1, col=1
    )
    
    # Scope comparison scatter plot
    fig.add_trace(
        go.Scatter(x=emissions['scope1_total'], y=emissions['scope2_electricity'],
                  mode='markers', name='Scope 1 vs 2',
                  marker=dict(color='rgba(74, 144, 226, 0.5)', size=3)),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, 
                     title_text="Emission Uncertainty Analysis")
    fig.update_xaxes(title_text="Scope 1 (kg CO2e)", row=1, col=2)
    fig.update_yaxes(title_text="Scope 2 (kg CO2e)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def display_sensitivity_analysis(results: Dict):
    """Display sensitivity analysis"""
    st.markdown("### Sensitivity Analysis")
    
    # Calculate correlation with total emissions
    emissions = results['emissions']
    samples = results['samples']
    
    correlations = []
    for param_id, param_values in samples.items():
        correlation = np.corrcoef(param_values, emissions['total_emissions'])[0, 1]
        correlations.append({
            'Parameter': param_id.replace('_', ' ').title(),
            'Correlation': correlation,
            'Abs_Correlation': abs(correlation)
        })
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x['Abs_Correlation'], reverse=True)
    
    # Create sensitivity chart
    param_names = [c['Parameter'] for c in correlations[:10]]  # Top 10
    correlation_values = [c['Correlation'] for c in correlations[:10]]
    
    fig = go.Figure(data=go.Bar(
        x=correlation_values,
        y=param_names,
        orientation='h',
        marker_color=['rgba(74, 144, 226, 0.7)' if x >= 0 else 'rgba(245, 101, 101, 0.7)' 
                     for x in correlation_values]
    ))
    
    fig.update_layout(
        title="Parameter Sensitivity (Correlation with Total Emissions)",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Parameters",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sensitivity table
    st.markdown("### Parameter Sensitivity Ranking")
    df_sensitivity = pd.DataFrame(correlations[:10])
    df_sensitivity['Correlation'] = df_sensitivity['Correlation'].round(3)
    df_sensitivity['Abs_Correlation'] = df_sensitivity['Abs_Correlation'].round(3)
    df_sensitivity = df_sensitivity.drop('Abs_Correlation', axis=1)
    
    st.dataframe(df_sensitivity, use_container_width=True, hide_index=True)

def display_export_options(results: Dict):
    """Display options to export results"""
    st.markdown("---")
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Statistics to CSV", use_container_width=True):
            statistics = results['statistics']
            
            # Create export data
            export_data = []
            for scope, stats in statistics.items():
                export_data.append({
                    'Emission_Source': scope,
                    'Mean_kg_CO2e': stats['mean'],
                    'Std_Dev_kg_CO2e': stats['std'],
                    'CV_Percent': stats['cv'],
                    'P5_kg_CO2e': stats['p5'],
                    'P25_kg_CO2e': stats['p25'],
                    'P50_kg_CO2e': stats['p50'],
                    'P75_kg_CO2e': stats['p75'],
                    'P95_kg_CO2e': stats['p95'],
                    'Min_kg_CO2e': stats['min'],
                    'Max_kg_CO2e': stats['max']
                })
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"uncertainty_analysis_{timestamp}.csv"
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("Export Raw Data to JSON", use_container_width=True):
            # Prepare data for JSON export (convert numpy arrays to lists)
            export_results = {
                'statistics': results['statistics'],
                'n_simulations': results['n_simulations'],
                'timestamp': datetime.now().isoformat()
            }
            
            json_str = json.dumps(export_results, indent=2)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"uncertainty_analysis_raw_{timestamp}.json"
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=filename,
                mime="application/json",
                use_container_width=True
            )

def main():
    """Main application function"""
    # Add global styling
    add_global_styles()
    
    # Display header
    display_header()
    
    # Main content
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Configuration", "Simulation", "Results"])
    
    with tab1:
        st.markdown("### Parameter Configuration")
        st.info("Modify parameter values, uncertainty levels, and distribution types to customize your analysis.")
        
        # Initialize simulator if not exists
        if 'simulator' not in st.session_state:
            st.session_state.simulator = initialize_default_parameters()
        
        # Display parameter configuration
        display_parameter_configuration(st.session_state.simulator)
    
    with tab2:
        st.markdown("### Monte Carlo Simulation")
        run_simulation_interface()
    
    with tab3:
        st.markdown("### Analysis Results")
        if 'simulation_results' in st.session_state:
            display_simulation_results(st.session_state.simulation_results)
        else:
            st.info("No simulation results available. Please run a simulation first.")
            
            # Show preview of what will be available
            st.markdown("#### Available Analysis:")
            st.markdown("""
            - **Summary Statistics**: Mean, standard deviation, confidence intervals
            - **Distribution Configuration**: Current parameter distribution types and parameters
            - **Distribution Plots**: Histograms and scatter plots of emission distributions  
            - **Sensitivity Analysis**: Parameter correlation analysis
            - **Export Options**: Download results in CSV or JSON format
            """)

def display_sidebar():
    """Display sidebar with additional information"""
    with st.sidebar:
        st.header("Monte Carlo Analysis")
        st.markdown("---")
        
        st.subheader("Current Configuration")
        
        if 'simulator' in st.session_state:
            simulator = st.session_state.simulator
            total_params = len(simulator.parameters)
            st.metric("Total Parameters", total_params)
            
            # Count by category
            categories = ['activity_data', 'emission_factor', 'gwp']
            for category in categories:
                count = sum(1 for p in simulator.parameters.values() if p.category == category)
                category_name = category.replace('_', ' ').title()
                st.text(f"{category_name}: {count}")
            
            st.markdown("---")
            st.subheader("Distribution Types")
            
            # Count by distribution type
            dist_counts = {'normal': 0, 'lognormal': 0, 'uniform': 0}
            for param in simulator.parameters.values():
                dist_counts[param.distribution_type] += 1
            
            for dist_type, count in dist_counts.items():
                if count > 0:
                    st.text(f"{dist_type.title()}: {count}")

        st.markdown("---")
        st.subheader("Distribution Guide")
        st.markdown("""
        **Normal**: Best for symmetric uncertainties around the mean value.
        
        **Lognormal**: Good for parameters that must be positive and have skewed uncertainty.
        
        **Uniform**: Use when all values in a range are equally likely.
        """)

# Run the application
if __name__ == "__main__":
    # Initialize sidebar
    display_sidebar()
    
    # Run main application
    main()