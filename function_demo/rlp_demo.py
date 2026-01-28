import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from copy import deepcopy
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RLP-SAA Carbon Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OptimizationMethod(Enum):
    """Optimization methods available"""
    SLSQP = "SLSQP (Sequential Least Squares Programming)"
    COBYLA = "COBYLA (Constrained Optimization by Linear Approximations)"
    TRUST_CONSTR = "TRUST_CONSTR (Trust Region Constrained Optimization)"


class DistributionType(Enum):
    """Distribution types for uncertainty"""
    NORMAL = "Normal"
    LOGNORMAL = "Lognormal"
    UNIFORM = "Uniform"
    TRIANGULAR = "Triangular"


@dataclass
class EmissionParameter:
    """Unified parameter with all properties"""
    name: str
    stub: str
    nominal_value: float
    unit: str
    scope: str
    
    is_optimizable: bool = False
    min_change_percent: float = -10.0
    max_change_percent: float = 10.0
    
    uncertainty_percent: float = 5.0
    distribution: DistributionType = DistributionType.NORMAL
    
    @property
    def optimization_min(self) -> float:
        return max(0, self.nominal_value * (1 + self.min_change_percent / 100))
    
    @property
    def optimization_max(self) -> float:
        return self.nominal_value * (1 + self.max_change_percent / 100)


@dataclass
class OptimizationResult:
    """Complete result including optimization and uncertainty analysis"""
    method: str
    success: bool
    message: str
    
    baseline_emissions: float
    optimized_emissions: float
    reduction_amount: float
    reduction_percent: float
    
    baseline_scope_breakdown: Dict[str, float]
    optimized_scope_breakdown: Dict[str, float]
    
    optimized_parameters: Dict[str, float]
    parameter_changes: List[Dict]
    
    mc_baseline: Dict
    mc_optimized: Dict
    
    sensitivity_ranking: List[Dict]
    
    saa_samples_used: int
    saa_objective_value: float
    
    iterations: int
    computation_time: float
    timestamp: str


@dataclass
class CarbonPricingScenario:
    """Carbon pricing scenario configuration with coverage rates"""
    name: str
    eu_ets_price: float
    eu_ets_coverage: float
    national_tax: float
    national_tax_coverage: float
    uncertainty_percent: float = 10.0
    
    @property
    def weighted_average_price(self) -> float:
        """Calculate weighted average carbon price based on coverage"""
        eu_component = self.eu_ets_price * (self.eu_ets_coverage / 100)
        national_component = self.national_tax * (self.national_tax_coverage / 100)
        return eu_component + national_component
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get breakdown of pricing components"""
        return {
            'EU ETS Component': self.eu_ets_price * (self.eu_ets_coverage / 100),
            'National Tax Component': self.national_tax * (self.national_tax_coverage / 100),
            'Weighted Average': self.weighted_average_price
        }

class IntegratedCarbonOptimizer:
    """Unified optimizer with Sample Average Approximation (SAA)"""
    
    def __init__(self):
        self.parameters: Dict[str, EmissionParameter] = {}
        self.emission_structure = {}
        self.grid_emission_factor = 0.495  
        self.initialize_default_parameters()
    
    def set_grid_emission_factor(self, factor: float):
        """Set custom grid emission factor"""
        self.grid_emission_factor = factor
        if 'grid_factor' in self.parameters:
            self.parameters['grid_factor'].nominal_value = factor
    
    def add_parameter(self, parameter: EmissionParameter):
        """Add parameter"""
        self.parameters[parameter.stub] = parameter
    
    def initialize_default_parameters(self):
        """Initialize default parameters"""
        
        # Scope 1 - Natural Gas
        self.add_parameter(EmissionParameter(
            name="Natural Gas Consumption", stub="ng_consumption",
            nominal_value=151091.0, unit="m3", scope="Scope1",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Natural Gas CO2 Factor", stub="ng_co2_factor",
            nominal_value=1.87904, unit="kg CO2/m3", scope="Scope1",
            is_optimizable=False
        ))
        self.add_parameter(EmissionParameter(
            name="Natural Gas CH4 Factor", stub="ng_ch4_factor",
            nominal_value=0.00093, unit="kg CH4/m3", scope="Scope1",
            is_optimizable=False
        ))
        self.add_parameter(EmissionParameter(
            name="Natural Gas N2O Factor", stub="ng_n2o_factor",
            nominal_value=0.00091, unit="kg N2O/m3", scope="Scope1",
            is_optimizable=False
        ))
        
        # Scope 1 - Diesel
        self.add_parameter(EmissionParameter(
            name="Diesel Consumption", stub="diesel_consumption",
            nominal_value=8831.991, unit="L", scope="Scope1",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Diesel CO2 Factor", stub="diesel_co2_factor",
            nominal_value=2.60603, unit="kg CO2/L", scope="Scope1",
            is_optimizable=False
        ))
        self.add_parameter(EmissionParameter(
            name="Diesel CH4 Factor", stub="diesel_ch4_factor",
            nominal_value=0.00383, unit="kg CH4/L", scope="Scope1",
            is_optimizable=False
        ))
        self.add_parameter(EmissionParameter(
            name="Diesel N2O Factor", stub="diesel_n2o_factor",
            nominal_value=0.03744, unit="kg N2O/L", scope="Scope1",
            is_optimizable=False
        ))
        
        # Scope 1 - GWP Factors
        self.add_parameter(EmissionParameter(
            name="CH4 GWP", stub="ch4_gwp",
            nominal_value=28.0, unit="CO2-eq", scope="Scope1",
            is_optimizable=False
        ))
        self.add_parameter(EmissionParameter(
            name="N2O GWP", stub="n2o_gwp",
            nominal_value=265.0, unit="CO2-eq", scope="Scope1",
            is_optimizable=False
        ))

        # Scope 2
        self.add_parameter(EmissionParameter(
            name="Electricity Consumption", stub="electricity_consumption",
            nominal_value=25461600.0, unit="kWh", scope="Scope2",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Grid Emission Factor", stub="grid_factor",
            nominal_value=self.grid_emission_factor, unit="kg CO2/kWh", scope="Scope2",
            is_optimizable=False
        ))
        
        # Scope 3
        self.add_parameter(EmissionParameter(
            name="Air Travel Distance", stub="air_travel_km",
            nominal_value=500000.0, unit="km", scope="Scope3",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Air Travel Factor", stub="air_travel_factor",
            nominal_value=0.255, unit="kg CO2/km", scope="Scope3",
            is_optimizable=False
        ))
        self.add_parameter(EmissionParameter(
            name="Waste Generated", stub="waste_kg",
            nominal_value=150000.0, unit="kg", scope="Scope3",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Waste Factor", stub="waste_factor",
            nominal_value=0.523, unit="kg CO2/kg", scope="Scope3",
            is_optimizable=False
        ))
        
        self._setup_emission_structure()
    
    def _setup_emission_structure(self):
        """Define emission calculations"""
        self.emission_structure = {
            'scope1_natural_gas': {
                'params': ['ng_consumption', 'ng_co2_factor', 'ng_ch4_factor', 
                          'ng_n2o_factor', 'ch4_gwp', 'n2o_gwp'],
                'formula': lambda p: (
                    p['ng_consumption'] * p['ng_co2_factor'] + 
                    p['ng_consumption'] * p['ng_ch4_factor'] * p['ch4_gwp'] +
                    p['ng_consumption'] * p['ng_n2o_factor'] * p['n2o_gwp']
                )
            },
            'scope1_diesel': {
                'params': ['diesel_consumption', 'diesel_co2_factor', 'diesel_ch4_factor',
                          'diesel_n2o_factor', 'ch4_gwp', 'n2o_gwp'],
                'formula': lambda p: (
                    p['diesel_consumption'] * p['diesel_co2_factor'] +
                    p['diesel_consumption'] * p['diesel_ch4_factor'] * p['ch4_gwp'] +
                    p['diesel_consumption'] * p['diesel_n2o_factor'] * p['n2o_gwp']
                )
            },
            'scope2_electricity': {
                'params': ['electricity_consumption', 'grid_factor'],
                'formula': lambda p: p['electricity_consumption'] * p['grid_factor']
            },
            'scope3_travel': {
                'params': ['air_travel_km', 'air_travel_factor'],
                'formula': lambda p: p['air_travel_km'] * p['air_travel_factor']
            },
            'scope3_waste': {
                'params': ['waste_kg', 'waste_factor'],
                'formula': lambda p: p['waste_kg'] * p['waste_factor']
            }
        }
    
    def calculate_emissions(self, parameter_values: Dict[str, float] = None) -> Dict:
        """Calculate total emissions"""
        if parameter_values is None:
            parameter_values = {stub: p.nominal_value for stub, p in self.parameters.items()}
        
        emissions = {}
        for source_name, source_info in self.emission_structure.items():
            params_dict = {}
            for stub in source_info['params']:
                if stub in parameter_values:
                    params_dict[stub] = parameter_values[stub]
                elif stub in self.parameters:
                    params_dict[stub] = self.parameters[stub].nominal_value
                else:
                    raise KeyError(f"Parameter '{stub}' not found")
            emissions[source_name] = source_info['formula'](params_dict)
        
        scope1 = sum(e for name, e in emissions.items() if 'scope1' in name)
        scope2 = sum(e for name, e in emissions.items() if 'scope2' in name)
        scope3 = sum(e for name, e in emissions.items() if 'scope3' in name)
        
        emissions['scope1_total'] = scope1
        emissions['scope2_total'] = scope2
        emissions['scope3_total'] = scope3
        emissions['total'] = scope1 + scope2 + scope3
        
        return emissions
    
    def calculate_carbon_cost(self, emissions_kg: float, pricing_scenario: CarbonPricingScenario) -> Dict[str, float]:
        """Calculate carbon cost using weighted average pricing based on coverage rates"""
        emissions_tons = emissions_kg / 1000
        
        eu_ets_covered_emissions = emissions_tons * (pricing_scenario.eu_ets_coverage / 100)
        national_covered_emissions = emissions_tons * (pricing_scenario.national_tax_coverage / 100)
        
        eu_ets_cost = eu_ets_covered_emissions * pricing_scenario.eu_ets_price
        national_cost = national_covered_emissions * pricing_scenario.national_tax
        total_cost = eu_ets_cost + national_cost
        
        effective_price = pricing_scenario.weighted_average_price
        
        return {
            'total_cost': total_cost,
            'eu_ets_cost': eu_ets_cost,
            'national_cost': national_cost,
            'emissions_tons': emissions_tons,
            'eu_ets_covered_tons': eu_ets_covered_emissions,
            'national_covered_tons': national_covered_emissions,
            'uncovered_tons': emissions_tons - eu_ets_covered_emissions - national_covered_emissions,
            'effective_price_per_ton': effective_price
        }
    
    def generate_samples(self, param: EmissionParameter, base_value: float, n_samples: int) -> np.ndarray:
        """Generate samples for a parameter based on its distribution"""
        if param.distribution == DistributionType.NORMAL:
            std = base_value * param.uncertainty_percent / 100
            samples = np.random.normal(base_value, std, n_samples)
        elif param.distribution == DistributionType.LOGNORMAL:
            mean_log = np.log(base_value)
            std_log = param.uncertainty_percent / 100
            samples = np.random.lognormal(mean_log, std_log, n_samples)
        elif param.distribution == DistributionType.UNIFORM:
            lower = base_value * (1 - param.uncertainty_percent / 100)
            upper = base_value * (1 + param.uncertainty_percent / 100)
            samples = np.random.uniform(lower, upper, n_samples)
        elif param.distribution == DistributionType.TRIANGULAR:
            lower = base_value * (1 - param.uncertainty_percent / 100)
            upper = base_value * (1 + param.uncertainty_percent / 100)
            samples = np.random.triangular(lower, base_value, upper, n_samples)
        else:
            std = base_value * param.uncertainty_percent / 100
            samples = np.random.normal(base_value, std, n_samples)
        
        return np.maximum(samples, 0)
    
    def monte_carlo_analysis(self, n_simulations: int, parameter_values: Dict[str, float]) -> Dict:
        """Run Monte Carlo simulation"""
        samples = {}
        
        for stub, param in self.parameters.items():
            base_value = parameter_values.get(stub, param.nominal_value)
            samples[stub] = self.generate_samples(param, base_value, n_simulations)
        
        emissions_array = np.zeros(n_simulations)
        for i in range(n_simulations):
            sim_params = {stub: samples[stub][i] for stub in samples}
            emissions_dict = self.calculate_emissions(sim_params)
            emissions_array[i] = emissions_dict['total']
        
        mean_emissions = np.mean(emissions_array)
        std_emissions = np.std(emissions_array)
        cv = (std_emissions / mean_emissions) * 100
        
        percentiles = np.percentile(emissions_array, [5, 25, 50, 75, 95])
        
        sensitivity = []
        for stub, param in self.parameters.items():
            if stub in samples:
                correlation = np.corrcoef(samples[stub], emissions_array)[0, 1]
                sensitivity.append({
                    'parameter': param.name,
                    'stub': stub,
                    'scope': param.scope,
                    'correlation': correlation,
                    'abs_correlation': abs(correlation)
                })
        
        sensitivity.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return {
            'data': emissions_array,
            'mean': mean_emissions,
            'std': std_emissions,
            'cv': cv,
            'p5': percentiles[0],
            'p25': percentiles[1],
            'p50': percentiles[2],
            'p75': percentiles[3],
            'p95': percentiles[4],
            'sensitivity': sensitivity
        }
    
    def optimize_with_uncertainty(self, method: str = 'SLSQP', 
                                  mc_simulations: int = 5000,
                                  saa_samples: int = 200,
                                  scope_filter: str = "All",
                                  use_saa: bool = True) -> OptimizationResult:
        """Integrated optimization with SAA (Sample Average Approximation)"""
        start_time = time.time()
        
        optimizable = [p for p in self.parameters.values() 
                      if p.is_optimizable and 
                      (scope_filter == "All" or p.scope == scope_filter)]
        
        if not optimizable:
            return self._create_failed_result(method, "No optimizable parameters found", start_time)
        
        baseline_params = {stub: p.nominal_value for stub, p in self.parameters.items()}
        baseline_emissions_dict = self.calculate_emissions(baseline_params)
        baseline_total = baseline_emissions_dict['total']
        
        mc_baseline = self.monte_carlo_analysis(mc_simulations, baseline_params)
        
        if use_saa:
            optim_samples = {}
            for param in optimizable:
                optim_samples[param.stub] = self.generate_samples(
                    param, param.nominal_value, saa_samples
                )
        
        use_scaling = 'COBYLA' in method.upper() or 'TRUST' in method.upper()

        if use_scaling:
            scale_factors = np.array([p.nominal_value for p in optimizable])

            if use_saa:
                def objective_saa(x_scaled):
                    x_real = x_scaled * scale_factors
                    total_em = 0.0
                    for i in range(saa_samples):
                        param_values = baseline_params.copy()
                        for j, param in enumerate(optimizable):
                            sample_val = optim_samples[param.stub][i]
                            if param.nominal_value != 0:
                                scaled_sample = sample_val * (x_real[j] / param.nominal_value)
                            else:
                                scaled_sample = sample_val
                            param_values[param.stub] = scaled_sample
                        total_em += self.calculate_emissions(param_values)['total']
                    return total_em / saa_samples
                
                objective = objective_saa
            else:
                def objective_det(x_scaled):
                    x_real = x_scaled * scale_factors
                    param_values = baseline_params.copy()
                    for i, param in enumerate(optimizable):
                        param_values[param.stub] = x_real[i]
                    emissions = self.calculate_emissions(param_values)
                    return emissions['total']
                
                objective = objective_det

            x0 = np.ones(len(optimizable))
            bounds = [
                (p.optimization_min / p.nominal_value, p.optimization_max / p.nominal_value)
                for p in optimizable
            ]
        else:
            if use_saa:
                def objective_saa(x):
                    total_em = 0.0
                    for i in range(saa_samples):
                        param_values = baseline_params.copy()
                        for j, param in enumerate(optimizable):
                            sample_val = optim_samples[param.stub][i]
                            if param.nominal_value != 0:
                                scaled_sample = sample_val * (x[j] / param.nominal_value)
                            else:
                                scaled_sample = sample_val
                            param_values[param.stub] = scaled_sample
                        total_em += self.calculate_emissions(param_values)['total']
                    return total_em / saa_samples
                
                objective = objective_saa
            else:
                def objective_det(x):
                    param_values = baseline_params.copy()
                    for i, param in enumerate(optimizable):
                        param_values[param.stub] = x[i]
                    emissions = self.calculate_emissions(param_values)
                    return emissions['total']
                
                objective = objective_det

            x0 = np.array([p.nominal_value for p in optimizable])
            bounds = [(p.optimization_min, p.optimization_max) for p in optimizable]

        if 'COBYLA' in method.upper():
            scipy_method = 'COBYLA'
            options = {'maxfun': 50000, 'rhobeg': 1.0, 'tol': 1e-6}
        elif 'TRUST' in method.upper():
            scipy_method = 'trust-constr'
            options = {'maxiter': 2000, 'xtol': 1e-6, 'gtol': 1e-6}
        else:
            scipy_method = 'SLSQP'
            options = {'maxiter': 2000, 'ftol': 1e-6}

        result = minimize(objective, x0, method=scipy_method, bounds=bounds, options=options)
        
        if not result.success:
            return self._create_failed_result(method, f"Optimization failed: {result.message}", 
                                             start_time, mc_baseline, getattr(result, 'nit', 0))

        optimized_params = baseline_params.copy()
        for i, param in enumerate(optimizable):
            optimized_params[param.stub] = (
                result.x[i] * param.nominal_value if use_scaling else result.x[i]
            )

        optimized_emissions_dict = self.calculate_emissions(optimized_params)
        optimized_total = optimized_emissions_dict['total']
        
        mc_optimized = self.monte_carlo_analysis(mc_simulations, optimized_params)
        
        reduction_amount = baseline_total - optimized_total
        reduction_percent = (reduction_amount / baseline_total) * 100
        
        parameter_changes = []
        for param in optimizable:
            original = param.nominal_value
            optimized = optimized_params[param.stub]
            change = optimized - original
            change_pct = (change / original) * 100 if original != 0 else 0
            
            parameter_changes.append({
                'parameter': param.name,
                'scope': param.scope,
                'unit': param.unit,
                'original': original,
                'optimized': optimized,
                'change': change,
                'change_percent': change_pct
            })
        
        return OptimizationResult(
            method=method,
            success=True,
            message=f"{'SAA' if use_saa else 'Deterministic'} optimization completed successfully",
            baseline_emissions=baseline_total,
            optimized_emissions=optimized_total,
            reduction_amount=reduction_amount,
            reduction_percent=reduction_percent,
            baseline_scope_breakdown={
                'scope1': baseline_emissions_dict['scope1_total'],
                'scope2': baseline_emissions_dict['scope2_total'],
                'scope3': baseline_emissions_dict['scope3_total']
            },
            optimized_scope_breakdown={
                'scope1': optimized_emissions_dict['scope1_total'],
                'scope2': optimized_emissions_dict['scope2_total'],
                'scope3': optimized_emissions_dict['scope3_total']
            },
            optimized_parameters={p.stub: optimized_params[p.stub] for p in optimizable},
            parameter_changes=parameter_changes,
            mc_baseline=mc_baseline,
            mc_optimized=mc_optimized,
            sensitivity_ranking=mc_optimized['sensitivity'],
            saa_samples_used=saa_samples if use_saa else 0,
            saa_objective_value=result.fun,
            iterations=result.nit if hasattr(result, 'nit') else 0,
            computation_time=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _create_failed_result(self, method, message, start_time, mc_baseline=None, iterations=0):
        """Helper to create failed optimization result"""
        baseline_params = {stub: p.nominal_value for stub, p in self.parameters.items()}
        baseline_emissions_dict = self.calculate_emissions(baseline_params)
        baseline_total = baseline_emissions_dict['total']
        
        if mc_baseline is None:
            mc_baseline = {'sensitivity': [], 'mean': baseline_total, 'std': 0, 'cv': 0,
                          'p5': baseline_total, 'p25': baseline_total, 'p50': baseline_total,
                          'p75': baseline_total, 'p95': baseline_total, 'data': np.array([baseline_total])}
        
        baseline_scope_safe = {
            'scope1': baseline_emissions_dict.get('scope1_total', 0),
            'scope2': baseline_emissions_dict.get('scope2_total', 0),
            'scope3': baseline_emissions_dict.get('scope3_total', 0)
        }

        return OptimizationResult(
            method=method,
            success=False,
            message=message,
            baseline_emissions=baseline_total,
            optimized_emissions=baseline_total,
            reduction_amount=0,
            reduction_percent=0,
            baseline_scope_breakdown=baseline_scope_safe,
            optimized_scope_breakdown=baseline_scope_safe,
            optimized_parameters={},
            parameter_changes=[],
            mc_baseline=mc_baseline,
            mc_optimized={},
            sensitivity_ranking=mc_baseline['sensitivity'],
            saa_samples_used=0,
            saa_objective_value=0,
            iterations=iterations,
            computation_time=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )

def setup_page():
    """Setup page styling"""
    st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    h1 {color: #1f77b4;}
    h2 {color: #2ca02c;}
    </style>
    """, unsafe_allow_html=True)


def display_header():
    st.title("RLP-SAA Carbon Emission Optimizer")
    st.markdown("**Robust Linear Programming with Sample Average Approximation**")
    st.markdown("---")


def display_sidebar(optimizer: IntegratedCarbonOptimizer):
    """Display sidebar"""
    with st.sidebar:
        st.header("Configuration")
        
        total_params = len(optimizer.parameters)
        optimizable = sum(1 for p in optimizer.parameters.values() if p.is_optimizable)
        
        st.metric("Total Parameters", total_params)
        st.metric("Optimizable Parameters", optimizable)
        st.metric("Fixed Parameters", total_params - optimizable)
        
        st.markdown("---")

def display_configuration_tab(optimizer: IntegratedCarbonOptimizer):
    """Configuration interface"""
    st.header("Parameter Configuration")
    
    st.subheader("Grid Emission Factor Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        custom_grid_factor = st.number_input(
            "Grid Emission Factor (kg CO2/kWh)",
            min_value=0.0,
            max_value=2.0,
            value=optimizer.grid_emission_factor,
            format="%.3f",
            help="Set the carbon intensity of your electricity grid. Default: 0.495 kg CO2/kWh (Denmark 2024)"
        )
        
        if st.button("Update Grid Factor", use_container_width=True):
            optimizer.set_grid_emission_factor(custom_grid_factor)
            st.success(f"Grid emission factor updated to {custom_grid_factor} kg CO2/kWh")
            st.rerun()
    
    with col2:
        st.info("""
        **Reference values:**
        - Denmark: 0.495
        - Germany: 0.420
        - France: 0.056
        - Poland: 0.766
        - EU Average: 0.328
        - USA: 0.417
        - China: 0.555
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scope_filter = st.selectbox(
            "Filter by Scope",
            ["All", "Scope1", "Scope2", "Scope3"],
            key="config_scope"
        )
    
    with col2:
        show_optimizable_only = st.checkbox("Show Optimizable Only", value=False)
    
    filtered_params = []
    for param in optimizer.parameters.values():
        if scope_filter != "All" and param.scope != scope_filter:
            continue
        if show_optimizable_only and not param.is_optimizable:
            continue
        filtered_params.append(param)
    
    st.info(f"Showing {len(filtered_params)} parameters")
    
    params_by_scope = {}
    for param in filtered_params:
        if param.scope not in params_by_scope:
            params_by_scope[param.scope] = []
        params_by_scope[param.scope].append(param)
    
    for scope_key, params in sorted(params_by_scope.items()):
        with st.expander(f"{scope_key} ({len(params)} parameters)", expanded=False):
            for param in params:
                st.markdown(f"**{param.name}**")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    new_value = st.number_input(
                        f"Value ({param.unit})",
                        value=float(param.nominal_value),
                        min_value=0.0,
                        format="%.4f",
                        key=f"value_{param.stub}"
                    )
                    param.nominal_value = new_value
                
                with col2:
                    new_uncertainty = st.number_input(
                        "Uncertainty (%)",
                        value=float(param.uncertainty_percent),
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f",
                        key=f"unc_{param.stub}"
                    )
                    param.uncertainty_percent = new_uncertainty
                
                with col3:
                    dist_options = [d.value for d in DistributionType]
                    new_dist = st.selectbox(
                        "Distribution",
                        dist_options,
                        index=dist_options.index(param.distribution.value),
                        key=f"dist_{param.stub}"
                    )
                    param.distribution = DistributionType[new_dist.upper().replace(" ", "_")]
                
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    is_opt = st.checkbox(
                        "Optimizable",
                        value=param.is_optimizable,
                        key=f"opt_{param.stub}"
                    )
                    param.is_optimizable = is_opt
                
                if is_opt:
                    with col5:
                        min_change = st.number_input(
                            "Min Change (%)",
                            value=float(param.min_change_percent),
                            min_value=-100.0,
                            max_value=0.0,
                            format="%.1f",
                            key=f"min_{param.stub}"
                        )
                        param.min_change_percent = min_change
                    
                    with col6:
                        max_change = st.number_input(
                            "Max Change (%)",
                            value=float(param.max_change_percent),
                            min_value=0.0,
                            max_value=100.0,
                            format="%.1f",
                            key=f"max_{param.stub}"
                        )
                        param.max_change_percent = max_change
                
                st.markdown("---")

def display_optimization_tab(optimizer: IntegratedCarbonOptimizer):
    """Unified optimization interface with SAA"""
    st.header("Integrated Optimization with Sample Average Approximation")
    st.subheader("Optimization Settings")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        method = st.selectbox(
            "Optimization Algorithm",
            [m.value for m in OptimizationMethod],
            help="SLSQP is recommended for most cases"
        )
        method_key = method.split('(')[0].strip()
    
    with col2:
        scope = st.selectbox(
            "Optimize Scope",
            ["All", "Scope1", "Scope2", "Scope3"],
            help="Select which scope to optimize"
        )
    
    with col3:
        use_saa = st.checkbox(
            "Use SAA (Robust)",
            value=True,
            help="Enable Sample Average Approximation for robust optimization"
        )
    
    with col4:
        if use_saa:
            saa_samples = st.select_slider(
                "SAA Samples",
                options=[50, 100, 200, 500, 1000],
                value=200,
                help="Samples for optimization objective"
            )
        else:
            saa_samples = 0
    
    col5, col6 = st.columns(2)
    with col5:
        mc_sims = st.select_slider(
            "Monte Carlo Simulations",
            options=[1000, 2500, 5000, 10000, 25000],
            value=5000,
            help="For final uncertainty analysis"
        )
    
    opt_count = sum(1 for p in optimizer.parameters.values()
                   if p.is_optimizable and 
                   (scope == "All" or p.scope == scope))
    
    with col6:
        st.metric("Optimizable Parameters", opt_count)
    
    if opt_count == 0:
        st.warning("No optimizable parameters found. Please configure at least one parameter as 'Optimizable' in the Configuration tab.")
        return
    
    st.markdown("---")
    
    if st.button("Run Optimization", use_container_width=True, type="primary"):
        with st.spinner(f"Running {'SAA' if use_saa else 'deterministic'} optimization..."):
            
            result = optimizer.optimize_with_uncertainty(
                method=method_key,
                mc_simulations=mc_sims,
                saa_samples=saa_samples,
                scope_filter=scope,
                use_saa=use_saa
            )

            st.session_state.result = result

        if result.success:
            st.success(f"{result.message}")
        else:
            st.error(f"{result.message}")

        display_optimization_results(result, use_saa)


def display_optimization_results(result: OptimizationResult, use_saa: bool):
    """Display optimization results with visualizations"""
    
    st.markdown("---")
    st.markdown("## Optimization Results")
    
    st.markdown("### Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Baseline Emissions",
            f"{result.baseline_emissions:,.0f} kg",
            help="Original total emissions"
        )
    
    with col2:
        st.metric(
            "Optimized Emissions",
            f"{result.optimized_emissions:,.0f} kg",
            delta=f"-{result.reduction_percent:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Reduction Achieved",
            f"{result.reduction_amount:,.0f} kg",
            help="Absolute reduction"
        )
    
    with col4:
        st.metric(
            "Computation Time",
            f"{result.computation_time:.2f}s",
            help=f"Iterations: {result.iterations}"
        )
    
    if use_saa:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("SAA Samples Used", f"{result.saa_samples_used:,}")
        with col2:
            st.metric("SAA Objective Value", f"{result.saa_objective_value:,.0f} kg")
    
    st.markdown("---")
    st.markdown("### Visualizations")
    
    # Chart 1: Emissions by Scope
    fig = go.Figure()
    
    scopes = ['Scope 1', 'Scope 2', 'Scope 3']
    baseline_values = [
        result.baseline_scope_breakdown['scope1'],
        result.baseline_scope_breakdown['scope2'],
        result.baseline_scope_breakdown['scope3']
    ]
    optimized_values = [
        result.optimized_scope_breakdown['scope1'],
        result.optimized_scope_breakdown['scope2'],
        result.optimized_scope_breakdown['scope3']
    ]
    
    fig.add_trace(go.Bar(
        name='Baseline',
        x=scopes,
        y=baseline_values,
        marker_color='rgba(255, 107, 107, 0.7)',
        text=[f"{v:,.0f}" for v in baseline_values],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Optimized',
        x=scopes,
        y=optimized_values,
        marker_color='rgba(78, 205, 196, 0.7)',
        text=[f"{v:,.0f}" for v in optimized_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Emissions by Scope: Baseline vs. Optimized",
        xaxis_title="Scope",
        yaxis_title="Emissions (kg CO2e)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Parameter Changes")
    
    changes_df = pd.DataFrame(result.parameter_changes)
    
    # Chart 2: Parameter Values
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name='Original',
        x=changes_df['parameter'],
        y=changes_df['original'],
        marker_color='rgba(255, 107, 107, 0.7)'
    ))
    fig2.add_trace(go.Bar(
        name='Optimized',
        x=changes_df['parameter'],
        y=changes_df['optimized'],
        marker_color='rgba(78, 205, 196, 0.7)'
    ))
    
    fig2.update_layout(
        title="Parameter Values: Original vs. Optimized",
        xaxis_title="Parameters",
        yaxis_title="Value",
        barmode='group',
        height=400,
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Table: Parameter Changes
    display_df = changes_df.copy()
    for col in ['original', 'optimized', 'change']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}")
    display_df['change_percent'] = display_df['change_percent'].apply(lambda x: f"{x:+.2f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Uncertainty Analysis (Monte Carlo)")
    
    tab1, tab2 = st.tabs(["Baseline Uncertainty", "Optimized Uncertainty"])
    
    with tab1:
        display_mc_results("Baseline", result.mc_baseline)
    
    with tab2:
        display_mc_results("Optimized", result.mc_optimized)
    st.markdown("---")
    st.markdown("### Uncertainty Reduction")
    
    col1, col2, col3, col4 = st.columns(4)
    
    baseline_cv = result.mc_baseline['cv']
    optimized_cv = result.mc_optimized['cv']
    cv_reduction = baseline_cv - optimized_cv
    
    with col1:
        st.metric("Baseline CV", f"{baseline_cv:.1f}%")
    with col2:
        st.metric("Optimized CV", f"{optimized_cv:.1f}%", 
                 delta=f"{-cv_reduction:.1f}%", 
                 delta_color="inverse")
    with col3:
        baseline_range = result.mc_baseline['p95'] - result.mc_baseline['p5']
        st.metric("Baseline 90% CI Range", f"{baseline_range:,.0f}")
    with col4:
        optimized_range = result.mc_optimized['p95'] - result.mc_optimized['p5']
        st.metric("Optimized 90% CI Range", f"{optimized_range:,.0f}",
                 delta=f"{baseline_range - optimized_range:,.0f}",
                 delta_color="inverse")
    
    # Chart 3: Distribution Comparison
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=result.mc_baseline['data'],
        name='Baseline',
        opacity=0.6,
        marker_color='rgba(255, 107, 107, 0.7)',
        nbinsx=50
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=result.mc_optimized['data'],
        name='Optimized',
        opacity=0.6,
        marker_color='rgba(78, 205, 196, 0.7)',
        nbinsx=50
    ))
    
    fig_dist.update_layout(
        title="Distribution Comparison: Baseline vs. Optimized",
        xaxis_title="Total Emissions (kg CO2e)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Sensitivity Analysis")
    
    top_n = min(10, len(result.sensitivity_ranking))
    top_sensitive = result.sensitivity_ranking[:top_n]
    
    param_names = [s['parameter'] for s in top_sensitive]
    correlations = [s['correlation'] for s in top_sensitive]
    
    fig_sens = go.Figure(data=go.Bar(
        x=correlations,
        y=param_names,
        orientation='h',
        marker_color=['rgba(74, 144, 226, 0.7)' if c >= 0 else 'rgba(245, 101, 101, 0.7)' 
                     for c in correlations],
        text=[f"{c:.3f}" for c in correlations],
        textposition='outside'
    ))
    
    fig_sens.update_layout(
        title=f"Top {top_n} Most Influential Parameters (After Optimization)",
        xaxis_title="Correlation with Total Emissions",
        yaxis_title="Parameters",
        height=max(400, top_n * 40),
        xaxis=dict(range=[-1, 1])
    )
    
    st.plotly_chart(fig_sens, use_container_width=True)
    
    st.markdown("---")
    if st.button("Export Complete Results", use_container_width=True):
        export_data = {
            'optimization': {
                'method': result.method,
                'use_saa': use_saa,
                'saa_samples': result.saa_samples_used,
                'saa_objective_value': result.saa_objective_value,
                'baseline_emissions': result.baseline_emissions,
                'optimized_emissions': result.optimized_emissions,
                'reduction_amount': result.reduction_amount,
                'reduction_percent': result.reduction_percent,
                'computation_time': result.computation_time,
                'iterations': result.iterations
            },
            'parameter_changes': result.parameter_changes,
            'baseline_uncertainty': {
                'mean': result.mc_baseline['mean'],
                'std': result.mc_baseline['std'],
                'cv': result.mc_baseline['cv'],
                'percentiles': {
                    'p5': result.mc_baseline['p5'],
                    'p25': result.mc_baseline['p25'],
                    'p50': result.mc_baseline['p50'],
                    'p75': result.mc_baseline['p75'],
                    'p95': result.mc_baseline['p95']
                }
            },
            'optimized_uncertainty': {
                'mean': result.mc_optimized['mean'],
                'std': result.mc_optimized['std'],
                'cv': result.mc_optimized['cv'],
                'percentiles': {
                    'p5': result.mc_optimized['p5'],
                    'p25': result.mc_optimized['p25'],
                    'p50': result.mc_optimized['p50'],
                    'p75': result.mc_optimized['p75'],
                    'p95': result.mc_optimized['p95']
                }
            },
            'sensitivity_ranking': result.sensitivity_ranking,
            'timestamp': result.timestamp
        }
        
        st.download_button(
            "Download Complete Analysis (JSON)",
            json.dumps(export_data, indent=2),
            f"saa_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


def display_mc_results(title: str, mc_result: Dict):
    """Display Monte Carlo results"""
    st.markdown(f"#### {title} Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{mc_result['mean']:,.0f} kg")
    with col2:
        st.metric("Std Dev", f"{mc_result['std']:,.0f} kg")
    with col3:
        st.metric("CV", f"{mc_result['cv']:.1f}%")
    with col4:
        ci_range = mc_result['p95'] - mc_result['p5']
        st.metric("90% CI Range", f"{ci_range:,.0f} kg")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=mc_result['data'],
        nbinsx=50,
        marker_color='rgba(74, 144, 226, 0.7)',
        name='Distribution'
    ))
    
    for percentile, value, name, color in [
        (5, mc_result['p5'], '5th', 'red'),
        (50, mc_result['p50'], 'Median', 'green'),
        (95, mc_result['p95'], '95th', 'red')
    ]:
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{name}<br>{value:,.0f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f"{title} Emission Distribution",
        xaxis_title="Total Emissions (kg CO2e)",
        yaxis_title="Frequency",
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    percentile_df = pd.DataFrame({
        'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
        'Emissions (kg CO2e)': [
            f"{mc_result['p5']:,.0f}",
            f"{mc_result['p25']:,.0f}",
            f"{mc_result['p50']:,.0f}",
            f"{mc_result['p75']:,.0f}",
            f"{mc_result['p95']:,.0f}"
        ]
    })
    
    st.dataframe(percentile_df, use_container_width=True, hide_index=True)

def display_carbon_pricing_tab(optimizer: IntegratedCarbonOptimizer):
    """Display carbon pricing analysis tab with weighted pricing"""
    st.header("Carbon Pricing Analysis")
    
    st.info("""
    **How Carbon Pricing Works:**
    
    Different emissions are subject to different carbon pricing mechanisms:
    - **EU ETS**: Covers specific sectors (electricity generation, large industrial facilities)
    - **National Carbon Tax**: May cover emissions not fully covered by EU ETS
    - **Coverage Overlap**: Some emissions may be covered by both, but typically one mechanism applies
    
    This calculator uses **weighted average pricing** based on coverage percentages you define.
    """)
    
    st.markdown("### Configure Pricing Scenarios")
    
    with st.expander("Current Scenario", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**EU ETS Configuration**")
            current_eu_ets = st.number_input(
                "EU ETS Price (EUR/ton)",
                min_value=0.0,
                value=85.0,
                help="Current EU ETS carbon price"
            )
            current_eu_coverage = st.number_input(
                "EU ETS Coverage (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                help="Percentage of total emissions covered by EU ETS"
            )
        
        with col2:
            st.markdown("**National Tax Configuration**")
            current_national = st.number_input(
                "National Tax (EUR/ton)",
                min_value=0.0,
                value=26.0,
                help="Denmark national carbon tax rate"
            )
            current_national_coverage = st.number_input(
                "National Tax Coverage (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                help="Percentage of total emissions covered by national carbon tax"
            )
        
        total_coverage = current_eu_coverage + current_national_coverage
        if total_coverage > 100:
            st.warning(f"Total coverage ({total_coverage:.2f}%) exceeds 100%. This may indicate overlapping coverage.")
        elif total_coverage < 100:
            st.info(f"Total coverage: {total_coverage:.2f}%. {100-total_coverage:.2f}% of emissions are not priced.")
    
    with st.expander("Future Scenario", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**EU ETS Configuration**")
            future_eu_ets = st.number_input(
                "Future EU ETS Price (EUR/ton)",
                min_value=0.0,
                value=120.0,
                help="Projected EU ETS carbon price (e.g., 2030)"
            )
            future_eu_coverage = st.number_input(
                "Future EU ETS Coverage (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                help="Expected future EU ETS coverage"
            )
        
        with col2:
            st.markdown("**National Tax Configuration**")
            future_national = st.number_input(
                "Future National Tax (EUR/ton)",
                min_value=0.0,
                value=35.0,
                help="Projected national carbon tax rate"
            )
            future_national_coverage = st.number_input(
                "Future National Tax Coverage (%)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                help="Expected future national tax coverage"
            )
        
        future_total_coverage = future_eu_coverage + future_national_coverage
        if future_total_coverage > 100:
            st.warning(f"Total coverage ({future_total_coverage:.2f}%) exceeds 100%. This may indicate overlapping coverage.")
        elif future_total_coverage < 100:
            st.info(f"Total coverage: {future_total_coverage:.2f}%. {100-future_total_coverage:.2f}% of emissions are not priced.")
    
    # Create scenario objects
    current_scenario = CarbonPricingScenario(
        name="Current (2024)",
        eu_ets_price=current_eu_ets,
        eu_ets_coverage=current_eu_coverage,
        national_tax=current_national,
        national_tax_coverage=current_national_coverage
    )
    
    future_scenario = CarbonPricingScenario(
        name="Future (2030)",
        eu_ets_price=future_eu_ets,
        eu_ets_coverage=future_eu_coverage,
        national_tax=future_national,
        national_tax_coverage=future_national_coverage
    )
    
    scenarios = {
        current_scenario.name: current_scenario,
        future_scenario.name: future_scenario
    }
    
    st.markdown("### Pricing Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{current_scenario.name}**")
        current_breakdown = current_scenario.get_breakdown()
        for key, value in current_breakdown.items():
            st.metric(key, f"EUR {value:.2f}/ton")
    
    with col2:
        st.markdown(f"**{future_scenario.name}**")
        future_breakdown = future_scenario.get_breakdown()
        for key, value in future_breakdown.items():
            st.metric(key, f"EUR {value:.2f}/ton")
    
    if 'result' in st.session_state:
        result = st.session_state.result
        display_cost_analysis(optimizer, result, scenarios, current_scenario, future_scenario)
    else:
        display_baseline_cost_sample(optimizer, scenarios)


def display_cost_analysis(optimizer, result, scenarios, current_scenario, future_scenario):
    """Display cost analysis when optimization results are available"""
    
    st.markdown("---")
    st.markdown("### Cost Analysis")
    
    baseline_emissions = result.baseline_emissions
    optimized_emissions = result.optimized_emissions
    
    cost_data = {}
    
    for scenario_name, scenario in scenarios.items():
        baseline_cost_info = optimizer.calculate_carbon_cost(baseline_emissions, scenario)
        optimized_cost_info = optimizer.calculate_carbon_cost(optimized_emissions, scenario)
        
        cost_data[scenario_name] = {
            'baseline': baseline_cost_info,
            'optimized': optimized_cost_info,
            'savings': baseline_cost_info['total_cost'] - optimized_cost_info['total_cost']
        }
    st.markdown("#### Annual Cost Comparison")
    
    comparison_data = []
    for scenario_name, costs in cost_data.items():
        comparison_data.append({
            'Scenario': scenario_name,
            'Baseline Cost (EUR)': f"EUR {costs['baseline']['total_cost']:,.0f}",
            'Optimized Cost (EUR)': f"EUR {costs['optimized']['total_cost']:,.0f}",
            'Annual Savings (EUR)': f"EUR {costs['savings']:,.0f}",
            'Effective Price (EUR/ton)': f"EUR {costs['baseline']['effective_price_per_ton']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    for scenario_name, scenario in scenarios.items():
        with st.expander(f"Detailed Breakdown - {scenario_name}"):
            costs = cost_data[scenario_name]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Baseline**")
                st.metric("Total Cost", f"EUR {costs['baseline']['total_cost']:,.0f}")
                st.metric("EU ETS Cost", f"EUR {costs['baseline']['eu_ets_cost']:,.0f}")
                st.metric("National Tax Cost", f"EUR {costs['baseline']['national_cost']:,.0f}")
                st.metric("EU ETS Covered Emissions", f"{costs['baseline']['eu_ets_covered_tons']:,.1f} tons")
                st.metric("National Tax Covered Emissions", f"{costs['baseline']['national_covered_tons']:,.1f} tons")
                st.metric("Uncovered Emissions", f"{costs['baseline']['uncovered_tons']:,.1f} tons")
            
            with col2:
                st.markdown("**Optimized**")
                st.metric("Total Cost", f"EUR {costs['optimized']['total_cost']:,.0f}", 
                         delta=f"-EUR {costs['savings']:,.0f}", delta_color="inverse")
                st.metric("EU ETS Cost", f"EUR {costs['optimized']['eu_ets_cost']:,.0f}")
                st.metric("National Tax Cost", f"EUR {costs['optimized']['national_cost']:,.0f}")
                st.metric("EU ETS Covered Emissions", f"{costs['optimized']['eu_ets_covered_tons']:,.1f} tons")
                st.metric("National Tax Covered Emissions", f"{costs['optimized']['national_covered_tons']:,.1f} tons")
                st.metric("Uncovered Emissions", f"{costs['optimized']['uncovered_tons']:,.1f} tons")
    
    st.markdown("#### Cost Comparison Visualization")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Annual Costs', 'Cost Savings'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    scenarios_list = list(scenarios.keys())
    baseline_costs = [cost_data[s]['baseline']['total_cost'] for s in scenarios_list]
    optimized_costs = [cost_data[s]['optimized']['total_cost'] for s in scenarios_list]
    savings = [cost_data[s]['savings'] for s in scenarios_list]
    
    fig.add_trace(
        go.Bar(name='Baseline', x=scenarios_list, y=baseline_costs, marker_color='#FF6B6B'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(name='Optimized', x=scenarios_list, y=optimized_costs, marker_color='#4ECDC4'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(name='Savings', x=scenarios_list, y=savings, marker_color='#45B7D1'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Scenario", row=1, col=1)
    fig.update_xaxes(title_text="Scenario", row=1, col=2)
    fig.update_yaxes(title_text="Annual Cost (EUR)", row=1, col=1)
    fig.update_yaxes(title_text="Annual Savings (EUR)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ROI Analysis")
    
    implementation_cost = st.number_input(
        "Implementation Cost (EUR)",
        min_value=0.0,
        value=100000.0,
        help="One-time cost to implement emission reduction measures"
    )
    
    current_cost_data = cost_data[current_scenario.name]
    future_cost_data = cost_data[future_scenario.name]
    
    annual_savings_current = current_cost_data['savings']
    annual_savings_future = future_cost_data['savings']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Implementation Cost", f"EUR {implementation_cost:,.0f}")
    
    with col2:
        if annual_savings_current > 0:
            payback_current = implementation_cost / annual_savings_current
            st.metric("Payback Period (Current Prices)", f"{payback_current:.1f} years")
        else:
            payback_current = None
            st.metric("Payback Period (Current Prices)", "N/A")
    
    with col3:
        if annual_savings_future > 0:
            payback_future = implementation_cost / annual_savings_future
            st.metric("Payback Period (Future Prices)", f"{payback_future:.1f} years")
        else:
            payback_future = None
            st.metric("Payback Period (Future Prices)", "N/A")
    
    # ROI Chart
    years = np.arange(0, 11)
    cumulative_savings_current = years * annual_savings_current - implementation_cost
    cumulative_savings_future = years * annual_savings_future - implementation_cost
    
    fig_roi = go.Figure()
    
    fig_roi.add_trace(go.Scatter(
        x=years,
        y=cumulative_savings_current,
        name='Current Prices',
        line=dict(color='rgba(74, 144, 226, 0.8)', width=3),
        fill='tozeroy',
        fillcolor='rgba(74, 144, 226, 0.2)'
    ))
    
    fig_roi.add_trace(go.Scatter(
        x=years,
        y=cumulative_savings_future,
        name='Future Prices',
        line=dict(color='rgba(46, 160, 67, 0.8)', width=3),
        fill='tozeroy',
        fillcolor='rgba(46, 160, 67, 0.2)'
    ))
    
    fig_roi.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    
    fig_roi.update_layout(
        title="Cumulative Net Savings Over 10 Years",
        xaxis_title="Years",
        yaxis_title="Cumulative Net Savings (EUR)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # Part 5: Export Pricing Analysis
    # -------------------------------------------------------------------------
    st.markdown("---")
    if st.button("Export Pricing Analysis", use_container_width=True):
        pricing_export = {
            'scenarios': {
                name: {
                    'eu_ets_price': s.eu_ets_price,
                    'eu_ets_coverage': s.eu_ets_coverage,
                    'national_tax': s.national_tax,
                    'national_tax_coverage': s.national_tax_coverage,
                    'weighted_average_price': s.weighted_average_price,
                    'breakdown': s.get_breakdown()
                }
                for name, s in scenarios.items()
            },
            'emissions': {
                'baseline_tons': baseline_emissions / 1000,
                'optimized_tons': optimized_emissions / 1000,
                'reduction_tons': (baseline_emissions - optimized_emissions) / 1000,
                'reduction_percent': ((baseline_emissions - optimized_emissions) / baseline_emissions * 100)
            },
            'costs': {
                name: {
                    'baseline_total': costs['baseline']['total_cost'],
                    'baseline_eu_ets': costs['baseline']['eu_ets_cost'],
                    'baseline_national': costs['baseline']['national_cost'],
                    'optimized_total': costs['optimized']['total_cost'],
                    'optimized_eu_ets': costs['optimized']['eu_ets_cost'],
                    'optimized_national': costs['optimized']['national_cost'],
                    'annual_savings': costs['savings']
                }
                for name, costs in cost_data.items()
            },
            'roi_analysis': {
                'implementation_cost': implementation_cost,
                'annual_savings_current': annual_savings_current,
                'annual_savings_future': annual_savings_future,
                'payback_period_current_years': payback_current if annual_savings_current > 0 else None,
                'payback_period_future_years': payback_future if annual_savings_future > 0 else None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        st.download_button(
            "Download Pricing Analysis (JSON)",
            json.dumps(pricing_export, indent=2),
            f"carbon_pricing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


def display_baseline_cost_sample(optimizer, scenarios):
    """Display sample cost calculation when no optimization results exist"""
    
    st.markdown("### Sample Cost Calculation (Baseline Only)")
    
    baseline = optimizer.calculate_emissions()
    baseline_emissions = baseline['total']
    
    st.info(f"Current baseline emissions: **{baseline_emissions/1000:,.1f} tons CO2e/year**")
    
    sample_costs = []
    for scenario_name, scenario in scenarios.items():
        cost_info = optimizer.calculate_carbon_cost(baseline_emissions, scenario)
        sample_costs.append({
            'Scenario': scenario_name,
            'Weighted Avg Price (EUR/ton)': f"EUR {scenario.weighted_average_price:.2f}",
            'EU ETS Cost': f"EUR {cost_info['eu_ets_cost']:,.0f}",
            'National Tax Cost': f"EUR {cost_info['national_cost']:,.0f}",
            'Total Annual Cost': f"EUR {cost_info['total_cost']:,.0f}"
        })
    
    sample_df = pd.DataFrame(sample_costs)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    st.warning("Run optimization to see cost savings analysis and full ROI calculations.")

def main():
    """Main application"""
    setup_page()
    
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = IntegratedCarbonOptimizer()
    
    optimizer = st.session_state.optimizer
    
    display_header()
    
    tab1, tab2, tab3 = st.tabs([
        "Configuration",
        "SAA Optimization",
        "Carbon Pricing"
    ])
    
    with tab1:
        display_configuration_tab(optimizer)
    
    with tab2:
        display_optimization_tab(optimizer)
    
    with tab3:
        display_carbon_pricing_tab(optimizer)
    
    display_sidebar(optimizer)


if __name__ == "__main__":
    main()