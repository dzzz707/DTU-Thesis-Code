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
    initial_sidebar_state="collapsed"
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
    
    # New fields for stability analysis
    stability_analysis: Optional[Dict] = None
    random_seed_used: Optional[int] = None


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

        # Scope 2 - Electricity
        self.add_parameter(EmissionParameter(
            name="Electricity Consumption", stub="electricity_consumption",
            nominal_value=1150000.0, unit="kWh", scope="Scope2",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Grid Emission Factor", stub="grid_factor",
            nominal_value=self.grid_emission_factor, unit="kg CO2/kWh", scope="Scope2",
            is_optimizable=False
        ))

        # Scope 3 - Business Travel
        self.add_parameter(EmissionParameter(
            name="Air Travel Distance", stub="air_travel_km",
            nominal_value=500000.0, unit="km", scope="Scope3",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Air Travel Emission Factor", stub="air_travel_factor",
            nominal_value=0.255, unit="kg CO2/km", scope="Scope3",
            is_optimizable=False
        ))

        # Scope 3 - Waste
        self.add_parameter(EmissionParameter(
            name="Waste Generated", stub="waste_kg",
            nominal_value=50000.0, unit="kg", scope="Scope3",
            is_optimizable=True
        ))
        self.add_parameter(EmissionParameter(
            name="Waste Emission Factor", stub="waste_factor",
            nominal_value=0.45, unit="kg CO2/kg", scope="Scope3",
            is_optimizable=False
        ))
        
        # Define emission calculation structure
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
            base_val = parameter_values.get(stub, param.nominal_value)
            if param.uncertainty_percent > 0:
                samples[stub] = self.generate_samples(param, base_val, n_simulations)
            else:
                samples[stub] = np.full(n_simulations, base_val)
        
        emissions_array = np.zeros(n_simulations)
        for i in range(n_simulations):
            sim_params = {stub: samples[stub][i] for stub in self.parameters.keys()}
            emissions_array[i] = self.calculate_emissions(sim_params)['total']
        
        mean_emissions = np.mean(emissions_array)
        std_emissions = np.std(emissions_array)
        cv = (std_emissions / mean_emissions * 100) if mean_emissions > 0 else 0
        percentiles = np.percentile(emissions_array, [5, 25, 50, 75, 95])
        
        sensitivity = []
        for stub, param in self.parameters.items():
            if param.uncertainty_percent > 0:
                correlation = np.corrcoef(samples[stub], emissions_array)[0, 1]
                sensitivity.append({
                    'parameter': param.name,
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
                                  use_saa: bool = True,
                                  stability_runs: int = 1,
                                  random_seed: Optional[int] = None,
                                  use_smart_init: bool = True) -> OptimizationResult:
        """
        Integrated optimization with SAA (Sample Average Approximation)
        
        New Parameters:
            stability_runs: Number of optimization runs (1 for demo mode, 5-10 for validation)
            random_seed: Fixed seed for reproducibility (None for random, 42 recommended)
            use_smart_init: Use intelligent initial point strategy
        """
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
    
        all_results = []
        all_reductions = []
        
        for run_idx in range(stability_runs):
            # Set seed for reproducibility
            if random_seed is not None:
                np.random.seed(random_seed + run_idx)
            
            # Generate SAA samples
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

                # Smart initialization
                if use_smart_init and random_seed is not None:
                    reduction_factor = 0.92 + 0.03 * (run_idx / max(stability_runs - 1, 1))
                    x0 = np.ones(len(optimizable)) * reduction_factor
                else:
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

                # Smart initialization
                if use_smart_init and random_seed is not None:
                    reduction_factor = 0.92 + 0.03 * (run_idx / max(stability_runs - 1, 1))
                    x0 = np.array([p.nominal_value * reduction_factor for p in optimizable])
                else:
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
                continue  # Skip failed runs

            optimized_params_run = baseline_params.copy()
            for i, param in enumerate(optimizable):
                optimized_params_run[param.stub] = (
                    result.x[i] * param.nominal_value if use_scaling else result.x[i]
                )

            optimized_emissions_dict_run = self.calculate_emissions(optimized_params_run)
            optimized_total_run = optimized_emissions_dict_run['total']
            
            reduction_amount_run = baseline_total - optimized_total_run
            reduction_percent_run = (reduction_amount_run / baseline_total) * 100
            
            all_reductions.append(reduction_percent_run)
            all_results.append({
                'optimized_total': optimized_total_run,
                'reduction_amount': reduction_amount_run,
                'reduction_percent': reduction_percent_run,
                'optimized_params': optimized_params_run.copy(),
                'optimized_emissions_dict': optimized_emissions_dict_run,
                'result_obj': result
            })
        
        stability_analysis = None
        if stability_runs > 1 and len(all_reductions) > 1:
            reduction_array = np.array(all_reductions)
            mean_reduction = np.mean(reduction_array)
            median_reduction = np.median(reduction_array)
            std_reduction = np.std(reduction_array, ddof=1)
            cv = std_reduction / mean_reduction if mean_reduction != 0 else float('inf')
            
            ci_95 = stats.t.interval(
                0.95,
                len(reduction_array) - 1,
                loc=mean_reduction,
                scale=stats.sem(reduction_array)
            )
            
            stability_analysis = {
                'runs': len(all_reductions),
                'mean_reduction': mean_reduction,
                'median_reduction': median_reduction,
                'std_reduction': std_reduction,
                'cv': cv,
                'min_reduction': np.min(reduction_array),
                'max_reduction': np.max(reduction_array),
                'ci_95_lower': ci_95[0],
                'ci_95_upper': ci_95[1],
                'is_stable': cv < 0.15,
                'all_reductions': all_reductions
            }
            
            # Select median result as "best"
            median_idx = np.argmin(np.abs(reduction_array - median_reduction))
            best_run = all_results[median_idx]
        else:
            # Single run or all failed
            if len(all_results) > 0:
                best_run = all_results[0]
            else:
                return self._create_failed_result(method, "All optimization runs failed", 
                                                 start_time, mc_baseline, 0)
        
        # Use best run results
        optimized_params = best_run['optimized_params']
        optimized_total = best_run['optimized_total']
        optimized_emissions_dict = best_run['optimized_emissions_dict']
        result = best_run['result_obj']
        reduction_amount = best_run['reduction_amount']
        reduction_percent = best_run['reduction_percent']
        
        mc_optimized = self.monte_carlo_analysis(mc_simulations, optimized_params)
        
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
            timestamp=datetime.now().isoformat(),
            stability_analysis=stability_analysis,
            random_seed_used=random_seed
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
            timestamp=datetime.now().isoformat(),
            stability_analysis=None,
            random_seed_used=None
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

def display_optimization_tab(optimizer: IntegratedCarbonOptimizer):
    """Combined optimization interface with inline configuration"""
    
    st.header("SAA Optimization")
    st.subheader("Execution Mode")
    
    col_mode1, col_mode2 = st.columns([3, 2])
    
    with col_mode1:
        exec_mode = st.radio(
            "Select execution mode",
            ["Demo Mode (Single Run)", "Validation Mode (Stability Analysis)"]
        )
    
    with col_mode2:
        if "Demo" in exec_mode:
            seed_options = {
                42: "Seed 42 (Academic Standard)"
            }
            selected_seed = st.selectbox(
                "Random Seed",
                list(seed_options.keys()),
                format_func=lambda x: seed_options[x],
                help="Fixed seed ensures reproducible results"
            )
            stability_runs = 1
        else:
            selected_seed = 42
            stability_runs = st.slider(
                "Number of Runs",
                min_value=5,
                max_value=15,
                value=10,
                help="More runs provide better statistical confidence"
            )
    
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
    
    with col1:
        method = st.selectbox(
            "Algorithm",
            [m.value for m in OptimizationMethod],
            help="SLSQP is recommended for most cases"
        )
        method_key = method.split('(')[0].strip()
    
    with col2:
        scope = st.selectbox(
            "Scope",
            ["All", "Scope1", "Scope2", "Scope3"]
        )
    
    with col3:
        use_saa = st.checkbox("SAA (Robust)", value=True)
    
    with col4:
        if use_saa:
            saa_samples = st.select_slider(
                "SAA Samples",
                options=[200, 500, 1000, 1500, 2000],
                value=500
            )
        else:
            saa_samples = 0
            st.write("")
    
    with col5:
        mc_sims = st.select_slider(
            "MC Simulations",
            options=[1000, 2500, 5000, 10000, 20000],
            value=5000
        )
    
    opt_count = sum(1 for p in optimizer.parameters.values()
                   if p.is_optimizable and 
                   (scope == "All" or p.scope == scope))
    
    # Run button
    run_clicked = st.button("Run Optimization", use_container_width=True, type="primary")
    
    if opt_count == 0:
        st.warning("No optimizable parameters. Enable some in Configuration below.")
    
    st.markdown("---")
    
    with st.expander("Configuration", expanded=False):
        display_configuration_section(optimizer)
    
    if run_clicked and opt_count > 0:
        progress_text = f"Running {'SAA' if use_saa else 'deterministic'} optimization"
        if "Validation" in exec_mode:
            progress_text += f" ({stability_runs} runs for stability analysis)"
        
        with st.spinner(progress_text):
            result = optimizer.optimize_with_uncertainty(
                method=method_key,
                mc_simulations=mc_sims,
                saa_samples=saa_samples,
                scope_filter=scope,
                use_saa=use_saa,
                stability_runs=stability_runs,
                random_seed=selected_seed,
                use_smart_init=True
            )
            st.session_state.result = result

        if result.success:
            st.success(f"{result.message}")
        else:
            st.error(f"{result.message}")

        display_optimization_results(result, use_saa)
    
    # Show previous results if available
    elif 'result' in st.session_state:
        st.info("Previous optimization results displayed below. Click 'Run Optimization' to update.")
        display_optimization_results(st.session_state.result, 
                                    st.session_state.result.saa_samples_used > 0)


def display_configuration_section(optimizer: IntegratedCarbonOptimizer):
    """Configuration section (inside expander)"""
    
    # Grid Factor
    col1, col2 = st.columns([2, 1])
    with col1:
        custom_grid_factor = st.number_input(
            "Grid Emission Factor (kg CO2/kWh)",
            min_value=0.0, max_value=2.0,
            value=optimizer.grid_emission_factor,
            format="%.3f"
        )
        if st.button("Update Grid Factor"):
            optimizer.set_grid_emission_factor(custom_grid_factor)
            st.success(f"Updated to {custom_grid_factor}")
            st.rerun()
    
    with col2:
        st.caption("Reference: DK=0.495, DE=0.420, FR=0.056, PL=0.766")
    
    st.markdown("---")
    
    # Parameter filters
    col1, col2 = st.columns(2)
    with col1:
        scope_filter = st.selectbox("Filter by Scope", ["All", "Scope1", "Scope2", "Scope3"], key="cfg_scope")
    with col2:
        show_opt_only = st.checkbox("Show Optimizable Only", value=False)
    
    # Filter parameters
    filtered_params = []
    for param in optimizer.parameters.values():
        if scope_filter != "All" and param.scope != scope_filter:
            continue
        if show_opt_only and not param.is_optimizable:
            continue
        filtered_params.append(param)
    
    # Group by scope
    params_by_scope = {}
    for param in filtered_params:
        if param.scope not in params_by_scope:
            params_by_scope[param.scope] = []
        params_by_scope[param.scope].append(param)
    
    # Display parameters
    for scope_key, params in sorted(params_by_scope.items()):
        st.markdown(f"**{scope_key}**")
        
        for param in params:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"{param.name}")
            
            with col2:
                st.write(f"{param.nominal_value:,.2f} {param.unit}")
            
            with col3:
                is_opt = st.checkbox("Optimize", value=param.is_optimizable, 
                                    key=f"opt_{param.stub}")
                param.is_optimizable = is_opt
            
            with col4:
                if is_opt:
                    unc = st.number_input("Uncertainty %", min_value=0.0, max_value=50.0,
                                        value=param.uncertainty_percent, step=1.0,
                                        key=f"unc_{param.stub}")
                    param.uncertainty_percent = unc
                else:
                    st.write("")
        
        st.markdown("")


def display_optimization_results(result: OptimizationResult, use_saa: bool):
    """Display optimization results with stability analysis"""
    
    st.header("Optimization Results")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Baseline Emissions",
            f"{result.baseline_emissions/1000:.3f} tons CO2e"
        )
    
    with col2:
        st.metric(
            "Optimized Emissions",
            f"{result.optimized_emissions/1000:.3f} tons CO2e",
            delta=f"-{result.reduction_percent:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Reduction Amount",
            f"{result.reduction_amount/1000:.3f} tons CO2e"
        )
    
    with col4:
        if result.random_seed_used is not None:
            st.metric(
                "Random Seed",
                f"{result.random_seed_used}"
            )
        else:
            st.metric(
                "Computation Time",
                f"{result.computation_time:.2f}s"
            )
    
    if result.stability_analysis is not None:
        st.markdown("---")
        st.subheader("Stability Analysis")
        
        stability = result.stability_analysis
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Statistical Summary**")
            summary_df = pd.DataFrame({
                'Metric': [
                    'Mean Reduction',
                    'Median Reduction',
                    'Standard Deviation',
                    'Coefficient of Variation',
                    'Min Reduction',
                    'Max Reduction'
                ],
                'Value': [
                    f"{stability['mean_reduction']:.2f}%",
                    f"{stability['median_reduction']:.2f}%",
                    f"{stability['std_reduction']:.2f}%",
                    f"{stability['cv']:.1%}",
                    f"{stability['min_reduction']:.2f}%",
                    f"{stability['max_reduction']:.2f}%"
                ]
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**95% Confidence Interval**")
            st.info(
                f"The true reduction effect is estimated to be between "
                f"**{stability['ci_95_lower']:.2f}%** and **{stability['ci_95_upper']:.2f}%** "
                f"with 95% confidence."
            )
            
            st.markdown("**Stability Assessment**")
            if stability['is_stable']:
                st.success(f"Algorithm is stable (CV = {stability['cv']:.1%} < 15%)")
            else:
                st.warning(f"High variability detected (CV = {stability['cv']:.1%} >= 15%). Consider increasing SAA samples or adjusting parameter bounds.")
        
        with col3:
            st.markdown("**Distribution of Results**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=stability['all_reductions'],
                nbinsx=15,
                marker_color='rgba(74, 144, 226, 0.7)',
                name='Reduction %'
            ))
            fig_hist.add_vline(
                x=stability['median_reduction'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {stability['median_reduction']:.2f}%",
                annotation_position="top"
            )
            fig_hist.update_layout(
                xaxis_title="Reduction (%)",
                yaxis_title="Frequency",
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Parameter Changes")
    
    if result.parameter_changes:
        changes_df = pd.DataFrame(result.parameter_changes)
        changes_df = changes_df[['parameter', 'scope', 'original', 'optimized', 'change', 'change_percent', 'unit']]
        changes_df.columns = ['Parameter', 'Scope', 'Original', 'Optimized', 'Change', 'Change %', 'Unit']
        
        st.dataframe(
            changes_df.style.format({
                'Original': '{:.2f}',
                'Optimized': '{:.2f}',
                'Change': '{:.2f}',
                'Change %': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Original',
            x=[c['parameter'] for c in result.parameter_changes],
            y=[c['original'] for c in result.parameter_changes],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Optimized',
            x=[c['parameter'] for c in result.parameter_changes],
            y=[c['optimized'] for c in result.parameter_changes],
            marker_color='darkblue'
        ))
        fig.update_layout(
            title="Parameter Comparison",
            barmode='group',
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No parameter changes (optimization may have failed or no optimizable parameters)")
        
    st.subheader("Emission Breakdown by Scope")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Baseline**")
        baseline_data = pd.DataFrame({
            'Scope': list(result.baseline_scope_breakdown.keys()),
            'Emissions (kg CO2e)': list(result.baseline_scope_breakdown.values())
        })
        fig_baseline = px.pie(
            baseline_data,
            values='Emissions (kg CO2e)',
            names='Scope',
            title=f"Total: {result.baseline_emissions/1000:.3f} tons"
        )
        fig_baseline.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_baseline, use_container_width=True)
    
    with col2:
        st.markdown("**Optimized**")
        optimized_data = pd.DataFrame({
            'Scope': list(result.optimized_scope_breakdown.keys()),
            'Emissions (kg CO2e)': list(result.optimized_scope_breakdown.values())
        })
        fig_optimized = px.pie(
            optimized_data,
            values='Emissions (kg CO2e)',
            names='Scope',
            title=f"Total: {result.optimized_emissions/1000:.3f} tons"
        )
        fig_optimized.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_optimized, use_container_width=True)
    
    if use_saa and result.mc_baseline and result.mc_optimized:
        st.subheader("Uncertainty Analysis (Monte Carlo)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Baseline Uncertainty**")
            mc_base_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'CV', 'P5', 'P50', 'P95'],
                'Value (kg CO2e)': [
                    result.mc_baseline['mean'],
                    result.mc_baseline['std'],
                    result.mc_baseline['cv'],
                    result.mc_baseline['p5'],
                    result.mc_baseline['p50'],
                    result.mc_baseline['p95']
                ]
            })
            st.dataframe(mc_base_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**Optimized Uncertainty**")
            mc_opt_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'CV', 'P5', 'P50', 'P95'],
                'Value (kg CO2e)': [
                    result.mc_optimized['mean'],
                    result.mc_optimized['std'],
                    result.mc_optimized['cv'],
                    result.mc_optimized['p5'],
                    result.mc_optimized['p50'],
                    result.mc_optimized['p95']
                ]
            })
            st.dataframe(mc_opt_df, hide_index=True, use_container_width=True)
        
        # Distribution comparison
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=result.mc_baseline['data'],
            name='Baseline',
            opacity=0.6,
            marker_color='red'
        ))
        fig_mc.add_trace(go.Histogram(
            x=result.mc_optimized['data'],
            name='Optimized',
            opacity=0.6,
            marker_color='green'
        ))
        fig_mc.update_layout(
            title="Emission Distribution Comparison",
            xaxis_title="Emissions (kg CO2e)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Sensitivity ranking
        if result.sensitivity_ranking:
            st.subheader("Sensitivity Analysis")
            sens_df = pd.DataFrame(result.sensitivity_ranking[:10])
            sens_df = sens_df[['parameter', 'correlation']]
            sens_df.columns = ['Parameter', 'Correlation']
            
            fig_sens = px.bar(
                sens_df,
                x='Correlation',
                y='Parameter',
                orientation='h',
                title="Top 10 Most Influential Parameters",
                color='Correlation',
                color_continuous_scale='RdYlGn_r'
            )
            fig_sens.update_layout(height=400)
            st.plotly_chart(fig_sens, use_container_width=True)
    
    st.markdown("---")
    if st.button("Export Results", use_container_width=True):
        export_data = {
            'optimization': {
                'method': result.method,
                'baseline_emissions_tons': result.baseline_emissions / 1000,
                'optimized_emissions_tons': result.optimized_emissions / 1000,
                'reduction_percent': result.reduction_percent,
                'random_seed': result.random_seed_used
            },
            'stability_analysis': result.stability_analysis if result.stability_analysis else {},
            'parameter_changes': result.parameter_changes,
            'scope_breakdown': {
                'baseline': result.baseline_scope_breakdown,
                'optimized': result.optimized_scope_breakdown
            },
            'timestamp': result.timestamp
        }
        
        st.download_button(
            "Download JSON",
            json.dumps(export_data, indent=2),
            f"optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


def display_carbon_pricing_tab(optimizer: IntegratedCarbonOptimizer):
    """Display carbon pricing analysis tab"""
    st.header("Carbon Pricing Analysis")
    
    # Pricing Configuration
    with st.expander("Pricing Scenarios", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current (2024)**")
            current_eu_ets = st.number_input("EU ETS (EUR/ton)", min_value=0.0, value=85.0, key="c_ets")
            current_eu_cov = st.number_input("EU ETS Coverage (%)", min_value=0.0, max_value=100.0, value=60.0, key="c_ets_cov")
            current_nat = st.number_input("National Tax (EUR/ton)", min_value=0.0, value=26.0, key="c_nat")
            current_nat_cov = st.number_input("National Tax Coverage (%)", min_value=0.0, max_value=100.0, value=40.0, key="c_nat_cov")
        
        with col2:
            st.markdown("**Future (2030)**")
            future_eu_ets = st.number_input("EU ETS (EUR/ton)", min_value=0.0, value=120.0, key="f_ets")
            future_eu_cov = st.number_input("EU ETS Coverage (%)", min_value=0.0, max_value=100.0, value=75.0, key="f_ets_cov")
            future_nat = st.number_input("National Tax (EUR/ton)", min_value=0.0, value=35.0, key="f_nat")
            future_nat_cov = st.number_input("National Tax Coverage (%)", min_value=0.0, max_value=100.0, value=25.0, key="f_nat_cov")
    
    # Create scenarios
    current_scenario = CarbonPricingScenario(
        name="Current (2024)", eu_ets_price=current_eu_ets, eu_ets_coverage=current_eu_cov,
        national_tax=current_nat, national_tax_coverage=current_nat_cov
    )
    future_scenario = CarbonPricingScenario(
        name="Future (2030)", eu_ets_price=future_eu_ets, eu_ets_coverage=future_eu_cov,
        national_tax=future_nat, national_tax_coverage=future_nat_cov
    )
    scenarios = {current_scenario.name: current_scenario, future_scenario.name: future_scenario}
    
    # Weighted prices display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Weighted Price", f"EUR {current_scenario.weighted_average_price:.2f}/ton")
    with col2:
        st.metric("Future Weighted Price", f"EUR {future_scenario.weighted_average_price:.2f}/ton")
    
    st.markdown("---")
    
    # Cost analysis
    if 'result' in st.session_state:
        result = st.session_state.result
        display_cost_analysis(optimizer, result, scenarios, current_scenario, future_scenario)
    else:
        display_baseline_cost_sample(optimizer, scenarios)


def display_cost_analysis(optimizer, result, scenarios, current_scenario, future_scenario):
    """Display cost analysis when optimization results are available"""
    
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
    
    # Cost comparison table
    comparison_data = []
    for scenario_name, costs in cost_data.items():
        comparison_data.append({
            'Scenario': scenario_name,
            'Baseline (EUR)': f"{costs['baseline']['total_cost']:,.0f}",
            'Optimized (EUR)': f"{costs['optimized']['total_cost']:,.0f}",
            'Savings (EUR)': f"{costs['savings']:,.0f}"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # Cost comparison chart
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Annual Costs', 'Savings'))
    
    scenarios_list = list(scenarios.keys())
    baseline_costs = [cost_data[s]['baseline']['total_cost'] for s in scenarios_list]
    optimized_costs = [cost_data[s]['optimized']['total_cost'] for s in scenarios_list]
    savings = [cost_data[s]['savings'] for s in scenarios_list]
    
    fig.add_trace(go.Bar(name='Baseline', x=scenarios_list, y=baseline_costs, marker_color='#FF6B6B'), row=1, col=1)
    fig.add_trace(go.Bar(name='Optimized', x=scenarios_list, y=optimized_costs, marker_color='#4ECDC4'), row=1, col=1)
    fig.add_trace(go.Bar(name='Savings', x=scenarios_list, y=savings, marker_color='#45B7D1'), row=1, col=2)
    
    fig.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Analysis
    st.markdown("### ROI Analysis")
    implementation_cost = st.number_input("Implementation Cost (EUR)", min_value=0.0, value=100000.0)
    
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
            st.metric("Payback (Current)", f"{payback_current:.1f} years")
        else:
            st.metric("Payback (Current)", "N/A")
    with col3:
        if annual_savings_future > 0:
            payback_future = implementation_cost / annual_savings_future
            st.metric("Payback (Future)", f"{payback_future:.1f} years")
        else:
            st.metric("Payback (Future)", "N/A")
    
    # ROI Chart
    years = np.arange(0, 11)
    cumulative_current = years * annual_savings_current - implementation_cost
    cumulative_future = years * annual_savings_future - implementation_cost
    
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=years, y=cumulative_current, name='Current Prices',
                                line=dict(color='rgba(74, 144, 226, 0.8)', width=3),
                                fill='tozeroy', fillcolor='rgba(74, 144, 226, 0.2)'))
    fig_roi.add_trace(go.Scatter(x=years, y=cumulative_future, name='Future Prices',
                                line=dict(color='rgba(46, 160, 67, 0.8)', width=3),
                                fill='tozeroy', fillcolor='rgba(46, 160, 67, 0.2)'))
    fig_roi.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    fig_roi.update_layout(title="Cumulative Net Savings (10 Years)",
                         xaxis_title="Years", yaxis_title="Cumulative Savings (EUR)", height=350)
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Export
    if st.button("Export Pricing Analysis", use_container_width=True):
        pricing_export = {
            'scenarios': {name: {'weighted_price': s.weighted_average_price} for name, s in scenarios.items()},
            'emissions': {'baseline_tons': baseline_emissions/1000, 'optimized_tons': optimized_emissions/1000},
            'costs': {name: {'baseline': c['baseline']['total_cost'], 'optimized': c['optimized']['total_cost'],
                            'savings': c['savings']} for name, c in cost_data.items()},
            'timestamp': datetime.now().isoformat()
        }
        st.download_button("Download JSON", json.dumps(pricing_export, indent=2),
                          f"pricing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                          mime="application/json", use_container_width=True)


def display_baseline_cost_sample(optimizer, scenarios):
    """Display sample cost calculation when no optimization results exist"""
    
    st.markdown("### Baseline Cost Calculation")
    
    baseline = optimizer.calculate_emissions()
    baseline_emissions = baseline['total']
    
    st.info(f"Baseline emissions: **{baseline_emissions/1000:,.3f} tons CO2e/year**")
    
    sample_costs = []
    for scenario_name, scenario in scenarios.items():
        cost_info = optimizer.calculate_carbon_cost(baseline_emissions, scenario)
        sample_costs.append({
            'Scenario': scenario_name,
            'Weighted Price (EUR/ton)': f"{scenario.weighted_average_price:.2f}",
            'Total Annual Cost': f"EUR {cost_info['total_cost']:,.0f}"
        })
    
    st.dataframe(pd.DataFrame(sample_costs), use_container_width=True, hide_index=True)
    st.warning("Run optimization to see cost savings analysis.")

def main():
    """Main application"""
    setup_page()
    
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = IntegratedCarbonOptimizer()
    
    optimizer = st.session_state.optimizer
    
    display_header()
    
    # Two tabs instead of three
    tab1, tab2 = st.tabs(["Optimization", "Cost Analysis"])
    
    with tab1:
        display_optimization_tab(optimizer)
    
    with tab2:
        display_carbon_pricing_tab(optimizer)


if __name__ == "__main__":
    main()