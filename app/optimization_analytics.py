import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
from model_manager import get_model_scope, is_total_model

RANDOM_SEED = 42

class OptimizationMethod(Enum):
    SLSQP = "SLSQP (Sequential Least Squares Programming)"
    COBYLA = "COBYLA (Constrained Optimization by Linear Approximations)"
    TRUST_CONSTR = "TRUST_CONSTR (Trust Region Constrained Optimization)"

@dataclass
class EmissionParameter:
    name: str
    stub: str
    nominal_value: float
    unit: str
    scope: str
    
    is_optimizable: bool = False
    min_change_percent: float = -10.0
    max_change_percent: float = 10.0
    uncertainty_percent: float = 5.0
    
    @property
    def optimization_min(self) -> float:
        """Calculate lower optimization bound"""
        return max(0, self.nominal_value * (1 + self.min_change_percent / 100))
    
    @property
    def optimization_max(self) -> float:
        """Calculate upper optimization bound"""
        return self.nominal_value * (1 + self.max_change_percent / 100)


@dataclass
class OptimizationResult:
    method: str
    success: bool
    message: str
    
    # Emission results
    baseline_emissions: float
    optimized_emissions: float
    reduction_amount: float
    reduction_percent: float
    
    # Scope breakdown
    baseline_scope_breakdown: Dict[str, float]
    optimized_scope_breakdown: Dict[str, float]
    
    # Parameter information
    optimized_parameters: Dict[str, float]
    parameter_changes: List[Dict]
    
    # Monte Carlo results
    mc_baseline: Dict
    mc_optimized: Dict
    
    # Sensitivity analysis
    sensitivity_ranking: List[Dict]
    
    # Optimization metadata
    saa_samples_used: int
    saa_objective_value: float
    iterations: int
    computation_time: float
    timestamp: str
    
    # Stability analysis (for multi-run mode)
    stability_analysis: Optional[Dict] = None
    random_seed_used: Optional[int] = None


@dataclass
class CarbonPricingScenario:
    name: str
    eu_ets_price: float
    eu_ets_coverage: float
    national_tax: float
    national_tax_coverage: float
    
    @property
    def weighted_average_price(self) -> float:
        eu_component = self.eu_ets_price * (self.eu_ets_coverage / 100)
        national_component = self.national_tax * (self.national_tax_coverage / 100)
        return eu_component + national_component
    
    def get_breakdown(self) -> Dict[str, float]:
        return {
            'EU ETS Component': self.eu_ets_price * (self.eu_ets_coverage / 100),
            'National Tax Component': self.national_tax * (self.national_tax_coverage / 100),
            'Weighted Average': self.weighted_average_price
        }

class IntegratedCarbonOptimizer:
 
    def __init__(self, manager, scale_factor=1.0):
        if manager is None:
            raise ValueError("ModelManager is required. Please provide a valid manager instance.")
        
        self.manager = manager
        self.scale_factor = scale_factor
        self.parameters: Dict[str, EmissionParameter] = {}
        self.grid_emission_factor = 0.495
        
        self.initialize_from_manager()
    
    def set_grid_emission_factor(self, factor: float):
        self.grid_emission_factor = factor
        if 'grid_factor' in self.parameters:
            self.parameters['grid_factor'].nominal_value = factor
    
    def add_parameter(self, parameter: EmissionParameter):
        self.parameters[parameter.stub] = parameter

    def initialize_from_manager(self):
        if not self.manager:
            return
        
        for model_name, model in self.manager.models.items():
            scope = get_model_scope(model_name)
            
            for stub, mode in model.data_modes.items():
                is_factor = self._is_emission_factor(mode.name, stub)
                is_optimizable = not is_factor
                unit = self._infer_unit(mode.name, stub)
                
                scaled_value = mode.value * self.scale_factor if is_optimizable else mode.value
                
                param = EmissionParameter(
                    name=mode.name,
                    stub=stub,
                    nominal_value=scaled_value,
                    unit=unit,
                    scope=scope,
                    is_optimizable=is_optimizable,
                    min_change_percent=-10.0 if is_optimizable else 0.0,
                    max_change_percent=10.0 if is_optimizable else 0.0,
                    uncertainty_percent=5.0
                )
                
                self.parameters[stub] = param
    
    def _is_emission_factor(self, name: str, stub: str) -> bool:
        name_lower = name.lower()
        stub_lower = stub.lower()
        factor_keywords = ['factor', 'gwp', 'xgwp', 'co2', 'ch4', 'n2o']
        return any(kw in name_lower or kw in stub_lower for kw in factor_keywords)
    
    def _infer_unit(self, name: str, stub: str) -> str:
        name_lower = name.lower()
        stub_lower = stub.lower()
        
        if 'm3' in stub_lower:
            return 'm³'
        elif any(x in stub_lower for x in ['ulp', 'diesel', 'lpg']) and 'data' in stub_lower:
            return 'L'
        elif 'electricity' in name_lower:
            return 'kWh'
        elif 'water' in name_lower:
            return 'L'
        elif 'gwp' in stub_lower or 'xgwp' in stub_lower:
            if 'co2' in stub_lower:
                return 't CO2/unit'
            elif 'ch4' in stub_lower:
                return 't CH4/unit'
            elif 'n2o' in stub_lower:
                return 't N2O/unit'
            else:
                return 'CO2-eq'
        elif any(x in stub_lower for x in ['r134a', 'r407c', 'r410a']):
            return 't'
        else:
            return 'unit'
    
    def calculate_emissions(self, parameter_values: Dict[str, float] = None) -> Dict:
        if self.manager:
            if parameter_values is None:
                parameter_values = {stub: p.nominal_value for stub, p in self.parameters.items()}
            
            if not hasattr(self, '_stub_to_model_cache'):
                self._stub_to_model_cache = {}
                for model_name, model in self.manager.models.items():
                    for stub in model.data_modes.keys():
                        self._stub_to_model_cache[stub] = model_name
            
            original_values = {}
            for stub, value in parameter_values.items():
                if stub in self._stub_to_model_cache:
                    model_name = self._stub_to_model_cache[stub]
                    original_values[(model_name, stub)] = self.manager.models[model_name].data_modes[stub].value
                    self.manager.models[model_name].data_modes[stub].value = value
            
            self.manager.calculate_all()
            
            if not hasattr(self, '_scope_stub_cache'):
                self._scope_stub_cache = {'Scope1': None, 'Scope2': None, 'Scope3': None}
                
                for model_name, model in self.manager.models.items():
                    scope_key = get_model_scope(model_name)
                    
                    if scope_key not in self._scope_stub_cache:
                        continue
                    
                    if is_total_model(model_name):
                        for stub, mode in model.formula_modes.items():
                            stub_lower = stub.lower()
                            scope_lower = scope_key.lower()
                            if (stub == scope_key or 
                                stub_lower.startswith(scope_lower) or
                                scope_lower in stub_lower):
                                if self._scope_stub_cache[scope_key] is None:
                                    self._scope_stub_cache[scope_key] = (model_name, stub)
                                break
                    else:
                        for stub, mode in model.formula_modes.items():
                            if stub in self._scope_stub_cache and self._scope_stub_cache[stub] is None:
                                self._scope_stub_cache[stub] = (model_name, stub)
                
                for scope_key in ['Scope1', 'Scope2', 'Scope3']:
                    if self._scope_stub_cache[scope_key] is None:
                        for model_name, model in self.manager.models.items():
                            if get_model_scope(model_name) == scope_key and not is_total_model(model_name):
                                for stub, mode in model.formula_modes.items():
                                    if 'co2e' in stub.lower():
                                        pass
            
            scope1 = scope2 = scope3 = 0
            if self._scope_stub_cache['Scope1']:
                model_name, stub = self._scope_stub_cache['Scope1']
                result = self.manager.models[model_name].formula_modes[stub].result
                scope1 = result if result else 0
            
            if self._scope_stub_cache['Scope2']:
                model_name, stub = self._scope_stub_cache['Scope2']
                result = self.manager.models[model_name].formula_modes[stub].result
                scope2 = result if result else 0
            
            if self._scope_stub_cache['Scope3']:
                model_name, stub = self._scope_stub_cache['Scope3']
                result = self.manager.models[model_name].formula_modes[stub].result
                scope3 = result if result else 0
            
            total = scope1 + scope2 + scope3
            
            for (model_name, stub), original_value in original_values.items():
                self.manager.models[model_name].data_modes[stub].value = original_value
            
            self.manager.calculate_all()
            
            return {
                'total': total,
                'Scope1': scope1,
                'Scope2': scope2,
                'Scope3': scope3,
                'scope1_total': scope1,
                'scope2_total': scope2,
                'scope3_total': scope3
            }
    
    def calculate_carbon_cost(self, emissions_tons: float, pricing_scenario: CarbonPricingScenario) -> Dict[str, float]:
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
        std = base_value * param.uncertainty_percent / 100
        samples = np.random.normal(base_value, std, n_samples)
        return np.maximum(samples, 0)
    
    def monte_carlo_analysis(self, n_simulations: int, parameter_values: Dict[str, float]) -> Dict:
        samples = {}
        
        for stub, param in self.parameters.items():
            base_val = parameter_values.get(stub, param.nominal_value)
            if param.uncertainty_percent > 0:
                samples[stub] = self.generate_samples(param, base_val, n_simulations)
            else:
                samples[stub] = np.full(n_simulations, base_val)
        
        emissions_array = self._monte_carlo_vectorized(samples, n_simulations)
        
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
                    'stub': stub,
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
    
    def _monte_carlo_vectorized(self, samples: Dict[str, np.ndarray], n_simulations: int) -> np.ndarray:
        if not hasattr(self, '_mc_coefficients'):
            self._build_mc_coefficients()
        
        emissions_array = np.full(n_simulations, self._mc_fixed_contribution)
        
        for stub, coef in self._mc_coefficients.items():
            if stub in samples:
                emissions_array += samples[stub] * coef
        
        return emissions_array
    
    def _build_mc_coefficients(self):
        baseline_params = {stub: p.nominal_value for stub, p in self.parameters.items()}
        baseline_result = self.calculate_emissions(baseline_params)
        baseline_total = baseline_result['total']
        
        coefficients = {}
        
        for stub, param in self.parameters.items():
            if param.uncertainty_percent > 0:
                delta = param.nominal_value * 0.001 if param.nominal_value != 0 else 0.001
                
                perturbed_params = baseline_params.copy()
                perturbed_params[stub] = param.nominal_value + delta
                
                perturbed_result = self.calculate_emissions(perturbed_params)
                
                coef = (perturbed_result['total'] - baseline_total) / delta
                coefficients[stub] = coef
        
        variable_contribution = sum(
            self.parameters[stub].nominal_value * coef 
            for stub, coef in coefficients.items()
        )
        self._mc_fixed_contribution = baseline_total - variable_contribution
        self._mc_coefficients = coefficients
    
    def _calculate_emissions_fast(self, parameter_values: Dict[str, float]) -> float:
        if hasattr(self, '_fast_calc_cache'):
            return self._fast_calc_cache(parameter_values)
        
        return self.calculate_emissions(parameter_values)['total']
    
    def _build_fast_calculator(self):
        baseline_result = self.calculate_emissions()
        
        coefficients = {}
        
        for stub, param in self.parameters.items():
            if param.is_optimizable:
                baseline_params = {s: p.nominal_value for s, p in self.parameters.items()}
                perturbed_params = baseline_params.copy()
                
                delta = param.nominal_value * 0.001
                if delta == 0:
                    delta = 0.001
                perturbed_params[stub] = param.nominal_value + delta
                
                perturbed_result = self.calculate_emissions(perturbed_params)
                
                coef = (perturbed_result['total'] - baseline_result['total']) / delta
                coefficients[stub] = coef
        
        optimizable_contribution = sum(
            self.parameters[stub].nominal_value * coefficients[stub] 
            for stub in coefficients
        )
        fixed_contribution = baseline_result['total'] - optimizable_contribution
        
        def fast_calc(param_values):
            total = fixed_contribution
            for stub, coef in coefficients.items():
                if stub in param_values:
                    total += param_values[stub] * coef
                else:
                    total += self.parameters[stub].nominal_value * coef
            return total
        
        self._fast_calc_cache = fast_calc
        self._fast_calc_coefficients = coefficients
        self._fast_calc_fixed = fixed_contribution
    
    def optimize_with_uncertainty(self, method: str = 'SLSQP', 
                                  mc_simulations: int = 5000,
                                  saa_samples: int = 500,
                                  scope_filter: str = "All",
                                  use_saa: bool = True,
                                  stability_runs: int = 1) -> OptimizationResult:
        
        start_time = time.time()
        
        # Clear cached calculators (in case parameters changed)
        if hasattr(self, '_fast_calc_cache'):
            del self._fast_calc_cache
        if hasattr(self, '_mc_coefficients'):
            del self._mc_coefficients
        
        optimizable = [p for p in self.parameters.values() 
                      if p.is_optimizable and 
                      (scope_filter == "All" or p.scope == scope_filter)]
        
        if not optimizable:
            return self._create_failed_result(method, "No optimizable parameters found", start_time)
        
        # Build fast calculator for optimization
        self._build_fast_calculator()
        
        baseline_params = {stub: p.nominal_value for stub, p in self.parameters.items()}
        baseline_emissions_dict = self.calculate_emissions(baseline_params)
        baseline_total = baseline_emissions_dict['total']
        
        # Check for zero baseline
        if baseline_total == 0:
            return self._create_failed_result(
                method, 
                "Baseline emissions are zero. Check if model is properly configured with emission data.",
                start_time
            )
        
        mc_baseline = self.monte_carlo_analysis(mc_simulations, baseline_params)
    
        all_results = []
        all_reductions = []
        
        for run_idx in range(stability_runs):
            # Set seed for reproducibility
            np.random.seed(RANDOM_SEED + run_idx)
            
            # Generate SAA samples
            if use_saa:
                optim_samples = {}
                for param in optimizable:
                    optim_samples[param.stub] = self.generate_samples(
                        param, param.nominal_value, saa_samples
                    )
            use_scaling = True 

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
                            total_em += self._calculate_emissions_fast(param_values)
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
                
                bounds = []
                for p in optimizable:
                    if p.nominal_value != 0:
                        bounds.append((p.optimization_min / p.nominal_value, 
                                      p.optimization_max / p.nominal_value))
                    else:
                        # Handle zero nominal value - use absolute bounds
                        bounds.append((0.0, 1.0))
            else:
                # This branch is now unreachable but kept for reference
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
                            total_em += self._calculate_emissions_fast(param_values)
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

            # Configure solver
            if 'COBYLA' in method.upper():
                scipy_method = 'COBYLA'
                options = {'maxfun': 50000, 'rhobeg': 1.0, 'tol': 1e-6}
            elif 'TRUST' in method.upper():
                scipy_method = 'trust-constr'
                options = {'maxiter': 2000, 'xtol': 1e-6, 'gtol': 1e-6}
            else:
                scipy_method = 'SLSQP'
                options = {'maxiter': 2000, 'ftol': 1e-8}  # Tighter tolerance for better convergence

            try:
                result = minimize(objective, x0, method=scipy_method, bounds=bounds, options=options)
            except Exception as e:
                # If optimization throws an exception, skip this run
                continue
            
            # Accept result if it shows ANY improvement, even if scipy reports failure
            has_result = hasattr(result, 'fun') and result.fun is not None
            if not has_result:
                continue
            
            # Check if result is valid (not NaN or Inf)
            if np.isnan(result.fun) or np.isinf(result.fun):
                continue

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
        
        # Stability analysis
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
            
            median_idx = np.argmin(np.abs(reduction_array - median_reduction))
            best_run = all_results[median_idx]
        else:
            if len(all_results) > 0:
                best_run = all_results[0]
            else:
                return self._create_failed_result(method, "All optimization runs failed", 
                                                 start_time, mc_baseline, 0)
        
        optimized_params = best_run['optimized_params']
        optimized_total = best_run['optimized_total']
        optimized_emissions_dict = best_run['optimized_emissions_dict']
        result = best_run['result_obj']
        reduction_amount = best_run['reduction_amount']
        reduction_percent = best_run['reduction_percent']
        
        mc_optimized = self.monte_carlo_analysis(mc_simulations, optimized_params)
        
        # Build parameter changes with bound information
        parameter_changes = []
        for param in optimizable:
            original = param.nominal_value
            optimized = optimized_params[param.stub]
            change = optimized - original
            change_pct = (change / original) * 100 if original != 0 else 0
            
            # Check if bounds are active (use 0.1% relative tolerance)
            tolerance = abs(original) * 0.001 if original != 0 else 1e-6
            at_lower_bound = abs(optimized - param.optimization_min) < tolerance
            at_upper_bound = abs(optimized - param.optimization_max) < tolerance
            
            parameter_changes.append({
                'parameter': param.name,
                'stub': param.stub,
                'scope': param.scope,
                'unit': param.unit,
                'original': original,
                'optimized': optimized,
                'change': change,
                'change_percent': change_pct,
                'lower_bound': param.optimization_min,
                'upper_bound': param.optimization_max,
                'at_lower_bound': at_lower_bound,
                'at_upper_bound': at_upper_bound
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
                'Scope1': baseline_emissions_dict['scope1_total'],
                'Scope2': baseline_emissions_dict['scope2_total'],
                'Scope3': baseline_emissions_dict['scope3_total']
            },
            optimized_scope_breakdown={
                'Scope1': optimized_emissions_dict['scope1_total'],
                'Scope2': optimized_emissions_dict['scope2_total'],
                'Scope3': optimized_emissions_dict['scope3_total']
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
            random_seed_used=RANDOM_SEED
        )
    
    def _create_failed_result(self, method, message, start_time, mc_baseline=None, iterations=0):
        baseline_params = {stub: p.nominal_value for stub, p in self.parameters.items()}
        baseline_emissions_dict = self.calculate_emissions(baseline_params)
        baseline_total = baseline_emissions_dict['total']
        
        if mc_baseline is None:
            mc_baseline = {'sensitivity': [], 'mean': baseline_total, 'std': 0, 'cv': 0,
                          'p5': baseline_total, 'p25': baseline_total, 'p50': baseline_total,
                          'p75': baseline_total, 'p95': baseline_total, 'data': np.array([baseline_total])}
        
        baseline_scope_safe = {
            'Scope1': baseline_emissions_dict.get('scope1_total', 0),
            'Scope2': baseline_emissions_dict.get('scope2_total', 0),
            'Scope3': baseline_emissions_dict.get('scope3_total', 0)
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
def create_box_plot_comparison(mc_baseline: Dict, mc_optimized: Dict) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=mc_baseline['data'],
        name='Baseline',
        marker_color='#E74C3C',
        boxpoints='outliers',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    fig.add_trace(go.Box(
        y=mc_optimized['data'],
        name='Optimized',
        marker_color='#27AE60',
        boxpoints='outliers',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    fig.update_layout(
        title='Emission Distribution Comparison (Box Plot)',
        yaxis_title='Emissions (tons CO2e)',
        showlegend=True,
        height=400
    )
    
    return fig

def create_waterfall_chart(baseline_breakdown: Dict, optimized_breakdown: Dict) -> go.Figure:
    baseline_total = sum(baseline_breakdown.values())
    optimized_total = sum(optimized_breakdown.values())
    
    scopes = list(baseline_breakdown.keys())
    reductions = [(baseline_breakdown[s] - optimized_breakdown[s]) for s in scopes]
    
    # Build text annotations for each bar
    baseline_text = f"{baseline_total:.2f}"
    reduction_texts = [f"-{r:.2f}" if r > 0 else f"+{abs(r):.2f}" if r < 0 else "" for r in reductions]
    optimized_text = f"{optimized_total:.2f}"
    
    fig = go.Figure(go.Waterfall(
        name="Emission Reduction",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(scopes) + ["total"],
        x=["Baseline"] + [f"{s} Reduction" for s in scopes] + ["Optimized"],
        y=[baseline_total] + [-r for r in reductions] + [0],
        text=[baseline_text] + reduction_texts + [optimized_text],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#27AE60"}},
        increasing={"marker": {"color": "#E74C3C"}},
        totals={"marker": {"color": "#3498DB"}}
    ))
    
    fig.update_layout(
        title="Emission Reduction Waterfall by Scope",
        yaxis_title="Emissions (tonnes CO2e)",
        showlegend=False,
        height=400
    )
    
    return fig


def create_parameter_bounds_chart(parameter_changes: List[Dict]) -> go.Figure:
    fig = go.Figure()
    
    params = [p['parameter'][:20] + '...' if len(p['parameter']) > 20 else p['parameter'] 
              for p in parameter_changes]
    
    # Normalize values to percentage of original
    for i, p in enumerate(parameter_changes):
        original = p['original']
        if original == 0:
            continue
            
        lower_pct = (p['lower_bound'] / original - 1) * 100
        upper_pct = (p['upper_bound'] / original - 1) * 100
        optimized_pct = (p['optimized'] / original - 1) * 100
        
        # Draw range bar
        fig.add_trace(go.Scatter(
            x=[lower_pct, upper_pct],
            y=[params[i], params[i]],
            mode='lines',
            line=dict(color='lightgray', width=8),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Mark bounds
        bound_color = '#E74C3C' if p['at_lower_bound'] or p['at_upper_bound'] else '#95A5A6'
        fig.add_trace(go.Scatter(
            x=[lower_pct, upper_pct],
            y=[params[i], params[i]],
            mode='markers',
            marker=dict(size=10, color=bound_color, symbol='line-ns'),
            showlegend=False,
            hovertemplate=f"Bounds: [{lower_pct:.1f}%, {upper_pct:.1f}%]<extra></extra>"
        ))
        
        # Original value (always at 0%)
        fig.add_trace(go.Scatter(
            x=[0],
            y=[params[i]],
            mode='markers',
            marker=dict(size=12, color='#3498DB', symbol='diamond'),
            showlegend=False,
            hovertemplate=f"Original: {original:,.2f}<extra></extra>"
        ))
        
        # Optimized value
        marker_color = '#E74C3C' if p['at_lower_bound'] or p['at_upper_bound'] else '#27AE60'
        fig.add_trace(go.Scatter(
            x=[optimized_pct],
            y=[params[i]],
            mode='markers',
            marker=dict(size=14, color=marker_color, symbol='circle'),
            showlegend=False,
            hovertemplate=f"Optimized: {p['optimized']:,.2f} ({optimized_pct:+.1f}%)<extra></extra>"
        ))
    
    # Add legend manually
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(size=10, color='#3498DB', symbol='diamond'),
                             name='Original Value'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(size=10, color='#27AE60', symbol='circle'),
                             name='Optimized Value'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(size=10, color='#E74C3C', symbol='circle'),
                             name='At Bound (Active)'))
    
    fig.update_layout(
        title='Parameter Optimization Results with Bounds',
        xaxis_title='Change from Original (%)',
        yaxis_title='',
        height=max(300, len(parameter_changes) * 40),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def create_cost_breakdown_chart(cost_data: Dict, scenarios: Dict) -> go.Figure:
    fig = go.Figure()
    
    categories = []
    eu_ets_costs = []
    national_costs = []
    
    for scenario_name in scenarios.keys():
        for state in ['baseline', 'optimized']:
            categories.append(f"{scenario_name}<br>({state.capitalize()})")
            eu_ets_costs.append(cost_data[scenario_name][state]['eu_ets_cost'])
            national_costs.append(cost_data[scenario_name][state]['national_cost'])
    
    fig.add_trace(go.Bar(
        name='EU ETS Cost',
        x=categories,
        y=eu_ets_costs,
        marker_color='#3498DB'
    ))
    
    fig.add_trace(go.Bar(
        name='National Carbon Tax',
        x=categories,
        y=national_costs,
        marker_color='#E67E22'
    ))
    
    fig.update_layout(
        title='Carbon Cost Breakdown by Pricing Mechanism',
        yaxis_title='Cost (EUR)',
        barmode='stack',
        height=400,
        showlegend=True
    )
    
    return fig

def create_scope_comparison_chart(baseline_breakdown: Dict, optimized_breakdown: Dict) -> go.Figure:
    scopes = list(baseline_breakdown.keys())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline',
        x=scopes,
        y=[baseline_breakdown[s] for s in scopes],
        marker_color='#E74C3C',
        text=[f"{baseline_breakdown[s]:.2f}" for s in scopes],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Optimized',
        x=scopes,
        y=[optimized_breakdown[s] for s in scopes],
        marker_color='#27AE60',
        text=[f"{optimized_breakdown[s]:.2f}" for s in scopes],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Emissions by Scope: Baseline vs Optimized',
        yaxis_title='Emissions (tonnes CO2e)',
        barmode='group',
        height=400
    )
    
    return fig

def display_optimization_tab(optimizer: IntegratedCarbonOptimizer):
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
            st.info(f"Using fixed seed: {RANDOM_SEED}")
            stability_runs = 1
        else:
            st.info(f"Using fixed seed: {RANDOM_SEED}")
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
                options=[100, 250, 500, 750, 1000],
                value=500,
                help="Lower = faster, Higher = more robust"
            )
        else:
            saa_samples = 0
            st.write("")
    
    with col5:
        mc_sims = st.select_slider(
            "MC Simulations",
            options=[1000, 2500, 5000, 7500, 10000],
            value=2500,
            help="For uncertainty analysis after optimization"
        )
    
    opt_count = sum(1 for p in optimizer.parameters.values()
                   if p.is_optimizable and 
                   (scope == "All" or p.scope == scope))
    
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
                stability_runs=stability_runs
            )
            st.session_state.result = result

        if result.success:
            st.success(f"{result.message}")
        else:
            st.error(f"{result.message}")

        display_optimization_results(result, use_saa)
    
    elif 'result' in st.session_state:
        st.info("Previous optimization results displayed below. Click 'Run Optimization' to update.")
        display_optimization_results(st.session_state.result, 
                                    st.session_state.result.saa_samples_used > 0)


def display_configuration_section(optimizer: IntegratedCarbonOptimizer):
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
        st.caption("Reference values:")
        st.caption("DK=0.495, DE=0.420, FR=0.056, PL=0.766")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        scope_filter = st.selectbox("Filter by Scope", ["All", "Scope1", "Scope2", "Scope3"], key="cfg_scope")
    with col2:
        show_opt_only = st.checkbox("Show Optimizable Only", value=False)
    
    filtered_params = []
    for param in optimizer.parameters.values():
        if scope_filter != "All" and param.scope != scope_filter:
            continue
        if show_opt_only and not param.is_optimizable:
            continue
        filtered_params.append(param)
    
    params_by_scope = {}
    for param in filtered_params:
        if param.scope not in params_by_scope:
            params_by_scope[param.scope] = []
        params_by_scope[param.scope].append(param)
    
    for scope_key, params in sorted(params_by_scope.items()):
        st.markdown(f"**{scope_key}**")
        
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 0.7, 0.7, 0.7])
        with col1:
            st.markdown("**Parameter**")
        with col2:
            st.markdown("**Value**")
        with col3:
            st.markdown("**Opt**")
        with col4:
            st.markdown("**Min%**")
        with col5:
            st.markdown("**Max%**")
        with col6:
            st.markdown("**Unc%**")
        
        for param in params:
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 0.7, 0.7, 0.7])
            
            with col1:
                st.write(f"{param.name}")
            
            with col2:
                st.write(f"{param.nominal_value:,.2f} {param.unit}")
            
            with col3:
                is_opt = st.checkbox("Optimize", value=param.is_optimizable, 
                                    key=f"opt_{param.stub}", label_visibility="collapsed")
                param.is_optimizable = is_opt
            
            with col4:
                if is_opt:
                    min_pct = st.number_input("Min%", min_value=-50.0, max_value=0.0,
                                            value=param.min_change_percent, step=5.0,
                                            key=f"min_{param.stub}", label_visibility="collapsed")
                    param.min_change_percent = min_pct
                else:
                    st.write("")
            
            with col5:
                if is_opt:
                    max_pct = st.number_input("Max%", min_value=0.0, max_value=50.0,
                                            value=param.max_change_percent, step=5.0,
                                            key=f"max_{param.stub}", label_visibility="collapsed")
                    param.max_change_percent = max_pct
                else:
                    st.write("")
            
            with col6:
                if is_opt:
                    unc = st.number_input("Unc%", min_value=0.0, max_value=50.0,
                                        value=param.uncertainty_percent, step=1.0,
                                        key=f"unc_{param.stub}", label_visibility="collapsed")
                    param.uncertainty_percent = unc
                else:
                    st.write("")
        
        st.markdown("")


def display_optimization_results(result: OptimizationResult, use_saa: bool):
    st.header("Optimization Results")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Baseline Emissions",
            f"{result.baseline_emissions:.3f} tonnes CO2e"
        )
    
    with col2:
        st.metric(
            "Optimized Emissions",
            f"{result.optimized_emissions:.3f} tonnes CO2e",
            delta=f"-{result.reduction_percent:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Reduction Amount",
            f"{result.reduction_amount:.3f} tonnes CO2e"
        )
    
    with col4:
        st.metric(
            "Computation Time",
            f"{result.computation_time:.2f}s"
        )
    
    # Stability Analysis Section
    if result.stability_analysis is not None:
        st.markdown("---")
        st.subheader("Algorithm Stability Analysis")
        
        stability = result.stability_analysis
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Statistical Summary**")
            summary_df = pd.DataFrame({
                'Metric': [
                    'Number of Runs',
                    'Mean Reduction',
                    'Median Reduction',
                    'Standard Deviation',
                    'Coefficient of Variation',
                    'Range (Min - Max)'
                ],
                'Value': [
                    f"{stability['runs']}",
                    f"{stability['mean_reduction']:.4f}%",
                    f"{stability['median_reduction']:.4f}%",
                    f"{stability['std_reduction']:.4f}%",
                    f"{stability['cv']:.2%}",
                    f"{stability['min_reduction']:.4f}% - {stability['max_reduction']:.4f}%"
                ]
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**95% Confidence Interval**")
            st.info(
                f"The true reduction effect is estimated to be between "
                f"**{stability['ci_95_lower']:.4f}%** and **{stability['ci_95_upper']:.4f}%** "
                f"with 95% confidence."
            )
            
            st.markdown("**Stability Assessment**")
            if stability['is_stable']:
                st.success(f"✓ Algorithm is stable (CV = {stability['cv']:.2%} < 15%)")
            else:
                st.warning(f"⚠ High variability detected (CV = {stability['cv']:.2%} ≥ 15%)")
        
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
    
    # Parameter Changes Section
    st.markdown("---")
    st.subheader("Parameter Optimization Results")
    
    if result.parameter_changes:
        # Parameter bounds visualization
        st.markdown("**Optimization Bounds Analysis**")
        fig_bounds = create_parameter_bounds_chart(result.parameter_changes)
        st.plotly_chart(fig_bounds, use_container_width=True)
        
        # Check for active bounds
        active_bounds = [p for p in result.parameter_changes if p.get('at_lower_bound') or p.get('at_upper_bound')]
        if active_bounds:
            st.warning(f"⚠ {len(active_bounds)} parameter(s) are at their optimization bounds (active constraints). "
                      "Consider relaxing bounds for potentially greater reduction.")
        
        # Detailed table
        st.markdown("**Detailed Parameter Changes**")
        changes_df = pd.DataFrame(result.parameter_changes)
        display_cols = ['parameter', 'scope', 'original', 'optimized', 'change_percent', 'unit']
        changes_df_display = changes_df[display_cols].copy()
        changes_df_display.columns = ['Parameter', 'Scope', 'Original', 'Optimized', 'Change %', 'Unit']
        
        st.dataframe(
            changes_df_display.style.format({
                'Original': '{:,.2f}',
                'Optimized': '{:,.2f}',
                'Change %': '{:+.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No parameter changes (optimization may have failed or no optimizable parameters)")
    
    # Scope Analysis Section
    st.markdown("---")
    st.subheader("Emission Analysis by Scope")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Grouped bar chart
        fig_scope = create_scope_comparison_chart(
            result.baseline_scope_breakdown, 
            result.optimized_scope_breakdown
        )
        st.plotly_chart(fig_scope, use_container_width=True)
    
    with col2:
        # Waterfall chart
        fig_waterfall = create_waterfall_chart(
            result.baseline_scope_breakdown,
            result.optimized_scope_breakdown
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Uncertainty Analysis Section
    if use_saa and result.mc_baseline and result.mc_optimized:
        st.markdown("---")
        st.subheader("Uncertainty Quantification (Monte Carlo)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistical Comparison**")
            comparison_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'CV (%)', 'P5', 'P50 (Median)', 'P95'],
                'Baseline (tonnes)': [
                    f"{result.mc_baseline['mean']:.3f}",
                    f"{result.mc_baseline['std']:.3f}",
                    f"{result.mc_baseline['cv']:.2f}",
                    f"{result.mc_baseline['p5']:.3f}",
                    f"{result.mc_baseline['p50']:.3f}",
                    f"{result.mc_baseline['p95']:.3f}"
                ],
                'Optimized (tonnes)': [
                    f"{result.mc_optimized['mean']:.3f}",
                    f"{result.mc_optimized['std']:.3f}",
                    f"{result.mc_optimized['cv']:.2f}",
                    f"{result.mc_optimized['p5']:.3f}",
                    f"{result.mc_optimized['p50']:.3f}",
                    f"{result.mc_optimized['p95']:.3f}"
                ]
            })
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**90% Confidence Interval**")
            baseline_ci = f"[{result.mc_baseline['p5']:.3f}, {result.mc_baseline['p95']:.3f}]"
            optimized_ci = f"[{result.mc_optimized['p5']:.3f}, {result.mc_optimized['p95']:.3f}]"
            st.info(f"**Baseline**: {baseline_ci} tonnes CO2e\n\n**Optimized**: {optimized_ci} tonnes CO2e")
        
        # Box plot comparison
        fig_box = create_box_plot_comparison(result.mc_baseline, result.mc_optimized)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram overlay
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=result.mc_baseline['data'],
            name='Baseline',
            opacity=0.6,
            marker_color='#E74C3C',
            nbinsx=50
        ))
        fig_mc.add_trace(go.Histogram(
            x=result.mc_optimized['data'],
            name='Optimized',
            opacity=0.6,
            marker_color='#27AE60',
            nbinsx=50
        ))
        fig_mc.update_layout(
            title="Monte Carlo Emission Distribution",
            xaxis_title="Emissions (tons CO2e)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Sensitivity Analysis
        if result.sensitivity_ranking:
            st.markdown("---")
            st.subheader("Sensitivity Analysis")
            st.markdown("*Parameters ranked by absolute Pearson correlation with total emissions*")
            
            sens_df = pd.DataFrame(result.sensitivity_ranking[:10])
            
            # Reverse order so rank #1 appears at top
            top_10_reversed = result.sensitivity_ranking[:10][::-1]
            
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Bar(
                x=[s['correlation'] for s in top_10_reversed],
                y=[s['parameter'] for s in top_10_reversed],
                orientation='h',
                marker=dict(
                    color=[s['correlation'] for s in top_10_reversed],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title='ρ')
                )
            ))
            fig_sens.update_layout(
                title="Top 10 Most Influential Parameters (Pearson Correlation)",
                xaxis_title="Correlation Coefficient (ρ)",
                yaxis_title="",
                height=400
            )
            fig_sens.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_sens, use_container_width=True)
    
    # Export Section
    st.markdown("---")
    if st.button("Export Results", use_container_width=True):
        export_data = {
            'metadata': {
                'method': result.method,
                'timestamp': result.timestamp,
                'random_seed': result.random_seed_used,
                'computation_time_seconds': result.computation_time
            },
            'optimization_results': {
                'baseline_emissions_tonnes': result.baseline_emissions,
                'optimized_emissions_tonnes': result.optimized_emissions,
                'reduction_percent': result.reduction_percent,
                'saa_samples_used': result.saa_samples_used
            },
            'stability_analysis': result.stability_analysis if result.stability_analysis else {},
            'parameter_changes': result.parameter_changes,
            'scope_breakdown': {
                'baseline': {k: v for k, v in result.baseline_scope_breakdown.items()},
                'optimized': {k: v for k, v in result.optimized_scope_breakdown.items()}
            },
            'uncertainty_analysis': {
                'baseline': {
                    'mean_tonnes': result.mc_baseline['mean'],
                    'std_tonnes': result.mc_baseline['std'],
                    'cv_percent': result.mc_baseline['cv'],
                    'p5_tonnes': result.mc_baseline['p5'],
                    'p95_tonnes': result.mc_baseline['p95']
                },
                'optimized': {
                    'mean_tonnes': result.mc_optimized['mean'] if result.mc_optimized else 0,
                    'std_tonnes': result.mc_optimized['std'] if result.mc_optimized else 0,
                    'cv_percent': result.mc_optimized['cv'] if result.mc_optimized else 0,
                    'p5_tonnes': result.mc_optimized['p5'] if result.mc_optimized else 0,
                    'p95_tonnes': result.mc_optimized['p95'] if result.mc_optimized else 0
                }
            },
            'sensitivity_ranking': result.sensitivity_ranking[:10] if result.sensitivity_ranking else []
        }
        
        st.download_button(
            "Download JSON",
            json.dumps(export_data, indent=2),
            f"optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


def display_carbon_pricing_tab(optimizer: IntegratedCarbonOptimizer):
    st.header("Carbon Pricing Analysis")
    
    with st.expander("Pricing Scenarios Configuration", expanded=True):
        col_year1, col_year2 = st.columns([1, 1])
        with col_year1:
            current_year = st.number_input("Current Year", min_value=2020, max_value=2050, value=2025, step=1, key="current_year")
        with col_year2:
            future_year = st.slider("Future Year", min_value=2030, max_value=2070, value=2040, step=5, key="future_year")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Current Scenario ({current_year})**")
            current_eu_ets = st.number_input("EU ETS Price (EUR/ton)", min_value=0.0, value=85.0, key="c_ets")
            current_eu_cov = st.number_input("EU ETS Coverage (%)", min_value=0.0, max_value=100.0, value=60.0, key="c_ets_cov")
            current_nat = st.number_input("National Carbon Tax (EUR/ton)", min_value=0.0, value=26.0, key="c_nat")
            current_nat_cov = st.number_input("National Tax Coverage (%)", min_value=0.0, max_value=100.0, value=40.0, key="c_nat_cov")
        
        with col2:
            st.markdown(f"**Future Scenario ({future_year})**")
            future_eu_ets = st.number_input("EU ETS Price (EUR/ton)", min_value=0.0, value=120.0, key="f_ets")
            future_eu_cov = st.number_input("EU ETS Coverage (%)", min_value=0.0, max_value=100.0, value=75.0, key="f_ets_cov")
            future_nat = st.number_input("National Carbon Tax (EUR/ton)", min_value=0.0, value=35.0, key="f_nat")
            future_nat_cov = st.number_input("National Tax Coverage (%)", min_value=0.0, max_value=100.0, value=25.0, key="f_nat_cov")
    
    current_scenario = CarbonPricingScenario(
        name=f"Current ({current_year})", eu_ets_price=current_eu_ets, eu_ets_coverage=current_eu_cov,
        national_tax=current_nat, national_tax_coverage=current_nat_cov
    )
    future_scenario = CarbonPricingScenario(
        name=f"Future ({future_year})", eu_ets_price=future_eu_ets, eu_ets_coverage=future_eu_cov,
        national_tax=future_nat, national_tax_coverage=future_nat_cov
    )
    scenarios = {current_scenario.name: current_scenario, future_scenario.name: future_scenario}
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Effective Price", f"EUR {current_scenario.weighted_average_price:.2f}/ton")
    with col2:
        st.metric("Future Effective Price", f"EUR {future_scenario.weighted_average_price:.2f}/ton")
    
    st.markdown("---")
    
    if 'result' in st.session_state:
        result = st.session_state.result
        display_cost_analysis(optimizer, result, scenarios, current_scenario, future_scenario)
    else:
        display_baseline_cost_sample(optimizer, scenarios)


def display_cost_analysis(optimizer, result, scenarios, current_scenario, future_scenario):
    st.subheader("Cost Impact Analysis")
    
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
            'Baseline Cost (EUR)': f"{costs['baseline']['total_cost']:,.0f}",
            'Optimized Cost (EUR)': f"{costs['optimized']['total_cost']:,.0f}",
            'Annual Savings (EUR)': f"{costs['savings']:,.0f}"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # Cost breakdown chart
    fig_breakdown = create_cost_breakdown_chart(cost_data, scenarios)
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Savings comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_compare = make_subplots(rows=1, cols=2, subplot_titles=('Annual Carbon Costs', 'Annual Savings'))
        
        scenarios_list = list(scenarios.keys())
        baseline_costs = [cost_data[s]['baseline']['total_cost'] for s in scenarios_list]
        optimized_costs = [cost_data[s]['optimized']['total_cost'] for s in scenarios_list]
        savings = [cost_data[s]['savings'] for s in scenarios_list]
        
        fig_compare.add_trace(go.Bar(name='Baseline', x=scenarios_list, y=baseline_costs, marker_color='#E74C3C'), row=1, col=1)
        fig_compare.add_trace(go.Bar(name='Optimized', x=scenarios_list, y=optimized_costs, marker_color='#27AE60'), row=1, col=1)
        fig_compare.add_trace(go.Bar(name='Savings', x=scenarios_list, y=savings, marker_color='#3498DB'), row=1, col=2)
        
        fig_compare.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    with col2:
        # Coverage visualization
        st.markdown("**Emission Coverage Analysis**")
        coverage_data = []
        for scenario_name, costs in cost_data.items():
            baseline_info = costs['baseline']
            coverage_data.append({
                'Scenario': scenario_name,
                'EU ETS Covered': f"{baseline_info['eu_ets_covered_tons']:.2f} tons",
                'National Tax Covered': f"{baseline_info['national_covered_tons']:.2f} tons",
                'Uncovered': f"{baseline_info['uncovered_tons']:.2f} tons"
            })
        st.dataframe(pd.DataFrame(coverage_data), use_container_width=True, hide_index=True)
    
    # ROI Analysis
    st.markdown("---")
    st.subheader("Return on Investment Analysis")
    
    baseline_tons = result.baseline_emissions 
    
    implementation_cost = st.number_input(
        "Implementation Cost (EUR)", 
        min_value=0.0, 
        value=500000.0,
        help="One-time cost to implement optimization measures"
    )
    
    cost_per_ton = implementation_cost / baseline_tons if baseline_tons > 0 else 0
    
    if cost_per_ton > 5000:
        st.error(f"Implementation cost ({cost_per_ton:,.0f} EUR/ton) is very high. Typical range: 50-3,000 EUR/ton.")
    elif cost_per_ton > 2000:
        st.warning(f"Implementation cost ({cost_per_ton:,.0f} EUR/ton) is suitable for major equipment investments.")
    else:
        st.success(f"Implementation cost ({cost_per_ton:,.0f} EUR/ton) is appropriate for operational improvements.")
    
    current_cost_data = cost_data[current_scenario.name]
    future_cost_data = cost_data[future_scenario.name]
    
    annual_savings_current = current_cost_data['savings']
    annual_savings_future = future_cost_data['savings']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Implementation Cost", f"EUR {implementation_cost:,.0f}")
    with col2:
        st.metric("Annual Savings (Current)", f"EUR {annual_savings_current:,.0f}")
    with col3:
        if annual_savings_current > 0:
            payback_current = implementation_cost / annual_savings_current
            st.metric("Payback Period (Current)", f"{payback_current:.1f} years")
        else:
            st.metric("Payback Period (Current)", "N/A")
    with col4:
        if annual_savings_future > 0:
            payback_future = implementation_cost / annual_savings_future
            st.metric("Payback Period (Future)", f"{payback_future:.1f} years")
        else:
            st.metric("Payback Period (Future)", "N/A")
    
    # ROI projection chart
    years = np.arange(0, 11)
    cumulative_current = years * annual_savings_current - implementation_cost
    cumulative_future = years * annual_savings_future - implementation_cost
    
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(
        x=years, y=cumulative_current, name=f'Current Prices ({current_scenario.name})',
        line=dict(color='#3498DB', width=3),
        fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    fig_roi.add_trace(go.Scatter(
        x=years, y=cumulative_future, name=f'Future Prices ({future_scenario.name})',
        line=dict(color='#27AE60', width=3),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.2)'
    ))
    fig_roi.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even Point")
    fig_roi.update_layout(
        title="Cumulative Net Savings Projection (10 Years)",
        xaxis_title="Years",
        yaxis_title="Cumulative Net Savings (EUR)",
        height=400
    )
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Export pricing analysis
    if st.button("Export Pricing Analysis", use_container_width=True):
        pricing_export = {
            'scenarios': {
                name: {
                    'eu_ets_price': s.eu_ets_price,
                    'eu_ets_coverage': s.eu_ets_coverage,
                    'national_tax': s.national_tax,
                    'national_tax_coverage': s.national_tax_coverage,
                    'weighted_average_price': s.weighted_average_price
                } for name, s in scenarios.items()
            },
            'emissions': {
                'baseline_tonnes': baseline_emissions,
                'optimized_tonnes': optimized_emissions
            },
            'costs': {
                name: {
                    'baseline_cost': c['baseline']['total_cost'],
                    'optimized_cost': c['optimized']['total_cost'],
                    'annual_savings': c['savings']
                } for name, c in cost_data.items()
            },
            'roi_analysis': {
                'implementation_cost': implementation_cost,
                'payback_current_years': implementation_cost / annual_savings_current if annual_savings_current > 0 else None,
                'payback_future_years': implementation_cost / annual_savings_future if annual_savings_future > 0 else None
            },
            'timestamp': datetime.now().isoformat()
        }
        st.download_button(
            "Download JSON",
            json.dumps(pricing_export, indent=2),
            f"pricing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


def display_baseline_cost_sample(optimizer, scenarios):
    st.markdown("### Baseline Cost Estimation")
    
    baseline = optimizer.calculate_emissions()
    baseline_emissions = baseline['total']
    
    st.info(f"Current baseline emissions: **{baseline_emissions:,.3f} tonnes CO2e/year**")
    
    sample_costs = []
    for scenario_name, scenario in scenarios.items():
        cost_info = optimizer.calculate_carbon_cost(baseline_emissions, scenario)
        sample_costs.append({
            'Scenario': scenario_name,
            'Effective Price (EUR/ton)': f"{scenario.weighted_average_price:.2f}",
            'Annual Cost (EUR)': f"{cost_info['total_cost']:,.0f}"
        })
    
    st.dataframe(pd.DataFrame(sample_costs), use_container_width=True, hide_index=True)
    st.warning("Run optimization to see cost savings analysis and ROI projections.")

class OptimizationAnalyticsModule:
    
    def __init__(self, manager):
        self.manager = manager
        self.optimizer = None
    
    def display(self):
        if not self.optimizer:
            self.optimizer = IntegratedCarbonOptimizer(manager=self.manager)
        
        tab1, tab2 = st.tabs(["Optimization", "Cost Analysis"])
        
        with tab1:
            display_optimization_tab(self.optimizer)
        
        with tab2:
            display_carbon_pricing_tab(self.optimizer)