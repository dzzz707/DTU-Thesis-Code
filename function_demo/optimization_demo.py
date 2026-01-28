import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from typing import Dict, List, Tuple, Optional
import json
from copy import deepcopy

# Import your existing model classes (simplified versions)
class DataMode:
    def __init__(self, name: str, stub: str, value: float):
        self.name = name
        self.stub = stub
        self.value = value

class FormulaMode:
    def __init__(self, name: str, stub: str, formula: str):
        self.name = name
        self.stub = stub
        self.formula = formula
        self.result = None

    def evaluate(self, variables: dict):
        try:
            self.result = eval(self.formula, {}, variables)
        except Exception as e:
            print(f"Error evaluating {self.name}: {e}")
        return self.result

class EmissionModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.data_modes = {}
        self.formula_modes = {}

    def add_data_mode(self, name, stub, value):
        mode = DataMode(name, stub, value)
        self.data_modes[stub] = mode

    def add_formula_mode(self, name, stub, formula):
        mode = FormulaMode(name, stub, formula)
        self.formula_modes[stub] = mode

    def calculate(self, external_vars=None):
        variables = {stub: mode.value for stub, mode in self.data_modes.items()}
        if external_vars:
            variables.update(external_vars)
        evaluated = set()
        for _ in range(len(self.formula_modes)):
            updated = False
            for stub, mode in self.formula_modes.items():
                if stub in evaluated:
                    continue
                result = mode.evaluate(variables)
                if result is not None:
                    variables[stub] = result
                    evaluated.add(stub)
                    updated = True
            if not updated:
                break
        return variables

class ScopeOptimizer:
    def __init__(self):
        self.models = {}
        self.setup_default_data()
    
    def setup_default_data(self):
        """Setup default emission models with realistic data"""
        # Fixed Emission (Scope 1)
        fixed_emission = EmissionModel("FixedEmission")
        fixed_emission.add_data_mode("Natural Gas Activity Data", "m3", 151091)
        fixed_emission.add_data_mode("N2O Emission Factor", "N2O_XGWP", 0.00091)
        fixed_emission.add_data_mode("CO2 Emission Factor", "CO2_XGWP", 1.87904)
        fixed_emission.add_data_mode("CH4 Emission Factor", "CH4_XGWP", 0.00093)
        fixed_emission.add_formula_mode("Natural Gas CO2e", "NaturalGasCO2e", "m3 * CO2_XGWP + m3 * CH4_XGWP + m3 * N2O_XGWP")
        fixed_emission.add_formula_mode("Fixed CO2e", "FixedCO2e", "NaturalGasCO2e")
        self.models["FixedEmission"] = fixed_emission

        # Fugitive Emission (Scope 1)
        fugitive_emission = EmissionModel("FugitiveEmission")
        fugitive_emission.add_data_mode("R134a Chiller Emission", "R134a", 231)
        fugitive_emission.add_data_mode("R134a Emission Factor", "R134a_XGWP", 130.05)
        fugitive_emission.add_data_mode("R407c Chiller Emission", "R407c", 280.5)
        fugitive_emission.add_data_mode("R407c Chiller Emission Factor", "R407c_XGWP", 162.18)
        fugitive_emission.add_data_mode("R410A Chiller Emission", "R410A_C", 96)
        fugitive_emission.add_data_mode("R410A Chiller Emission Factor", "R410A_C_XGWP", 191.718)
        fugitive_emission.add_data_mode("R410A AC Emission", "R410A_AC", 88)
        fugitive_emission.add_data_mode("R410A AC Emission Factor", "R410A_AC_XGWP", 124.053)
        fugitive_emission.add_data_mode("Extinguisher Emission", "Extinguisher", 20.5)
        fugitive_emission.add_data_mode("Extinguisher Emission Factor", "Extinguisher_XGWP", 1)
        fugitive_emission.add_data_mode("Septic Tank CH4 Emission", "SepticTankCH4_kg", 764894.2933)
        fugitive_emission.add_data_mode("CH4 Emission Factor", "SepticTankCH4_XGWP", 0.10672)
        fugitive_emission.add_formula_mode("Chiller CO2e", "ChillerCO2e", "R134a * R134a_XGWP + R407c * R407c_XGWP + R410A_C * R410A_C_XGWP")
        fugitive_emission.add_formula_mode("Building CO2e", "BuildingCO2e", "R410A_AC * R410A_AC_XGWP + Extinguisher * Extinguisher_XGWP")
        fugitive_emission.add_formula_mode("Septic Tank CO2e", "SepticTankCO2e", "SepticTankCH4_kg * SepticTankCH4_XGWP")
        fugitive_emission.add_formula_mode("Fugitive CO2e", "FugitiveCO2e", "ChillerCO2e + BuildingCO2e + SepticTankCO2e")
        self.models["FugitiveEmission"] = fugitive_emission

        # Mobile Emission (Scope 1)
        mobile_emission = EmissionModel("MobileEmission")
        mobile_emission.add_data_mode("ULP92 Consumption", "ULP92", 159.22)
        mobile_emission.add_data_mode("ULP92 CH4 Factor", "ULP92_CH4_XGWP", 0.02278)
        mobile_emission.add_data_mode("ULP92 CO2 Factor", "ULP92_CO2_XGWP", 2.263)
        mobile_emission.add_data_mode("ULP92 N2O Factor", "ULP92_N2O_XGWP", 0.07132)
        mobile_emission.add_data_mode("ULP98 Consumption", "ULP98", 5582.08)
        mobile_emission.add_data_mode("ULP98 CH4 Factor", "ULP98_CH4_XGWP", 0.02278)
        mobile_emission.add_data_mode("ULP98 CO2 Factor", "ULP98_CO2_XGWP", 2.26313)
        mobile_emission.add_data_mode("ULP98 N2O Factor", "ULP98_N2O_XGWP", 0.07132)
        mobile_emission.add_data_mode("ULP95 Consumption", "ULP95", 9217.254)
        mobile_emission.add_data_mode("ULP95 CH4 Factor", "ULP95_CH4_XGWP", 0.02278)
        mobile_emission.add_data_mode("ULP95 CO2 Factor", "ULP95_CO2_XGWP", 2.263)
        mobile_emission.add_data_mode("ULP95 N2O Factor", "ULP95_N2O_XGWP", 0.07132)
        mobile_emission.add_data_mode("Diesel Data1", "DieselData1", 1114.92)
        mobile_emission.add_data_mode("Diesel Data2", "DieselData2", 2121.581)
        mobile_emission.add_data_mode("Diesel Data3", "DieselData3", 5595.49)
        mobile_emission.add_data_mode("Diesel CH4 Factor", "Diesel_CH4_XGWP", 0.00383)
        mobile_emission.add_data_mode("Diesel CO2 Factor", "Diesel_CO2_XGWP", 2.60603)
        mobile_emission.add_data_mode("Diesel N2O Factor", "Diesel_N2O_XGWP", 0.03744)
        mobile_emission.add_data_mode("LPG Consumption", "LPGData", 3955.968)
        mobile_emission.add_data_mode("LPG CH4 Factor", "LPG_CH4_XGWP", 0.040805)
        mobile_emission.add_data_mode("LPG CO2 Factor", "LPG_CO2_XGWP", 1.75288)
        mobile_emission.add_data_mode("LPG N2O Factor", "LPG_N2O_XGWP", 0.00152)
        mobile_emission.add_formula_mode("ULP92 Mower", "ULP92_Mower", "ULP92 * (ULP92_CH4_XGWP + ULP92_CO2_XGWP + ULP92_N2O_XGWP)")
        mobile_emission.add_formula_mode("ULP98 Company Car", "ULP98_CompanyCar", "ULP98 * (ULP98_CH4_XGWP + ULP98_CO2_XGWP + ULP98_N2O_XGWP)")
        mobile_emission.add_formula_mode("ULP95 Company Car", "ULP95_CompanyCar", "ULP95 * (ULP95_CH4_XGWP + ULP95_CO2_XGWP + ULP95_N2O_XGWP)")
        mobile_emission.add_formula_mode("Motor Petrol CO2e", "MotorPetrolCO2e", "ULP92_Mower + ULP98_CompanyCar + ULP95_CompanyCar")
        mobile_emission.add_formula_mode("Diesel Company Car", "Diesel_CompanyCar", "DieselData1 * (Diesel_CH4_XGWP + Diesel_CO2_XGWP + Diesel_N2O_XGWP)")
        mobile_emission.add_formula_mode("Diesel Stacker", "Diesel_Stacker", "DieselData2 * (Diesel_CH4_XGWP + Diesel_CO2_XGWP + Diesel_N2O_XGWP)")
        mobile_emission.add_formula_mode("Diesel Truck", "Diesel_Truck", "DieselData3 * (Diesel_CH4_XGWP + Diesel_CO2_XGWP + Diesel_N2O_XGWP)")
        mobile_emission.add_formula_mode("Diesel CO2e", "DieselCO2e", "Diesel_CompanyCar + Diesel_Stacker + Diesel_Truck")
        mobile_emission.add_formula_mode("LPG CO2e", "LPGCO2e", "LPGData * (LPG_CH4_XGWP + LPG_CO2_XGWP + LPG_N2O_XGWP)")
        mobile_emission.add_formula_mode("Mobile CO2e", "MobileCO2e", "MotorPetrolCO2e + DieselCO2e + LPGCO2e")
        self.models["MobileEmission"] = mobile_emission

        # Scope2
        scope2 = EmissionModel("Scope2")
        scope2.add_data_mode("Electricity Usage", "ElectricityData", 25461600)
        scope2.add_data_mode("Electricity CO2 Factor", "Electricity_CO2_XGWP", 0.495)
        scope2.add_formula_mode("Purchased Electricity CO2e", "PurchasedElectricityCO2e", "ElectricityData * Electricity_CO2_XGWP")
        scope2.add_formula_mode("Scope 2 Total", "Scope2", "PurchasedElectricityCO2e")
        self.models["Scope2"] = scope2

        # Scope3
        scope3 = EmissionModel("Scope3")
        scope3.add_data_mode("Tap Water Usage", "TapWaterData", 100000)
        scope3.add_data_mode("Tap Water CO2 Factor", "TapWater_CO2_XGWP", 0.233)
        scope3.add_formula_mode("Tap Water CO2e", "TapWaterCO2e", "TapWaterData * TapWater_CO2_XGWP")
        scope3.add_formula_mode("Scope 3 Total", "Scope3", "TapWaterCO2e")
        self.models["Scope3"] = scope3

        # Scope1 (aggregates)
        scope1 = EmissionModel("Scope1")
        scope1.add_formula_mode("Scope 1 Total", "Scope1", "FixedCO2e + FugitiveCO2e + MobileCO2e")
        self.models["Scope1"] = scope1

    def get_scope_mapping(self):
        """Get mapping of scopes to their sub-models"""
        return {
            "Scope1": {
                "name": "Scope 1",
                "models": ["FixedEmission", "FugitiveEmission", "MobileEmission"],
                "total_var": "Scope1"
            },
            "Scope2": {
                "name": "Scope 2",
                "models": ["Scope2"],
                "total_var": "Scope2"
            },
            "Scope3": {
                "name": "Scope 3",
                "models": ["Scope3"],
                "total_var": "Scope3"
            }
        }

    def calculate_all(self):
        """Calculate all models"""
        all_vars = {}
        for model in self.models.values():
            result = model.calculate(all_vars)
            all_vars.update(result)
        return all_vars

    def get_scope_total(self, scope_name: str) -> float:
        """Get total emissions for a scope"""
        results = self.calculate_all()
        scope_mapping = self.get_scope_mapping()
        if scope_name in scope_mapping:
            total_var = scope_mapping[scope_name]["total_var"]
            return results.get(total_var, 0.0)
        return 0.0

    def get_all_scopes_total(self) -> float:
        """Get total emissions for all scopes combined"""
        results = self.calculate_all()
        scope_mapping = self.get_scope_mapping()
        total = 0.0
        for scope_info in scope_mapping.values():
            total_var = scope_info["total_var"]
            total += results.get(total_var, 0.0)
        return total

    def get_optimizable_parameters(self, scope_name: str) -> List[Dict]:
        """Get parameters that can be optimized for a scope"""
        scope_mapping = self.get_scope_mapping()
        
        # Handle "All Scopes" case
        if scope_name == "AllScopes":
            parameters = []
            for scope_key in scope_mapping.keys():
                parameters.extend(self.get_optimizable_parameters(scope_key))
            return parameters
        
        if scope_name not in scope_mapping:
            return []
        
        parameters = []
        for model_name in scope_mapping[scope_name]["models"]:
            if model_name in self.models:
                model = self.models[model_name]
                for stub, mode in model.data_modes.items():
                    # Skip parameters that are emission factors
                    if "Factor" not in mode.name:
                        parameters.append({
                            "model": model_name,
                            "stub": stub,
                            "name": mode.name,
                            "current_value": mode.value,
                            "mode": mode,
                            "scope": scope_name
                        })
        return parameters

    def sensitivity_analysis(self, scope_name: str, perturbation: float = 0.01) -> List[Dict]:
        """Perform sensitivity analysis using numerical differentiation"""
        if scope_name == "AllScopes":
            baseline_total = self.get_all_scopes_total()
        else:
            baseline_total = self.get_scope_total(scope_name)
            
        parameters = self.get_optimizable_parameters(scope_name)
        
        sensitivities = []
        
        for param in parameters:
            original_value = param["mode"].value
            
            # Positive perturbation
            param["mode"].value = original_value * (1 + perturbation)
            if scope_name == "AllScopes":
                positive_total = self.get_all_scopes_total()
            else:
                positive_total = self.get_scope_total(scope_name)
            
            # Negative perturbation
            param["mode"].value = original_value * (1 - perturbation)
            if scope_name == "AllScopes":
                negative_total = self.get_all_scopes_total()
            else:
                negative_total = self.get_scope_total(scope_name)
            
            # Restore original value
            param["mode"].value = original_value
            
            # Calculate sensitivity (derivative)
            if original_value != 0:
                sensitivity = (positive_total - negative_total) / (2 * perturbation * original_value)
                relative_sensitivity = abs(sensitivity * original_value / baseline_total) if baseline_total != 0 else 0
            else:
                sensitivity = 0
                relative_sensitivity = 0
            
            sensitivities.append({
                "model": param["model"],
                "stub": param["stub"],
                "name": param["name"],
                "current_value": original_value,
                "sensitivity": sensitivity,
                "relative_sensitivity": relative_sensitivity,
                "impact_per_percent": sensitivity * original_value * 0.01,
                "scope": param.get("scope", scope_name)
            })
        
        # Sort by relative sensitivity (descending)
        sensitivities.sort(key=lambda x: x["relative_sensitivity"], reverse=True)
        return sensitivities

    def optimize_scope(self, scope_name: str, constraints: Dict, method: str = "L-BFGS-B") -> Dict:
        """Optimize scope emissions with constraints"""
        parameters = self.get_optimizable_parameters(scope_name)
        if not parameters:
            return {"success": False, "message": "No parameters found for optimization"}
        
        # Store original values
        original_values = {}
        for param in parameters:
            original_values[param["stub"]] = param["mode"].value
        
        # Setup optimization
        x0 = []  # Initial values
        bounds = []  # Bounds for optimization
        param_map = []  # Mapping from optimization variables to parameters
        
        for param in parameters:
            stub = param["stub"]
            if stub in constraints:
                x0.append(param["current_value"])
                bounds.append((constraints[stub]["min"], constraints[stub]["max"]))
                param_map.append(param)
        
        if not x0:
            return {"success": False, "message": "No constrained parameters found"}
        
        def objective(x):
            """Objective function: minimize scope emissions"""
            # Set parameter values
            for i, param in enumerate(param_map):
                param["mode"].value = x[i]
            
            # Calculate and return scope total
            if scope_name == "AllScopes":
                return self.get_all_scopes_total()
            else:
                return self.get_scope_total(scope_name)
        
        # Run optimization
        try:
            result = minimize(
                objective, 
                x0, 
                method=method, 
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                # Get optimized values
                optimized_values = {}
                for i, param in enumerate(param_map):
                    optimized_values[param["stub"]] = result.x[i]
                
                return {
                    "success": True,
                    "original_total": objective(x0),
                    "optimized_total": result.fun,
                    "reduction": objective(x0) - result.fun,
                    "reduction_percent": ((objective(x0) - result.fun) / objective(x0)) * 100 if objective(x0) > 0 else 0,
                    "optimized_values": optimized_values,
                    "iterations": result.nit,
                    "message": result.message
                }
            else:
                return {
                    "success": False,
                    "message": f"Optimization failed: {result.message}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Optimization error: {str(e)}"
            }
        
        finally:
            # Restore original values
            for param in parameters:
                if param["stub"] in original_values:
                    param["mode"].value = original_values[param["stub"]]

def main():
    st.set_page_config(
        page_title="Carbon Emission Scope Optimizer",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("Carbon Emission Scope Optimizer")
    st.markdown("Optimize emission parameters within constraints to minimize scope-specific carbon emissions")
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = ScopeOptimizer()
    
    optimizer = st.session_state.optimizer
    
    # Scope selection with "All Scopes" option
    scope_mapping = optimizer.get_scope_mapping()
    
    # Create options dictionary with "All Scopes" first
    scope_options = {"AllScopes": "All Scopes"}
    scope_options.update({k: v["name"] for k, v in scope_mapping.items()})
    
    selected_scope = st.selectbox(
        "Select Scope to Optimize",
        list(scope_options.keys()),
        format_func=lambda x: scope_options[x]
    )
    
    if selected_scope:
        st.markdown("---")
        
        # Current emissions display
        if selected_scope == "AllScopes":
            # Show all scopes breakdown
            st.subheader("Current Emissions Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            scope1_total = optimizer.get_scope_total("Scope1")
            scope2_total = optimizer.get_scope_total("Scope2")
            scope3_total = optimizer.get_scope_total("Scope3")
            all_total = scope1_total + scope2_total + scope3_total
            
            with col1:
                st.metric("Scope 1", f"{scope1_total:,.2f} kg CO2e")
            with col2:
                st.metric("Scope 2", f"{scope2_total:,.2f} kg CO2e")
            with col3:
                st.metric("Scope 3", f"{scope3_total:,.2f} kg CO2e")
            with col4:
                st.metric("**Total**", f"{all_total:,.2f} kg CO2e")
            
            # Create pie chart for scope distribution
            scope_data = {
                "Scope": ["Scope 1", "Scope 2", "Scope 3"],
                "Emissions": [scope1_total, scope2_total, scope3_total],
                "Percentage": [
                    (scope1_total / all_total * 100) if all_total > 0 else 0,
                    (scope2_total / all_total * 100) if all_total > 0 else 0,
                    (scope3_total / all_total * 100) if all_total > 0 else 0
                ]
            }
            
            fig = px.pie(
                values=scope_data["Emissions"],
                names=scope_data["Scope"],
                title="Emissions Distribution by Scope",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Single scope display
            current_total = optimizer.get_scope_total(selected_scope)
            st.metric(f"Current {selected_scope} Emissions", f"{current_total:,.2f} kg CO2e")
        
        # Get parameters for selected scope
        parameters = optimizer.get_optimizable_parameters(selected_scope)
        
        if not parameters:
            st.warning("No optimizable parameters found for this scope")
            return
        
        # Sensitivity Analysis Section
        st.subheader("Sensitivity Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("Run Sensitivity Analysis", type="primary"):
                with st.spinner("Analyzing parameter sensitivities..."):
                    sensitivities = optimizer.sensitivity_analysis(selected_scope)
                    st.session_state.sensitivities = sensitivities
        
        with col1:
            if 'sensitivities' in st.session_state:
                sensitivities = st.session_state.sensitivities
                
                # Create sensitivity chart
                sens_df = pd.DataFrame(sensitivities[:15])  # Top 15 most sensitive
                
                fig = px.bar(
                    sens_df,
                    x='relative_sensitivity',
                    y='name',
                    orientation='h',
                    title=f"Top 15 Most Sensitive Parameters ({scope_options[selected_scope]})",
                    labels={'relative_sensitivity': 'Relative Sensitivity', 'name': 'Parameter'},
                    color='scope' if selected_scope == "AllScopes" else None,
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'] if selected_scope == "AllScopes" else None
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sensitivity table
                st.markdown("### Sensitivity Results")
                display_df = pd.DataFrame(sensitivities)
                display_df['Impact per 1% Change'] = display_df['impact_per_percent'].apply(lambda x: f"{x:,.2f} kg CO2e")
                display_df['Relative Sensitivity'] = display_df['relative_sensitivity'].apply(lambda x: f"{x:.4f}")
                
                # Add scope column for all scopes view
                columns_to_show = ['name', 'current_value', 'Relative Sensitivity', 'Impact per 1% Change']
                if selected_scope == "AllScopes":
                    columns_to_show.insert(1, 'scope')
                
                st.dataframe(
                    display_df[columns_to_show],
                    use_container_width=True,
                    hide_index=True
                )

        # Constraint setup
        st.markdown("---")
        st.markdown("#### Set Parameter Constraints")
        
        # Master checkbox for select/deselect all
        col1, col2 = st.columns([1, 5])
        with col1:
            # Calculate current state
            excluded_count = sum(1 for param in parameters if st.session_state.get(f"exclude_{param['stub']}", False))
            total_count = len(parameters)
            
            if excluded_count == 0:
                # All selected
                master_state = True
                master_label = "All parameters included"
            elif excluded_count == total_count:
                # None selected  
                master_state = False
                master_label = "All parameters excluded"
            else:
                # Partial selection - we'll handle this as False but show different text
                master_state = False
                master_label = f"{total_count - excluded_count}/{total_count} parameters included"
            
            master_checkbox = st.checkbox(
                master_label,
                value=master_state,
                key="master_param_selection",
                help="Click to toggle all parameters"
            )
            
            # Handle master checkbox changes
            if st.session_state.get("master_param_selection") != master_state:
                # User clicked the master checkbox
                if excluded_count == 0:
                    # Currently all selected, so exclude all
                    for param in parameters:
                        st.session_state[f"exclude_{param['stub']}"] = True
                else:
                    # Currently some or all excluded, so include all
                    for param in parameters:
                        st.session_state[f"exclude_{param['stub']}"] = False
                st.rerun()
        
        constraints = {}
        
        # Group parameters by scope if showing all scopes
        if selected_scope == "AllScopes":
            # Group parameters by scope for better organization
            scope_params = {}
            for param in parameters:
                scope = param.get("scope", "Unknown")
                if scope not in scope_params:
                    scope_params[scope] = []
                scope_params[scope].append(param)
            
            # Display parameters grouped by scope
            for scope, scope_param_list in scope_params.items():
                scope_name = scope_mapping.get(scope, {}).get("name", scope)
                st.markdown(f"#### {scope_name}")
                
                for param in scope_param_list:
                    with st.expander(f"{param['name']} ({param['model']})", expanded=False):
                        # Add exclude checkbox at the top
                        col_exclude, col_spacer = st.columns([1, 3])
                        with col_exclude:
                            is_excluded = st.checkbox(
                                "Exclude from optimization",
                                value=st.session_state.get(f"exclude_{param['stub']}", False),
                                key=f"exclude_{param['stub']}",
                                help="Check to exclude this parameter from optimization"
                            )
                        
                        if not is_excluded:
                            col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 2])
                            
                            with col1:
                                st.metric("Current Value", f"{param['current_value']:,.2f}")
                            
                            with col2:
                                decrease_percent = st.number_input(
                                    "Decrease %",
                                    value=10.0,
                                    min_value=0.0,
                                    max_value=100.0,
                                    step=1.0,
                                    format="%.1f",
                                    key=f"decrease_pct_{param['stub']}",
                                    help="Maximum percentage decrease (positive value)"
                                )
                            
                            with col3:
                                increase_percent = st.number_input(
                                    "Increase %", 
                                    value=10.0,
                                    min_value=0.0,
                                    max_value=200.0,
                                    step=1.0,
                                    format="%.1f",
                                    key=f"increase_pct_{param['stub']}",
                                    help="Maximum percentage increase (positive value)"
                                )
                            
                            with col4:
                                # Calculate actual values based on percentages
                                min_val = param['current_value'] * (1 - decrease_percent/100)
                                max_val = param['current_value'] * (1 + increase_percent/100)
                                
                                st.markdown("**Range:**")
                                st.text(f"Min: {min_val:,.2f}")
                                st.text(f"Max: {max_val:,.2f}")
                                st.text(f"(-{decrease_percent:.1f}% / +{increase_percent:.1f}%)")
                            
                            # Add to constraints if not excluded and percentages are not zero
                            if decrease_percent != 0 or increase_percent != 0:
                                constraints[param['stub']] = {
                                    "min": max(0, min_val),  # Ensure non-negative
                                    "max": max_val,
                                    "name": param['name'],
                                    "decrease_percent": decrease_percent,
                                    "increase_percent": increase_percent,
                                    "scope": param.get('scope', scope)
                                }
                        else:
                            # Show parameter info but grayed out
                            st.markdown(f"*Parameter excluded from optimization*")
                            st.text(f"Current Value: {param['current_value']:,.2f}")
        else:
            # Single scope - original layout
            for i, param in enumerate(parameters):
                with st.expander(f"{param['name']} ({param['model']})", expanded=False):
                    # Add exclude checkbox at the top
                    col_exclude, col_spacer = st.columns([1, 3])
                    with col_exclude:
                        is_excluded = st.checkbox(
                            "Exclude from optimization",
                            value=st.session_state.get(f"exclude_{param['stub']}", False),
                            key=f"exclude_{param['stub']}",
                            help="Check to exclude this parameter from optimization"
                        )
                    
                    if not is_excluded:
                        col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 2])
                        
                        with col1:
                            st.metric("Current Value", f"{param['current_value']:,.2f}")
                        
                        with col2:
                            decrease_percent = st.number_input(
                                "Decrease %",
                                value=10.0,
                                min_value=0.0,
                                max_value=100.0,
                                step=1.0,
                                format="%.1f",
                                key=f"decrease_pct_{param['stub']}",
                                help="Maximum percentage decrease (positive value)"
                            )
                        
                        with col3:
                            increase_percent = st.number_input(
                                "Increase %", 
                                value=10.0,
                                min_value=0.0,
                                max_value=200.0,
                                step=1.0,
                                format="%.1f",
                                key=f"increase_pct_{param['stub']}",
                                help="Maximum percentage increase (positive value)"
                            )
                        
                        with col4:
                            # Calculate actual values based on percentages
                            min_val = param['current_value'] * (1 - decrease_percent/100)
                            max_val = param['current_value'] * (1 + increase_percent/100)
                            
                            st.markdown("**Range:**")
                            st.text(f"Min: {min_val:,.2f}")
                            st.text(f"Max: {max_val:,.2f}")
                            st.text(f"(-{decrease_percent:.1f}% / +{increase_percent:.1f}%)")
                        
                        # Add to constraints if not excluded and percentages are not zero
                        if decrease_percent != 0 or increase_percent != 0:
                            constraints[param['stub']] = {
                                "min": max(0, min_val),  # Ensure non-negative
                                "max": max_val,
                                "name": param['name'],
                                "decrease_percent": decrease_percent,
                                "increase_percent": increase_percent
                            }
                    else:
                        # Show parameter info but grayed out
                        st.markdown(f"*Parameter excluded from optimization*")
                        st.text(f"Current Value: {param['current_value']:,.2f}")
        
        # Optimization controls
        st.markdown("#### Optimization Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["L-BFGS-B", "SLSQP", "TNC"],
                help="L-BFGS-B is recommended for most cases"
            )
        
        with col2:
            st.metric("Parameters to Optimize", len(constraints))
        
        with col3:
            optimization_button_text = "Run Optimization" if selected_scope == "AllScopes" else "Run Optimization"
            if st.button(optimization_button_text, type="primary", disabled=len(constraints) == 0):
                if constraints:
                    with st.spinner("Running optimization..."):
                        result = optimizer.optimize_scope(selected_scope, constraints, optimization_method)
                        st.session_state.optimization_result = result
                else:
                    st.warning("Please set constraints for at least one parameter")
        
        # Optimization Results
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            
            st.markdown("---")
            st.subheader("Optimization Results")
            
            if result["success"]:

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Emissions", f"{result['original_total']:,.2f} kg CO2e")
                with col2:
                    st.metric("Optimized Emissions", f"{result['optimized_total']:,.2f} kg CO2e")

                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Reduction", f"{result['reduction']:,.2f} kg CO2e")
                with col4:
                    st.metric("Reduction %", f"{result['reduction_percent']:.2f}%")
                
                # Show scope-wise breakdown for All Scopes optimization
                if selected_scope == "AllScopes":
                    st.markdown("#### Scope-wise Impact Analysis")
                    
                    # Calculate original scope totals
                    original_scope1 = optimizer.get_scope_total("Scope1")
                    original_scope2 = optimizer.get_scope_total("Scope2") 
                    original_scope3 = optimizer.get_scope_total("Scope3")
                    
                    # Apply optimized values temporarily to calculate new totals
                    original_values = {}
                    for param in parameters:
                        if param['stub'] in result['optimized_values']:
                            original_values[param['stub']] = param['mode'].value
                            param['mode'].value = result['optimized_values'][param['stub']]
                    
                    # Calculate optimized scope totals
                    optimized_scope1 = optimizer.get_scope_total("Scope1")
                    optimized_scope2 = optimizer.get_scope_total("Scope2")
                    optimized_scope3 = optimizer.get_scope_total("Scope3")
                    
                    # Restore original values
                    for param in parameters:
                        if param['stub'] in original_values:
                            param['mode'].value = original_values[param['stub']]
                    
                    # Create comparison chart
                    scope_comparison = pd.DataFrame({
                        'Scope': ['Scope 1', 'Scope 2', 'Scope 3'],
                        'Original': [original_scope1, original_scope2, original_scope3],
                        'Optimized': [optimized_scope1, optimized_scope2, optimized_scope3],
                        'Reduction': [
                            original_scope1 - optimized_scope1,
                            original_scope2 - optimized_scope2,
                            original_scope3 - optimized_scope3
                        ]
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Original',
                        x=scope_comparison['Scope'],
                        y=scope_comparison['Original'],
                        marker_color='lightcoral'
                    ))
                    fig.add_trace(go.Bar(
                        name='Optimized',
                        x=scope_comparison['Scope'],
                        y=scope_comparison['Optimized'],
                        marker_color='lightgreen'
                    ))
                    
                    fig.update_layout(
                        title="Scope-wise Emissions: Original vs Optimized",
                        xaxis_title="Scopes",
                        yaxis_title="Emissions (kg CO2e)",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Scope breakdown table
                    st.markdown("##### Scope Breakdown")
                    scope_comparison['Reduction %'] = scope_comparison.apply(
                        lambda row: (row['Reduction'] / row['Original'] * 100) if row['Original'] > 0 else 0, axis=1
                    )
                    
                    display_scope_df = scope_comparison.copy()
                    for col in ['Original', 'Optimized', 'Reduction']:
                        display_scope_df[col] = display_scope_df[col].apply(lambda x: f"{x:,.2f}")
                    display_scope_df['Reduction %'] = display_scope_df['Reduction %'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(display_scope_df, use_container_width=True, hide_index=True)
                
                # Show optimized parameter values
                st.markdown("#### Optimized Parameter Values")
                
                comparison_data = []
                for param in parameters:
                    if param['stub'] in result['optimized_values']:
                        comparison_data.append({
                            "Parameter": param['name'],
                            "Model": param['model'],
                            "Scope": param.get('scope', 'N/A'),
                            "Original": param['current_value'],
                            "Optimized": result['optimized_values'][param['stub']],
                            "Change": result['optimized_values'][param['stub']] - param['current_value'],
                            "Change %": ((result['optimized_values'][param['stub']] - param['current_value']) / param['current_value']) * 100 if param['current_value'] != 0 else 0
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Original',
                        x=comp_df['Parameter'],
                        y=comp_df['Original'],
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Optimized',
                        x=comp_df['Parameter'],
                        y=comp_df['Optimized'],
                        marker_color='darkblue'
                    ))
                    
                    fig.update_layout(
                        title="Parameter Values: Original vs Optimized",
                        xaxis_title="Parameters",
                        yaxis_title="Value",
                        barmode='group',
                        height=400,
                        xaxis={'tickangle': 45}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed comparison table
                    st.markdown("#### Detailed Comparison")
                    display_comp_df = comp_df.copy()
                    display_comp_df['Original'] = display_comp_df['Original'].apply(lambda x: f"{x:,.2f}")
                    display_comp_df['Optimized'] = display_comp_df['Optimized'].apply(lambda x: f"{x:,.2f}")
                    display_comp_df['Change'] = display_comp_df['Change'].apply(lambda x: f"{x:+,.2f}")
                    display_comp_df['Change %'] = display_comp_df['Change %'].apply(lambda x: f"{x:+.2f}%")
                    
                    # Show scope column for all scopes view
                    columns_to_show = ['Parameter', 'Model', 'Original', 'Optimized', 'Change', 'Change %']
                    if selected_scope == "AllScopes":
                        columns_to_show.insert(2, 'Scope')
                    
                    st.dataframe(display_comp_df[columns_to_show], use_container_width=True, hide_index=True)

                if selected_scope == "AllScopes":
                    st.success(f"{optimization_method} Algorithm: Optimization across all scopes completed successfully!")
                else:
                    st.success(f"{optimization_method} Algorithm: {scope_options[selected_scope]} optimization completed successfully!")

                # Export results
                if st.button("Export Optimization Results", key="export_results"):
                    export_data = {
                        "scope": selected_scope,
                        "scope_name": scope_options[selected_scope],
                        "optimization_method": optimization_method,
                        "results": result,
                        "parameter_comparison": comparison_data,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    # Add scope breakdown for all scopes
                    if selected_scope == "AllScopes":
                        export_data["scope_breakdown"] = scope_comparison.to_dict('records')
                    
                    st.download_button(
                        label="Download Results (JSON)",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"{selected_scope}_optimization_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            else:
                st.error(f"Optimization failed: {result['message']}")

if __name__ == "__main__":
    main()