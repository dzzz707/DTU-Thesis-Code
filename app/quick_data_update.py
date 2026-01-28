import streamlit as st
import pandas as pd
import ast
import json
from typing import Dict, List
from datetime import datetime
from model_manager import get_model_scope, is_total_model

class QuickDataUpdateModule:
    
    def __init__(self, manager):
        self.manager = manager
        # Initialize change history if not exists
        if 'change_history' not in st.session_state:
            st.session_state.change_history = []
    
    def display(self):
        st.header("Quick Data Update")
        st.markdown("Quickly modify emission factors and activity data without changing calculation logic")
        
        # Add function selection buttons
        col1, col2, col3 = st.columns(3)
                
        with col1:
            if st.button("Update by Scope", use_container_width=True):
                st.session_state.selected_function = "scope_update"

        with col2:
            if st.button("Parameter Management", use_container_width=True):
                st.session_state.selected_function = "parameter_management"

        with col3:
            if st.button("Quick Overview", use_container_width=True):
                st.session_state.selected_function = "overview"
        
        st.markdown("---")
        
        # Display selected function
        if st.session_state.selected_function == "scope_update" or st.session_state.selected_function is None:
            self.display_scope_based_update()
        elif st.session_state.selected_function == "parameter_management":
            self.display_parameter_management() 
        elif st.session_state.selected_function == "overview":
            self.display_enhanced_overview()
    
    def log_change(self, model_name: str, parameter_name: str, old_value: float, new_value: float, variable_stub: str):
        change_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "parameter": parameter_name,
            "variable": variable_stub,
            "old_value": old_value,
            "new_value": new_value,
            "change": new_value - old_value,
            "change_percent": ((new_value - old_value) / old_value * 100) if old_value != 0 else float('inf')
        }
        st.session_state.change_history.append(change_record)
        
        # Keep only last 100 records to prevent excessive memory usage
        if len(st.session_state.change_history) > 100:
            st.session_state.change_history = st.session_state.change_history[-100:]
    
    def get_scope_structure(self) -> Dict:
        # Initialize scope structure
        scope_structure = {}
        
        # First, identify all Total models (e.g., Scope1_Total, Scope2_Total)
        total_models = {}
        source_models = {"Scope1": [], "Scope2": [], "Scope3": []}
        
        for model_name in self.manager.models:
            scope = get_model_scope(model_name)
            
            if scope in ["Scope1", "Scope2", "Scope3"]:
                if is_total_model(model_name):
                    total_models[scope] = model_name
                else:
                    source_models[scope].append(model_name)
        
        # Build scope structure for each detected scope
        name_mapping = {
            "Scope1": "Scope 1 - Direct Emissions",
            "Scope2": "Scope 2 - Purchased Electricity", 
            "Scope3": "Scope 3 - Other Indirect Emissions"
        }
        
        for scope_key in ["Scope1", "Scope2", "Scope3"]:
            # Only include scope if it has models
            has_total = scope_key in total_models
            has_sources = len(source_models[scope_key]) > 0
            
            if has_total or has_sources:
                scope_structure[scope_key] = {
                    "name": name_mapping.get(scope_key, f"{scope_key} - Auto-detected"),
                    "icon": "",
                    "total_model": total_models.get(scope_key),
                    "sub_models": source_models[scope_key],
                    "sub_results": []
                }
                
                # Analyze formulas to find sub-results from total model
                if has_total:
                    total_model_name = total_models[scope_key]
                    total_model = self.manager.models[total_model_name]
                    
                    for stub, mode in total_model.formula_modes.items():
                        # Parse variables referenced in formula
                        formula_vars = self.extract_variables(mode.formula)
                        scope_structure[scope_key]["sub_results"] = list(formula_vars)
        
        return scope_structure
    
    def extract_variables(self, expr: str) -> set:
        try:
            return {node.id for node in ast.walk(ast.parse(expr)) if isinstance(node, ast.Name)}
        except:
            return set()
    
    def show_confirmation_dialog(self):
        @st.dialog("Confirm Parameter Changes")
        def confirmation_dialog():
            st.markdown("### The following parameters will be changed:")
            
            # Display changes in a table
            for change in st.session_state.pending_changes:
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.text(f"{change['model']}.{change['parameter']}")
                    with col2:
                        st.text(f"{change['old_value']:.6f}")
                    with col3:
                        st.text(f"{change['new_value']:.6f}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Confirm Changes", type="primary", use_container_width=True):
                    # Apply all changes
                    for change in st.session_state.pending_changes:
                        # Log the change
                        self.log_change(change['model'], change['parameter'], 
                                      change['old_value'], change['new_value'], change['variable'])
                        # Apply the change
                        change['mode'].value = change['new_value']
                    
                    # Calculate and clean up
                    self.manager.calculate_all()
                    st.session_state.show_confirmation = False
                    st.session_state.pending_changes = []
                    st.success("Parameters updated successfully!")
                    st.rerun()
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_confirmation = False
                    st.session_state.pending_changes = []
                    st.rerun()
        
        confirmation_dialog()
    
    @st.dialog("Confirm Parameter Addition")
    def show_add_parameter_confirmation(self):
        pending_add = st.session_state.get('pending_add_parameter', {})
        
        if not pending_add:
            st.error("No pending parameter addition found.")
            return
        
        st.markdown("### Parameter to be Added:")
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Model: {pending_add['model']}")
            st.text(f"Name: {pending_add['name']}")
        with col2:
            st.text(f"Variable: {pending_add['stub']}")
            st.text(f"Value: {pending_add['value']}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_add_confirm = False
                if 'pending_add_parameter' in st.session_state:
                    del st.session_state.pending_add_parameter
                st.rerun()
        
        with col2:
            if st.button("Confirm Add", use_container_width=True, type="primary"):
                try:
                    model = self.manager.models[pending_add['model']]
                    model.add_data_mode(pending_add['name'], pending_add['stub'], pending_add['value'])
                    
                    # Log the addition
                    self.log_change(pending_add['model'], pending_add['name'], 0.0, pending_add['value'], pending_add['stub'])
                    
                    # Clear confirmation state
                    st.session_state.show_add_confirm = False
                    if 'pending_add_parameter' in st.session_state:
                        del st.session_state.pending_add_parameter
                    
                    st.success(f"Added parameter '{pending_add['name']}' to {pending_add['model']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding parameter: {str(e)}")

    @st.dialog("Confirm Parameter Deletion")
    def show_delete_parameter_confirmation(self):
        pending_delete = st.session_state.get('pending_delete_parameter', {})
        
        if not pending_delete:
            st.error("No pending parameter deletion found.")
            return
        
        st.markdown("### Parameter to be Deleted:")
        st.text(f"Model: {pending_delete['model']}")
        st.text(f"Name: {pending_delete['name']}")
        st.text(f"Variable: {pending_delete['stub']}")
        st.text(f"Current Value: {pending_delete['value']}")
        
        st.warning("This action cannot be undone!")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_delete_confirm = False
                if 'pending_delete_parameter' in st.session_state:
                    del st.session_state.pending_delete_parameter
                st.rerun()
        
        with col2:
            if st.button("Confirm Delete", use_container_width=True, type="primary"):
                try:
                    model = self.manager.models[pending_delete['model']]
                    
                    # Log the deletion
                    self.log_change(pending_delete['model'], pending_delete['name'], 
                                pending_delete['value'], 0.0, pending_delete['stub'])
                    
                    # Delete the parameter
                    del model.data_modes[pending_delete['stub']]
                    
                    # Clear confirmation state
                    st.session_state.show_delete_confirm = False
                    if 'pending_delete_parameter' in st.session_state:
                        del st.session_state.pending_delete_parameter
                    
                    st.success(f"Deleted parameter '{pending_delete['name']}' from {pending_delete['model']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting parameter: {str(e)}")

    def display_calculation_summary(self):
        if not hasattr(st.session_state, 'scope_selector_quick') or not st.session_state.scope_selector_quick:
            return
        
        selected_scope = st.session_state.scope_selector_quick
        scope_mapping = self.get_scope_structure()
        
        if selected_scope and selected_scope in scope_mapping:
            scope_info = scope_mapping[selected_scope]
            
            # Display calculation summary
            st.markdown("---")
            st.subheader("Calculation Summary")
            
            # Get Scope total from the total model
            scope_result = None
            total_model_name = scope_info.get('total_model')
            
            if total_model_name and total_model_name in self.manager.models:
                total_model = self.manager.models[total_model_name]
                for stub, mode in total_model.formula_modes.items():
                    if mode.result is not None:
                        scope_result = mode.result
                        break
            
            if scope_result is not None:
                # Display total and components
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(f"{selected_scope} Total", f"{scope_result:.2f} tonnes CO2e")
                
                with col2:
                    # Display component breakdown
                    sub_results = []
                    for var in scope_info['sub_results']:
                        for model_name, model in self.manager.models.items():
                            for stub, mode in model.formula_modes.items():
                                if stub == var and mode.result is not None:
                                    sub_results.append({
                                        "name": mode.name,
                                        "value": mode.result,
                                        "percentage": (mode.result / scope_result * 100) if scope_result > 0 else 0
                                    })
                                    break
                    
                    if sub_results:
                        st.markdown("**Component Breakdown:**")
                        for item in sub_results:
                            st.text(f"• {item['name']}: {item['value']:.2f} tonnes ({item['percentage']:.1f}%)")

    def display_scope_based_update(self):
        if not self.manager:
            st.error("Model not initialized")
            return

        st.subheader("Update Mode Selection")
        update_mode = st.radio(
            "Choose update method:",
            ["Manual Update", "JSON File Upload"],
            horizontal=True,
            key="update_mode_selector"
        )
        
        st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

        if update_mode == "Manual Update":
            self.display_manual_update()
        else:
            self.display_json_update()
        
        self.display_calculation_summary()

    def display_manual_update(self):
        scope_mapping = self.get_scope_structure()
        
        if not scope_mapping:
            st.warning("No scope models detected. Please ensure your model names follow the naming convention (Scope1_*, Scope2_*, Scope3_*)")
            return
        
        # Scope selection
        scope_names = list(scope_mapping.keys())
        selected_scope = st.selectbox(
            "Select Emission Scope",
            scope_names,
            format_func=lambda x: scope_mapping[x]['name'],
            key="scope_selector_quick"
        )
        
        if selected_scope:
            scope_info = scope_mapping[selected_scope]
            
            # Collect all data parameters grouped by model
            model_data = {}
            
            # Collect parameters from sub-models
            for sub_model_name in scope_info['sub_models']:
                if sub_model_name in self.manager.models:
                    model = self.manager.models[sub_model_name]
                    if model.data_modes:
                        model_data[sub_model_name] = []
                        for stub, mode in model.data_modes.items():
                            model_data[sub_model_name].append({
                                "mode": mode,
                                "stub": stub,
                                "key": f"{sub_model_name}_{stub}"
                            })
            
            # Also check Total model for data parameters
            total_model_name = scope_info.get('total_model')
            if total_model_name and total_model_name in self.manager.models:
                total_model = self.manager.models[total_model_name]
                if total_model.data_modes:
                    model_data[total_model_name] = []
                    for stub, mode in total_model.data_modes.items():
                        model_data[total_model_name].append({
                            "mode": mode,
                            "stub": stub,
                            "key": f"{total_model_name}_{stub}"
                        })
            
            if model_data:
                st.subheader("Edit Parameters")
                
                # Create form for batch updates
                with st.form(key="quick_update_form"):
                    updates = {}
                    
                    # Use expander to display each model's parameters
                    for model_name, params in model_data.items():
                        # Calculate model total if available
                        model_total = None
                        if model_name in self.manager.models:
                            model_obj = self.manager.models[model_name]
                            for stub, mode in model_obj.formula_modes.items():
                                if stub.endswith("CO2e") and mode.result:
                                    model_total = mode.result
                                    break
                        
                        # Create expander title
                        expander_title = f"{model_name}"
                        if model_total is not None:
                            expander_title += f" (Total: {model_total:.2f} tonnes CO2e)"
                        
                        # Use expander for each model's parameters
                        with st.expander(expander_title, expanded=(model_name == total_model_name)):
                            for param_data in params:
                                mode = param_data['mode']
                                key = param_data['key']
                                
                                col1, col2, col3 = st.columns([3, 2, 1])
                                
                                with col1:
                                    st.text(f"{mode.name}")
                                
                                with col2:
                                    new_value = st.number_input(
                                        f"Value",
                                        value=float(mode.value),
                                        format="%.6f",
                                        key=f"quick_{key}",
                                        label_visibility="collapsed"
                                    )
                                    updates[key] = (mode, new_value, model_name)
                                
                                with col3:
                                    st.text(f"{mode.stub}")
                    
                    st.markdown("---")
                    submitted = st.form_submit_button("Save Changes", type="primary", use_container_width=True)
                    
                    if submitted:
                        # Check for changes and show confirmation dialog
                        changes_to_make = []
                        for key, (mode, new_value, model_name) in updates.items():
                            if new_value != mode.value:
                                changes_to_make.append({
                                    "model": model_name,
                                    "parameter": mode.name,
                                    "variable": mode.stub,
                                    "old_value": mode.value,
                                    "new_value": new_value,
                                    "mode": mode
                                })
                        
                        if changes_to_make:
                            # Show confirmation dialog using session state
                            st.session_state.pending_changes = changes_to_make
                            st.session_state.show_confirmation = True
                            st.rerun()
                        else:
                            st.info("No changes detected")
                
        # Show confirmation dialog if needed
        if hasattr(st.session_state, 'show_confirmation') and st.session_state.show_confirmation:
            self.show_confirmation_dialog()

    def display_json_update(self):
        st.subheader("JSON File Upload")
        
        uploaded_file = st.file_uploader(
            "Upload JSON parameter file",
            type=['json'],
            help="Upload a JSON file with parameter updates"
        )
        
        if uploaded_file:
            try:
                json_data = json.load(uploaded_file)
                
                st.markdown("### File Preview")
                with st.expander("View uploaded data", expanded=True):
                    st.json(json_data)
                
                # Validate JSON format
                validation_result = self.validate_json_format(json_data)
                
                if validation_result["valid"]:
                    st.success("JSON format is valid!")
                    
                    updates_preview = []
                    unchanged_count = 0 

                    for model_name, params in json_data.items():
                        if model_name in self.manager.models:
                            model = self.manager.models[model_name]
                            for param_stub, new_value in params.items():
                                if param_stub in model.data_modes:
                                    old_value = model.data_modes[param_stub].value
                                    param_name = model.data_modes[param_stub].name
                                    
                                    # Log the change if values differ
                                    if old_value != new_value:
                                        updates_preview.append({
                                            "Model": model_name,
                                            "Parameter": param_name,
                                            "Variable": param_stub,
                                            "Current Value": f"{old_value:.6f}",
                                            "New Value": f"{new_value:.6f}",
                                            "Change": f"{new_value - old_value:+.6f}"
                                        })
                                    else:
                                        unchanged_count += 1

                    if updates_preview:
                        st.markdown("### Changed Parameters Only")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Parameters to Update", len(updates_preview))
                        with col2:
                            st.metric("Unchanged Parameters", unchanged_count)
                        with col3:
                            total_params = len(updates_preview) + unchanged_count
                            st.metric("Total Parameters", total_params)
                        
                        df_preview = pd.DataFrame(updates_preview)
                        st.dataframe(df_preview, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Apply JSON Updates", type="primary", use_container_width=True):
                                self.apply_json_updates(json_data)
                        
                        with col2:
                            if st.button("Cancel", use_container_width=True):
                                st.rerun()
                    else:
                        if unchanged_count > 0:
                            st.info(f"All {unchanged_count} parameters in the JSON file have the same values as current parameters. No changes needed.")
                        else:
                            st.warning("No valid parameters found for update")
                            
                else:
                    st.error("Invalid JSON format!")
                    for error in validation_result["errors"]:
                        st.error(f"- {error}")
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON file format")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.markdown("### JSON Format Guidelines")
            st.markdown("""
            Upload a JSON file with the following structure:
            - Top level: Model names as keys (must follow naming convention: Scope1_*, Scope2_*, Scope3_*)
            - Second level: Parameter variable stubs as keys with new values
            """)
            
            # Provide download link for example JSON
            st.markdown("### Export Current Parameters")
            if st.button("Download Current Parameters as JSON", use_container_width=True):
                current_data = self.export_current_parameters_json()
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(current_data, indent=2),
                    file_name="current_parameters.json",
                    mime="application/json"
                )    

    def display_parameter_management(self):
        # Sub-function selection using buttons
        st.markdown("### Operation Selection")
        operation_mode = st.radio(
            "Choose operation:",
            ["Add Parameter", "Delete Parameter", "Edit Parameter"],
            horizontal=True,
            key="param_operation_selector"
        )
        
        st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

        if operation_mode == "Add Parameter":
            self.display_add_parameter()
        elif operation_mode == "Delete Parameter":
            self.display_delete_parameter()
        else:
            self.display_edit_parameter()

    def display_add_parameter(self):
        st.markdown("### Add New Parameter")
        
        # Model selection
        model_options = list(self.manager.models.keys())
        selected_model = st.selectbox("Select Model", model_options, key="add_param_model")
        
        if selected_model:
            with st.form("add_parameter_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    param_name = st.text_input("Parameter Name", placeholder="e.g., Solar Panel Efficiency")
                    param_stub = st.text_input("Variable Stub", placeholder="e.g., Solar_Efficiency")
                    
                with col2:
                    param_value = st.number_input("Default Value", value=0.0, format="%.6f")
                    param_unit = st.text_input("Unit", placeholder="e.g., kWh/m2")
                
                param_description = st.text_area("Description (Optional)", placeholder="Brief description of this parameter")
                
                submitted = st.form_submit_button("Add Parameter", type="primary")
                
                if submitted:
                    if not param_name or not param_stub:
                        st.error("Parameter name and variable stub are required")
                    else:
                        # Check if stub already exists
                        existing_vars = set()
                        for m in self.manager.models.values():
                            existing_vars.update(m.data_modes.keys())
                            existing_vars.update(m.formula_modes.keys())
                        
                        if param_stub in existing_vars:
                            st.error(f"Variable stub '{param_stub}' already exists")
                        else:
                            # Store pending addition for confirmation
                            st.session_state.pending_add_parameter = {
                                'model': selected_model,
                                'name': param_name,
                                'stub': param_stub,
                                'value': param_value
                            }
                            st.session_state.show_add_confirm = True
                            st.rerun()

        # Show confirmation dialog if needed
        if hasattr(st.session_state, 'show_add_confirm') and st.session_state.show_add_confirm:
            self.show_add_parameter_confirmation()

    def display_delete_parameter(self):
        st.markdown("### Delete Parameter")
        
        # Model selection
        model_options = list(self.manager.models.keys())
        selected_model = st.selectbox("Select Model", model_options, key="delete_param_model")
        
        if selected_model:
            model = self.manager.models[selected_model]
            
            if not model.data_modes:
                st.info("No parameters found in this model")
                return
            
            # Parameter selection
            param_options = [(stub, mode.name) for stub, mode in model.data_modes.items()]
            selected_param = st.selectbox(
                "Select Parameter to Delete",
                param_options,
                format_func=lambda x: f"{x[1]} ({x[0]})",
                key="delete_param_selection"
            )
            
            if selected_param:
                stub, name = selected_param
                mode = model.data_modes[stub]
                
                # Show parameter details
                with st.expander("Parameter Details", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Name: {mode.name}")
                        st.text(f"Variable: {stub}")
                    with col2:
                        st.text(f"Current Value: {mode.value}")
                
                # Check dependencies
                dependencies = self.check_parameter_dependencies(stub)
                if dependencies:
                    st.warning("This parameter is used in the following formulas:")
                    for dep in dependencies:
                        st.text(f"• {dep['model']}.{dep['formula']} uses {stub}")
                    st.error("Cannot delete parameter that is used in formulas")
                else:
                    st.success("This parameter is safe to delete (no dependencies found)")
                    
                    if st.button("Delete Parameter", type="primary", use_container_width=True):
                        # Store pending deletion for confirmation
                        st.session_state.pending_delete_parameter = {
                            'model': selected_model,
                            'name': mode.name,
                            'stub': stub,
                            'value': mode.value
                        }
                        st.session_state.show_delete_confirm = True
                        st.rerun()
        # Show confirmation dialog if needed
        if hasattr(st.session_state, 'show_delete_confirm') and st.session_state.show_delete_confirm:
            self.show_delete_parameter_confirmation()

    def display_edit_parameter(self):
        st.markdown("### Edit Parameter Properties")
        
        # Model selection
        model_options = list(self.manager.models.keys())
        selected_model = st.selectbox("Select Model", model_options, key="edit_param_model")
        
        if selected_model:
            model = self.manager.models[selected_model]
            
            if not model.data_modes:
                st.info("No parameters found in this model")
                return
            
            # Parameter selection
            param_options = [(stub, mode.name) for stub, mode in model.data_modes.items()]
            selected_param = st.selectbox(
                "Select Parameter to Edit",
                param_options,
                format_func=lambda x: f"{x[1]} ({x[0]})",
                key="edit_param_selection"
            )
            
            if selected_param:
                stub, current_name = selected_param
                mode = model.data_modes[stub]
                
                with st.form("edit_parameter_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        new_name = st.text_input("Parameter Name", value=mode.name)
                        new_stub = st.text_input("Variable Stub", value=stub)
                        
                    with col2:
                        current_value = st.number_input("Current Value", value=float(mode.value), format="%.6f")
                        new_unit = st.text_input("Unit", placeholder="e.g., kg CO2e/kWh")
                    
                    new_description = st.text_area("Description", placeholder="Parameter description")
                    
                    submitted = st.form_submit_button("Update Parameter", type="primary")
                    
                    if submitted:
                        if not new_name or not new_stub:
                            st.error("Parameter name and variable stub are required")
                        else:
                            # Check if new stub conflicts with existing ones
                            if new_stub != stub:
                                existing_vars = set()
                                for m in self.manager.models.values():
                                    existing_vars.update(m.data_modes.keys())
                                    existing_vars.update(m.formula_modes.keys())
                                
                                if new_stub in existing_vars:
                                    st.error(f"Variable stub '{new_stub}' already exists")
                                    return
                                
                                # Check dependencies if changing stub
                                dependencies = self.check_parameter_dependencies(stub)
                                if dependencies:
                                    st.error("Cannot change variable stub for parameter used in formulas:")
                                    for dep in dependencies:
                                        st.text(f"• {dep['model']}.{dep['formula']}")
                                    return
                            
                            try:
                                # Update parameter properties
                                if new_stub != stub:
                                    # Create new parameter with new stub
                                    new_mode = type(mode)(new_name, new_stub, current_value)
                                    model.data_modes[new_stub] = new_mode
                                    # Remove old parameter
                                    del model.data_modes[stub]
                                    
                                    # Log stub change
                                    self.log_change(selected_model, f"Renamed {current_name}", 0, 0, f"{stub} -> {new_stub}")
                                else:
                                    # Update existing parameter
                                    mode.name = new_name
                                    if current_value != mode.value:
                                        old_value = mode.value
                                        mode.value = current_value
                                        self.log_change(selected_model, new_name, old_value, current_value, stub)
                                
                                st.success(f"Updated parameter properties successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating parameter: {str(e)}")

    def check_parameter_dependencies(self, param_stub: str) -> list:
        dependencies = []
        
        for model_name, model in self.manager.models.items():
            for formula_stub, formula_mode in model.formula_modes.items():
                if param_stub in self.extract_variables(formula_mode.formula):
                    dependencies.append({
                        "model": model_name,
                        "formula": formula_mode.name,
                        "formula_stub": formula_stub
                    })
        
        return dependencies

    def display_enhanced_overview(self):
        st.markdown("### View Selection")
        view_mode = st.radio(
            "Choose view:",
            ["Emissions Overview", "Change History"],
            horizontal=True,
            key="overview_view_selector"
        )
        
        st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

        if view_mode == "Emissions Overview":
            self.display_emissions_overview()
        else: 
            self.display_change_history()

    def display_emissions_overview(self):
        if not self.manager:
            st.error("Model not initialized")
            return
        
        scope_totals = {"Scope1": 0, "Scope2": 0, "Scope3": 0}
        
        for model_name, model in self.manager.models.items():
            scope = get_model_scope(model_name)
            
            # Look for total in Total models
            if is_total_model(model_name) and scope in scope_totals:
                for stub, mode in model.formula_modes.items():
                    if mode.result is not None:
                        scope_totals[scope] = mode.result
                        break
        
        scope1_total = scope_totals["Scope1"]
        scope2_total = scope_totals["Scope2"]
        scope3_total = scope_totals["Scope3"]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Scope 1", f"{scope1_total:.2f} tonnes CO2e", help="Direct emissions")
        
        with col2:
            st.metric("Scope 2", f"{scope2_total:.2f} tonnes CO2e", help="Indirect emissions from electricity")
        
        with col3:
            st.metric("Scope 3", f"{scope3_total:.2f} tonnes CO2e", help="Other indirect emissions")
        
        with col4:
            total = scope1_total + scope2_total + scope3_total
            st.metric("Total", f"{total:.2f} tonnes CO2e", help="Total emissions")
        
        # Add a simple chart
        if total > 0:
            st.markdown("### Emission Distribution")
            chart_data = pd.DataFrame({
                'Scope': ['Scope 1', 'Scope 2', 'Scope 3'],
                'Emissions': [scope1_total, scope2_total, scope3_total]
            })
            st.bar_chart(chart_data.set_index('Scope'))
        
        # Additional model-level breakdown
        st.markdown("### Model Breakdown")
        scope_summary = []
        for model_name, model in self.manager.models.items():
            scope = get_model_scope(model_name)
            
            if scope in ["Scope1", "Scope2", "Scope3"]:
                # Find the main result for each model
                main_result = None
                
                # Look for a formula result
                for stub, mode in model.formula_modes.items():
                    if mode.result is not None:
                        # Prefer results ending with CO2e or matching model pattern
                        if stub.endswith("CO2e") or stub.endswith("_Total"):
                            main_result = mode.result
                            break
                        elif main_result is None:
                            main_result = mode.result
                
                if main_result is not None:
                    scope_summary.append({
                        "Model": model_name,
                        "Scope": scope,
                        "Value (tonnes CO2e)": f"{main_result:.2f}",
                        "Percentage": f"{(main_result/total*100):.1f}%" if total > 0 else "0%"
                    })
        
        if scope_summary:
            df_summary = pd.DataFrame(scope_summary)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # Show parameter count summary
        st.markdown("### System Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", len(self.manager.models))
        
        with col2:
            total_data_params = sum(len(model.data_modes) for model in self.manager.models.values())
            st.metric("Data Parameters", total_data_params)
        
        with col3:
            total_formulas = sum(len(model.formula_modes) for model in self.manager.models.values())
            st.metric("Calculation Formulas", total_formulas)
        
        with col4:
            calculated_results = sum(1 for model in self.manager.models.values() 
                                for mode in model.formula_modes.values() 
                                if mode.result is not None)
            st.metric("Calculated Results", calculated_results)


    def display_batch_operations(self):
        st.subheader("Batch Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Export Data")
            if st.button("Export to Excel", use_container_width=True):
                try:
                    # Collect all parameters
                    data = []
                    for model_name, model in self.manager.models.items():
                        for stub, mode in model.data_modes.items():
                            data.append({
                                "Model": model_name,
                                "Parameter": mode.name,
                                "Variable": stub,
                                "Value": mode.value
                            })
                    
                    # Create DataFrame
                    df = pd.DataFrame(data)
                    
                    # Convert to CSV for download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="carbon_parameters.csv",
                        mime="text/csv"
                    )
                    st.success("Export ready for download!")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            st.markdown("### Import Data")
            uploaded_file = st.file_uploader(
                "Upload Excel/CSV file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload a file with parameter updates"
            )
            
            if uploaded_file:
                if st.button("Import Parameters", use_container_width=True):
                    try:
                        # Read file
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        # Update parameters
                        updated_count = 0
                        for _, row in df.iterrows():
                            model_name = row.get('Model')
                            variable = row.get('Variable')
                            value = row.get('Value')
                            
                            if model_name in self.manager.models:
                                model = self.manager.models[model_name]
                                if variable in model.data_modes:
                                    old_value = model.data_modes[variable].value
                                    model.data_modes[variable].value = float(value)
                                    # Log the change
                                    self.log_change(model_name, model.data_modes[variable].name, old_value, float(value), variable)
                                    updated_count += 1
                        
                        self.manager.calculate_all()
                        st.success(f"Imported {updated_count} parameters successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {str(e)}")
    
    def display_change_history(self):
        st.subheader("Parameter Change History")
        
        if not st.session_state.change_history:
            st.info("No parameter changes recorded yet.")
            st.markdown("**Tip:** Make some parameter changes in the 'Update by Scope' section to see the modification history here.")
            return
        
        # Control panel
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Filter by model directly
            all_models = set()
            for record in st.session_state.change_history:
                all_models.add(record['model'])
            
            # Create model list with "All Models" at the beginning
            model_list = ["All Models"] + sorted(list(all_models))
            
            selected_model = st.selectbox(
                "Filter by Model",
                model_list,
                key="history_model_filter"
            )
        
        with col2:
            # Filter by time period
            time_filter = st.selectbox(
                "Time Period",
                ["All Time", "Last Day", "Last Week", "Last Month"],
                key="history_time_filter"
            )
        
        with col3:
            # Clear history button
            if st.button("Clear History", use_container_width=True, help="Clear all change records"):
                st.session_state.change_history = []
                st.success("History cleared!")
                st.rerun()
        
        # Apply filters
        filtered_history = st.session_state.change_history.copy()
        
        # Model filter
        if selected_model != "All Models":
            filtered_history = [record for record in filtered_history if record['model'] == selected_model]
        
        # Time filter
        if time_filter != "All Time":
            from datetime import datetime, timedelta
            now = datetime.now()
            
            if time_filter == "Last Day":
                cutoff = now - timedelta(days=1)
            elif time_filter == "Last Week":
                cutoff = now - timedelta(weeks=1)
            elif time_filter == "Last Month":
                cutoff = now - timedelta(days=30)
            
            filtered_history = [
                record for record in filtered_history 
                if datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S") >= cutoff
            ]
        
        if not filtered_history:
            st.warning("No records found for the selected filters.")
            return
        
        # Sort by timestamp (newest first)
        filtered_history.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Display summary statistics
        st.markdown("### Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Changes", len(filtered_history))
        
        with col2:
            unique_params = len(set(f"{record['model']}.{record['variable']}" for record in filtered_history))
            st.metric("Parameters Modified", unique_params)
        
        with col3:
            unique_models = len(set(record['model'] for record in filtered_history))
            st.metric("Models Affected", unique_models)
        
        with col4:
            if filtered_history:
                latest_change = filtered_history[0]['timestamp'].split(' ')[0]  # Show only date part (YYYY-MM-DD)
                st.metric("Latest Change", latest_change)
        
        # Display detailed history
        st.markdown("### Change Records")
        
        # Create DataFrame for better display
        history_data = []
        for record in filtered_history:
            # Format change with appropriate color coding
            change_val = record['change']
            
            history_data.append({
                "Time": record['timestamp'].split(' ')[1],  # Only time part
                "Date": record['timestamp'].split(' ')[0],  # Only date part
                "Model": record['model'],
                "Parameter": record['parameter'],
                "Variable": record['variable'],
                "Old Value": f"{record['old_value']:.6f}",
                "New Value": f"{record['new_value']:.6f}",
                "Change": f"{change_val:+.6f}",
            })
        
        # Display as dataframe with pagination
        df_history = pd.DataFrame(history_data)
        
        # Show records in pages
        records_per_page = 20
        total_pages = (len(df_history) - 1) // records_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}",
                key="history_page"
            )
            
            start_idx = (page - 1) * records_per_page
            end_idx = start_idx + records_per_page
            df_display = df_history.iloc[start_idx:end_idx]
        else:
            df_display = df_history
        
        # Style the dataframe
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Parameter": st.column_config.TextColumn("Parameter", width="large"),
                "Variable": st.column_config.TextColumn("Variable", width="small"),
                "Old Value": st.column_config.TextColumn("Old Value", width="small"),
                "New Value": st.column_config.TextColumn("New Value", width="small"),
                "Change": st.column_config.TextColumn("Change", width="small"),
            }
        )

    def validate_json_format(self, json_data):
        errors = []
        
        if not isinstance(json_data, dict):
            errors.append("Root should be a JSON object")
            return {"valid": False, "errors": errors}
        
        for model_name, params in json_data.items():
            if not isinstance(params, dict):
                errors.append(f"Parameters for {model_name} should be a JSON object")
                continue
                
            if model_name not in self.manager.models:
                errors.append(f"Model '{model_name}' not found")
                continue
                
            model = self.manager.models[model_name]
            for param_stub, value in params.items():
                if param_stub not in model.data_modes:
                    errors.append(f"Parameter '{param_stub}' not found in model '{model_name}'")
                
                if not isinstance(value, (int, float)):
                    errors.append(f"Value for '{param_stub}' should be a number")
        
        return {"valid": len(errors) == 0, "errors": errors}

    def apply_json_updates(self, json_data):
        try:
            update_count = 0
            
            for model_name, params in json_data.items():
                if model_name in self.manager.models:
                    model = self.manager.models[model_name]
                    
                    for param_stub, new_value in params.items():
                        if param_stub in model.data_modes:
                            old_value = model.data_modes[param_stub].value
                            param_name = model.data_modes[param_stub].name
                            
                            # Only update if the value has changed
                            if old_value != float(new_value):
                                # Log the change
                                self.log_change(model_name, param_name, old_value, float(new_value), param_stub)
                                
                                # Update the parameter value
                                model.data_modes[param_stub].value = float(new_value)
                                update_count += 1
            
            self.manager.calculate_all()
            
            if update_count > 0:
                st.success(f"Successfully updated {update_count} parameters from JSON file!")
            else:
                st.info("No parameters were changed (all values were already current)")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error applying updates: {str(e)}")

    def export_current_parameters_json(self):
        export_data = {}
        
        for model_name, model in self.manager.models.items():
            if model.data_modes:  # Only include models with data parameters
                export_data[model_name] = {}
                for stub, mode in model.data_modes.items():
                    export_data[model_name][stub] = mode.value
        
        return export_data