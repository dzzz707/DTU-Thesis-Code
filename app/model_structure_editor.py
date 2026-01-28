import streamlit as st
import ast
import json
import os
import base64
from typing import Set, Dict, List
import pandas as pd
from datetime import datetime
from model_manager import get_model_scope

class ModelStructureEditorModule:
    
    def __init__(self, manager):
        self.manager = manager
        # Initialize formula change history if not exists
        if 'formula_change_history' not in st.session_state:
            st.session_state.formula_change_history = []
    
    def display(self):
        st.header("Model Structure Editor")
        st.markdown("Edit calculation formulas and add new emission sources")
        
        # Add function selection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Edit Formulas", use_container_width=True):
                st.session_state.selected_function = "edit_formulas"
        
        with col2:
            if st.button("Source Management", use_container_width=True):
                st.session_state.selected_function = "source_management"
        
        with col3:
            if st.button("View Dependencies", use_container_width=True):
                st.session_state.selected_function = "view_dependencies"
        
        with col4:
            if st.button("Change History", use_container_width=True):
                st.session_state.selected_function = "change_history"
        
        st.markdown("---")
        
        # Display selected function
        if st.session_state.selected_function == "edit_formulas" or st.session_state.selected_function is None:
            self.display_formula_editor()
        elif st.session_state.selected_function == "source_management":
            self.display_source_management()
        elif st.session_state.selected_function == "view_dependencies":
            self.display_dependency_viewer()
        elif st.session_state.selected_function == "change_history":
            self.display_change_history()

    def log_formula_change(self, model_name: str, formula_name: str, old_formula: str, new_formula: str, variable_stub: str):
        change_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "formula_name": formula_name,
            "variable": variable_stub,
            "old_formula": old_formula,
            "new_formula": new_formula,
            "change_type": "Formula Update"
        }
        st.session_state.formula_change_history.append(change_record)
        
        # Keep only last 100 records to prevent excessive memory usage
        if len(st.session_state.formula_change_history) > 100:
            st.session_state.formula_change_history = st.session_state.formula_change_history[-100:]
    
    def log_source_addition(self, model_name: str, source_name: str, variable_stub: str, formula: str, params: List):
        change_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "formula_name": source_name,
            "variable": variable_stub,
            "old_formula": "N/A (New Source)",
            "new_formula": formula,
            "change_type": "New Source Added",
            "parameters_added": len(params)
        }
        st.session_state.formula_change_history.append(change_record)
        
        # Keep only last 100 records
        if len(st.session_state.formula_change_history) > 100:
            st.session_state.formula_change_history = st.session_state.formula_change_history[-100:]
    
    def log_source_deletion(self, model_name: str, source_name: str, variable_stub: str, formula: str, deleted_params: List):
        change_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "formula_name": source_name,
            "variable": variable_stub,
            "old_formula": formula,
            "new_formula": "DELETED",
            "change_type": "Source Deleted",
            "parameters_deleted": len(deleted_params)
        }
        st.session_state.formula_change_history.append(change_record)
        
        # Keep only last 100 records
        if len(st.session_state.formula_change_history) > 100:
            st.session_state.formula_change_history = st.session_state.formula_change_history[-100:]

    def log_model_deletion(self, model_name: str, param_count: int, source_count: int):
        change_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "formula_name": f"Model: {model_name}",
            "variable": "ENTIRE_MODEL",
            "old_formula": f"{param_count} parameters, {source_count} sources",
            "new_formula": "DELETED",
            "change_type": "Model Deleted",
            "parameters_deleted": param_count,
            "sources_deleted": source_count
        }
        st.session_state.formula_change_history.append(change_record)
        
        # Keep only last 100 records
        if len(st.session_state.formula_change_history) > 100:
            st.session_state.formula_change_history = st.session_state.formula_change_history[-100:]

    def get_scope_model_mapping(self) -> Dict[str, List[str]]:
        scope_mapping = {
            "Scope 1": [],
            "Scope 2": [],
            "Scope 3": []
        }
        
        for model_name in self.manager.models:
            scope = get_model_scope(model_name)
            
            if scope == "Scope1":
                scope_mapping["Scope 1"].append(model_name)
            elif scope == "Scope2":
                scope_mapping["Scope 2"].append(model_name)
            elif scope == "Scope3":
                scope_mapping["Scope 3"].append(model_name)
        
        return scope_mapping
    
    def get_variable_name_mapping(self):
        variable_mapping = {}
        for model in self.manager.models.values():
            # Add data modes (parameters)
            for stub, mode in model.data_modes.items():
                variable_mapping[stub] = mode.name
            # Add formula modes
            for stub, mode in model.formula_modes.items():
                variable_mapping[stub] = mode.name
        return variable_mapping
    
    def extract_variables(self, expr: str) -> Set:
        try:
            return {node.id for node in ast.walk(ast.parse(expr)) if isinstance(node, ast.Name)}
        except:
            return set()
    
    def display_formula_editor(self):
        st.subheader("Formula Editor")
        
        if not self.manager:
            st.error("Model not initialized")
            return
        
        # Update mode selection
        update_mode = st.radio(
            "Choose update method:",
            ["Manual Update", "JSON File Upload"],
            horizontal=True,
            key="formula_update_mode_selector"
        )
        
        st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

        if update_mode == "Manual Update":
            self.display_manual_formula_update()
        else:
            self.display_json_formula_update()
    
    def display_manual_formula_update(self):
        
        st.markdown("""
        <style>
        .variable-list {
            background: #1a202c;
            border: 1px solid #4a5568;
            border-radius: 8px;
            padding: 12px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 14px;
            line-height: 1.8;
            color: #e2e8f0;
            max-height: 200px;
            overflow-y: auto;
        }

        .variable-item {
            padding: 6px 12px;
            margin: 3px 0;
            border-left: 4px solid #4a90e2;
            background: rgba(74, 144, 226, 0.1);
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }

        .variable-item:hover {
            background: rgba(74, 144, 226, 0.2);
        }

        .description-item {
            padding: 8px 12px;
            margin: 4px 0;
            border-left: 4px solid #48bb78;
            background: rgba(72, 187, 120, 0.1);
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.5;
        }

        .description-item strong {
            color: #4a90e2;
        }

        .variable-list::-webkit-scrollbar {
            width: 8px;
        }

        .variable-list::-webkit-scrollbar-track {
            background: #2d3748;
            border-radius: 4px;
        }

        .variable-list::-webkit-scrollbar-thumb {
            background: #4a5568;
            border-radius: 4px;
        }

        .variable-list::-webkit-scrollbar-thumb:hover {
            background: #718096;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get scope to model mapping
        scope_mapping = self.get_scope_model_mapping()
        
        # Check if any scope has models
        has_models = any(len(models) > 0 for models in scope_mapping.values())
        
        if not has_models:
            st.warning("No models found. Please ensure your model names follow the naming convention (Scope1_*, Scope2_*, Scope3_*)")
            return
        
        # Filter out empty scopes for selection
        available_scopes = [scope for scope, models in scope_mapping.items() if len(models) > 0]
        
        # Scope and model selection
        selected_scope = st.selectbox(
            "Select Scope",
            available_scopes,
            key="scope_selector_edit"
        )
        
        if selected_scope:
            available_models = scope_mapping[selected_scope]
            
            if not available_models:
                st.warning(f"No models found for {selected_scope}")
                return
            
            selected_model = st.selectbox(
                f"Select Model in {selected_scope}",
                available_models,
                key="model_selector_edit"
            )
            
            if selected_model:
                model = self.manager.models[selected_model]
                
                if model.formula_modes:
                    formula_names = list(model.formula_modes.keys())
                    selected_formula = st.selectbox(
                        "Select Formula",
                        formula_names,
                        format_func=lambda x: f"{model.formula_modes[x].name} ({x})",
                        key="formula_selector"
                    )
                    
                    if selected_formula:
                        formula_mode = model.formula_modes[selected_formula]
                        
                        # Show current formula info
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("**Current Formula:**")
                            st.code(f"{selected_formula} = {formula_mode.formula}")
                        
                        with col2:
                            if hasattr(formula_mode, 'result') and formula_mode.result is not None:
                                st.metric("Current Result", f"{formula_mode.result:.6f}")
                        
                        # Smart formula editor with helper tools
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Main formula input
                            new_formula = st.text_area(
                                "Edit Formula",
                                value=formula_mode.formula,
                                height=120,
                                help="Type your formula using variable names. Use +, -, *, /, and parentheses for operations.",
                                key=f"formula_editor_{selected_formula}"
                            )
                        
                        with col2:
                            # Variable browser with better formatting
                            st.markdown("**Variable Reference:**")

                            # Show descriptions in expandable area
                            data_vars = self.get_organized_variables()

                            with st.expander("View descriptions", expanded=False):
                                descriptions_html = ""
                                for var, desc in data_vars.items():
                                    descriptions_html += f'<div class="description-item"><strong>{var}:</strong> {desc}</div>'
                                
                                st.markdown(f'<div class="variable-list">{descriptions_html}</div>', unsafe_allow_html=True)
                        
                        # Real-time validation and preview
                        if new_formula != formula_mode.formula:
                            st.markdown("---")
                            validation_result = self.validate_formula_realtime(new_formula)
                            
                            if validation_result['valid']:
                                st.success("Formula syntax is valid")
                                if validation_result['warnings']:
                                    for warning in validation_result['warnings']:
                                        st.warning(f"Warning: {warning}")
                            else:
                                st.error(f"Error: {validation_result['error']}")
                            
                            # Preview changes
                            st.markdown("**Preview:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text("Current:")
                                st.code(formula_mode.formula)
                            with col2:
                                st.text("New:")
                                st.code(new_formula)
                        
                        # Apply button
                        st.markdown("---")
                        if st.button("Apply Changes", type="primary", use_container_width=True, 
                                    key=f"apply_changes_{selected_formula}"):
                            # Validate formula first
                            validation = self.validate_formula_realtime(new_formula)
                            
                            # Store pending update for confirmation
                            st.session_state.pending_formula_update_manual = {
                                'model_name': selected_model,
                                'formula_name': formula_mode.name,
                                'formula_stub': selected_formula,
                                'formula_mode': formula_mode,
                                'old_formula': formula_mode.formula,
                                'new_formula': new_formula,
                                'warnings': validation.get('warnings', [])
                            }
                            st.session_state.show_formula_update_confirm = True
                            st.rerun()
                        
                        # Quick examples
                        with st.expander("Formula Examples", expanded=False):
                            st.markdown("""
                            **Basic Examples:**
                            - `variable1 + variable2`
                            - `(a + b) * c`
                            - `value / 1000`
                            
                            **Real Examples:**
                            - `m3 * CO2_XGWP + m3 * CH4_XGWP`
                            - `(ULP92 + ULP95) * EmissionFactor`
                            - `ElectricityData * Electricity_CO2_XGWP / 1000`
                            """)
                else:
                    st.info("No formulas found in this model")
        # Show confirmation dialog if needed
        if hasattr(st.session_state, 'show_formula_update_confirm') and st.session_state.show_formula_update_confirm:
            self.show_formula_update_confirmation()

    def get_organized_variables(self):
        data_vars = {}
        
        for model in self.manager.models.values():
            # Data parameters only
            for stub, mode in model.data_modes.items():
                data_vars[stub] = mode.name
        
        return data_vars

    def validate_formula_realtime(self, formula):
        result = {'valid': False, 'error': '', 'warnings': []}
        
        try:
            # Basic checks
            if not formula.strip():
                result['error'] = "Formula cannot be empty"
                return result
            
            # Syntax check
            ast.parse(formula)
            
            # Variable existence check
            variables = self.extract_variables(formula)
            all_vars = set()
            for m in self.manager.models.values():
                all_vars.update(m.data_modes.keys())
                all_vars.update(m.formula_modes.keys())
            
            missing_vars = variables - all_vars
            if missing_vars:
                result['warnings'].append(f"Unknown variables: {', '.join(missing_vars)}")
            
            # Check for common issues
            if formula.count('(') != formula.count(')'):
                result['error'] = "Unmatched parentheses"
                return result
            
            # Check for consecutive operators
            import re
            if re.search(r'[+\-*/]{2,}', formula):
                result['warnings'].append("Consecutive operators detected")
            
            result['valid'] = True
            return result
            
        except SyntaxError as e:
            result['error'] = f"Syntax error: {str(e)}"
            return result
        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
            return result

    def apply_formula_changes(self, formula_mode, new_formula, model_name, formula_stub):
        try:
            if new_formula == formula_mode.formula:
                st.info("No changes detected")
                return False
            
            validation = self.validate_formula_realtime(new_formula)
            if not validation['valid']:
                st.error(f"Cannot apply invalid formula: {validation['error']}")
                return False
            
            # Log the change
            self.log_formula_change(
                model_name,
                formula_mode.name, 
                formula_mode.formula,
                new_formula,
                formula_stub
            )
            
            # Apply change
            formula_mode.formula = new_formula
            self.manager.calculate_all()
            self.save_formulas_to_file()
            
            return True
            
        except Exception as e:
            st.error(f"Error applying changes: {str(e)}")
            return False

    def display_json_formula_update(self):
        st.subheader("JSON Formula Update")
        
        uploaded_file = st.file_uploader(
            "Upload JSON formula file",
            type=['json'],
            help="Upload a JSON file with formula updates"
        )
        
        if uploaded_file:
            try:
                json_data = json.load(uploaded_file)
                
                st.markdown("### File Preview")
                with st.expander("View uploaded data", expanded=True):
                    st.json(json_data)
                
                # Validate JSON format for formulas
                validation_result = self.validate_formula_json_format(json_data)
                
                if validation_result["valid"]:
                    st.success("JSON format is valid!")
                    
                    updates_preview = []
                    unchanged_count = 0 

                    for model_name, formulas in json_data.items():
                        if model_name in self.manager.models:
                            model = self.manager.models[model_name]
                            for formula_stub, new_formula in formulas.items():
                                if formula_stub in model.formula_modes:
                                    old_formula = model.formula_modes[formula_stub].formula
                                    formula_name = model.formula_modes[formula_stub].name
                                    
                                    # Log the change if formulas differ
                                    if old_formula != new_formula:
                                        updates_preview.append({
                                            "Model": model_name,
                                            "Formula Name": formula_name,
                                            "Variable": formula_stub,
                                            "Current Formula": old_formula,
                                            "New Formula": new_formula
                                        })
                                    else:
                                        unchanged_count += 1

                    if updates_preview:
                        st.markdown("### Changed Formulas Only")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Formulas to Update", len(updates_preview))
                        with col2:
                            st.metric("Unchanged Formulas", unchanged_count)
                        with col3:
                            total_formulas = len(updates_preview) + unchanged_count
                            st.metric("Total Formulas", total_formulas)
                        
                        df_preview = pd.DataFrame(updates_preview)
                        st.dataframe(df_preview, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Apply JSON Formula Updates", type="primary", use_container_width=True):
                                self.apply_json_formula_updates(json_data)
                        
                        with col2:
                            if st.button("Cancel", use_container_width=True):
                                st.rerun()
                    else:
                        if unchanged_count > 0:
                            st.info(f"All {unchanged_count} formulas in the JSON file are identical to current formulas. No changes needed.")
                        else:
                            st.warning("No valid formulas found for update")
                            
                else:
                    st.error("Invalid JSON format!")
                    for error in validation_result["errors"]:
                        st.error(f"- {error}")
                    
                    # Display expected JSON format
                    st.markdown("### Expected JSON Format")
                    example_json = {
                        "Scope1_FixedCombustion": {
                            "NaturalGasCO2e": "m3 * CO2_XGWP + m3 * CH4_XGWP",
                            "FixedCO2e": "NaturalGasCO2e"
                        },
                        "Scope1_Mobile": {
                            "MobileCO2e": "MotorPetrolCO2e + DieselCO2e + LPGCO2e"
                        }
                    }
                    st.json(example_json)
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON file format")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.markdown("### JSON Format Guidelines")
            st.markdown("""
            Upload a JSON file with the following structure:
            - Top level: Model names as keys
            - Second level: Formula variable stubs as keys with formula expressions as values
            """)
            
            # Provide download link for current formulas
            st.markdown("### Export Current Formulas")
            if st.button("Download Current Formulas as JSON", use_container_width=True):
                current_formulas = self.export_current_formulas_json()
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(current_formulas, indent=2),
                    file_name="current_formulas.json",
                    mime="application/json"
                )

    def display_source_management(self):

        st.subheader("Source & Model Management")
        
        # Operation mode selection
        operation_mode = st.radio(
            "Choose operation mode:",
            ["Manual Modification", "JSON Modification"],
            horizontal=True,
            key="operation_mode_selector"
        )
        
        st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

        if operation_mode == "Manual Modification":
            self.display_manual_operations()
        else:
            self.display_json_operations()

    def display_manual_operations(self):
        st.markdown("### Manual Modifications")
        
        # Manual operation mode selection
        management_mode = st.radio(
            "Choose operation:",
            ["Add New Source", "Delete Source", "Delete Model"],
            horizontal=True,
            key="manual_operation_type"
        )
        
        st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

        # Display the appropriate interface based on selection
        if management_mode == "Add New Source":
            self.display_manual_source_addition()
        elif management_mode == "Delete Source":
            self.display_delete_emission_source()
        else:  # Delete Model
            self.display_delete_model()

    def display_json_operations(self):
        self.display_json_source_addition()

    def check_parameter_dependencies(self, variable_stub: str) -> List[Dict]:
        dependencies = []
        
        for model_name, model in self.manager.models.items():
            for formula_stub, formula_mode in model.formula_modes.items():
                # Skip if the variable is the same as the formula stub
                if formula_stub == variable_stub:
                    continue
                    
                # Check if the variable is used in this formula
                formula_vars = self.extract_variables(formula_mode.formula)
                if variable_stub in formula_vars:
                    dependencies.append({
                        "model": model_name,
                        "formula": formula_mode.name,
                        "formula_stub": formula_stub
                    })
        
        return dependencies

    def display_delete_emission_source(self):
        st.markdown("### Delete Emission Source")
        
        if not self.manager.models:
            st.warning("No models found")
            return
        
        # Model selection
        model_options = list(self.manager.models.keys())
        selected_model = st.selectbox("Select Model", model_options, key="delete_source_model")
        
        if selected_model:
            model = self.manager.models[selected_model]
            
            if not model.formula_modes:
                st.info("No emission sources found in this model")
                return
            
            # Source selection
            source_options = [(stub, mode.name) for stub, mode in model.formula_modes.items()]
            selected_source = st.selectbox(
                "Select Emission Source to Delete",
                source_options,
                format_func=lambda x: f"{x[1]} ({x[0]})",
                key="delete_source_selection"
            )
            
            if selected_source:
                stub, name = selected_source
                formula_mode = model.formula_modes[stub]
                
                # Show source details
                with st.expander("Source Details", expanded=True):
                    st.text(f"Name: {formula_mode.name}")
                    st.text(f"Variable: {stub}")
                    st.text(f"Formula: {formula_mode.formula}")
                    if hasattr(formula_mode, 'result') and formula_mode.result is not None:
                        st.text(f"Current Result: {formula_mode.result:.6f}")
                
                # Analyze dependencies and related parameters
                formula_vars = self.extract_variables(formula_mode.formula)
                related_params = []
                dependent_formulas = []
                
                # Find related parameters (used only by this formula)
                for var in formula_vars:
                    if var in model.data_modes:
                        usage_count = 0
                        for other_stub, other_mode in model.formula_modes.items():
                            if other_stub != stub and var in self.extract_variables(other_mode.formula):
                                usage_count += 1
                        
                        # Also check usage in other models
                        for other_model_name, other_model in self.manager.models.items():
                            if other_model_name != selected_model:
                                for other_stub, other_mode in other_model.formula_modes.items():
                                    if var in self.extract_variables(other_mode.formula):
                                        usage_count += 1
                        
                        if usage_count == 0:  # Only used by this formula
                            related_params.append({
                                "stub": var,
                                "name": model.data_modes[var].name,
                                "value": model.data_modes[var].value
                            })
                
                # Find dependent formulas
                dependent_formulas = self.check_parameter_dependencies(stub)
                
                # Display analysis
                if related_params:
                    st.markdown("### Related Parameters (will be deleted)")
                    st.info("These parameters are only used by this emission source:")
                    for param in related_params:
                        st.text(f"• {param['name']} ({param['stub']}): {param['value']}")
                
                if dependent_formulas:
                    st.markdown("### Dependencies Found")
                    st.error("This emission source is used by other formulas. Cannot delete:")
                    for dep in dependent_formulas:
                        st.text(f"• {dep['model']}.{dep['formula']} uses {stub}")
                    return
                else:
                    st.success("This emission source is safe to delete (no dependencies found)")
                
                # Confirmation
                st.markdown("---")
                st.warning("This action will permanently delete the emission source and its related parameters!")
                
                confirm_text = f"Type 'DELETE {name}' to confirm:"
                confirmation = st.text_input(confirm_text, key="delete_source_confirm")
                
                if confirmation == f"DELETE {name}":
                    if st.button("Confirm Deletion", type="primary", use_container_width=True):
                        try:
                            # Log the deletion
                            self.log_source_deletion(selected_model, name, stub, formula_mode.formula, related_params)
                            
                            # Delete the formula
                            del model.formula_modes[stub]
                            
                            # Delete related parameters
                            for param in related_params:
                                del model.data_modes[param['stub']]
                            
                            # Recalculate and save
                            self.manager.calculate_all()
                            self.save_formulas_to_file()
                            
                            st.success(f"Successfully deleted emission source '{name}' and {len(related_params)} related parameters!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error deleting emission source: {str(e)}")
                elif confirmation and confirmation != f"DELETE {name}":
                    st.error("Confirmation text does not match. Please type exactly as shown.")

    def display_delete_model(self):
        st.markdown("### Delete Model")
        
        if not self.manager.models:
            st.warning("No models found")
            return
        
        # Model selection
        model_options = list(self.manager.models.keys())
        selected_model = st.selectbox("Select Model to Delete", model_options, key="delete_model_selection")
        
        if selected_model:
            model = self.manager.models[selected_model]
            
            # Show model details
            with st.expander("Model Details", expanded=True):
                st.text(f"Model Name: {selected_model}")
                st.text(f"Data Parameters: {len(model.data_modes)}")
                st.text(f"Emission Sources: {len(model.formula_modes)}")
                
                if model.data_modes:
                    st.markdown("**Parameters:**")
                    for stub, mode in model.data_modes.items():
                        st.text(f"• {mode.name} ({stub})")
                
                if model.formula_modes:
                    st.markdown("**Emission Sources:**")
                    for stub, mode in model.formula_modes.items():
                        st.text(f"• {mode.name} ({stub})")
            
            # Check dependencies
            dependencies = []
            model_vars = set(model.data_modes.keys()) | set(model.formula_modes.keys())
            
            for other_model_name, other_model in self.manager.models.items():
                if other_model_name != selected_model:
                    for formula_stub, formula_mode in other_model.formula_modes.items():
                        formula_vars = self.extract_variables(formula_mode.formula)
                        used_vars = formula_vars & model_vars
                        if used_vars:
                            dependencies.append({
                                "model": other_model_name,
                                "formula": formula_mode.name,
                                "variables": list(used_vars)
                            })
            
            # Display dependencies
            if dependencies:
                st.markdown("### Dependencies Found")
                st.error("This model's variables are used by other models. Cannot delete:")
                for dep in dependencies:
                    st.text(f"• {dep['model']}.{dep['formula']} uses: {', '.join(dep['variables'])}")
                return
            else:
                st.success("This model is safe to delete (no dependencies found)")
            
            # Confirmation
            st.markdown("---")
            st.warning("This action will permanently delete the entire model and all its contents!")
            
            confirm_text = f"Type 'DELETE {selected_model}' to confirm:"
            confirmation = st.text_input(confirm_text, key="delete_model_confirm")
            
            if confirmation == f"DELETE {selected_model}":
                if st.button("Confirm Deletion", type="primary", use_container_width=True):
                    try:
                        # Log the deletion
                        self.log_model_deletion(selected_model, len(model.data_modes), len(model.formula_modes))
                        
                        # Delete the model
                        del self.manager.models[selected_model]
                        
                        # Recalculate remaining models
                        self.manager.calculate_all()
                        self.save_formulas_to_file()
                        
                        st.success(f"Successfully deleted model '{selected_model}'!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error deleting model: {str(e)}")
            elif confirmation and confirmation != f"DELETE {selected_model}":
                st.error("Confirmation text does not match. Please type exactly as shown.")

    def display_manual_source_addition(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Scope Selection")
            scope_options = ["Scope 1", "Scope 2", "Scope 3"]
            selected_scope = st.selectbox(
                "Select Scope", 
                scope_options,
                key="scope_selector_add_source"
            )
        
        with col2:
            st.markdown("### Model Selection")
            # Get scope to model mapping
            scope_mapping = self.get_scope_model_mapping()
            
            # Determine available models based on selected scope
            available_models = scope_mapping.get(selected_scope, [])
            model_options = available_models + ["Create New Model"]
            
            selected_model_option = st.selectbox(
                f"Select Model for {selected_scope}",
                model_options,
                key="model_selector_add_source",
                help=f"Choose an existing model in {selected_scope} or create a new one"
            )
        
        # Handle new model creation (if needed)
        new_model_name = None
        if selected_model_option == "Create New Model":
            st.markdown("### New Model Configuration")
            
            # Generate suggested prefix based on scope
            scope_prefix = selected_scope.replace(" ", "")  # "Scope 1" -> "Scope1"
            
            new_model_name = st.text_input(
                "New Model Name", 
                placeholder=f"e.g., {scope_prefix}_NewSource",
                help=f"Enter a unique name for the new model. Must start with '{scope_prefix}_'",
                key="new_model_name_input"
            )
            
            if new_model_name:
                # Validate naming convention
                if not new_model_name.startswith(scope_prefix):
                    st.error(f"Model name must start with '{scope_prefix}_' to be properly categorized under {selected_scope}")
                    new_model_name = None
            else:
                st.warning(f"Please enter a name for the new model (must start with '{scope_prefix}_')")

        with st.form("add_source_form"):
            st.markdown("### Basic Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                source_name = st.text_input(
                    "Source Name", 
                    placeholder="e.g., Electric Vehicle",
                    help="Descriptive name for this emission source"
                )
                
                variable_name = st.text_input(
                    "Variable Name (Stub)", 
                    placeholder="e.g., EV_CO2e",
                    help="Unique variable identifier for calculations"
                )
                
            with col2:
                unit = st.text_input(
                    "Unit", 
                    placeholder="e.g., kg CO2e",
                    help="Unit of measurement for this emission source"
                )

            st.markdown("---") 
            
            st.markdown("### Parameters")
            
            # Initialize parameter list in session state if not exists
            if 'temp_parameters' not in st.session_state:
                st.session_state.temp_parameters = [{"name": "", "stub": "", "value": 0.0}]
            
            # Display current parameters with better styling
            for i, param in enumerate(st.session_state.temp_parameters):
                with st.container():
                    st.markdown(f"**Parameter {i+1}**")
                    col1, col2, col3, col4 = st.columns([3, 3, 2, 1])
                    
                    with col1:
                        param_name = st.text_input(
                            "Parameter Name", 
                            value=param["name"],
                            key=f"param_name_{i}",
                            placeholder="e.g., Activity Data"
                        )
                        st.session_state.temp_parameters[i]["name"] = param_name
                        
                    with col2:
                        param_stub = st.text_input(
                            "Variable Stub", 
                            value=param["stub"],
                            key=f"param_stub_{i}",
                            placeholder="e.g., EV_kWh"
                        )
                        st.session_state.temp_parameters[i]["stub"] = param_stub
                        
                    with col3:
                        param_value = st.number_input(
                            "Default Value", 
                            value=param["value"],
                            key=f"param_value_{i}",
                            format="%.6f",
                            step=0.000001
                        )
                        st.session_state.temp_parameters[i]["value"] = param_value
            
            # Add parameter button with better styling
            button_col1, button_col2, spacer = st.columns([1.5, 1.5, 5])
            with button_col1:
                add_param = st.form_submit_button("Add", use_container_width=True)
            with button_col2:
                if len(st.session_state.temp_parameters) > 1:
                    remove_last = st.form_submit_button("Remove", use_container_width=True)
                else:
                    remove_last = False
            
            # Handle parameter operations
            if add_param:
                st.session_state.temp_parameters.append({"name": "", "stub": "", "value": 0.0})
                st.rerun()
            elif remove_last and len(st.session_state.temp_parameters) > 1:
                st.session_state.temp_parameters.pop()
                st.rerun()
            
            st.markdown("---") 
            
            st.markdown("### Formula Configuration")
            formula = st.text_area(
                "Calculation Formula",
                placeholder="e.g., EV_kWh * EV_CO2_XGWP",
                height=100,
                help="Use the parameter variables defined above in your formula"
            )
            st.markdown("---")
            
            # Submit button with better styling
            col1, col2, col3 = st.columns([2, 3, 2])
            with col2:
                submitted = st.form_submit_button(
                    "Add Emission Source", 
                    type="primary", 
                    use_container_width=True
                )
            
            if submitted:
                # Determine model category
                model_category = selected_model_option
                if selected_model_option == "Create New Model":
                    if new_model_name:
                        model_category = new_model_name
                    else:
                        st.error("Please enter a name for the new model")
                        return
                
                # Validation with better error messages
                if not source_name:
                    st.error("Please enter a source name")
                elif not variable_name:
                    st.error("Please enter a variable name")
                elif not formula:
                    st.error("Please enter a calculation formula")
                else:
                    # Validate parameters
                    valid_params = []
                    for param in st.session_state.temp_parameters:
                        if param["name"] and param["stub"]:
                            valid_params.append((param["name"], param["stub"], param["value"]))
                    
                    if not valid_params:
                        st.error("Please enter at least one valid parameter (name and variable required)")
                    else:
                        # Check if variable name already exists
                        existing_vars = set()
                        for m in self.manager.models.values():
                            existing_vars.update(m.data_modes.keys())
                            existing_vars.update(m.formula_modes.keys())
                        
                        if variable_name in existing_vars:
                            st.error(f"Variable name '{variable_name}' already exists. Please choose a different name.")
                        else:
                            # Store pending addition for confirmation
                            st.session_state.pending_add_source = {
                                'model_name': model_category,
                                'source_name': source_name,
                                'variable_name': variable_name,
                                'formula': formula,
                                'valid_params': valid_params
                            }
                            st.session_state.show_add_source_confirm = True
                            st.rerun()

        # Show confirmation dialog if needed
        if hasattr(st.session_state, 'show_add_source_confirm') and st.session_state.show_add_source_confirm:
            self.show_add_source_confirmation()

    def display_json_source_addition(self):        
        uploaded_file = st.file_uploader(
            "Upload JSON source definition file",
            type=['json'],
            help="Upload a JSON file with new emission source definitions"
        )
        
        if uploaded_file:
            try:
                json_data = json.load(uploaded_file)
                st.markdown("### File Preview")
                with st.expander("View uploaded data", expanded=True):
                    st.json(json_data)
                
                # Validate JSON format for new sources
                validation_result = self.validate_source_json_format(json_data)
                
                if validation_result["valid"]:
                    st.success("JSON format is valid!")
                    
                    # Preview sources to be added
                    sources_preview = []
                    
                    for model_name, sources in json_data.items():
                        for source_data in sources:
                            sources_preview.append({
                                "Model": model_name,
                                "Source Name": source_data["source_name"],
                                "Variable": source_data["variable_stub"],
                                "Formula": source_data["formula"],
                                "Parameters": len(source_data.get("parameters", []))
                            })
                    
                    if sources_preview:
                        st.markdown("### Sources to be Added")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Sources", len(sources_preview))
                        with col2:
                            unique_models = len(set(s["Model"] for s in sources_preview))
                            st.metric("Models Affected", unique_models)
                        with col3:
                            total_params = sum(s["Parameters"] for s in sources_preview)
                            st.metric("Total Parameters", total_params)
                        
                        df_preview = pd.DataFrame(sources_preview)
                        st.dataframe(df_preview, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Add All Sources", type="primary", use_container_width=True):
                                self.apply_json_source_additions(json_data)
                        
                        with col2:
                            if st.button("Cancel", use_container_width=True):
                                st.rerun()
                    else:
                        st.warning("No valid sources found in the JSON file")
                        
                else:
                    st.error("Invalid JSON format!")
                    for error in validation_result["errors"]:
                        st.error(f"- {error}")
                    
                    # Display expected JSON format
                    st.markdown("### Expected JSON Format")
                    example_json = {
                        "Scope1_Mobile": [
                            {
                                "source_name": "Electric Vehicle Charging",
                                "variable_stub": "EV_CO2e",
                                "formula": "EV_kWh * EV_CO2_Factor",
                                "parameters": [
                                    {"name": "EV Energy Consumption", "stub": "EV_kWh", "value": 1000.0},
                                    {"name": "EV CO2 Factor", "stub": "EV_CO2_Factor", "value": 0.495}
                                ]
                            }
                        ],
                        "Scope3_Travel": [
                            {
                                "source_name": "Business Travel",
                                "variable_stub": "BusinessTravel_CO2e",
                                "formula": "Travel_km * Travel_CO2_Factor",
                                "parameters": [
                                    {"name": "Travel Distance", "stub": "Travel_km", "value": 5000.0},
                                    {"name": "Travel CO2 Factor", "stub": "Travel_CO2_Factor", "value": 0.21}
                                ]
                            }
                        ]
                    }
                    st.json(example_json)
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON file format")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.markdown("### JSON Format Guidelines")
            st.markdown("""
            Upload a JSON file with the following structure:
            - Top level: Model names as keys (must follow naming convention: Scope1_*, Scope2_*, Scope3_*)
            - Second level: Array of source objects with:
              - source_name: Display name for the emission source
              - variable_stub: Unique variable identifier
              - formula: Calculation formula using parameter stubs
              - parameters: Array of parameter objects (name, stub, value)
            """)

            # Provide download link for current sources
            st.markdown("### Export Current Sources")
            if st.button("Download Current Sources as JSON", use_container_width=True):
                current_sources = self.export_current_sources_json()
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(current_sources, indent=2),
                    file_name="current_sources.json",
                    mime="application/json"
                )

    @st.dialog("Confirm Formula Update", width="large")
    def show_formula_confirmation_modal(self):
        """Show confirmation modal for formula updates as a floating dialog"""
        pending_update = st.session_state.get('pending_formula_update', {})
        
        if not pending_update:
            st.error("No pending update found.")
            return
        
        # Add some spacing and better layout
        st.markdown("###") # Add space at top
        
        # Show what will be changed with better formatting
        st.markdown("### Current Formula:")
        st.code(f"{pending_update['selected_formula']} = {pending_update['formula_mode'].formula}", 
                language="python")
        
        st.markdown("### New Formula:")
        st.code(f"{pending_update['selected_formula']} = {pending_update['new_formula']}", 
                language="python")
        
        # Add separator
        st.markdown("---")
        
        # Show warnings if any with better formatting
        if pending_update['missing_vars']:
            st.warning(f"**Warning**: The following variables were not found in the system:")
            # Display missing variables in a more readable format
            for var in pending_update['missing_vars']:
                st.markdown(f"   • `{var}`")
            st.markdown("Please make sure these variables exist or the formula may not calculate correctly.")
        else:
            st.success("**All variables in the formula were found in the system.**")
        
        # Add some spacing before buttons
        st.markdown("###")
        
        # Action buttons with better spacing
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            # Empty column for spacing
            pass
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_formula_confirm_modal = False
                if 'pending_formula_update' in st.session_state:
                    del st.session_state.pending_formula_update
                st.rerun()
        
        with col3:
            if st.button("Confirm Save", use_container_width=True, type="primary"):
                try:
                    # Log the formula change before applying
                    self.log_formula_change(
                        pending_update['selected_model'],
                        pending_update['formula_mode'].name,
                        pending_update['formula_mode'].formula,
                        pending_update['new_formula'],
                        pending_update['selected_formula']
                    )
                    
                    # Apply the changes
                    formula_mode = pending_update['formula_mode']
                    formula_mode.formula = pending_update['new_formula']
                    
                    # Recalculate all models
                    self.manager.calculate_all()
                    
                    # Save to formula.json
                    self.save_formulas_to_file()
                    
                    # Clear modal state
                    st.session_state.show_formula_confirm_modal = False
                    if 'pending_formula_update' in st.session_state:
                        del st.session_state.pending_formula_update
                    
                    st.success("Formula updated and saved successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error saving formula: {str(e)}")

    @st.dialog("Confirm Formula Update")
    def show_formula_update_confirmation(self):
        pending_update = st.session_state.get('pending_formula_update_manual', {})
        
        if not pending_update:
            st.error("No pending formula update found.")
            return
        
        st.markdown("### Formula to be Updated:")
        st.markdown(f"**Model**: {pending_update['model_name']}")
        st.markdown(f"**Formula**: {pending_update['formula_name']}")
        
        st.markdown("### Changes:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Current Formula:**")
            st.code(pending_update['old_formula'], language="python")
        with col2:
            st.markdown("**New Formula:**")
            st.code(pending_update['new_formula'], language="python")
        
        if pending_update.get('warnings'):
            st.warning("Validation warnings found. Please review before applying.")
            for warning in pending_update['warnings']:
                st.text(f"• {warning}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_formula_update_confirm = False
                if 'pending_formula_update_manual' in st.session_state:
                    del st.session_state.pending_formula_update_manual
                st.rerun()
        
        with col2:
            if st.button("Confirm Update", use_container_width=True, type="primary"):
                try:
                    # Apply the formula change
                    formula_mode = pending_update['formula_mode']
                    old_formula = formula_mode.formula
                    
                    # Log the change
                    self.log_formula_change(
                        pending_update['model_name'],
                        formula_mode.name,
                        old_formula,
                        pending_update['new_formula'],
                        pending_update['formula_stub']
                    )
                    
                    # Update formula
                    formula_mode.formula = pending_update['new_formula']
                    self.manager.calculate_all()
                    self.save_formulas_to_file()
                    
                    # Clear confirmation state
                    st.session_state.show_formula_update_confirm = False
                    if 'pending_formula_update_manual' in st.session_state:
                        del st.session_state.pending_formula_update_manual
                    
                    st.success("Formula updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating formula: {str(e)}")

    @st.dialog("Confirm New Source Addition")
    def show_add_source_confirmation(self):
        pending_add = st.session_state.get('pending_add_source', {})
        
        if not pending_add:
            st.error("No pending source addition found.")
            return
        
        st.markdown("### Source to be Added:")
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Model: {pending_add['model_name']}")
            st.text(f"Source Name: {pending_add['source_name']}")
            st.text(f"Variable: {pending_add['variable_name']}")
        with col2:
            st.text(f"Formula: {pending_add['formula']}")
            st.text(f"Parameters: {len(pending_add['valid_params'])}")
        
        if pending_add['valid_params']:
            st.markdown("### Parameters to be Added:")
            for param_name, param_stub, param_value in pending_add['valid_params']:
                st.text(f"• {param_name} ({param_stub}): {param_value}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_add_source_confirm = False
                if 'pending_add_source' in st.session_state:
                    del st.session_state.pending_add_source
                st.rerun()
        
        with col2:
            if st.button("Confirm Add Source", use_container_width=True, type="primary"):
                try:
                    # Get or create model
                    model_name = pending_add['model_name']
                    if model_name not in self.manager.models:
                        from model_manager import EmissionModel
                        self.manager.models[model_name] = EmissionModel(model_name)
                    
                    model = self.manager.models[model_name]
                    
                    # Add parameters
                    for param_name, param_stub, param_value in pending_add['valid_params']:
                        if param_stub not in model.data_modes:
                            model.add_data_mode(param_name, param_stub, param_value)
                    
                    # Add formula
                    model.add_formula_mode(pending_add['source_name'], pending_add['variable_name'], pending_add['formula'])
                    
                    # Log the addition
                    self.log_source_addition(model_name, pending_add['source_name'], 
                                        pending_add['variable_name'], pending_add['formula'], pending_add['valid_params'])
                    
                    # Recalculate and save
                    self.manager.calculate_all()
                    self.save_formulas_to_file()
                    
                    # Clear state
                    st.session_state.show_add_source_confirm = False
                    if 'pending_add_source' in st.session_state:
                        del st.session_state.pending_add_source
                    if 'temp_parameters' in st.session_state:
                        st.session_state.temp_parameters = [{"name": "", "stub": "", "value": 0.0}]
                    
                    st.success(f"Successfully added '{pending_add['source_name']}' to {model_name}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding emission source: {str(e)}")

    def validate_formula_json_format(self, json_data):
        errors = []
        
        if not isinstance(json_data, dict):
            errors.append("Root should be a JSON object")
            return {"valid": False, "errors": errors}
        
        for model_name, formulas in json_data.items():
            if not isinstance(formulas, dict):
                errors.append(f"Formulas for {model_name} should be a JSON object")
                continue
                
            if model_name not in self.manager.models:
                errors.append(f"Model '{model_name}' not found")
                continue
                
            model = self.manager.models[model_name]
            for formula_stub, formula_expr in formulas.items():
                if formula_stub not in model.formula_modes:
                    errors.append(f"Formula '{formula_stub}' not found in model '{model_name}'")
                
                if not isinstance(formula_expr, str):
                    errors.append(f"Formula expression for '{formula_stub}' should be a string")
                else:
                    # Try to parse the formula expression
                    try:
                        ast.parse(formula_expr)
                    except SyntaxError:
                        errors.append(f"Invalid formula syntax for '{formula_stub}': {formula_expr}")
        
        return {"valid": len(errors) == 0, "errors": errors}

    def validate_source_json_format(self, json_data):
        errors = []
        
        if not isinstance(json_data, dict):
            errors.append("Root should be a JSON object")
            return {"valid": False, "errors": errors}
        
        for model_name, sources in json_data.items():
            if not isinstance(sources, list):
                errors.append(f"Sources for {model_name} should be an array")
                continue
            
            for i, source in enumerate(sources):
                if not isinstance(source, dict):
                    errors.append(f"Source {i+1} in {model_name} should be an object")
                    continue
                
                # Check required fields
                required_fields = ["source_name", "variable_stub", "formula"]
                for field in required_fields:
                    if field not in source:
                        errors.append(f"Source {i+1} in {model_name} missing required field: {field}")
                    elif not isinstance(source[field], str):
                        errors.append(f"Field '{field}' in source {i+1} of {model_name} should be a string")
                
                # Check parameters if present
                if "parameters" in source:
                    if not isinstance(source["parameters"], list):
                        errors.append(f"Parameters in source {i+1} of {model_name} should be an array")
                    else:
                        for j, param in enumerate(source["parameters"]):
                            if not isinstance(param, dict):
                                errors.append(f"Parameter {j+1} in source {i+1} of {model_name} should be an object")
                                continue
                            
                            param_required = ["name", "stub", "value"]
                            for pfield in param_required:
                                if pfield not in param:
                                    errors.append(f"Parameter {j+1} in source {i+1} of {model_name} missing field: {pfield}")
                                elif pfield in ["name", "stub"] and not isinstance(param[pfield], str):
                                    errors.append(f"Parameter field '{pfield}' should be a string")
                                elif pfield == "value" and not isinstance(param[pfield], (int, float)):
                                    errors.append(f"Parameter value should be a number")
                
                # Check if variable stub already exists
                if "variable_stub" in source:
                    existing_vars = set()
                    for m in self.manager.models.values():
                        existing_vars.update(m.data_modes.keys())
                        existing_vars.update(m.formula_modes.keys())
                    
                    if source["variable_stub"] in existing_vars:
                        errors.append(f"Variable '{source['variable_stub']}' already exists in the system")
        
        return {"valid": len(errors) == 0, "errors": errors}

    def apply_json_formula_updates(self, json_data):
        try:
            update_count = 0
            
            for model_name, formulas in json_data.items():
                if model_name in self.manager.models:
                    model = self.manager.models[model_name]
                    
                    for formula_stub, new_formula in formulas.items():
                        if formula_stub in model.formula_modes:
                            old_formula = model.formula_modes[formula_stub].formula
                            formula_name = model.formula_modes[formula_stub].name
                            
                            # Only update if the formula has changed
                            if old_formula != new_formula:
                                # Log the change
                                self.log_formula_change(model_name, formula_name, old_formula, new_formula, formula_stub)
                                
                                # Update the formula
                                model.formula_modes[formula_stub].formula = new_formula
                                update_count += 1
            
            self.manager.calculate_all()
            self.save_formulas_to_file()
            
            if update_count > 0:
                st.success(f"Successfully updated {update_count} formulas from JSON file!")
            else:
                st.info("No formulas were changed (all formulas were already current)")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error applying formula updates: {str(e)}")

    def apply_json_source_additions(self, json_data):
        try:
            added_sources = 0
            added_parameters = 0
            
            for model_name, sources in json_data.items():
                # Get or create model
                if model_name not in self.manager.models:
                    try:
                        from model_manager import EmissionModel
                        self.manager.models[model_name] = EmissionModel(model_name)
                        st.info(f"Created new model: {model_name}")
                    except ImportError:
                        st.error(f"Cannot create model {model_name}. Please check model_manager.py exists.")
                        continue
                
                model = self.manager.models[model_name]
                
                for source in sources:
                    source_name = source["source_name"]
                    variable_stub = source["variable_stub"]
                    formula = source["formula"]
                    parameters = source.get("parameters", [])
                    
                    # Add parameters to model
                    for param in parameters:
                        param_name = param["name"]
                        param_stub = param["stub"]
                        param_value = param["value"]
                        
                        # Check if parameter doesn't already exist
                        if param_stub not in model.data_modes:
                            if hasattr(model, 'add_data_mode'):
                                model.add_data_mode(param_name, param_stub, param_value)
                            else:
                                from model_manager import DataMode
                                model.data_modes[param_stub] = DataMode(param_name, param_stub, param_value)
                            added_parameters += 1
                    
                    # Add formula to model
                    if hasattr(model, 'add_formula_mode'):
                        model.add_formula_mode(source_name, variable_stub, formula)
                    else:
                        from model_manager import FormulaMode
                        model.formula_modes[variable_stub] = FormulaMode(source_name, variable_stub, formula)
                    
                    # Log the new source addition
                    self.log_source_addition(model_name, source_name, variable_stub, formula, parameters)
                    added_sources += 1
            
            # Recalculate and save
            self.manager.calculate_all()
            self.save_formulas_to_file()
            
            st.success(f"Successfully added {added_sources} emission sources with {added_parameters} parameters!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error adding emission sources: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    def export_current_formulas_json(self):
        export_data = {}
        
        for model_name, model in self.manager.models.items():
            if model.formula_modes:  # Only include models with formulas
                export_data[model_name] = {}
                for stub, mode in model.formula_modes.items():
                    export_data[model_name][stub] = mode.formula
        
        return export_data

    def export_current_sources_json(self):
        export_data = {}
        
        for model_name, model in self.manager.models.items():
            if model.formula_modes:  # Only include models with formulas
                export_data[model_name] = []
                for stub, mode in model.formula_modes.items():
                    # Get related parameters for this formula
                    formula_vars = self.extract_variables(mode.formula)
                    parameters = []
                    
                    for var in formula_vars:
                        if var in model.data_modes:
                            data_mode = model.data_modes[var]
                            parameters.append({
                                "name": data_mode.name,
                                "stub": var,
                                "value": data_mode.value
                            })
                    
                    source_data = {
                        "source_name": mode.name,
                        "variable_stub": stub,
                        "formula": mode.formula,
                        "parameters": parameters
                    }
                    export_data[model_name].append(source_data)
        
        return export_data

    def display_change_history(self):
        st.subheader("Change History")
        
        if not st.session_state.formula_change_history:
            st.info("No changes recorded yet.")
            st.markdown("**Tip:** Make changes in other sections to see the modification history here.")
            return
        
        # Control panel
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Filter by model
            all_models = set()
            for record in st.session_state.formula_change_history:
                all_models.add(record['model'])
            
            model_list = ["All Models"] + sorted(list(all_models))
            selected_model = st.selectbox(
                "Filter by Model",
                model_list,
                key="change_history_model_filter"
            )
        
        with col2:
            # Filter by change type
            change_types = ["All Types", "Formula Update", "New Source Added", "Source Deleted", "Model Deleted"]
            selected_change_type = st.selectbox(
                "Change Type",
                change_types,
                key="change_history_type_filter"
            )
        
        with col3:
            # Clear history button
            if st.button("Clear History", use_container_width=True, help="Clear all change records"):
                st.session_state.formula_change_history = []
                st.success("Change history cleared!")
                st.rerun()
        
        # Apply filters
        filtered_history = st.session_state.formula_change_history.copy()
        
        if selected_model != "All Models":
            filtered_history = [record for record in filtered_history if record['model'] == selected_model]
        
        if selected_change_type != "All Types":
            filtered_history = [record for record in filtered_history if record['change_type'] == selected_change_type]
        
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
            formula_updates = len([r for r in filtered_history if r['change_type'] == 'Formula Update'])
            st.metric("Formula Updates", formula_updates)
        
        with col3:
            sources_added = len([r for r in filtered_history if r['change_type'] == 'New Source Added'])
            sources_deleted = len([r for r in filtered_history if r['change_type'] == 'Source Deleted'])
            st.metric("Source Changes", f"+{sources_added}/-{sources_deleted}")
        
        with col4:
            models_deleted = len([r for r in filtered_history if r['change_type'] == 'Model Deleted'])
            st.metric("Models Deleted", models_deleted)
        
        # Display detailed history
        st.markdown("### Change Records")
        
        # Create DataFrame for display
        history_data = []
        for record in filtered_history:
            # Format display based on change type
            if record['change_type'] == 'Source Deleted':
                old_display = f"Source: {record['old_formula'][:30]}..."
                new_display = "DELETED"
            elif record['change_type'] == 'Model Deleted':
                old_display = record['old_formula']
                new_display = "DELETED"
            else:
                old_display = record['old_formula'][:50] + "..." if len(record['old_formula']) > 50 else record['old_formula']
                new_display = record['new_formula'][:50] + "..." if len(record['new_formula']) > 50 else record['new_formula']
            
            # Additional info
            extra_info = ""
            if 'parameters_added' in record and record['parameters_added']:
                extra_info = f"{record['parameters_added']} params"
            elif 'parameters_deleted' in record and record['parameters_deleted']:
                extra_info = f"{record['parameters_deleted']} params del"
            elif 'sources_deleted' in record and record['sources_deleted']:
                extra_info = f"{record['sources_deleted']} sources del"
            
            history_data.append({
                "Date": record['timestamp'].split(' ')[0],
                "Time": record['timestamp'].split(' ')[1],
                "Model": record['model'],
                "Target": record['formula_name'],
                "Variable": record['variable'],
                "Type": record['change_type'],
                "Old": old_display,
                "New": new_display,
                "Extra": extra_info
            })
        
        # Display with pagination
        records_per_page = 15
        total_pages = (len(history_data) - 1) // records_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}",
                key="change_history_page"
            )
            
            start_idx = (page - 1) * records_per_page
            end_idx = start_idx + records_per_page
            df_display = pd.DataFrame(history_data[start_idx:end_idx])
        else:
            df_display = pd.DataFrame(history_data)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Target": st.column_config.TextColumn("Target", width="large"),
                "Variable": st.column_config.TextColumn("Variable", width="small"),
                "Type": st.column_config.TextColumn("Type", width="medium"),
                "Old": st.column_config.TextColumn("Old", width="large"),
                "New": st.column_config.TextColumn("New", width="large"),
                "Extra": st.column_config.TextColumn("Info", width="small")
            }
        )

    def visualize_model_structure(self, output_path="../data/graphs/model_structure"):
        try:
            from graphviz import Digraph
        except ImportError:
            st.error("Graphviz not installed.")
            return None
        
        try:
            # Ensure output directory exists
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create directed graph
            dot = Digraph(name="ModelStructure", format="png")
            dot.attr(rankdir="LR")  # Left to right layout
            dot.attr('node', fontname='Arial', fontsize='10')
            dot.attr('edge', fontname='Arial', fontsize='8')
            
            # Collect all variables first
            all_vars = {}
            for model in self.manager.models.values():
                for stub, mode in model.data_modes.items():
                    all_vars[stub] = mode.value
            
            # Calculate all formulas to get results
            for model in self.manager.models.values():
                for stub, mode in model.formula_modes.items():
                    mode.evaluate(all_vars)
                    all_vars[stub] = mode.result
            
            # Create subgraphs for each model
            for model in self.manager.models.values():
                with dot.subgraph(name=f"cluster_{model.model_name}") as sub:
                    sub.attr(label=f"Model: {model.model_name}", 
                            style="dashed", 
                            color="blue",
                            fontsize="12",
                            fontname="Arial Bold")
                    
                    # Add data nodes (input parameters)
                    for stub, mode in model.data_modes.items():
                        sub.node(stub, 
                                shape="box", 
                                style="filled", 
                                fillcolor="#E6F3FF",  # Light blue
                                color="#4A90E2")      # Border blue
                    
                    # Add formula nodes (computed values)
                    for stub, mode in model.formula_modes.items():
                        sub.node(stub, 
                                shape="ellipse", 
                                style="filled", 
                                fillcolor="#E6FFE6",  # Light green
                                color="#50C878")      # Border green
                        
                        # Add edges from dependencies to this formula
                        dependencies = self.extract_variables(mode.formula)
                        for var in dependencies:
                            if var in all_vars:  # Only add edge if variable exists
                                dot.edge(var, stub, color="#666666")
            
            # Render the graph
            dot.render(output_path, cleanup=True)
            return f"{output_path}.png"
            
        except Exception as e:
            st.error(f"Error generating graph: {str(e)}")
            return None
    
    def display_dependency_viewer(self):
        st.subheader("Dependency Viewer")
        
        if not self.manager:
            st.error("Model not initialized")
            return
        
        # Create dependency graph
        st.markdown("### Formula Dependencies")
        
        # Add visualization option
        view_type = st.radio("View Type", ["Tree View", "Table View", "Graph View"], horizontal=True)
        
        if view_type == "Tree View":
            # Get variable name mapping
            variable_mapping = self.get_variable_name_mapping()
            
            for model_name, model in self.manager.models.items():
                if model.formula_modes:
                    with st.expander(f"📂 {model_name}"):
                        for stub, mode in model.formula_modes.items():
                            st.markdown(f"**{mode.name}** (`{stub}`)")
                            st.code(f"{stub} = {mode.formula}")
                            
                            # Extract and show dependencies with variable names
                            deps = self.extract_variables(mode.formula)
                            if deps:
                                st.markdown("Dependencies:")
                                for dep in deps:
                                    # Show variable name instead of model location
                                    if dep in variable_mapping:
                                        st.markdown(f"  - `{dep}` ({variable_mapping[dep]})")
                                    else:
                                        st.markdown(f"  - `{dep}` ⚠️ (not found)")
                            
                            # Show result if calculated
                            if hasattr(mode, 'result') and mode.result is not None:
                                st.info(f"Result: {mode.result:.6f}")
        
        elif view_type == "Table View":
            # Create table view of all formulas
            variable_mapping = self.get_variable_name_mapping()
            data = []
            for model_name, model in self.manager.models.items():
                for stub, mode in model.formula_modes.items():
                    deps = self.extract_variables(mode.formula)
                    # Create dependency display with variable names
                    dep_display = []
                    for dep in deps:
                        if dep in variable_mapping:
                            dep_display.append(f"{dep} ({variable_mapping[dep]})")
                        else:
                            dep_display.append(f"{dep} (not found)")
                    
                    result_text = "Not calculated"
                    if hasattr(mode, 'result') and mode.result is not None:
                        result_text = f"{mode.result:.6f}"
                    
                    data.append({
                        "Model": model_name,
                        "Name": mode.name,
                        "Variable": stub,
                        "Formula": mode.formula,
                        "Dependencies": ", ".join(dep_display) if dep_display else "None",
                        "Result": result_text
                    })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No formulas found")
        
        else: 
            output_path = "../data/graphs/model_structure"
            
            with st.spinner("Generating model structure graph..."):
                graph_path = self.visualize_model_structure(output_path)
            
            # Display the graph
            if graph_path and os.path.exists(graph_path):
                try:
                    with open(graph_path, "rb") as img_file:
                        img_data = img_file.read()
                        st.image(img_data, caption="Model Structure Graph", use_column_width=True)
                    
                    # Add download button
                    with open(graph_path, "rb") as file:
                        st.download_button(
                            label="Download Graph",
                            data=file.read(),
                            file_name=os.path.basename(graph_path),
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        
                except Exception as e:
                    st.error(f"Error displaying graph: {str(e)}")
            else:
                st.error("Failed to generate graph. Please ensure Graphviz is installed.")
    
    def find_variable_location(self, variable: str) -> str:
        for model_name, model in self.manager.models.items():
            if variable in model.data_modes:
                return f"{model_name} (data)"
            if variable in model.formula_modes:
                return f"{model_name} (formula)"
        return None
    
    def save_formulas_to_file(self):
        if not hasattr(self.manager, 'formula_file'):
            # Default path if formula_file attribute doesn't exist
            formula_file = "../data/formula.json"
        else:
            formula_file = self.manager.formula_file
            
        formulas = {}
        for model_name, model in self.manager.models.items():
            if model.formula_modes:
                formulas[model_name] = {}
                for formula_name, mode in model.formula_modes.items():
                    formulas[model_name][mode.name] = {
                        "stub": mode.stub,
                        "formula": mode.formula
                    }
             
        try:
            import os
            os.makedirs(os.path.dirname(formula_file), exist_ok=True)
            with open(formula_file, "w", encoding="utf-8") as f:
                json.dump(formulas, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving formulas: {str(e)}")