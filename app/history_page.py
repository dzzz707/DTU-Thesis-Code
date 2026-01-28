import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

class HistoryPageModule:
    
    def __init__(self, manager):
        self.manager = manager
    
    def display(self):
        st.header("Change History")
        st.markdown("Comprehensive view of all modifications made across the system")
        st.markdown("---")
        
        # Get combined history
        combined_history = self.get_combined_history()
        
        if not combined_history:
            self.display_empty_state()
            return
        
        # Control panel
        self.display_control_panel()
        
        # Apply filters and display content
        filtered_history = self.apply_filters(combined_history)
        
        if not filtered_history:
            st.warning("No records found for the selected filters.")
            return
        
        # Display summary and detailed history
        self.display_summary_statistics(filtered_history)
        self.display_detailed_history(filtered_history)
    
    def get_combined_history(self) -> List[Dict]:
        combined = []
        
        # Add Quick Data Update history
        if 'change_history' in st.session_state and st.session_state.change_history:
            for record in st.session_state.change_history:
                normalized_record = {
                    "timestamp": record['timestamp'],
                    "source_module": "Quick Data Update",
                    "change_type": "Parameter Update",
                    "model": record['model'],
                    "target_name": record['parameter'],
                    "target_variable": record['variable'],
                    "old_value": str(record['old_value']),
                    "new_value": str(record['new_value']),
                    "change_amount": record.get('change', 0),
                    "change_percent": record.get('change_percent', 0),
                    "details": f"Updated parameter from {record['old_value']:.6f} to {record['new_value']:.6f}",
                    "original_record": record
                }
                combined.append(normalized_record)
        
        # Add Model Structure Editor history
        if 'formula_change_history' in st.session_state and st.session_state.formula_change_history:
            for record in st.session_state.formula_change_history:
                normalized_record = {
                    "timestamp": record['timestamp'],
                    "source_module": "Model Structure Editor",
                    "change_type": record['change_type'],
                    "model": record['model'],
                    "target_name": record['formula_name'],
                    "target_variable": record['variable'],
                    "old_value": record['old_formula'],
                    "new_value": record['new_formula'],
                    "change_amount": None,
                    "change_percent": None,
                    "details": self.format_formula_change_details(record),
                    "original_record": record,
                    "parameters_added": record.get('parameters_added', 0)
                }
                combined.append(normalized_record)
        
        # Sort by timestamp (newest first)
        combined.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return combined
    
    def format_formula_change_details(self, record: Dict) -> str:
        if record['change_type'] == "New Source Added":
            params_text = ""
            if 'parameters_added' in record and record['parameters_added'] > 0:
                params_text = f" with {record['parameters_added']} parameter(s)"
            return f"Added new emission source '{record['formula_name']}'{params_text}"
        else:
            return f"Updated formula from '{record['old_formula']}' to '{record['new_formula']}'"
    
    def display_empty_state(self):
        st.info("No modification history found.")
        
        # Add some vertical spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Use columns with smaller gap
        col1, gap, col2 = st.columns([3, 1, 3])
        
        with col1:
            # Use simple container without custom styling
            st.markdown("""
            ### Quick Data Update
            - Modify emission factors and activity data
            - Changes will be tracked automatically
            """)
            
            # Add spacing before button for alignment
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            if st.button("Go to Quick Data Update", use_container_width=True, type="primary", key="goto_quick_data"):
                st.session_state.selected_tab = 1
                st.session_state.selected_function = None
                st.query_params["tab"] = "1"
                st.rerun()
        
        with gap:
            # Empty gap column
            st.write("")
        
        with col2:
            # Use simple container without custom styling
            st.markdown("""
            ### Model Structure Editor
            - Edit calculation formulas and add new sources
            - Formula changes will be logged
            """)
            
            # Add same spacing before button for alignment
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            if st.button("Go to Model Structure Editor", use_container_width=True, type="primary", key="goto_model_editor"):
                st.session_state.selected_tab = 2
                st.session_state.selected_function = None
                st.query_params["tab"] = "2"
                st.rerun()
        
        # Add bottom spacing
        st.markdown("<br>", unsafe_allow_html=True)

    def display_control_panel(self):
        st.subheader("Filters & Controls")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            # Source module filter
            source_options = ["All Sources", "Quick Data Update", "Model Structure Editor"]
            selected_source = st.selectbox(
                "Filter by Source",
                source_options,
                key="history_source_filter"
            )
        
        with col2:
            # Model filter
            combined_history = self.get_combined_history()
            all_models = set()
            for record in combined_history:
                all_models.add(record['model'])
            
            model_list = ["All Models"] + sorted(list(all_models))
            selected_model = st.selectbox(
                "Filter by Model",
                model_list,
                key="history_model_filter"
            )
        
        with col3:
            # Time filter
            time_filter = st.selectbox(
                "Time Period",
                ["All Time", "Last Day", "Last Week", "Last Month"],
                key="history_time_filter"
            )
        
        st.markdown("---")
    
    def apply_filters(self, combined_history: List[Dict]) -> List[Dict]:
        filtered_history = combined_history.copy()
        
        # Source module filter
        selected_source = st.session_state.get('history_source_filter', 'All Sources')
        if selected_source != "All Sources":
            filtered_history = [
                record for record in filtered_history 
                if record['source_module'] == selected_source
            ]
        
        # Model filter
        selected_model = st.session_state.get('history_model_filter', 'All Models')
        if selected_model != "All Models":
            filtered_history = [
                record for record in filtered_history 
                if record['model'] == selected_model
            ]
        
        # Time filter
        time_filter = st.session_state.get('history_time_filter', 'All Time')
        if time_filter != "All Time":
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
        
        return filtered_history
    
    def display_summary_statistics(self, filtered_history: List[Dict]):
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Changes", len(filtered_history))
        
        with col2:
            # Count parameter updates
            param_updates = len([r for r in filtered_history if r['change_type'] == 'Parameter Update'])
            st.metric("Parameter Updates", param_updates)
        
        with col3:
            # Count formula changes
            formula_changes = len([r for r in filtered_history if r['change_type'] in ['Formula Update', 'New Source Added']])
            st.metric("Formula Changes", formula_changes)
        
        with col4:
            if filtered_history:
                latest_change = filtered_history[0]['timestamp'].split(' ')[0]
                st.metric("Latest Change", latest_change)
        
        # Source module breakdown
        if len(filtered_history) > 0:
            st.markdown("### Source Module Breakdown")
            
            source_counts = {}
            for record in filtered_history:
                source = record['source_module']
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Display as columns
            cols = st.columns(len(source_counts))
            for i, (source, count) in enumerate(source_counts.items()):
                with cols[i]:
                    percentage = (count / len(filtered_history)) * 100
                    st.metric(
                        source.replace(" ", "\n"), 
                        count, 
                        f"{percentage:.1f}%"
                    )
    
    def display_detailed_history(self, filtered_history: List[Dict]):
        st.subheader("Detailed Change Records")
        
        if not filtered_history:
            return
        
        # Create DataFrame for display
        history_data = []
        for record in filtered_history:
            # Format timestamp
            timestamp_obj = datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S")
            
            # Format values based on change type
            if record['change_type'] == 'Parameter Update':
                old_display = f"{float(record['old_value']):.6f}"
                new_display = f"{float(record['new_value']):.6f}"
                change_info = ""
                if record['change_amount'] is not None:
                    change_info = f"{record['change_amount']:+.6f}"
                    if record['change_percent'] not in [None, float('inf')]:
                        change_info += f" ({record['change_percent']:+.1f}%)"
            else:
                # Formula changes - truncate for display
                old_display = record['old_value'][:50] + "..." if len(record['old_value']) > 50 else record['old_value']
                new_display = record['new_value'][:50] + "..." if len(record['new_value']) > 50 else record['new_value']
                change_info = ""
                if 'parameters_added' in record and record['parameters_added']:
                    change_info = f"{record['parameters_added']} params"
            
            history_data.append({
                "Date": timestamp_obj.strftime("%Y-%m-%d"),
                "Time": timestamp_obj.strftime("%H:%M:%S"),
                "Source": record['source_module'],
                "Type": record['change_type'],
                "Model": record['model'],
                "Target": record['target_name'],
                "Variable": record['target_variable'],
                "Old Value": old_display,
                "New Value": new_display,
                "Change": change_info
            })
        
        # Pagination
        records_per_page = 25
        total_pages = (len(history_data) - 1) // records_per_page + 1
        
        if total_pages > 1:
            col1, col2 = st.columns([1, 4])
            with col1:
                page = st.selectbox(
                    "Page",
                    range(1, total_pages + 1),
                    format_func=lambda x: f"Page {x} of {total_pages}",
                    key="detailed_history_page"
                )
            
            start_idx = (page - 1) * records_per_page
            end_idx = start_idx + records_per_page
            df_display = pd.DataFrame(history_data[start_idx:end_idx])
        else:
            df_display = pd.DataFrame(history_data)
        
        # Display table
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Source": st.column_config.TextColumn("Source", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="medium"),
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Target": st.column_config.TextColumn("Target", width="large"),
                "Variable": st.column_config.TextColumn("Variable", width="medium"),
                "Old Value": st.column_config.TextColumn("Old Value", width="medium"),
                "New Value": st.column_config.TextColumn("New Value", width="medium"),
                "Change": st.column_config.TextColumn("Change", width="small")
            }
        )
        
        # Detailed record viewer
        st.markdown("### Record Details")
        if filtered_history:
            selected_record_idx = st.selectbox(
                "Select a record to view full details:",
                range(len(filtered_history)),
                format_func=lambda x: f"{filtered_history[x]['timestamp']} - {filtered_history[x]['source_module']} - {filtered_history[x]['target_name']}",
                key="detailed_record_viewer"
            )
            
            if selected_record_idx is not None:
                record = filtered_history[selected_record_idx]
                
                with st.expander("Full Record Details", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**Basic Information:**")
                        st.text(f"Timestamp: {record['timestamp']}")
                        st.text(f"Source Module: {record['source_module']}")
                        st.text(f"Change Type: {record['change_type']}")
                        st.text(f"Model: {record['model']}")
                        st.text(f"Target: {record['target_name']}")
                        st.text(f"Variable: {record['target_variable']}")
                        
                        if record['change_type'] == 'Parameter Update':
                            if record['change_amount'] is not None:
                                st.text(f"Change Amount: {record['change_amount']:+.6f}")
                            if record['change_percent'] not in [None, float('inf')]:
                                st.text(f"Change Percent: {record['change_percent']:+.1f}%")
                        
                        if 'parameters_added' in record and record['parameters_added']:
                            st.text(f"Parameters Added: {record['parameters_added']}")
                    
                    with col2:
                        st.markdown("**Change Details:**")
                        if record['change_type'] == 'Parameter Update':
                            col2a, col2b = st.columns(2)
                            with col2a:
                                st.markdown("*Old Value:*")
                                st.code(record['old_value'])
                            with col2b:
                                st.markdown("*New Value:*")
                                st.code(record['new_value'])
                        else:
                            st.markdown("*Old Formula/Value:*")
                            st.code(record['old_value'], language="python")
                            st.markdown("*New Formula/Value:*")
                            st.code(record['new_value'], language="python")
                        
                        st.markdown("*Description:*")
                        st.text(record['details'])

        # Actions section
        st.markdown("---")
        st.markdown("### Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export History", use_container_width=True, help="Export history to CSV"):
                self.export_history()

        with col2:
            if st.button("Clear All History", use_container_width=True, help="Clear all history records"):
                self.clear_all_history()
    
    def export_history(self):
        try:
            combined_history = self.get_combined_history()
            
            # Prepare data for export
            export_data = []
            for record in combined_history:
                export_data.append({
                    "Timestamp": record['timestamp'],
                    "Source_Module": record['source_module'],
                    "Change_Type": record['change_type'],
                    "Model": record['model'],
                    "Target_Name": record['target_name'],
                    "Target_Variable": record['target_variable'],
                    "Old_Value": record['old_value'],
                    "New_Value": record['new_value'],
                    "Change_Amount": record.get('change_amount', ''),
                    "Change_Percent": record.get('change_percent', ''),
                    "Details": record['details'],
                    "Parameters_Added": record.get('parameters_added', '')
                })
            
            # Create CSV
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"carbon_calculator_history_{timestamp}.csv"
            
            st.download_button(
                label="Download History CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
            st.success("History export ready for download!")
            
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    def clear_all_history(self):
        @st.dialog("Clear All History")
        def confirm_clear():
            st.warning("This action will permanently delete all change history records from both Quick Data Update and Model Structure Editor.")
            st.markdown("Are you sure you want to continue?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Cancel", use_container_width=True):
                    st.rerun()
            
            with col2:
                if st.button("Clear All History", use_container_width=True, type="primary"):
                    # Clear all histories
                    if 'change_history' in st.session_state:
                        st.session_state.change_history = []
                    if 'formula_change_history' in st.session_state:
                        st.session_state.formula_change_history = []
                    
                    st.success("All history records cleared!")
                    st.rerun()
        
        confirm_clear()