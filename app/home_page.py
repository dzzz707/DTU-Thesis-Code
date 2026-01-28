import streamlit as st

class HomePageModule:
    
    def __init__(self, manager):
        self.manager = manager
    
    def display(self):
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Hero section with simple introduction
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; color: #4a90e2;">
                🌱 Carbon Emission Calculator
            </h1>
            <p style="font-size: 1.3rem; color: #8e8ea0; margin-bottom: 3rem; max-width: 600px; margin-left: auto; margin-right: auto;">
                A comprehensive system for managing and calculating your organization's carbon footprint across all emission scopes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Function navigation cards
        self.display_function_cards()
        
        # Simple footer
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6c757d;">
            <p>Select a function above to get started with your carbon emission management.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_function_cards(self):
        # First row
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            # Quick Data Update Card
            if st.button("Quick Data Update", key="home_nav_quick", use_container_width=True, type="primary"):
                st.session_state.selected_tab = 1
                st.session_state.selected_function = None
                st.query_params["tab"] = "1"
                st.rerun()
            
            st.markdown("""
            <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
                <p style="color: #8e8ea0; margin: 0;">
                    Quickly modify emission factors and activity data without changing calculation logic
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Model Structure Editor Card
            if st.button("Model Structure Editor", key="home_nav_editor", use_container_width=True, type="primary"):
                st.session_state.selected_tab = 2
                st.session_state.selected_function = None
                st.query_params["tab"] = "2"
                st.rerun()
            
            st.markdown("""
            <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
                <p style="color: #8e8ea0; margin: 0;">
                    Edit calculation formulas, add new emission sources, and manage model dependencies
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row
        col3, col4 = st.columns(2, gap="large")
        
        with col3:
            # AI Smart Assistant Card
            if st.button("AI Smart Assistant", key="home_nav_ai", use_container_width=True, type="primary"):
                st.session_state.selected_tab = 3
                st.session_state.selected_function = None
                st.query_params["tab"] = "3"
                st.rerun()
            
            st.markdown("""
            <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
                <p style="color: #8e8ea0; margin: 0;">
                    Use AI to analyze ISO documents, get intelligent suggestions, and automate updates
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Change History Card
            if st.button("Change History", key="home_nav_history", use_container_width=True, type="primary"):
                st.session_state.selected_tab = 4
                st.session_state.selected_function = None
                st.query_params["tab"] = "4"
                st.rerun()
            
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <p style="color: #8e8ea0; margin: 0;">
                    View comprehensive history of all modifications made across the system
                </p>
            </div>
            """, unsafe_allow_html=True)