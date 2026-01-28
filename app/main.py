import streamlit as st
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_manager import ModelManager, get_model_template
from home_page import HomePageModule
from quick_data_update import QuickDataUpdateModule
from model_structure_editor import ModelStructureEditorModule
from ai_smart_assistant import AISmartAssistantModule
from history_page import HistoryPageModule

st.set_page_config(
    page_title="Carbon Emission Calculator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_global_styles():
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem !important;
        padding-left: 6rem !important;
        padding-right: 6rem !important;
        padding-bottom: 2rem !important;
        max-width: none !important;
    }
    
    .css-1d391kg {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
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
    
    .stButton > button[kind="primary"],
    button[key*="new_chat_btn"],
    button[key*="clear_all_btn"],
    button[key*="nav_"],
    button[data-testid="baseButton-primary"] {
        background-color: #1f4e79 !important;
        border-color: #4a90e2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    button[key*="new_chat_btn"]:hover,
    button[key*="clear_all_btn"]:hover,
    button[key*="nav_"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #2d5aa0 !important;
        border-color: #5ba3f5 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
    }
    
    button[key="nav_home"],
    button[key="nav_quick_update"],
    button[key="nav_model_editor"], 
    button[key="nav_ai_assistant"],
    button[key="nav_history"] {
        background-color: #1f4e79 !important;
        border-color: #4a90e2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        color: #e2e8f0 !important;
    }
    
    .stButton > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #4a5568 !important;
        border-color: #4a90e2 !important;
        color: #ffffff !important;
    }
    
    .stFormSubmitButton > button {
        background-color: #1f4e79 !important;
        border-color: #4a90e2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stFormSubmitButton > button:hover {
        background-color: #2d5aa0 !important;
        border-color: #5ba3f5 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
    }
    
    .stSelectbox > div > div {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        color: #e2e8f0 !important;
        margin: 0.5rem 0 !important;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        color: #e2e8f0 !important;
        margin: 0.25rem 0 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 1px #4a90e2 !important;
    }
    
    .metric-container {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 10px;
        padding: 1rem;
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
    
    .stAlert {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    .stSlider [data-testid="stTickBar"] > div,
    .stSlider div[class*="st-emotion-cache"],
    [data-baseweb="slider"] [class*="st-emotion-cache"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        background-color: transparent !important;
    }
    
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
    
    .stDataFrame {
        background-color: #2d3748 !important;
        margin: 0.5rem 0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
        padding: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748 !important;
        border-radius: 8px 8px 0 0 !important;
        color: #8e8ea0 !important;
        padding: 0.75rem 1.5rem !important;
        border: 1px solid #4a5568 !important;
        border-bottom: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3d4758 !important;
        color: #e2e8f0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important;
        color: #ffffff !important;
        border-color: #4a90e2 !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem 0 !important;
    }
    
    .stCheckbox > label {
        color: #e2e8f0 !important;
    }
    
    .stCheckbox > label > span {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
    }
    
    .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    .stSlider > div > div > div {
        background-color: #4a90e2 !important;
    }
    
    .stProgress > div > div {
        background-color: #4a90e2 !important;
    }
    
    hr {
        border-color: #4a5568 !important;
        margin: 1rem 0 !important;
    }
    
    .stCaption {
        color: #8e8ea0 !important;
    }
    
    .stCodeBlock {
        background-color: #1a202c !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px !important;
    }
    
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    .stMarkdown a {
        color: #4a90e2 !important;
    }
    
    .stMarkdown a:hover {
        color: #5ba3f5 !important;
    }
    
    .stAlert {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1f4e79 !important;
        color: #ffffff !important;
    }
    
    .stDateInput > div > div {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
    }
    
    .stTimeInput > div > div {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
    }
    
    .stFileUploader > div {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #4a90e2 !important;
    }
    
    .stColorPicker > div {
        background-color: #2d3748 !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a202c;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #718096;
    }
    
    .stDialog {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 12px !important;
    }
    
    .stButton > button:disabled {
        background-color: #1a202c !important;
        border-color: #2d3748 !important;
        color: #4a5568 !important;
        cursor: not-allowed !important;
        opacity: 0.6 !important;
    }
    
    @media (max-width: 768px) {
        .stButton > button {
            font-size: 0.85rem !important;
            padding: 0.5rem 0.7rem !important;
            min-height: 2.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    if 'selected_tab' not in st.session_state:
        params = st.query_params
        tab_index = params.get('tab', '0')
        try:
            st.session_state.selected_tab = int(tab_index)
        except:
            st.session_state.selected_tab = 0  # Default to Home tab
    
    # Initialize sub-function selection
    if 'selected_function' not in st.session_state:
        st.session_state.selected_function = None
    
    # Initialize manager as empty (no default data loaded)
    if 'manager' not in st.session_state:
        st.session_state.manager = ModelManager()
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
    
    # Other session state variables
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0


def display_model_upload_section():
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    with st.expander("Model & Data", expanded=not st.session_state.manager.is_model_loaded()):
        # Check if model is loaded
        if st.session_state.manager.is_model_loaded():
            st.success("Model Loaded")
            
            # Show statistics
            total_models = len(st.session_state.manager.models)
            total_params = sum(len(model.data_modes) for model in st.session_state.manager.models.values())
            total_formulas = sum(len(model.formula_modes) for model in st.session_state.manager.models.values())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Models", total_models)
            with col2:
                st.metric("Parameters", total_params)
            
            st.caption(f"{total_formulas} formulas loaded")
            
            st.markdown("---")
            
            # Export current model button
            export_data = st.session_state.manager.export_complete_model()
            export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Export Current Model",
                data=export_json,
                file_name="exported_model.json",
                mime="application/json",
                key="download_export_btn",
                use_container_width=True
            )
            
            # Clear model button
            if st.button("Clear Model", key="clear_model_btn", use_container_width=True):
                st.session_state.manager.clear_models()
                st.session_state.last_uploaded_file = None  # Reset upload tracking
                st.rerun()
        
        else:
            st.warning("No Model Loaded")
            st.caption("Upload a model file to start using the system.")
        
        st.markdown("---")
        
        # Download template section
        st.caption("**Get Started**")
        template_json = get_model_template()
        st.download_button(
            label="Download Template",
            data=template_json,
            file_name="model_template.json",
            mime="application/json",
            key="download_template_btn",
            use_container_width=True
        )
        
        st.markdown("")
        
        # Upload section
        st.caption("**Upload Model**")
        uploaded_file = st.file_uploader(
            "Choose JSON file",
            type=["json"],
            key="model_uploader",
            label_visibility="collapsed"
        )
        
        # Process uploaded file only if it's a new file
        if uploaded_file is not None:
            # Check if this is a new file (not already processed)
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.last_uploaded_file != file_id:
                # This is a new file, process it
                try:
                    # Reset file position to beginning
                    uploaded_file.seek(0)
                    success, message, stats = st.session_state.manager.load_complete_model(uploaded_file)
                    
                    if success:
                        st.session_state.last_uploaded_file = file_id  # Mark as processed
                        st.success(f"{message}")
                        st.caption(f"{stats['models']} models, {stats['parameters']} parameters, {stats['formulas']} formulas")
                        st.rerun()
                    else:
                        st.error(f"{message}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def display_sidebar():
    """Display the sidebar with navigation and model upload"""
    with st.sidebar:
        st.header("🌱 Carbon Calculator")
        st.markdown("---")
        
        # Model Upload Section
        display_model_upload_section()
        
        st.markdown("---")
        st.subheader("Navigation")
        
        # Check if model is loaded for enabling/disabling navigation
        model_loaded = st.session_state.manager.is_model_loaded()
        
        # Home (always enabled)
        if st.button("Home", 
                    key="nav_home",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_tab == 0 else "secondary"):
            st.session_state.selected_tab = 0
            st.session_state.selected_function = None
            st.query_params["tab"] = "0"
            st.rerun()
        
        # Quick Data Update (requires model)
        if st.button("Quick Data Update", 
                    key="nav_quick_update",
                    use_container_width=True,
                    disabled=not model_loaded,
                    type="primary" if st.session_state.selected_tab == 1 else "secondary"):
            if model_loaded:
                st.session_state.selected_tab = 1
                st.session_state.selected_function = None
                st.query_params["tab"] = "1"
                st.rerun()
        
        # Model Editor (requires model)
        if st.button("Model Structure Editor", 
                    key="nav_model_editor",
                    use_container_width=True,
                    disabled=not model_loaded,
                    type="primary" if st.session_state.selected_tab == 2 else "secondary"):
            if model_loaded:
                st.session_state.selected_tab = 2
                st.session_state.selected_function = None
                st.query_params["tab"] = "2"
                st.rerun()
        
        # AI Assistant (requires model)
        if st.button("AI Smart Assistant", 
                    key="nav_ai_assistant",
                    use_container_width=True,
                    disabled=not model_loaded,
                    type="primary" if st.session_state.selected_tab == 3 else "secondary"):
            if model_loaded:
                st.session_state.selected_tab = 3
                st.session_state.selected_function = None
                st.query_params["tab"] = "3"
                st.rerun()
        
        # Change History (requires model)
        if st.button("Change History", 
                    key="nav_history",
                    use_container_width=True,
                    disabled=not model_loaded,
                    type="primary" if st.session_state.selected_tab == 4 else "secondary"):
            if model_loaded:
                st.session_state.selected_tab = 4
                st.session_state.selected_function = None
                st.query_params["tab"] = "4"
                st.rerun()
        
        # Show hint if no model loaded
        if not model_loaded:
            st.markdown("")
            st.info("Upload a model to enable all features")
        
        # Footer
        st.markdown("---")
        st.caption("Carbon Calculator v2.0")


def display_breadcrumb():
    tabs = ["Home", "Quick Data Update", "Model Structure Editor", "AI Smart Assistant", "Change History"]
    current_tab = tabs[st.session_state.selected_tab]
    
    # Only show breadcrumb for non-home pages
    if st.session_state.selected_tab != 0:
        breadcrumb = f"Home > {current_tab}"
        
        # Handle sub-functions for non-home tabs
        if st.session_state.selected_function:
            function_names = {
                "scope_update": "Update by Scope",
                "overview": "Quick Overview", 
                "change_history": "Change History",
                "edit_formulas": "Edit Formulas",
                "add_source": "Add New Source",
                "view_dependencies": "View Dependencies",
                "formula_history": "Formula History"
            }
            
            if st.session_state.selected_function in function_names:
                breadcrumb += f" > {function_names[st.session_state.selected_function]}"
        
        st.caption(breadcrumb)


def display_no_model_warning():
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; color: #ecc94b;">
            ⚠️ No Model Loaded
        </h1>
        <p style="font-size: 1.3rem; color: #8e8ea0; margin-bottom: 2rem; max-width: 600px; margin-left: auto; margin-right: auto;">
            Please upload a carbon emission model file to use this feature.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### How to get started:
        
        1. **Download the template** from the sidebar (Download Template)
        2. **Edit the template** with your emission sources and data
        3. **Upload your model** using the file uploader in the sidebar
        
        ---
        
        ### Template includes:
        
        | Scope | Example Calculations |
        |-------|---------------------|
        | **Scope 1** | Addition, subtraction, multiplication |
        | **Scope 2** | Division, efficiency calculations |
        | **Scope 3** | Power/exponent, square root, complex formulas |
        
        ---
        
        ### Supported operators:
        
        - `+` Addition
        - `-` Subtraction  
        - `*` Multiplication
        - `/` Division
        - `**` Power (e.g., `x ** 2` for square, `x ** 0.5` for square root)
        - `()` Parentheses for grouping
        """)


def main():

    add_global_styles()
    initialize_session_state()
    display_sidebar()
    display_breadcrumb()
    
    # Check if model is loaded
    model_loaded = st.session_state.manager.is_model_loaded()
    
    # Initialize modules with proper error handling
    try:
        home_module = HomePageModule(st.session_state.manager)
        
        # Display selected tab content
        if st.session_state.selected_tab == 0:
            home_module.display()
            
        elif not model_loaded:
            display_no_model_warning()
            
        else:
            # Only initialize other modules if model is loaded
            quick_data_module = QuickDataUpdateModule(st.session_state.manager)
            ai_assistant_module = AISmartAssistantModule(st.session_state.manager, st.session_state.api_key)
            model_editor_module = ModelStructureEditorModule(st.session_state.manager)
            history_module = HistoryPageModule(st.session_state.manager)
            
            if st.session_state.selected_tab == 1:
                quick_data_module.display()
            elif st.session_state.selected_tab == 2:
                model_editor_module.display()
            elif st.session_state.selected_tab == 3:
                ai_assistant_module.display()
            elif st.session_state.selected_tab == 4:
                history_module.display()
            
    except Exception as e:
        st.error(f"Error initializing modules: {str(e)}")
        st.error("Please check that all required files are present:")
        st.code("""
Required files:
- home_page.py
- model_manager.py
- quick_data_update.py
- model_structure_editor.py
- ai_smart_assistant.py
- history_page.py
        """)


if __name__ == "__main__":
    main()