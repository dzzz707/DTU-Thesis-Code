import streamlit as st
from ai_chatbox import AISmartAssistantModule as ChatModule
from optimization_analytics import OptimizationAnalyticsModule

class AISmartAssistantModule:
    
    def __init__(self, manager, api_key=None):
        self.manager = manager
        self.api_key = api_key
        self.chat_module = None
        self.analytics_module = OptimizationAnalyticsModule(manager=manager)
    
    def _initialize_chat_module(self):
        if self.chat_module is None:
            self.chat_module = ChatModule(manager=self.manager)
            if self.api_key and hasattr(st.session_state, 'api_key'):
                st.session_state.chat_api_key = st.session_state.api_key
    
    def display(self):
        # Initialize chat module if needed
        self._initialize_chat_module()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; color: #4a90e2;">
                AI Smart Assistant
            </h1>
            <p style="font-size: 1.1rem; color: #8e8ea0;">
                Intelligent analysis and optimization for carbon emission management
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["AI Chat Assistant", "Optimization & Cost Analysis"])
        
        with tab1:
            self.chat_module.display()
        
        with tab2:
            self.analytics_module.display()