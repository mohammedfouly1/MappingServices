# config.py
import os
from pathlib import Path
import streamlit as st

class Config:
    """Configuration settings with Streamlit Cloud support"""
    
    # For Streamlit Cloud, use secrets management
    @staticmethod
    def get_api_key():
        # Try Streamlit secrets first (for cloud deployment)
        try:
            return st.secrets["OPENAI_API_KEY"]
        except:
            # Fall back to environment variable (for local)
            return os.getenv("OPENAI_API_KEY", "")
    
    # File paths - will be handled by file uploader in Streamlit
    excel_path = None
    prompt_path = None
    
    # API settings - can be overridden by Streamlit secrets
    @staticmethod
    def get_settings():
        try:
            # Try to load from Streamlit secrets
            return {
                "api_key": st.secrets.get("OPENAI_API_KEY", ""),
                "model": st.secrets.settings.get("model", "gpt-4o"),
                "max_tokens": st.secrets.settings.get("max_tokens", 16000),
                "temperature": st.secrets.settings.get("temperature", 0.2),
                "top_p": st.secrets.settings.get("top_p", 0.9),
                "threshold": st.secrets.settings.get("threshold", 80)
            }
        except:
            # Default settings
            return {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": "gpt-4o",
                "max_tokens": 16000,
                "temperature": 0.2,
                "top_p": 0.9,
                "threshold": 80
            }
    
    # Load settings
    settings = get_settings.__func__()
    api_key = get_api_key.__func__()
    model = settings.get("model", "gpt-4o")
    max_tokens = settings.get("max_tokens", 16000)
    temperature = settings.get("temperature", 0.2)
    top_p = settings.get("top_p", 0.9)
    threshold = settings.get("threshold", 80)
    
    # Optimization settings
    use_compact_json = True
    abbreviate_keys = True
    
    # Batch settings
    max_batch_size = 200
    wait_between_batches = 120