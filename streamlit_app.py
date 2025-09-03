# streamlit_app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import io
import sys
import time
import json
import base64
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os

# Import our modules
from config import Config
from input_handler import SendInputParts
from result_processor import get_dataframes, reset_dataframes, save_dataframes_to_excel

# Set page config
st.set_page_config(
    page_title="Mapping Medical Services",
    page_icon="🩺",  # stethoscope = general medical services
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 18px;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    pre {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitConsoleCapture:
    """Capture console output for Streamlit display"""
    def __init__(self, text_element):
        self.text_element = text_element
        self.output = []
        self.old_stdout = sys.stdout
        
    def write(self, text):
        # Write to original stdout
        self.old_stdout.write(text)
        
        # Remove ANSI color codes for display
        import re
        clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        
        # Capture for Streamlit
        if clean_text.strip():
            self.output.append(clean_text)
            # Update the text element with last 50 lines
            self.text_element.text('\n'.join(self.output[-50:]))
    
    def flush(self):
        self.old_stdout.flush()

def load_prompt_from_file(prompt_type):
    """Load prompt text from the appropriate file based on type"""
    prompt_files = {
        "Lab": "LabPrompt.txt",
        "Radiology": "RadPrompt.txt",
        "Service": "ServicePrompt.txt"
    }
    
    prompt_file = prompt_files.get(prompt_type)
    if not prompt_file:
        return None
    
    try:
        # Check if file exists
        file_path = Path(prompt_file)
        if not file_path.exists():
            st.error(f"❌ Prompt file '{prompt_file}' not found! Please ensure the file exists in the project folder.")
            return None
            
        # Read the file
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            prompt_text = f.read().strip()
            
        if not prompt_text:
            st.warning(f"⚠️ Prompt file '{prompt_file}' is empty!")
            return None
            
        return prompt_text
        
    except Exception as e:
        st.error(f"❌ Error reading prompt file '{prompt_file}': {str(e)}")
        return None

def main():
    # Header
    st.title("🩺 Mapping Medical Services")
    st.markdown("### AI-Powered Mapping System with Batch Processing")
    
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'console_output' not in st.session_state:
        st.session_state.console_output = []
    if 'selected_prompt_type' not in st.session_state:
        st.session_state.selected_prompt_type = None
    if 'uploaded_file_content' not in st.session_state:
        st.session_state.uploaded_file_content = None
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=Config.api_key,
            help="Enter your OpenAI API key"
        )
        if api_key:
            Config.api_key = api_key
        
        st.divider()
        
        # Model Settings
        st.subheader("🤖 Model Settings")
        
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to use"
        )
        Config.model = model
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=Config.temperature,
            step=0.1,
            help="Controls randomness (0=focused, 1=creative)"
        )
        Config.temperature = temperature
        
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=Config.top_p,
            step=0.1,
            help="Controls diversity of output"
        )
        Config.top_p = top_p
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1000,
            max_value=32000,
            value=Config.max_tokens,
            step=1000,
            help="Maximum tokens for response"
        )
        Config.max_tokens = max_tokens
        
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0,
            max_value=100,
            value=Config.threshold,
            step=5,
            help="Minimum similarity score for valid mapping"
        )
        Config.threshold = threshold
        
        st.divider()
        
        # Batch Settings
        st.subheader("📦 Batch Settings")
        
        max_batch_size = st.number_input(
            "Max Batch Size",
            min_value=50,
            max_value=500,
            value=Config.max_batch_size,
            step=50,
            help="Maximum rows per batch"
        )
        Config.max_batch_size = max_batch_size
        
        wait_time = st.number_input(
            "Wait Between Batches (seconds)",
            min_value=0,
            max_value=300,
            value=Config.wait_between_batches,
            step=30,
            help="Delay between API calls"
        )
        Config.wait_between_batches = wait_time
        
        st.divider()
        
        # Optimization Settings
        st.subheader("⚡ Optimization")
        
        use_compact = st.checkbox(
            "Use Compact JSON",
            value=Config.use_compact_json,
            help="Reduces token usage by ~60%"
        )
        Config.use_compact_json = use_compact
        
        abbreviate = st.checkbox(
            "Abbreviate Keys",
            value=Config.abbreviate_keys,
            help="Use short key names (c/n instead of code/name)"
        )
        Config.abbreviate_keys = abbreviate
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Input", "🔄 Processing", "📊 Results", "📈 Analytics"])
    
    with tab1:
        st.header("Data Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Excel File")
            uploaded_file = st.file_uploader(
                "Upload Excel file with 'First Group' and 'Second Group' sheets",
                type=['xlsx', 'xls'],
                help="Excel file must contain two sheets: 'First Group' and 'Second Group'"
            )
            
            if uploaded_file:
                try:
                    # Store the file content in session state
                    st.session_state.uploaded_file_content = uploaded_file.read()
                    uploaded_file.seek(0)  # Reset file pointer for preview
                    
                    # Preview the uploaded file
                    excel_data = pd.ExcelFile(uploaded_file)
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    st.info(f"Sheets found: {', '.join(excel_data.sheet_names)}")
                    
                    # Show preview of data
                    if 'First Group' in excel_data.sheet_names:
                        df_first = pd.read_excel(excel_data, sheet_name='First Group', header=None)
                        st.write("**First Group Preview:**")
                        st.dataframe(df_first.head(), use_container_width=True)
                        st.caption(f"Total rows: {len(df_first)}")
                    
                    if 'Second Group' in excel_data.sheet_names:
                        df_second = pd.read_excel(excel_data, sheet_name='Second Group', header=None)
                        st.write("**Second Group Preview:**")
                        st.dataframe(df_second.head(), use_container_width=True)
                        st.caption(f"Total rows: {len(df_second)}")
                    
                    excel_data.close()  # Close the ExcelFile object
                        
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
        
        with col2:
            st.subheader("📝 Prompt Selection")
            st.info("Select the type of mapping to load the appropriate prompt")
            
            # Create three columns for the selection buttons
            button_col1, button_col2, button_col3 = st.columns(3)
            
            with button_col1:
                if st.button("🧪 Lab", use_container_width=True, type="primary"):
                    st.session_state.selected_prompt_type = "Lab"
                    
            with button_col2:
                if st.button("📷 Radiology", use_container_width=True, type="primary"):
                    st.session_state.selected_prompt_type = "Radiology"
                    
            with button_col3:
                if st.button("🔧 Service", use_container_width=True, type="primary"):
                    st.session_state.selected_prompt_type = "Service"
            
            # Alternative: Radio buttons
            st.divider()
            st.write("**Or select using radio buttons:**")
            prompt_type_radio = st.radio(
                "Select Prompt Type:",
                ["Lab", "Radiology", "Service"],
                horizontal=True,
                index=None
            )
            
            if prompt_type_radio:
                st.session_state.selected_prompt_type = prompt_type_radio
            
            # Display selected prompt type and load prompt
            if st.session_state.selected_prompt_type:
                st.success(f"✅ Selected: **{st.session_state.selected_prompt_type}** Mapping")
                
                # Load and display prompt
                prompt_text = load_prompt_from_file(st.session_state.selected_prompt_type)
                if prompt_text:
                    st.write("**Prompt Preview:**")
                    with st.expander("View Full Prompt", expanded=False):
                        st.text_area(
                            "Prompt Content",
                            value=prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text,
                            height=150,
                            disabled=True
                        )
                    st.caption(f"Prompt length: {len(prompt_text)} characters")
                else:
                    st.error(f"Failed to load prompt for {st.session_state.selected_prompt_type}")
            else:
                st.warning("⚠️ Please select a prompt type")
        
        st.divider()
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 Start Mapping Process",
                type="primary",
                use_container_width=True,
                disabled=not (uploaded_file and st.session_state.selected_prompt_type and Config.api_key)
            ):
                if not Config.api_key:
                    st.error("❌ Please enter your OpenAI API key in the sidebar")
                elif not uploaded_file:
                    st.error("❌ Please upload an Excel file")
                elif not st.session_state.selected_prompt_type:
                    st.error("❌ Please select a prompt type (Lab, Radiology, or Service)")
                else:
                    st.session_state.processing = True
                    st.rerun()
    
    with tab2:
        st.header("Processing")
        
        if st.session_state.processing:
            # Load prompt based on selection
            prompt_text = load_prompt_from_file(st.session_state.selected_prompt_type)
            
            if not prompt_text:
                st.error(f"❌ Failed to load {st.session_state.selected_prompt_type} prompt")
                st.session_state.processing = False
                st.stop()
            
            # Create temporary files using tempfile module
            temp_excel_fd, temp_excel_path = tempfile.mkstemp(suffix='.xlsx')
            temp_prompt_fd, temp_prompt_path = tempfile.mkstemp(suffix='.txt')
            
            try:
                # Write Excel content to temp file
                with os.fdopen(temp_excel_fd, 'wb') as f:
                    f.write(st.session_state.uploaded_file_content)
                
                # Write prompt to temp file
                with os.fdopen(temp_prompt_fd, 'w', encoding='utf-8') as f:
                    f.write(prompt_text)
                
                # Reset DataFrames
                reset_dataframes()
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                console_output = st.empty()
                
                # Capture console output
                console_capture = StreamlitConsoleCapture(console_output)
                old_stdout = sys.stdout
                sys.stdout = console_capture
                
                try:
                    status_text.text(f"Starting {st.session_state.selected_prompt_type} mapping process...")
                    progress_bar.progress(10)
                    
                    # Call the processing function
                    results = SendInputParts(
                        excel_path=temp_excel_path,
                        prompt_path=temp_prompt_path,
                        verbose=True
                    )
                    
                    progress_bar.progress(90)
                    
                    if results:
                        st.session_state.results = results
                        status_text.text("✅ Processing completed successfully!")
                        progress_bar.progress(100)
                        
                        # Show summary
                        st.success(f"✅ {st.session_state.selected_prompt_type} Mapping completed successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Mappings", len(results.get("mappings", [])))
                        with col2:
                            stats = results.get("statistics", {})
                            st.metric("Mapped Items", stats.get("mapped_count", 0))
                        with col3:
                            st.metric("Avg Score", f"{stats.get('avg_score', 0):.1f}")
                    else:
                        st.error("❌ Processing failed. Check the console output for details.")
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
                    st.session_state.processing = False
                    
            finally:
                # Clean up temp files - use try/except to handle any errors
                try:
                    if os.path.exists(temp_excel_path):
                        os.unlink(temp_excel_path)
                except:
                    pass  # Ignore errors during cleanup
                
                try:
                    if os.path.exists(temp_prompt_path):
                        os.unlink(temp_prompt_path)
                except:
                    pass  # Ignore errors during cleanup
        else:
            st.info("👈 Please upload data and select a prompt type in the Input tab to start processing")
    
    with tab3:
        st.header("Results")
        
        if st.session_state.results:
            # Get DataFrames
            dataframes = get_dataframes()
            
            # Display mapping results
            st.subheader("📊 Mapping Results")
            df_mappings = dataframes.get('ApiMapping')
            
            if df_mappings is not None and not df_mappings.empty:
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_mapped = st.checkbox("Show Mapped Only", value=False)
                with col2:
                    min_score = st.slider("Minimum Score", 0, 100, 0)
                
                # Apply filters
                filtered_df = df_mappings.copy()
                if show_mapped:
                    filtered_df = filtered_df[filtered_df['Second Group Code'].notna()]
                filtered_df = filtered_df[filtered_df['Similarity Score'] >= min_score]
                
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download buttons
                st.subheader("📥 Download Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Excel download
                    excel_buffer = io.BytesIO()
                    save_dataframes_to_excel(excel_buffer)
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_buffer,
                        file_name=f"{st.session_state.selected_prompt_type}_mapping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # CSV download
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name=f"{st.session_state.selected_prompt_type}_mappings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # JSON download
                    json_str = json.dumps(st.session_state.results, indent=2, default=str)
                    st.download_button(
                        label="🔧 Download JSON",
                        data=json_str,
                        file_name=f"{st.session_state.selected_prompt_type}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.info("No mapping results available yet")
        else:
            st.info("👈 No results yet. Please process data first.")
    
    with tab4:
        st.header("Analytics")
        
        if st.session_state.results:
            dataframes = get_dataframes()
            df_mappings = dataframes.get('ApiMapping')
            df_calls = dataframes.get('ApiCall')
            
            if df_mappings is not None and not df_mappings.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score distribution
                    st.subheader("📊 Score Distribution")
                    fig = px.histogram(
                        df_mappings,
                        x='Similarity Score',
                        nbins=20,
                        title="Distribution of Similarity Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Mapping status
                    st.subheader("🎯 Mapping Status")
                    mapped_count = df_mappings['Second Group Code'].notna().sum()
                    unmapped_count = df_mappings['Second Group Code'].isna().sum()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Mapped', 'Unmapped'],
                        values=[mapped_count, unmapped_count],
                        hole=.3
                    )])
                    fig.update_layout(title="Mapping Status Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # API Call Statistics
                if df_calls is not None and not df_calls.empty:
                    st.subheader("📈 API Call Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Calls", len(df_calls))
                    with col2:
                        st.metric("Avg Latency", f"{df_calls['Latency'].mean():.2f}s")
                    with col3:
                        st.metric("Total Tokens", f"{df_calls['Total Tokens'].sum():,}")
                    with col4:
                        st.metric("Prompt Type", st.session_state.selected_prompt_type)
                    
                    # Token usage over time
                    if len(df_calls) > 1:
                        st.subheader("🔤 Token Usage")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_calls.index,
                            y=df_calls['Input Tokens'],
                            mode='lines+markers',
                            name='Input Tokens'
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_calls.index,
                            y=df_calls['Output Tokens'],
                            mode='lines+markers',
                            name='Output Tokens'
                        ))
                        fig.update_layout(
                            title="Token Usage per API Call",
                            xaxis_title="Call Number",
                            yaxis_title="Tokens"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 No analytics data available. Please process data first.")
    
    # Footer
    st.divider()
    st.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
            Laboratory Mapping Service v2.0 | Selected Mode: {st.session_state.selected_prompt_type or 'None'} | 
            Model: {Config.model} | Temperature: {Config.temperature} | Threshold: {Config.threshold}
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()