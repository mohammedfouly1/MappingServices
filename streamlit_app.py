# streamlit_app.py - Complete updated version with all fixes
import streamlit as st
import pandas as pd
import json
import time
import sys
import io
from pathlib import Path
import traceback
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import your modules
from config import Config
from input_handler import SendInputParts, SaveResults
from result_processor import display_dataframe_summary, reset_dataframes, get_dataframes
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

# Page configuration
st.set_page_config(
    page_title="Laboratory Mapping Service",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .console-output {
        background-color: #0e1117;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'console_output' not in st.session_state:
    st.session_state.console_output = []
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = None
if 'user_params' not in st.session_state:
    st.session_state.user_params = {}
if 'excel_file' not in st.session_state:
    st.session_state.excel_file = None
if 'prompt_text' not in st.session_state:
    st.session_state.prompt_text = ""

class StreamlitConsoleCapture:
    """Capture console output for Streamlit display"""
    def __init__(self, console_placeholder):
        self.console_placeholder = console_placeholder
        self.output_buffer = []
        
    def write(self, text):
        if text and text != '\n':
            # Strip ANSI color codes for web display
            import re
            clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
            self.output_buffer.append(clean_text)
            # Update console in real-time
            self.console_placeholder.markdown(
                f'<div class="console-output">{"".join(self.output_buffer[-100:])}</div>',
                unsafe_allow_html=True
            )
        sys.__stdout__.write(text)
    
    def flush(self):
        pass

def fix_dataframe_types(df):
    """Fix mixed data types in DataFrame columns for Streamlit display"""
    df_fixed = df.copy()
    for col in df_fixed.columns:
        # Convert mixed types to string for display
        try:
            df_fixed[col] = df_fixed[col].astype(str)
            # Replace 'nan' strings with empty strings for better display
            df_fixed[col] = df_fixed[col].replace('nan', '')
            df_fixed[col] = df_fixed[col].replace('None', '')
        except:
            pass
    return df_fixed

def main():
    # Header
    st.title("🔬 Laboratory Mapping Service")
    st.markdown("### AI-Powered Laboratory Test Mapping System")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("🔑 API Settings")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get('api_key', Config.api_key),
            help="Enter your OpenAI API key"
        )
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to use"
        )
        
        # Advanced Settings (Collapsible)
        with st.expander("🎛️ Advanced Parameters", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.2,
                    step=0.1,
                    help="Controls randomness in responses (0=deterministic, 2=very random)"
                )
                
                top_p = st.slider(
                    "Top P",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.1,
                    help="Nucleus sampling parameter (alternative to temperature)"
                )
                
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=32000,
                    value=16000,
                    step=1000,
                    help="Maximum tokens in response"
                )
            
            with col2:
                max_batch_size = st.number_input(
                    "Max Batch Size",
                    min_value=10,
                    max_value=500,
                    value=200,
                    step=10,
                    help="Maximum rows per batch"
                )
                
                wait_between_batches = st.number_input(
                    "Wait Between Batches (seconds)",
                    min_value=0,
                    max_value=300,
                    value=120,
                    step=10,
                    help="Seconds to wait between API calls"
                )
                
                threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0,
                    max_value=100,
                    value=80,
                    step=5,
                    help="Minimum similarity score to consider a match"
                )
        
        # Optimization Settings
        with st.expander("⚡ Optimization Settings"):
            use_compact_json = st.checkbox(
                "Use Compact JSON",
                value=True,
                help="Reduce token usage with compact format"
            )
            
            abbreviate_keys = st.checkbox(
                "Abbreviate Keys",
                value=True,
                help="Use abbreviated keys (c/n instead of code/name)"
            )
        
        # Store parameters in session state
        st.session_state.user_params = {
            'api_key': api_key,
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'max_batch_size': max_batch_size,
            'wait_between_batches': wait_between_batches,
            'threshold': threshold,
            'use_compact_json': use_compact_json,
            'abbreviate_keys': abbreviate_keys
        }
        
        # Display current settings
        st.divider()
        st.subheader("📊 Current Settings")
        settings_data = [
            ["Model", model],
            ["Temperature", f"{temperature:.1f}"],
            ["Top P", f"{top_p:.1f}"],
            ["Max Batch Size", str(max_batch_size)],
            ["Wait Between Batches", f"{wait_between_batches}s"],
            ["Threshold", f"{threshold}%"]
        ]
        settings_df = pd.DataFrame(settings_data, columns=["Parameter", "Value"])
        # Fix data types for display
        settings_df = fix_dataframe_types(settings_df)
        st.dataframe(settings_df, hide_index=True, use_container_width=True)
    
    # Main content area
    tabs = st.tabs(["📁 Data Input", "🔄 Processing", "📊 Results", "📈 Analytics"])
    
    with tabs[0]:
        st.header("Data Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Excel File")
            excel_file = st.file_uploader(
                "Upload Excel file with 'First Group' and 'Second Group' sheets",
                type=['xlsx', 'xls'],
                help="Excel file should contain two sheets: 'First Group' and 'Second Group'"
            )
            
            if excel_file:
                st.session_state.excel_file = excel_file
                try:
                    excel_data = pd.ExcelFile(excel_file)
                    st.success(f"✓ File loaded: {excel_file.name}")
                    st.info(f"Sheets found: {', '.join(excel_data.sheet_names)}")
                    
                    # Preview data
                    if 'First Group' in excel_data.sheet_names:
                        st.write("**First Group Preview:**")
                        df_first = pd.read_excel(excel_data, sheet_name='First Group', header=None)
                        df_first = fix_dataframe_types(df_first)
                        st.dataframe(df_first.head(), use_container_width=True)
                        st.caption(f"Total rows: {len(df_first)}")
                    
                    if 'Second Group' in excel_data.sheet_names:
                        st.write("**Second Group Preview:**")
                        df_second = pd.read_excel(excel_data, sheet_name='Second Group', header=None)
                        df_second = fix_dataframe_types(df_second)
                        st.dataframe(df_second.head(), use_container_width=True)
                        st.caption(f"Total rows: {len(df_second)}")
                        
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
        
        with col2:
            st.subheader("📝 Prompt Text")
            
            # Default laboratory mapping prompt
            default_prompt = """You will receive 2 Json Arrays. First json array is a table with two columns "First Group Code", "First Group Name". Second json array is a table with two columns "Second Group Code", "Second Group Name".

Your task is to choose every Laboratory Examinations "First Group Name" from the first group (you MUST output ALL first group Laboratory Examinations even if there are no matching Laboratory Examinations from the second group), search and compare each Laboratory Examination in all services of the second group to select and choose the most similar Laboratory Examination.

Mapping should be based on word meaning and Laboratory Examination service name understanding and NOT depend on keyword or string similarity. Use your medical knowledge to understand each Laboratory Examination details including: technique (smear, culture, centrifuge, microscope, etc), approach (ELISA, immunofluorescence), substrate measured (IgM, IgG, Cholesterol), organism included (hepatitis B, HIV, chlamydia, etc), substrate level (direct/indirect/total bilirubin), anatomical site of sample (blood, CSF, urine), test type (quantitative or qualitative), chemical state (free T3 vs total T3), antigen or antibodies.

Your response should be a JSON array with each item having 6 columns:
1. First Group Code (exact value)
2. First Group Name (exact value)
3. Second Group Code (exact value or null)
4. Second Group Name (exact value or null)
5. Similarity Score (1-100)
6. Similarity Reason

Output ALL first group items. If no match, use null for Second Group fields and score <80."""
            
            prompt_text = st.text_area(
                "Enter or paste your prompt",
                height=200,
                value=default_prompt,
                placeholder="Enter the prompt for mapping...",
                help="This prompt will guide the AI in mapping the items"
            )
            
            if prompt_text:
                st.session_state.prompt_text = prompt_text
                st.success(f"✓ Prompt loaded ({len(prompt_text)} characters)")
        
        # Process button
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Processing", 
                        type="primary", 
                        use_container_width=True,
                        disabled=st.session_state.processing):
                
                if not st.session_state.excel_file:
                    st.error("Please upload an Excel file")
                elif not st.session_state.prompt_text:
                    st.error("Please enter a prompt")
                elif not api_key:
                    st.error("Please enter your OpenAI API key")
                else:
                    st.session_state.processing = True
                    st.rerun()
    
    with tabs[1]:
        st.header("Processing")
        
        if st.session_state.processing:
            # Update Config with user parameters
            update_config_with_user_params(st.session_state.user_params)
            
            # Console output area
            st.subheader("🖥️ Console Output")
            console_placeholder = st.empty()
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Capture console output
            old_stdout = sys.stdout
            console_capture = StreamlitConsoleCapture(console_placeholder)
            sys.stdout = console_capture
            
            try:
                # Save uploaded files temporarily
                with open("temp_input.xlsx", "wb") as f:
                    f.write(st.session_state.excel_file.getvalue())
                
                with open("temp_prompt.txt", "w", encoding='utf-8') as f:
                    f.write(st.session_state.prompt_text)
                
                # Reset DataFrames
                reset_dataframes()
                
                # Update status
                status_text.text("Processing data...")
                progress_bar.progress(10)
                
                # Call processing function with user parameters
                results = SendInputParts(
                    excel_path="temp_input.xlsx",
                    prompt_path="temp_prompt.txt",
                    verbose=True,
                    temperature=st.session_state.user_params['temperature'],
                    top_p=st.session_state.user_params['top_p'],
                    model=st.session_state.user_params['model'],
                    max_batch_size=st.session_state.user_params['max_batch_size'],
                    wait_between_batches=st.session_state.user_params['wait_between_batches']
                )
                
                progress_bar.progress(90)
                
                if results:
                    st.session_state.results = results
                    st.session_state.dataframes = get_dataframes()
                    status_text.text("Processing complete!")
                    progress_bar.progress(100)
                    st.success("✓ Processing completed successfully!")
                    
                    # Save results
                    SaveResults(results)
                else:
                    st.error("✗ Processing failed. Check console output for details.")
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.code(traceback.format_exc())
            
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                st.session_state.processing = False
                
                # Clean up temp files
                try:
                    Path("temp_input.xlsx").unlink(missing_ok=True)
                    Path("temp_prompt.txt").unlink(missing_ok=True)
                except:
                    pass
        else:
            st.info("Click 'Start Processing' in the Data Input tab to begin")
    
    with tabs[2]:
        st.header("Results")
        
        if st.session_state.results and st.session_state.dataframes:
            # Display DataFrames
            st.subheader("📊 Mapping Results")
            
            df_mappings = st.session_state.dataframes.get('ApiMapping')
            if df_mappings is not None and not df_mappings.empty:
                # Add filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_score = st.slider("Minimum Score Filter", 0, 100, 0, key="min_score_filter")
                with col2:
                    show_unmapped = st.checkbox("Show Unmapped Items", value=True, key="show_unmapped")
                with col3:
                    search_term = st.text_input("Search", placeholder="Search in results...", key="search_term")
                
                # Filter dataframe
                filtered_df = df_mappings.copy()
                
                # Ensure Similarity Score is numeric
                filtered_df['Similarity Score'] = pd.to_numeric(filtered_df['Similarity Score'], errors='coerce').fillna(0)
                
                # Apply filters
                filtered_df = filtered_df[filtered_df['Similarity Score'] >= min_score]
                
                if not show_unmapped:
                    filtered_df = filtered_df[filtered_df['Second Group Code'].notna()]
                    filtered_df = filtered_df[filtered_df['Second Group Code'] != '']
                    filtered_df = filtered_df[filtered_df['Second Group Code'] != 'None']
                
                if search_term:
                    mask = filtered_df.apply(lambda x: search_term.lower() in str(x).lower(), axis=1)
                    filtered_df = filtered_df[mask.any(axis=1)]
                
                # Fix data types for display
                display_df = fix_dataframe_types(filtered_df)
                
                # Display filtered results
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Mappings", len(df_mappings))
                with col2:
                    mapped = df_mappings['Second Group Code'].notna().sum()
                    st.metric("Mapped Items", mapped)
                with col3:
                    unmapped = df_mappings['Second Group Code'].isna().sum()
                    st.metric("Unmapped Items", unmapped)
                with col4:
                    # Ensure numeric for mean calculation
                    numeric_scores = pd.to_numeric(df_mappings['Similarity Score'], errors='coerce')
                    avg_score = numeric_scores.mean() if not numeric_scores.isna().all() else 0
                    st.metric("Avg Score", f"{avg_score:.1f}")
                
                # Download buttons
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Excel download
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='Mappings')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"mapping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # CSV download
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"mapping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # JSON download
                    def prepare_json_results(results):
                        """Prepare results for JSON serialization"""
                        json_safe_results = {}
                        
                        for key, value in results.items():
                            if key == 'dataframes':
                                continue
                            elif isinstance(value, pd.DataFrame):
                                json_safe_results[key] = value.to_dict('records')
                            else:
                                json_safe_results[key] = value
                        
                        if df_mappings is not None and not df_mappings.empty:
                            json_safe_results['mappings'] = df_mappings.to_dict('records')
                        
                        return json_safe_results
                    
                    json_safe_results = prepare_json_results(st.session_state.results)
                    json_str = json.dumps(json_safe_results, indent=2, ensure_ascii=False, default=str)
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"mapping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.warning("No mapping results available")
        else:
            st.info("No results yet. Process data first.")
    
    with tabs[3]:
        st.header("Analytics")
        
        if st.session_state.dataframes:
            df_mappings = st.session_state.dataframes.get('ApiMapping')
            df_calls = st.session_state.dataframes.get('ApiCall')
            
            if df_mappings is not None and not df_mappings.empty:
                # Ensure numeric data types for analytics
                df_mappings['Similarity Score'] = pd.to_numeric(df_mappings['Similarity Score'], errors='coerce').fillna(0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score distribution
                    st.subheader("📊 Score Distribution")
                    valid_scores = df_mappings[df_mappings['Similarity Score'] > 0]
                    if not valid_scores.empty:
                        fig = px.histogram(
                            valid_scores,
                            x='Similarity Score',
                            nbins=20,
                            title="Distribution of Similarity Scores"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid scores to display")
                
                with col2:
                    # Mapping status pie chart
                    st.subheader("🥧 Mapping Status")
                    mapped_count = df_mappings['Second Group Code'].notna().sum()
                    unmapped_count = df_mappings['Second Group Code'].isna().sum()
                    
                    if mapped_count > 0 or unmapped_count > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Mapped', 'Unmapped'],
                            values=[mapped_count, unmapped_count],
                            hole=.3
                        )])
                        fig.update_layout(title="Mapping Status Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No mapping data to display")
                
                # API Call Statistics
                if df_calls is not None and not df_calls.empty:
                    st.subheader("📈 API Call Statistics")
                    
                    # Ensure numeric data types
                    numeric_columns = ['Latency', 'Input Tokens', 'Output Tokens', 'Total Tokens']
                    for col in numeric_columns:
                        if col in df_calls.columns:
                            df_calls[col] = pd.to_numeric(df_calls[col], errors='coerce').fillna(0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total API Calls", len(df_calls))
                    with col2:
                        avg_latency = df_calls['Latency'].mean() if 'Latency' in df_calls.columns else 0
                        st.metric("Avg Latency", f"{avg_latency:.2f}s")
                    with col3:
                        total_input = df_calls['Input Tokens'].sum() if 'Input Tokens' in df_calls.columns else 0
                        st.metric("Total Input Tokens", f"{int(total_input):,}")
                    with col4:
                        total_output = df_calls['Output Tokens'].sum() if 'Output Tokens' in df_calls.columns else 0
                        st.metric("Total Output Tokens", f"{int(total_output):,}")
                    
                    # Token usage over time
                    if len(df_calls) > 1 and 'Input Tokens' in df_calls.columns and 'Output Tokens' in df_calls.columns:
                        st.subheader("📊 Token Usage Over Time")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(df_calls) + 1)),
                            y=df_calls['Input Tokens'],
                            mode='lines+markers',
                            name='Input Tokens'
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(df_calls) + 1)),
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
                    
                    # Display parameters used
                    st.subheader("⚙️ Parameters Used")
                    params_data = [
                        ["Model", st.session_state.user_params.get('model', 'N/A')],
                        ["Temperature", str(st.session_state.user_params.get('temperature', 'N/A'))],
                        ["Top P", str(st.session_state.user_params.get('top_p', 'N/A'))],
                        ["Max Batch Size", str(st.session_state.user_params.get('max_batch_size', 'N/A'))],
                        ["Wait Between Batches", f"{st.session_state.user_params.get('wait_between_batches', 'N/A')}s"]
                    ]
                    params_df = pd.DataFrame(params_data, columns=["Parameter", "Value"])
                    params_df = fix_dataframe_types(params_df)
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
        else:
            st.info("No analytics data available. Process data first.")

def update_config_with_user_params(params):
    """Update Config class with user parameters"""
    Config.api_key = params['api_key']
    Config.model = params['model']
    Config.temperature = params['temperature']
    Config.top_p = params['top_p']
    Config.max_tokens = params['max_tokens']
    Config.max_batch_size = params['max_batch_size']
    Config.wait_between_batches = params['wait_between_batches']
    Config.threshold = params['threshold']
    Config.use_compact_json = params['use_compact_json']
    Config.abbreviate_keys = params['abbreviate_keys']

if __name__ == "__main__":
    main()