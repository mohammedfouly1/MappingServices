# streamlit_app.py
import streamlit as st
import pandas as pd
import time
import json
import sys
import io
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import os
import threading
from contextlib import redirect_stdout, redirect_stderr
import queue

# Import your existing modules
from config import Config
from input_handler import SendInputParts, SaveResults
from result_processor import display_dataframe_summary, reset_dataframes, get_dataframes, save_dataframes_to_excel
from optimization_utils import create_compact_item

# Configure Streamlit page
st.set_page_config(
    page_title="Laboratory Mapping Service",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .console-output {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        padding: 10px;
        border-radius: 5px;
        height: 400px;
        overflow-y: auto;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'console_output' not in st.session_state:
    st.session_state.console_output = []
if 'first_group_data' not in st.session_state:
    st.session_state.first_group_data = None
if 'second_group_data' not in st.session_state:
    st.session_state.second_group_data = None
if 'console_text' not in st.session_state:
    st.session_state.console_text = ""

class OutputCapture:
    """Enhanced output capture that handles all print statements including colorama"""
    def __init__(self, console_placeholder, status_placeholder=None):
        self.console_placeholder = console_placeholder
        self.status_placeholder = status_placeholder
        self.output_lines = []
        self.buffer = ""
        
    def write(self, text):
        # Handle colorama escape sequences
        import re
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        
        # Add to buffer
        self.buffer += clean_text
        
        # If we have a complete line, process it
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Keep the last incomplete line in buffer
            self.buffer = lines[-1]
            # Process complete lines
            for line in lines[:-1]:
                if line.strip():  # Only add non-empty lines
                    self.output_lines.append(line)
                    st.session_state.console_output.append(line)
                    st.session_state.console_text += line + "\n"
                    
                    # Update display
                    self.update_display()
                    
                    # Update status if it's a status message
                    if self.status_placeholder and any(keyword in line for keyword in ['Step', 'Processing', 'Completed', '✓', '✗']):
                        self.status_placeholder.text(line[:100])  # Show first 100 chars in status
    
    def flush(self):
        if self.buffer.strip():
            self.output_lines.append(self.buffer)
            st.session_state.console_output.append(self.buffer)
            st.session_state.console_text += self.buffer + "\n"
            self.buffer = ""
            self.update_display()
    
    def update_display(self):
        # Display last 50 lines in console
        display_lines = self.output_lines[-50:]
        console_text = "\n".join(display_lines)
        
        # Create HTML for colored console output
        html_output = f"""
        <div style="background-color: #1e1e1e; color: #d4d4d4; font-family: 'Courier New', monospace; 
                    font-size: 12px; padding: 10px; border-radius: 5px; height: 400px; 
                    overflow-y: auto; white-space: pre-wrap;">
        {self._colorize_output(console_text)}
        </div>
        """
        self.console_placeholder.markdown(html_output, unsafe_allow_html=True)
    
    def _colorize_output(self, text):
        """Add HTML colors based on content patterns"""
        # Replace special characters with HTML entities
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Color patterns
        patterns = [
            (r'(✓.*)', '<span style="color: #4CAF50;">\\1</span>'),  # Success (green)
            (r'(✗.*)', '<span style="color: #f44336;">\\1</span>'),  # Error (red)
            (r'(\[Step \d+\].*)', '<span style="color: #FFC107;">\\1</span>'),  # Steps (yellow)
            (r'(Starting.*|Processing.*|Completed.*)', '<span style="color: #2196F3;">\\1</span>'),  # Info (blue)
            (r'(Error:.*)', '<span style="color: #f44336;">\\1</span>'),  # Errors
            (r'(Warning:.*)', '<span style="color: #FF9800;">\\1</span>'),  # Warnings
            (r'(═+)', '<span style="color: #9C27B0;">\\1</span>'),  # Separators (purple)
            (r'(•.*)', '<span style="color: #00BCD4;">\\1</span>'),  # Bullet points (cyan)
        ]
        
        import re
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        return text

def load_and_preview_excel(file_path: str) -> tuple:
    """Load Excel file and return data from both sheets"""
    try:
        excel_data = pd.ExcelFile(file_path)
        
        # Read First Group
        first_group_df = None
        if 'First Group' in excel_data.sheet_names:
            first_group_df = pd.read_excel(excel_data, sheet_name='First Group', header=None)
            first_group_df.columns = ['Code', 'Name']
            # Convert Code column to string to avoid type issues
            first_group_df['Code'] = first_group_df['Code'].astype(str)
            first_group_df['Name'] = first_group_df['Name'].astype(str)
            # Remove empty rows
            first_group_df = first_group_df.dropna(how='all')
            first_group_df = first_group_df[first_group_df['Code'] != 'nan']
        
        # Read Second Group
        second_group_df = None
        if 'Second Group' in excel_data.sheet_names:
            second_group_df = pd.read_excel(excel_data, sheet_name='Second Group', header=None)
            second_group_df.columns = ['Code', 'Name']
            # Convert Code column to string to avoid type issues
            second_group_df['Code'] = second_group_df['Code'].astype(str)
            second_group_df['Name'] = second_group_df['Name'].astype(str)
            # Remove empty rows
            second_group_df = second_group_df.dropna(how='all')
            second_group_df = second_group_df[second_group_df['Code'] != 'nan']
        
        return first_group_df, second_group_df, excel_data.sheet_names
    
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None, None, []

def run_processing_with_capture(excel_path, prompt_path, console_placeholder, progress_bar, status_text):
    """Run the processing with full output capture"""
    
    # Create output capture
    output_capture = OutputCapture(console_placeholder, status_text)
    
    # Capture both stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect all output to our capture
        sys.stdout = output_capture
        sys.stderr = output_capture
        
        # Reset DataFrames
        status_text.text("Resetting DataFrames...")
        progress_bar.progress(5)
        reset_dataframes()
        progress_bar.progress(10)
        
        # Call the main processing function
        status_text.text("Starting data processing...")
        progress_bar.progress(20)
        
        # This will capture ALL print statements from SendInputParts and its called functions
        results = SendInputParts(
            excel_path=str(excel_path),
            prompt_path=str(prompt_path),
            verbose=True
        )
        
        progress_bar.progress(90)
        
        if results:
            status_text.text("Saving results...")
            SaveResults(results)
            progress_bar.progress(100)
            status_text.text("✅ Processing completed successfully!")
        else:
            status_text.text("❌ Processing failed!")
        
        # Flush any remaining output
        output_capture.flush()
        
        return results
        
    except Exception as e:
        # Capture exception output
        import traceback
        sys.stdout.write(f"\n❌ Error occurred: {str(e)}\n")
        sys.stdout.write(traceback.format_exc())
        output_capture.flush()
        status_text.text(f"❌ Error: {str(e)}")
        return None
        
    finally:
        # Always restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def main():
    # Header
    st.title("🔬 Laboratory Mapping Service")
    st.markdown("### AI-Powered Item Mapping Between Laboratory Groups")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration Settings")
        
        # File Input Section
        st.subheader("📁 Input Files")
        
        # Excel file upload
        excel_file = st.file_uploader(
            "Upload Excel File (.xlsx)",
            type=['xlsx'],
            help="Excel file should contain 'First Group' and 'Second Group' sheets"
        )
        
        # Prompt file upload
        prompt_file = st.file_uploader(
            "Upload Prompt File (.txt)",
            type=['txt'],
            help="Text file containing the mapping prompt"
        )
        
        st.markdown("---")
        
        # API Settings
        st.subheader("🔌 API Settings")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your OpenAI API key"
        )
        
        model = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to use"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1000,
                max_value=128000,
                value=16000,
                step=1000,
                help="Maximum tokens for response"
            )
        
        with col2:
            threshold = st.number_input(
                "Similarity Threshold",
                min_value=0,
                max_value=100,
                value=80,
                step=5,
                help="Minimum similarity score to consider a match"
            )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.1,
            help="Controls randomness in responses (lower = more deterministic)"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Controls diversity of responses"
        )
        
        st.markdown("---")
        
        # Optimization Settings
        st.subheader("⚡ Optimization Settings")
        
        use_compact_json = st.checkbox(
            "Use Compact JSON",
            value=True,
            help="Reduces token usage by using abbreviated JSON format"
        )
        
        abbreviate_keys = st.checkbox(
            "Abbreviate Keys",
            value=True,
            help="Uses short key names to save tokens"
        )
        
        st.markdown("---")
        
        # Batch Settings
        st.subheader("📦 Batch Settings")
        
        max_batch_size = st.number_input(
            "Max Batch Size",
            min_value=10,
            max_value=500,
            value=200,
            step=10,
            help="Maximum rows per API call"
        )
        
        wait_between_batches = st.number_input(
            "Wait Between Batches (seconds)",
            min_value=0,
            max_value=300,
            value=120,
            step=30,
            help="Delay between batch API calls"
        )
        
        st.markdown("---")
        
        # Console Output Settings
        st.subheader("🖥️ Console Settings")
        
        auto_scroll = st.checkbox(
            "Auto-scroll Console",
            value=True,
            help="Automatically scroll to show latest output"
        )
        
        show_timestamps = st.checkbox(
            "Show Timestamps",
            value=False,
            help="Add timestamps to console output"
        )
    
    # Main content area
    if excel_file and prompt_file:
        # Save uploaded files temporarily
        excel_path = Path(f"temp_{excel_file.name}")
        prompt_path = Path(f"temp_{prompt_file.name}")
        
        with open(excel_path, "wb") as f:
            f.write(excel_file.getbuffer())
        
        with open(prompt_path, "wb") as f:
            f.write(prompt_file.getbuffer())
        
        # Update Config with user inputs
        Config.excel_path = excel_path
        Config.prompt_path = prompt_path
        Config.api_key = api_key
        Config.model = model
        Config.max_tokens = max_tokens
        Config.temperature = temperature
        Config.top_p = top_p
        Config.threshold = threshold
        Config.use_compact_json = use_compact_json
        Config.abbreviate_keys = abbreviate_keys
        Config.max_batch_size = max_batch_size
        Config.wait_between_batches = wait_between_batches
        
        # Load and preview data
        first_group_df, second_group_df, sheet_names = load_and_preview_excel(str(excel_path))
        
        if first_group_df is not None and second_group_df is not None:
            # Store in session state
            st.session_state.first_group_data = first_group_df
            st.session_state.second_group_data = second_group_df
            
            # Display data preview
            st.header("📊 Data Preview")
            
            # Create tabs for data display
            tab1, tab2, tab3, tab4 = st.tabs(["First Group", "Second Group", "Prompt", "Statistics"])
            
            with tab1:
                st.subheader(f"First Group Data ({len(first_group_df)} rows)")
                st.dataframe(first_group_df, height=400)
                
                # Show sample in compact format
                if use_compact_json and len(first_group_df) > 0:
                    with st.expander("View Compact Format Preview (for API)"):
                        sample_compact = []
                        for idx, row in first_group_df.head(3).iterrows():
                            compact = create_compact_item(str(row['Code']), str(row['Name']), "first")
                            sample_compact.append(compact)
                        st.json(sample_compact)
            
            with tab2:
                st.subheader(f"Second Group Data ({len(second_group_df)} rows)")
                st.dataframe(second_group_df, height=400)
                
                # Show sample in compact format
                if use_compact_json and len(second_group_df) > 0:
                    with st.expander("View Compact Format Preview (for API)"):
                        sample_compact = []
                        for idx, row in second_group_df.head(3).iterrows():
                            compact = create_compact_item(str(row['Code']), str(row['Name']), "second")
                            sample_compact.append(compact)
                        st.json(sample_compact)
            
            with tab3:
                st.subheader("Prompt Text")
                prompt_text = prompt_file.read().decode('utf-8-sig')
                st.text_area("Prompt Content", value=prompt_text, height=200, disabled=True)
                st.info(f"Prompt length: {len(prompt_text)} characters (~{len(prompt_text)//4} tokens)")
            
            with tab4:
                st.subheader("📈 Dataset Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("First Group Rows", len(first_group_df))
                with col2:
                    st.metric("Second Group Rows", len(second_group_df))
                with col3:
                    st.metric("Total Rows", len(first_group_df) + len(second_group_df))
                
                # Batch calculation
                total_rows = len(first_group_df) + len(second_group_df)
                if total_rows > max_batch_size:
                    st.warning(f"⚠️ Dataset exceeds max batch size ({max_batch_size}). Batching will be required.")
                    
                    # Better batch estimation
                    import math
                    n1 = len(first_group_df)
                    n2 = len(second_group_df)
                    
                    # Simplified batch calculation
                    if total_rows <= max_batch_size:
                        estimated_batches = 1
                    else:
                        # This is a simplified estimate
                        f = max_batch_size // 2
                        s = max_batch_size - f
                        b1 = math.ceil(n1 / f)
                        b2 = math.ceil(n2 / s)
                        estimated_batches = b1 * b2
                    
                    estimated_time = estimated_batches * (wait_between_batches + 10) / 60
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Estimated Batches", f"~{estimated_batches}")
                    with col2:
                        st.metric("Estimated Time", f"~{estimated_time:.1f} minutes")
                else:
                    st.success(f"✅ Dataset within batch size limit. Single API call will be used.")
            
            st.markdown("---")
            
            # Process button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Start Mapping Process", type="primary", key="start_button"):
                    if not api_key:
                        st.error("❌ Please provide an OpenAI API key")
                    else:
                        st.session_state.processing_started = True
                        st.session_state.processing_complete = False
                        st.session_state.console_output = []
                        st.session_state.console_text = ""
            
            # Processing section
            if st.session_state.processing_started and not st.session_state.processing_complete:
                st.header("🔄 Processing")
                
                # Create two columns for console and live stats
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("📟 Console Output")
                    console_container = st.container()
                    console_placeholder = console_container.empty()
                
                with col2:
                    st.subheader("📊 Live Stats")
                    stats_container = st.container()
                    with stats_container:
                        progress_metric = st.empty()
                        time_metric = st.empty()
                        status_metric = st.empty()
                
                # Progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Start time tracking
                start_time = time.time()
                
                # Run processing with output capture
                results = run_processing_with_capture(
                    excel_path,
                    prompt_path,
                    console_placeholder,
                    progress_bar,
                    status_text
                )
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                time_metric.metric("Elapsed Time", f"{elapsed_time:.1f}s")
                
                # Store results
                st.session_state.results = results
                st.session_state.processing_complete = True
                
                if results:
                    st.success("✅ Processing completed successfully!")
                    progress_metric.metric("Status", "✅ Complete")
                else:
                    st.error("❌ Processing failed. Check console output for details.")
                    progress_metric.metric("Status", "❌ Failed")
            
            # Display results if processing is complete
            if st.session_state.processing_complete and st.session_state.results:
                st.header("📊 Results")
                
                # Create result tabs
                result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(
                    ["Mapping Results", "API Statistics", "Console Log", "Download"]
                )
                
                with result_tab1:
                    st.subheader("Mapping Results")
                    
                    # Get the DataFrames
                    api_call_df, api_mapping_df = get_dataframes()
                    
                    if not api_mapping_df.empty:
                        # Ensure proper typing
                        api_mapping_df = api_mapping_df.copy()
                        for col in ['First Group Code', 'Second Group Code']:
                            if col in api_mapping_df.columns:
                                api_mapping_df[col] = api_mapping_df[col].astype(str)
                        
                        # Display summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Mappings", len(api_mapping_df))
                        with col2:
                            unique_codes = api_mapping_df['First Group Code'].nunique()
                            st.metric("Unique First Codes", unique_codes)
                        with col3:
                            matched = api_mapping_df['Second Group Code'].notna().sum()
                            st.metric("Matched Items", matched)
                        with col4:
                            avg_score = api_mapping_df['Similarity Score'].mean()
                            st.metric("Avg Score", f"{avg_score:.1f}")
                        
                        # Display the mapping DataFrame
                        st.dataframe(api_mapping_df, height=400)
                        
                        # Score distribution
                        if 'Similarity Score' in api_mapping_df.columns:
                            st.subheader("Score Distribution")
                            try:
                                import plotly.express as px
                                fig = px.histogram(
                                    api_mapping_df,
                                    x='Similarity Score',
                                    nbins=20,
                                    title="Distribution of Similarity Scores"
                                )
                                st.plotly_chart(fig)
                            except ImportError:
                                # Fallback to simple stats if plotly not available
                                st.write("Score Statistics:")
                                st.write(api_mapping_df['Similarity Score'].describe())
                    else:
                        st.warning("No mapping results available")
                
                with result_tab2:
                    st.subheader("API Call Statistics")
                    
                    if not api_call_df.empty:
                        st.dataframe(api_call_df, height=300)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total API Calls", len(api_call_df))
                        with col2:
                            total_tokens = api_call_df['Tokens Input'].sum() + api_call_df['Tokens OutPut'].sum()
                            st.metric("Total Tokens", f"{total_tokens:,}")
                        with col3:
                            avg_latency = api_call_df['Latency Per seconds'].mean()
                            st.metric("Avg Latency", f"{avg_latency:.2f}s")
                    else:
                        st.warning("No API call statistics available")
                
                with result_tab3:
                    st.subheader("📜 Complete Console Log")
                    
                    # Display the full console output
                    if st.session_state.console_text:
                        # Add download button for console log
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            st.download_button(
                                label="📥 Download Log",
                                data=st.session_state.console_text,
                                file_name=f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                        # Display console output in a scrollable container
                        st.text_area(
                            "Console Output",
                            value=st.session_state.console_text,
                            height=500,
                            disabled=True
                        )
                    else:
                        st.info("No console output available")
                
                with result_tab4:
                    st.subheader("📥 Download Results")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Download Excel file
                        if not api_mapping_df.empty:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                api_call_df.to_excel(writer, sheet_name='ApiCall', index=False)
                                api_mapping_df.to_excel(writer, sheet_name='ApiMapping', index=False)
                            output.seek(0)
                            
                            st.download_button(
                                label="📊 Download Excel",
                                data=output.getvalue(),
                                file_name=f"results_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    with col2:
                        # Download JSON results
                        if st.session_state.results:
                            json_results = {k: v for k, v in st.session_state.results.items() 
                                          if k != "dataframes"}
                            json_str = json.dumps(json_results, ensure_ascii=False, indent=2)
                            
                            st.download_button(
                                label="📄 Download JSON",
                                data=json_str,
                                file_name=f"results_{timestamp}.json",
                                mime="application/json"
                            )
                    
                    with col3:
                        # Download console log
                        if st.session_state.console_text:
                            st.download_button(
                                label="📝 Download Log",
                                data=st.session_state.console_text,
                                file_name=f"console_{timestamp}.txt",
                                mime="text/plain"
                            )
            
            # Clean up temporary files
            if st.session_state.processing_complete:
                try:
                    if excel_path.exists():
                        excel_path.unlink()
                    if prompt_path.exists():
                        prompt_path.unlink()
                except:
                    pass
        
        else:
            if first_group_df is None:
                st.error("❌ Excel file must contain a 'First Group' sheet")
            if second_group_df is None:
                st.error("❌ Excel file must contain a 'Second Group' sheet")
    
    else:
        # Instructions when no files are uploaded
        st.info("""
        ### 📋 Getting Started
        
        1. **Upload Files**: Use the sidebar to upload your Excel file and prompt text file
        2. **Configure Settings**: Adjust API and optimization settings as needed
        3. **Review Data**: Preview your data in the tabs above
        4. **Start Process**: Click the "Start Mapping Process" button
        5. **Monitor Progress**: Watch real-time console output during processing
        6. **Download Results**: Get your results in Excel, JSON, or text format
        
        #### Excel File Requirements:
        - Must contain two sheets: "First Group" and "Second Group"
        - Each sheet should have two columns: Code and Name
        - No headers required
        
        #### Features:
        - ✅ Real-time console output with color coding
        - ✅ Complete capture of all print statements
        - ✅ Progress tracking and status updates
        - ✅ Downloadable logs and results
        """)

if __name__ == "__main__":
    main()