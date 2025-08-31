Laboratory Mapping Service - Complete Project Documentation
📋 Project Overview
The Laboratory Mapping Service is an AI-powered system that maps laboratory test items between two different groups using OpenAI's GPT models. It features intelligent batching, deduplication, token optimization, and both CLI and web interfaces.

🏗️ Architecture

┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                       │
│                  (streamlit_app.py)                      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                   Input Handler                          │
│                (input_handler.py)                        │
│  • Reads Excel files (First Group, Second Group)        │
│  • Loads prompt text                                     │
│  • Validates data                                        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Batch Dispatcher                         │
│              (batch_dispatcher.py)                       │
│  • Calculates optimal batch strategy                     │
│  • Splits large datasets into manageable chunks          │
│  • Manages API call scheduling                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  API Mapping Engine                      │
│                 (api_mapping.py)                         │
│  • Formats prompts (compact/standard)                    │
│  • Calls OpenAI API                                      │
│  • Handles responses                                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Result Processor                         │
│              (result_processor.py)                       │
│  • Deduplicates mappings                                 │
│  • Creates DataFrames                                    │
│  • Generates statistics                                  │
│  • Exports to Excel/CSV                                  │
└─────────────────────────────────────────────────────────┘
📁 File Structure

laboratory-mapping-service/
│
├── 🎯 Core Components
│   ├── main.py                 # CLI entry point
│   ├── streamlit_app.py        # Web UI entry point
│   ├── config.py               # Configuration management
│   └── optimization_utils.py   # Token optimization utilities
│
├── 🔄 Processing Modules
│   ├── input_handler.py        # Input processing & validation
│   ├── batch_dispatcher.py     # Batch management & optimization
│   ├── api_mapping.py          # OpenAI API integration
│   └── result_processor.py     # Result deduplication & DataFrames
│
├── 📋 Configuration Files
│   ├── requirements.txt        # Python dependencies
│   ├── requirements_streamlit.txt # Additional Streamlit deps
│   └── .gitignore             # Git exclusions
│
└── 📊 Sample Data (not in repo)
    ├── Input.xlsx             # Sample input file
    └── Prompt.txt             # Sample prompt text
🔧 Module Details
1. config.py - Configuration Management

Key Features:
• Streamlit Cloud integration with secrets management
• Environment variable fallback for local development
• Dynamic configuration loading
• Support for both cloud and local deployments

Settings:
• API Key management (secure)
• Model selection (gpt-4o, gpt-4, etc.)
• Token limits and temperature control
• Batch size and timing configuration
• Optimization toggles (compact JSON, abbreviated keys)
2. input_handler.py - Data Input Processing

Functions:
• SendInputParts(): Main orchestrator
  - Loads Excel files with error handling
  - Validates data integrity
  - Creates both full and compact JSON formats
  - Initiates batch processing

• SaveResults(): Result persistence
  - Saves JSON results
  - Exports DataFrames to Excel
  - Generates timestamped outputs

Data Flow:
Excel → Pandas DataFrames → JSON Lists → Batch Dispatcher
3. batch_dispatcher.py - Intelligent Batching

Key Algorithm:
• Optimal batch splitting to minimize API calls
• Dynamic calculation based on dataset size
• Handles datasets of any size efficiently

Functions:
• calculate_optimal_batch_split(): 
  - Finds best split (f, s) where f + s = max_batch_size
  - Minimizes total batches = ceil(n1/f) × ceil(n2/s)
  
• Dispatcher():
  - Routes to direct processing or batching
  - Manages wait times between batches
  - Accumulates results across batches

Optimization:
For n1=300, n2=400, max_batch=200:
• Calculates f=86, s=114
• Results in 4×4=16 batches instead of naive 35 batches
4. api_mapping.py - OpenAI Integration

Features:
• Dual format support (compact/standard JSON)
• Token optimization through abbreviated keys
• Robust error handling and retry logic
• Response parsing with fallback strategies

Compact Format Example:
Instead of:
{"First Group Code": "123", "First Group Name": "Test A"}

Uses:
{"c": "123", "n": "Test A"}

Result: ~60% token reduction
5. result_processor.py - Data Processing & Deduplication

Core Features:
• Intelligent deduplication by First Group Code
• Keeps highest similarity score for duplicates
• Accumulates results across batches
• Real-time statistics generation

DataFrames:
1. ApiCall DataFrame:
   - Timestamp, Model, Latency
   - Token usage (input/output)
   - Parameters used

2. ApiMapping DataFrame:
   - Unique mappings (deduplicated)
   - Similarity scores and reasons
   - Filterable by threshold

Deduplication Algorithm:
• Maintains order of first occurrence
• Updates only if higher score found
• Filters scores below threshold
• Ensures one mapping per First Group Code
6. optimization_utils.py - Token Optimization

Strategies:
• Compact JSON with abbreviated keys
• Minimal separators in JSON
• Efficient prompt structuring

Token Savings:
• Standard format: ~4 tokens per item
• Compact format: ~1.5 tokens per item
• Result: 60-70% token reduction
7. streamlit_app.py - Web Interface

Features:
• Drag-and-drop file upload
• Real-time console output capture
• Interactive configuration panel
• Live processing visualization
• Downloadable results (Excel, JSON, logs)

UI Components:
• Sidebar: Configuration settings
• Main Area: Data preview, processing, results
• Console: Real-time colored output
• Charts: Score distribution, token usage

Session Management:
• Preserves state between interactions
• Handles file uploads securely
• Manages processing lifecycle
🚀 Key Features
1. Intelligent Batching
Automatically detects when batching is needed
Calculates optimal batch sizes to minimize API calls
Manages timing between batches to avoid rate limits
2. Deduplication System
Ensures one mapping per First Group Code
Keeps highest similarity score
Maintains original order
Works across multiple batches
3. Token Optimization
Compact JSON format reduces tokens by 60-70%
Abbreviated keys (c/n instead of code/name)
Efficient prompt structuring
Automatic format selection
4. Real-time Monitoring
Live console output with color coding
Progress bars and status updates
Token usage tracking
Performance metrics
5. Flexible Deployment
Local CLI execution
Streamlit web interface
Cloud deployment ready
Docker support
📊 Data Flow

graph TD
    A[Excel Input] -->|Load| B[Input Handler]
    B --> C{Dataset Size?}
    C -->|Small| D[Direct Processing]
    C -->|Large| E[Batch Dispatcher]
    E --> F[Calculate Batches]
    F --> G[Process Batch 1]
    G --> H[Process Batch 2..N]
    D --> I[API Mapping]
    H --> I
    I --> J[Result Processor]
    J --> K[Deduplication]
    K --> L[DataFrames]
    L --> M[Excel Output]
    L --> N[JSON Output]
🔐 Security Features
API Key Management

Environment variables for local
Streamlit secrets for cloud
Never stored in code
Input Validation

File type checking
Data integrity validation
Error boundaries
Secure Cloud Deployment

HTTPS only
Optional authentication
Secrets management
📈 Performance Optimizations
Token Reduction: 60-70% fewer tokens through compact JSON
Batch Optimization: Minimizes API calls through intelligent splitting
Deduplication: Reduces redundant processing
Caching: Session state management in Streamlit
Parallel Processing: Ready for async implementation
🌐 Deployment Options
Local Development

# CLI Mode
python main.py

# Web Interface
streamlit run streamlit_app.py
Cloud Deployment
Streamlit Cloud: Free, easy, GitHub integration
Heroku: Scalable, custom domain support
Google Cloud Run: Serverless, auto-scaling
AWS EC2/ECS: Full control, enterprise features
📝 Usage Example
Input Excel Structure

Sheet: "First Group"
| Code | Name        |
|------|-------------|
| A001 | Blood Test  |
| A002 | Urine Test  |

Sheet: "Second Group"
| Code | Name          |
|------|---------------|
| B101 | CBC Analysis  |
| B102 | Urinalysis    |
Prompt Example

Map laboratory tests from First Group to Second Group.
Consider test names, purposes, and methodologies.
Provide similarity scores (0-100) and reasons.
Output Structure

{
  "mappings": [
    {
      "First Group Code": "A001",
      "First Group Name": "Blood Test",
      "Second Group Code": "B101",
      "Second Group Name": "CBC Analysis",
      "Similarity Score": 85,
      "Similarity Reason": "Both are blood analysis tests"
    }
  ]
}
🔄 Update & Maintenance
Adding New Features
Update relevant module
Test locally with main.py
Test web UI with streamlit_app.py
Push to GitHub for auto-deployment
Monitoring
API call logs in ApiCall DataFrame
Token usage tracking
Performance metrics
Error logging
📚 Dependencies

Core:
• openai>=1.0.0      # AI integration
• pandas>=1.3.0      # Data processing
• colorama>=0.4.4    # Console colors
• openpyxl>=3.0.0    # Excel handling

Web Interface:
• streamlit>=1.28.0  # Web framework
• plotly>=5.17.0     # Visualizations
🎯 Best Practices
For Large Datasets

Use batch size 150-200
Set wait time 120-180 seconds
Enable compact JSON
For Accuracy

Use gpt-4o or gpt-4
Set temperature 0.2-0.3
Threshold 70-80
For Cost Optimization

Enable all optimizations
Use gpt-3.5-turbo for testing
Monitor token usage
🚦 Status Codes & Indicators

✓ Success (Green)
✗ Error (Red)
⚠ Warning (Yellow)
• Information (White)
═ Separator (Magenta)
→ Mapping (Cyan)
📞 Support & Contribution
This system is designed for laboratory data mapping but can be adapted for any similar mapping tasks between two datasets. The modular architecture allows easy customization and extension.

Version: 2.0.0
Last Updated: November 2024
Author: Laboratory Mapping Service Team
License: MIT