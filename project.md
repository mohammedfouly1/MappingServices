Laboratory Mapping Service - Complete Project Documentation
Project Overview
This is an AI-powered Laboratory Mapping Service that intelligently maps laboratory items between two groups using OpenAI's GPT API. The system is designed to handle large datasets efficiently through intelligent batching, token optimization, and deduplication strategies.

Core Purpose
The service reads two groups of laboratory items from Excel files, uses AI to find similar items between groups based on semantic understanding, and outputs structured mapping results with similarity scores and explanations.

Architecture Overview

┌─────────────┐
│   main.py   │  Entry Point
└──────┬──────┘
       │
┌──────▼──────────────┐
│ input_handler.py    │  Data Input & Validation
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ batch_dispatcher.py │  Intelligent Batching
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ api_mapping.py      │  OpenAI API Interface
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ result_processor.py │  Result Processing & Deduplication
└─────────────────────┘
Module Descriptions
1. main.py - Application Entry Point
Purpose: Orchestrates the entire application flow

Key Functions:

main(): Main execution function that:
Initializes colored console output
Displays configuration settings
Resets DataFrames for fresh start
Calls input handler to process data
Displays results summary
Saves output files
Data Flow:


Start → Reset DataFrames → Call SendInputParts → Display Summary → Save Results → End
2. config.py - Configuration Management
Purpose: Centralized configuration for all settings

Key Settings:

File Paths: Excel input and prompt text locations
API Settings:
Model: gpt-4o
Max tokens: 16000
Temperature: 0.2 (low randomness for consistency)
Threshold: 80 (minimum similarity score)
Optimization Flags:
use_compact_json: Reduces token usage
abbreviate_keys: Uses short key names
Batch Settings:
max_batch_size: 200 rows maximum per API call
wait_between_batches: 120 seconds delay
3. input_handler.py - Data Input and Validation
Purpose: Reads and prepares data from Excel files

Key Functions:

SendInputParts(excel_path, prompt_path, verbose)
Reads Excel sheets: "First Group" and "Second Group"
Counts rows: Tracks first_group_count and second_group_count
Creates two formats:
Full format: {"First Group Code": "101", "First Group Name": "Blood Test"}
Compact format: {"c": "101", "n": "Blood Test"}
Validates data: Ensures no empty groups
Sends to Dispatcher: Passes all data with row counts
Data Transformation:


Excel Cell → DataFrame → Python Dict → Compact JSON → Dispatcher
SaveResults(results, output_path)
Saves JSON results (without DataFrames)
Calls Excel export for DataFrames
4. batch_dispatcher.py - Intelligent Batch Management
Purpose: Handles large datasets by creating optimal batches

Key Functions:

calculate_optimal_batch_split(n1, n2, max_batch_size)
Algorithm: Minimizes total batches using mathematical optimization
Strategy: Tests all combinations where f + s = 200
Returns: Optimal split with batch plan
Example Calculations:

500 × 1000 items → 50 batches (100×100 split)
300 × 300 items → 9 batches (100×100 split)
1 × 20000 items → 101 batches (1×199 split)
Dispatcher(first_group_list, second_group_list, ...)
Decision Logic:
If total ≤ 200: Single API call
If total > 200: Create batch plan
Batch Processing:
Processes each batch sequentially
Waits 2 minutes between batches
Accumulates results across all batches
Batch Pattern:


For each First Group Block:
    For each Second Group Block:
        Create Batch → Call API → Process Results → Wait
5. api_mapping.py - OpenAI API Interface
Purpose: Handles all OpenAI API interactions

Key Functions:

PerformMapping(first_group, second_group, prompt, ...)
Prompt Optimization:
Compact mode: Uses abbreviated JSON format
Standard mode: Full descriptive format
API Call Management:
Configures GPT parameters
Handles response format enforcement
Measures latency
Token Optimization:
Reduces prompt size by ~60% in compact mode
Example: {"First Group Code": "101"} → {"c":"101"}
parse_optimized_response(response_text, is_compact, verbose)
Parses JSON response
Handles malformed JSON with regex fallback
Expands compact format back to full format
API Response Format:


{
  "mappings": [
    {
      "fc": "101",
      "fn": "Blood Test",
      "sc": "201", 
      "sn": "CBC",
      "s": 95,
      "r": "Both are blood tests"
    }
  ]
}
6. result_processor.py - Result Processing & Deduplication
Purpose: Processes API results and maintains unique mappings

Key Functions:

deduplicate_mappings(incoming_mappings, existing_df)
Deduplication Logic:
Keeps only highest score for each "First Group Code"
Maintains original insertion order
Filters scores ≤ 50
Algorithm:
Build index by First Group Code
For each new mapping:
If code exists and score higher: Replace
If code new: Add
If score lower: Skip
ProcessMappingResults(mappings, response, elapsed_time, ...)
Creates DataFrames:
api_call_df: API call metadata
api_mapping_df: Deduplicated mappings
Applies Filters:
Threshold filtering (default 80)
Score validation
Statistics Tracking:
Token usage
Latency metrics
Score distributions
DataFrame Schemas:

ApiCall DataFrame:

Column	Type	Description
TimeStamp	str	Call timestamp
Model ID	str	GPT model used
Latency Per seconds	float	API response time
Tokens Input	int	Input tokens
Tokens Output	int	Output tokens
ApiMapping DataFrame:

Column	Type	Description
TimeStamp	str	Processing time
First Group Code	str	Source item code
First Group Name	str	Source item name
Second Group Code	str/null	Matched item code
Second Group Name	str/null	Matched item name
Similarity Score	int	Score 1-100
Similarity Reason	str	Explanation
7. optimization_utils.py - Token Optimization
Purpose: Reduces API costs through JSON compression

Key Functions:

create_compact_item(code, name, group_type)
Converts full format to compact
Reduces character count by ~70%
expand_compact_result(item, group_type)
Restores full format from compact
Maintains data integrity
Compression Example:


# Full Format (52 chars)
{"First Group Code": "101", "First Group Name": "Test"}

# Compact Format (20 chars)
{"c":"101","n":"Test"}
Data Flow Sequence

graph TD
    A[Excel Input] --> B[SendInputParts]
    B --> C{Row Count Check}
    C -->|≤200| D[Direct API Call]
    C -->|>200| E[Calculate Batches]
    E --> F[Process Batch 1]
    F --> G[Wait 2 min]
    G --> H[Process Batch 2]
    H --> I[...]
    I --> J[Combine Results]
    D --> K[Deduplicate]
    J --> K
    K --> L[Save Excel/JSON]
Key Features
1. Intelligent Batching
Automatically detects when batching is needed
Optimizes batch sizes to minimize API calls
Implements grid-sweep pattern for complete coverage
2. Token Optimization
Compact JSON format reduces tokens by 60-70%
Abbreviated keys (c instead of code)
Efficient prompt structuring
3. Deduplication System
Ensures unique First Group Codes
Keeps highest scoring matches
Maintains insertion order
4. Comprehensive Logging
Colored console output for clarity
Progress tracking for long operations
Detailed error messages with tracebacks
5. Flexible Output
Excel files with multiple sheets
JSON for programmatic access
CSV for data analysis
Usage Example

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run the application
python main.py
Expected Console Output:


============================================================
Laboratory Mapping Service - DataFrame Version
============================================================

Optimization Settings:
  • Compact JSON: True
  • Model: gpt-4o
  • Max Tokens: 16000

[Step 1] Opening Excel file...
✓ Excel file opened successfully

[Step 2] Reading 'First Group' sheet...
✓ Processed 500 items from First Group

[Step 3] Reading 'Second Group' sheet...
✓ Processed 1000 items from Second Group

DISPATCHER: Batch Processing Handler
⚠ Total rows (1500) > max batch size (200)
Calculating optimal batch strategy...

Batch Plan:
  • Total batches: 50
  • Estimated time: ~105 minutes

Processing Batch 1/50...
✓ Batch 1 completed successfully
Waiting 120 seconds before next batch...
Error Handling
The system includes comprehensive error handling:

File Not Found: Clear messages with path information
API Failures: Automatic retry with fallback modes
Invalid Data: Skip invalid rows with logging
Malformed JSON: Regex extraction fallback
Performance Characteristics
Small datasets (≤200 rows): ~10-30 seconds
Medium datasets (500-1000 rows): ~20-40 minutes with batching
Large datasets (5000+ rows): ~2-4 hours with optimal batching
Token usage: ~4 tokens per character (optimized to ~1.5 with compact mode)
Extension Points
For future development, the system can be extended:

Alternative AI Providers: Replace api_mapping.py
Different Data Sources: Modify input_handler.py
Custom Batching Strategies: Update batch_dispatcher.py
Additional Processing: Extend result_processor.py
New Output Formats: Add exporters to result_processor.py
Dependencies

pandas>=1.3.0      # DataFrame operations
openai>=1.0.0      # GPT API client
colorama>=0.4.4    # Colored console output
openpyxl>=3.0.0    # Excel file handling
Configuration Customization
Modify config.py to adjust:

File paths for different environments
API parameters for different use cases
Batch sizes based on rate limits
Thresholds for quality control
This modular architecture ensures maintainability, testability, and scalability for future enhancements.