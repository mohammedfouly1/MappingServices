# result_processor.py
import time
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from colorama import Fore

from config import Config

# Global DataFrames for accumulating results
api_call_df = pd.DataFrame()
api_mapping_df = pd.DataFrame()


def deduplicate_mappings(incoming_mappings: List[Dict], existing_df: pd.DataFrame = None) -> List[Dict]:
    """
    Deduplicate mappings based on First Group Code, keeping highest similarity score.
    
    Args:
        incoming_mappings: New mappings to add
        existing_df: Existing DataFrame with previous mappings
    
    Returns:
        List of deduplicated mappings maintaining original order
    """
    
    # Initialize final list and index
    if existing_df is not None and not existing_df.empty:
        # Convert existing DataFrame to list of dicts
        final = existing_df.to_dict('records')
    else:
        final = []
    
    # Build index map by First Group Code
    byCode = {}
    for idx, row in enumerate(final):
        code = row.get("First Group Code")
        if code:
            byCode[code] = idx
    
    # Process incoming mappings
    for row in incoming_mappings:
        # Validate row has all required keys
        required_keys = [
            "First Group Code",
            "First Group Name", 
            "Second Group Code",
            "Second Group Name",
            "Similarity Score",  # Note: Our system uses this name
            "Similarity Reason"  # Note: Our system uses this name
        ]
        
        # Check if all keys exist (with some flexibility for naming)
        if not all(k in row or k.replace("Similarity ", "similarity ") in row for k in required_keys[:2]):
            continue  # Skip invalid row
            
        # Get similarity score
        try:
            score = int(row.get("Similarity Score", row.get("similarity score", 0)))
        except (TypeError, ValueError):
            continue  # Skip if score is not an integer
        
        # Apply score filter (> 50 for acceptance, but we'll use Config.threshold)
        if score <= 50:  # Hard minimum threshold
            continue
        
        # Get First Group Code
        code = row.get("First Group Code", "")
        if not code:
            continue
        
        # Check if code already exists
        if code not in byCode:
            # Add new row
            final.append(row)
            byCode[code] = len(final) - 1
        else:
            # Compare scores
            existing_idx = byCode[code]
            existing_score = int(final[existing_idx].get("Similarity Score", 
                                                         final[existing_idx].get("similarity score", 0)))
            
            if score > existing_score:
                # Replace with higher score, maintain position
                final[existing_idx] = row
            # If score <= existing_score, do nothing (keep existing)
    
    return final


def ProcessMappingResults(mappings: List[Dict], 
                          response: Any, 
                          elapsed_time: float,
                          verbose: bool = True,
                          reset_dataframes: bool = False,
                          batch_info: Dict = None) -> Dict:
    """
    Process and format mapping results into structured DataFrames with deduplication.
    
    Args:
        mappings: List of mapping results from API
        response: Raw API response object
        elapsed_time: Time taken for API call
        verbose: If True, prints detailed information
        reset_dataframes: If True, resets the global dataframes before processing
        batch_info: Optional batch information for tracking
    
    Returns:
        Dictionary containing both dataframes and current response data
    """
    
    global api_call_df, api_mapping_df
    
    print(f"\n{Fore.YELLOW}Processing mapping results into DataFrames...")
    
    # Reset dataframes if requested
    if reset_dataframes:
        api_call_df = pd.DataFrame()
        api_mapping_df = pd.DataFrame()
        print(f"{Fore.GREEN}✓ DataFrames reset")
    
    # Generate timestamp for this API call
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Apply threshold filtering to mappings
    threshold = Config.threshold
    filtered_mappings = []
    
    for mapping in mappings:
        # Ensure similarity score is an integer
        try:
            score = int(mapping.get("similarity score", 0))
        except (TypeError, ValueError):
            score = 0
        
        mapping["similarity score"] = score
        
        # Apply threshold - set to None if below threshold
        if score < threshold:
            mapping["Second Group Code"] = None
            mapping["Second Group Name"] = None
        
        filtered_mappings.append(mapping)
    
    # Get token usage from response
    usage = response.usage
    
    # ===== Create ApiCall DataFrame Entry =====
    api_call_entry = {
        "TimeStamp": timestamp,
        "Model ID": Config.model,
        "Latency Per seconds": round(elapsed_time, 3),
        "Tokens Input": usage.prompt_tokens,
        "Tokens OutPut": usage.completion_tokens,
        "Temperature": Config.temperature,
        "Top P": Config.top_p,
        "Max tokens": Config.max_tokens
    }
    
    # Add batch info if provided
    if batch_info:
        api_call_entry["Batch Number"] = batch_info.get("batch_number", "")
        api_call_entry["Total Batches"] = batch_info.get("total_batches", "")
    
    # Append to ApiCall DataFrame
    api_call_df = pd.concat([api_call_df, pd.DataFrame([api_call_entry])], ignore_index=True)
    
    # ===== Create ApiMapping DataFrame Entries with Deduplication =====
    
    # Prepare new mapping entries
    new_mapping_entries = []
    for mapping in filtered_mappings:
        mapping_entry = {
            "TimeStamp": timestamp,
            "First Group Code": mapping.get("First Group Code", ""),
            "First Group Name": mapping.get("First Group Name", ""),
            "Second Group Code": mapping.get("Second Group Code", None),
            "Second Group Name": mapping.get("Second Group Name", None),
            "Similarity Score": mapping.get("similarity score", 0),
            "Similarity Reason": mapping.get("reason for similarity score", "")
        }
        new_mapping_entries.append(mapping_entry)
    
    # Deduplicate against existing data
    if not api_mapping_df.empty:
        # Get existing data (without timestamp for deduplication)
        existing_data = api_mapping_df.drop(columns=['TimeStamp'], errors='ignore')
        
        # Perform deduplication
        deduplicated_entries = deduplicate_mappings(new_mapping_entries, existing_data)
        
        # Create new DataFrame with deduplicated data
        if deduplicated_entries:
            # Reset api_mapping_df with deduplicated data
            api_mapping_df = pd.DataFrame(deduplicated_entries)
            
            # Count how many were replaced/added
            original_count = len(existing_data)
            new_count = len(api_mapping_df)
            replaced_count = sum(1 for entry in new_mapping_entries 
                               if any(entry.get("First Group Code") == row.get("First Group Code") 
                                    for row in existing_data.to_dict('records')))
            
            if verbose:
                print(f"{Fore.CYAN}Deduplication Summary:")
                print(f"{Fore.WHITE}  • Previous mappings: {original_count}")
                print(f"{Fore.WHITE}  • New mappings received: {len(new_mapping_entries)}")
                print(f"{Fore.WHITE}  • Mappings after deduplication: {new_count}")
                if replaced_count > 0:
                    print(f"{Fore.WHITE}  • Mappings updated (higher score): {replaced_count}")
    else:
        # First batch - apply deduplication on new entries only
        deduplicated_entries = deduplicate_mappings(new_mapping_entries, None)
        
        if deduplicated_entries:
            api_mapping_df = pd.DataFrame(deduplicated_entries)
            
            if verbose and len(deduplicated_entries) < len(new_mapping_entries):
                print(f"{Fore.CYAN}Initial Deduplication:")
                print(f"{Fore.WHITE}  • Original mappings: {len(new_mapping_entries)}")
                print(f"{Fore.WHITE}  • After deduplication: {len(deduplicated_entries)}")
    
    # Print DataFrame information if verbose
    if verbose:
        print(f"\n{Fore.CYAN}DataFrame Statistics:")
        print(f"{Fore.WHITE}  • ApiCall DataFrame:")
        print(f"    - Total API calls recorded: {len(api_call_df)}")
        print(f"    - Latest call timestamp: {timestamp}")
        print(f"    - Latest tokens used: {usage.total_tokens}")
        
        print(f"\n{Fore.WHITE}  • ApiMapping DataFrame:")
        print(f"    - Total unique mappings: {len(api_mapping_df)}")
        print(f"    - Mappings from this call: {len(new_mapping_entries)}")
        
        # Show score distribution
        if not api_mapping_df.empty and 'Similarity Score' in api_mapping_df.columns:
            score_stats = api_mapping_df['Similarity Score'].describe()
            print(f"\n{Fore.CYAN}Similarity Score Statistics:")
            print(f"{Fore.WHITE}    - Mean: {score_stats['mean']:.1f}")
            print(f"{Fore.WHITE}    - Min: {score_stats['min']:.0f}")
            print(f"{Fore.WHITE}    - Max: {score_stats['max']:.0f}")
            print(f"{Fore.WHITE}    - Scores > 50: {(api_mapping_df['Similarity Score'] > 50).sum()}")
            print(f"{Fore.WHITE}    - Scores > {Config.threshold}: {(api_mapping_df['Similarity Score'] > Config.threshold).sum()}")
        
        # Show sample of current API call entry
        print(f"\n{Fore.CYAN}Current API Call Entry:")
        for key, value in api_call_entry.items():
            print(f"{Fore.WHITE}    {key}: {value}")
        
        # Show sample mappings
        if deduplicated_entries:
            print(f"\n{Fore.CYAN}Sample Unique Mappings (first 3):")
            for i, entry in enumerate(deduplicated_entries[:3], 1):
                print(f"\n{Fore.WHITE}  [{i}] Code: {entry['First Group Code']} | {entry['First Group Name']}")
                if entry['Second Group Code']:
                    print(f"      → {entry['Second Group Name']}")
                    print(f"      Score: {entry['Similarity Score']}")
                else:
                    print(f"      → No match (score: {entry['Similarity Score']})")
    
    # Create result dictionary with DataFrames and current response data
    result = {
        "timestamp": timestamp,
        "model": Config.model,
        "latency_seconds": round(elapsed_time, 3),
        "tokens": {
            "input": usage.prompt_tokens,
            "output": usage.completion_tokens,
            "total": usage.total_tokens
        },
        "parameters": {
            "temperature": Config.temperature,
            "top_p": Config.top_p,
            "max_tokens": Config.max_tokens,
            "threshold": threshold,
            "compact_mode": Config.use_compact_json
        },
        "mappings": filtered_mappings,  # Original mappings for this call
        "deduplicated_mappings": deduplicated_entries,  # Deduplicated across all calls
        "dataframes": {
            "api_call": api_call_df,
            "api_mapping": api_mapping_df
        }
    }
    
    return result


def get_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the current state of both DataFrames.
    
    Returns:
        Tuple of (api_call_df, api_mapping_df)
    """
    global api_call_df, api_mapping_df
    return api_call_df.copy(), api_mapping_df.copy()


def save_dataframes_to_excel(output_path: str = None) -> bool:
    """
    Save both DataFrames to an Excel file with separate sheets.
    
    Args:
        output_path: Path for output Excel file (uses default if None)
    
    Returns:
        True if saved successfully, False otherwise
    """
    global api_call_df, api_mapping_df
    
    if not output_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"mapping_results_{timestamp}.xlsx"
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save ApiCall DataFrame
            if not api_call_df.empty:
                api_call_df.to_excel(writer, sheet_name='ApiCall', index=False)
                print(f"{Fore.GREEN}✓ ApiCall DataFrame saved ({len(api_call_df)} records)")
            
            # Save ApiMapping DataFrame (deduplicated)
            if not api_mapping_df.empty:
                api_mapping_df.to_excel(writer, sheet_name='ApiMapping', index=False)
                print(f"{Fore.GREEN}✓ ApiMapping DataFrame saved ({len(api_mapping_df)} unique mappings)")
                
                # Show deduplication stats
                unique_codes = api_mapping_df['First Group Code'].nunique()
                total_rows = len(api_mapping_df)
                if unique_codes == total_rows:
                    print(f"{Fore.WHITE}  • All mappings are unique by First Group Code")
                else:
                    print(f"{Fore.YELLOW}  • Warning: {total_rows - unique_codes} duplicate First Group Codes found")
        
        print(f"{Fore.GREEN}✓ DataFrames saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ Error saving DataFrames: {str(e)}")
        return False


def save_dataframes_to_csv(output_prefix: str = None) -> bool:
    """
    Save both DataFrames to separate CSV files.
    
    Args:
        output_prefix: Prefix for output CSV files (uses default if None)
    
    Returns:
        True if saved successfully, False otherwise
    """
    global api_call_df, api_mapping_df
    
    if not output_prefix:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_prefix = f"mapping_results_{timestamp}"
    
    try:
        # Save ApiCall DataFrame
        if not api_call_df.empty:
            api_call_path = f"{output_prefix}_apicall.csv"
            api_call_df.to_csv(api_call_path, index=False, encoding='utf-8-sig')
            print(f"{Fore.GREEN}✓ ApiCall DataFrame saved to: {api_call_path}")
        
        # Save ApiMapping DataFrame
        if not api_mapping_df.empty:
            api_mapping_path = f"{output_prefix}_apimapping.csv"
            api_mapping_df.to_csv(api_mapping_path, index=False, encoding='utf-8-sig')
            print(f"{Fore.GREEN}✓ ApiMapping DataFrame saved to: {api_mapping_path}")
            print(f"{Fore.WHITE}  • Total unique mappings: {len(api_mapping_df)}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ Error saving DataFrames to CSV: {str(e)}")
        return False


def display_dataframe_summary():
    """
    Display a summary of the current DataFrames with deduplication info.
    """
    global api_call_df, api_mapping_df
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}DataFrame Summary")
    print(f"{Fore.CYAN}{'='*60}")
    
    # ApiCall DataFrame Summary
    print(f"\n{Fore.YELLOW}ApiCall DataFrame:")
    if not api_call_df.empty:
        print(f"{Fore.WHITE}  • Total API calls: {len(api_call_df)}")
        print(f"{Fore.WHITE}  • Total tokens used: {api_call_df['Tokens Input'].sum() + api_call_df['Tokens OutPut'].sum()}")
        print(f"{Fore.WHITE}  • Average latency: {api_call_df['Latency Per seconds'].mean():.2f} seconds")
        print(f"\n{Fore.CYAN}  Last 3 API calls:")
        print(api_call_df.tail(3).to_string(index=False))
    else:
        print(f"{Fore.WHITE}  No API calls recorded yet")
    
    # ApiMapping DataFrame Summary
    print(f"\n{Fore.YELLOW}ApiMapping DataFrame (Deduplicated):")
    if not api_mapping_df.empty:
        print(f"{Fore.WHITE}  • Total unique mappings: {len(api_mapping_df)}")
        print(f"{Fore.WHITE}  • Unique First Group Codes: {api_mapping_df['First Group Code'].nunique()}")
        
        # Check for any remaining duplicates (shouldn't be any)
        duplicates = api_mapping_df.duplicated(subset=['First Group Code'], keep=False).sum()
        if duplicates > 0:
            print(f"{Fore.YELLOW}  • Warning: {duplicates} duplicate First Group Codes found")
        else:
            print(f"{Fore.GREEN}  • All First Group Codes are unique ✓")
        
        print(f"{Fore.WHITE}  • Average similarity score: {api_mapping_df['Similarity Score'].mean():.2f}")
        print(f"{Fore.WHITE}  • Matched items (non-null Second Group): {api_mapping_df['Second Group Code'].notna().sum()}")
        print(f"{Fore.WHITE}  • Score distribution:")
        print(f"    - Scores > 50: {(api_mapping_df['Similarity Score'] > 50).sum()}")
        print(f"    - Scores > {Config.threshold}: {(api_mapping_df['Similarity Score'] > Config.threshold).sum()}")
        print(f"    - Scores = 100: {(api_mapping_df['Similarity Score'] == 100).sum()}")
        
        print(f"\n{Fore.CYAN}  Top 5 mappings by score:")
        display_cols = ['First Group Code', 'First Group Name', 'Second Group Name', 'Similarity Score']
        top_mappings = api_mapping_df.nlargest(5, 'Similarity Score')[display_cols]
        print(top_mappings.to_string(index=False))
    else:
        print(f"{Fore.WHITE}  No mappings recorded yet")


def reset_dataframes():
    """Reset global DataFrames"""
    global api_call_df, api_mapping_df
    api_call_df = pd.DataFrame()
    api_mapping_df = pd.DataFrame()
    print(f"{Fore.GREEN}✓ DataFrames reset")


def get_mapping_stats() -> Dict:
    """
    Get statistics about the current mapping DataFrame.
    
    Returns:
        Dictionary with mapping statistics
    """
    global api_mapping_df
    
    if api_mapping_df.empty:
        return {
            "total_mappings": 0,
            "unique_first_codes": 0,
            "matched_items": 0,
            "avg_score": 0
        }
    
    return {
        "total_mappings": len(api_mapping_df),
        "unique_first_codes": api_mapping_df['First Group Code'].nunique(),
        "matched_items": api_mapping_df['Second Group Code'].notna().sum(),
        "avg_score": api_mapping_df['Similarity Score'].mean(),
        "score_distribution": {
            "above_50": (api_mapping_df['Similarity Score'] > 50).sum(),
            "above_threshold": (api_mapping_df['Similarity Score'] > Config.threshold).sum(),
            "perfect_100": (api_mapping_df['Similarity Score'] == 100).sum()
        }
    }