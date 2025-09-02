# result_processor.py
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from colorama import Fore
import json

from config import Config

# Global DataFrames
df_api_call = pd.DataFrame(columns=[
    'Timestamp', 'Model', 'Temperature', 'Top P', 'Max Batch Size', 
    'Wait Time', 'Latency', 'Input Tokens', 'Output Tokens', 
    'Total Tokens', 'Total Mappings', 'Mapped Count', 
    'Unmapped Count', 'Avg Score'
])

df_api_mapping = pd.DataFrame(columns=[
    'First Group Code', 'First Group Name', 'Second Group Code',
    'Second Group Name', 'Similarity Score', 'Similarity Reason'
])

# Dictionary to track seen First Group Codes and their best scores
seen_first_codes = {}


def reset_dataframes():
    """Reset the global DataFrames and tracking dictionary"""
    global df_api_call, df_api_mapping, seen_first_codes
    
    df_api_call = pd.DataFrame(columns=[
        'Timestamp', 'Model', 'Temperature', 'Top P', 'Max Batch Size', 
        'Wait Time', 'Latency', 'Input Tokens', 'Output Tokens', 
        'Total Tokens', 'Total Mappings', 'Mapped Count', 
        'Unmapped Count', 'Avg Score'
    ])
    
    df_api_mapping = pd.DataFrame(columns=[
        'First Group Code', 'First Group Name', 'Second Group Code',
        'Second Group Name', 'Similarity Score', 'Similarity Reason'
    ])
    
    seen_first_codes = {}
    
    print(f"{Fore.CYAN}DataFrames and tracking dictionary reset")


def get_dataframes():
    """Return the current state of DataFrames"""
    global df_api_call, df_api_mapping
    return {
        'ApiCall': df_api_call.copy(),
        'ApiMapping': df_api_mapping.copy()
    }


def ProcessMappingResults(mappings: List[Dict], 
                         response: Any, 
                         elapsed_time: float, 
                         verbose: bool = True,
                         reset_before_processing: bool = True) -> Optional[Dict]:
    """
    Process mapping results with deduplication and DataFrame creation.
    Tracks the parameters used for this API call.
    
    Args:
        mappings: List of mapping dictionaries
        response: OpenAI API response object
        elapsed_time: Time taken for API call
        verbose: If True, prints detailed information
        reset_dataframes: If True, resets DataFrames before processing
    
    Returns:
        Dictionary with processed results and dataframes
    """
    
    global df_api_call, df_api_mapping, seen_first_codes
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}Processing Mapping Results with Deduplication")
    print(f"{Fore.MAGENTA}Using Parameters:")
    print(f"{Fore.WHITE}  • Model: {Config.model}")
    print(f"{Fore.WHITE}  • Temperature: {Config.temperature}")
    print(f"{Fore.WHITE}  • Top P: {Config.top_p}")
    print(f"{Fore.WHITE}  • Threshold: {Config.threshold}")
    print(f"{Fore.MAGENTA}{'='*60}\n")
    
    # Reset if requested (for first batch)
    if reset_before_processing:
        reset_dataframes()
    
    if not mappings:
        print(f"{Fore.RED}✗ No mappings to process")
        return None
    
    # Extract token usage from response
    try:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
    except:
        input_tokens = output_tokens = total_tokens = 0
    
    # Process mappings with deduplication
    new_mappings = []
    updated_mappings = 0
    duplicate_count = 0
    
    for mapping in mappings:
        first_code = mapping.get("First Group Code", "")
        first_name = mapping.get("First Group Name", "")
        second_code = mapping.get("Second Group Code")
        second_name = mapping.get("Second Group Name")
        score = mapping.get("similarity score", 0)
        reason = mapping.get("reason for similarity score", "")
        
        # Check if we've seen this First Group Code before
        if first_code in seen_first_codes:
            # Compare scores
            existing_score = seen_first_codes[first_code]['score']
            if score > existing_score:
                # Update with better score
                seen_first_codes[first_code] = {
                    'score': score,
                    'index': seen_first_codes[first_code]['index']  # Keep original index
                }
                
                # Update the DataFrame row
                idx = seen_first_codes[first_code]['index']
                df_api_mapping.loc[idx] = [
                    first_code, first_name, second_code, 
                    second_name, score, reason
                ]
                updated_mappings += 1
                
                if verbose:
                    print(f"{Fore.YELLOW}↑ Updated: {first_code} - Score improved from {existing_score} to {score}")
            else:
                duplicate_count += 1
                if verbose:
                    print(f"{Fore.BLUE}⊡ Duplicate: {first_code} - Keeping existing score {existing_score} (new: {score})")
        else:
            # New mapping - add it
            new_row = pd.DataFrame([{
                'First Group Code': first_code,
                'First Group Name': first_name,
                'Second Group Code': second_code,
                'Second Group Name': second_name,
                'Similarity Score': score,
                'Similarity Reason': reason
            }])
            
            # Get the index where this will be added
            new_index = len(df_api_mapping)
            df_api_mapping = pd.concat([df_api_mapping, new_row], ignore_index=True)
            
            # Track this First Group Code
            seen_first_codes[first_code] = {
                'score': score,
                'index': new_index
            }
            
            new_mappings.append(mapping)
            
            if verbose and len(new_mappings) <= 3:
                print(f"{Fore.GREEN}✓ New: {first_code} → {second_code} (Score: {score})")
    
    # Calculate statistics
    total_mappings = len(mappings)
    unique_mappings = len(seen_first_codes)
    mapped_count = sum(1 for _, data in seen_first_codes.items() 
                      if df_api_mapping.loc[data['index'], 'Second Group Code'] is not None 
                      and pd.notna(df_api_mapping.loc[data['index'], 'Second Group Code']))
    unmapped_count = unique_mappings - mapped_count
    
    # Calculate average score for non-null mappings
    valid_scores = df_api_mapping[df_api_mapping['Similarity Score'] > 0]['Similarity Score']
    avg_score = valid_scores.mean() if not valid_scores.empty else 0
    
    # Filter by threshold
    above_threshold = df_api_mapping[df_api_mapping['Similarity Score'] >= Config.threshold]
    below_threshold = df_api_mapping[df_api_mapping['Similarity Score'] < Config.threshold]
    
    # Add to API call DataFrame with parameters
    api_call_data = pd.DataFrame([{
        'Timestamp': datetime.now(),
        'Model': Config.model,
        'Temperature': Config.temperature,
        'Top P': Config.top_p,
        'Max Batch Size': Config.max_batch_size,
        'Wait Time': Config.wait_between_batches,
        'Latency': elapsed_time,
        'Input Tokens': input_tokens,
        'Output Tokens': output_tokens,
        'Total Tokens': total_tokens,
        'Total Mappings': total_mappings,
        'Mapped Count': mapped_count,
        'Unmapped Count': unmapped_count,
        'Avg Score': avg_score
    }])
    
    df_api_call = pd.concat([df_api_call, api_call_data], ignore_index=True)
    
    # Print summary
    print(f"\n{Fore.CYAN}Deduplication Summary:")
    print(f"{Fore.WHITE}  • Total mappings received: {total_mappings}")
    print(f"{Fore.WHITE}  • New mappings added: {len(new_mappings)}")
    print(f"{Fore.WHITE}  • Mappings updated (better score): {updated_mappings}")
    print(f"{Fore.WHITE}  • Duplicates ignored: {duplicate_count}")
    print(f"{Fore.WHITE}  • Total unique mappings: {unique_mappings}")
    
    print(f"\n{Fore.CYAN}Mapping Statistics:")
    print(f"{Fore.WHITE}  • Mapped items: {mapped_count}")
    print(f"{Fore.WHITE}  • Unmapped items: {unmapped_count}")
    print(f"{Fore.WHITE}  • Average similarity score: {avg_score:.2f}")
    print(f"{Fore.WHITE}  • Above threshold ({Config.threshold}): {len(above_threshold)}")
    print(f"{Fore.WHITE}  • Below threshold: {len(below_threshold)}")
    
    print(f"\n{Fore.CYAN}Token Usage:")
    print(f"{Fore.WHITE}  • Input tokens: {input_tokens:,}")
    print(f"{Fore.WHITE}  • Output tokens: {output_tokens:,}")
    print(f"{Fore.WHITE}  • Total tokens: {total_tokens:,}")
    
    print(f"\n{Fore.CYAN}Parameters Used:")
    print(f"{Fore.WHITE}  • Model: {Config.model}")
    print(f"{Fore.WHITE}  • Temperature: {Config.temperature}")
    print(f"{Fore.WHITE}  • Top P: {Config.top_p}")
    print(f"{Fore.WHITE}  • Max Batch Size: {Config.max_batch_size}")
    print(f"{Fore.WHITE}  • Wait Between Batches: {Config.wait_between_batches}s")
    
    # Return results
    return {
        "mappings": list(df_api_mapping.to_dict('records')),
        "statistics": {
            "total_mappings": total_mappings,
            "unique_mappings": unique_mappings,
            "new_mappings": len(new_mappings),
            "updated_mappings": updated_mappings,
            "duplicate_count": duplicate_count,
            "mapped_count": mapped_count,
            "unmapped_count": unmapped_count,
            "avg_score": avg_score,
            "above_threshold": len(above_threshold),
            "below_threshold": len(below_threshold)
        },
        "token_usage": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        },
        "parameters_used": {
            "model": Config.model,
            "temperature": Config.temperature,
            "top_p": Config.top_p,
            "max_batch_size": Config.max_batch_size,
            "wait_between_batches": Config.wait_between_batches,
            "threshold": Config.threshold
        },
        "dataframes": get_dataframes()
    }


def display_dataframe_summary():
    """Display summary of the DataFrames"""
    global df_api_call, df_api_mapping
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}DataFrame Summary")
    print(f"{Fore.MAGENTA}{'='*60}\n")
    
    print(f"{Fore.CYAN}API Call DataFrame:")
    print(f"{Fore.WHITE}  • Shape: {df_api_call.shape}")
    print(f"{Fore.WHITE}  • Columns: {list(df_api_call.columns)}")
    if not df_api_call.empty:
        print(f"\n{Fore.CYAN}Last API Call:")
        last_call = df_api_call.iloc[-1]
        print(f"{Fore.WHITE}  • Model: {last_call['Model']}")
        print(f"{Fore.WHITE}  • Temperature: {last_call['Temperature']}")
        print(f"{Fore.WHITE}  • Top P: {last_call['Top P']}")
        print(f"{Fore.WHITE}  • Latency: {last_call['Latency']:.2f}s")
        print(f"{Fore.WHITE}  • Total Tokens: {last_call['Total Tokens']:,}")
    
    print(f"\n{Fore.CYAN}API Mapping DataFrame:")
    print(f"{Fore.WHITE}  • Shape: {df_api_mapping.shape}")
    print(f"{Fore.WHITE}  • Unique First Group Codes: {df_api_mapping['First Group Code'].nunique()}")
    if not df_api_mapping.empty:
        print(f"{Fore.WHITE}  • Average Score: {df_api_mapping['Similarity Score'].mean():.2f}")
        print(f"{Fore.WHITE}  • Mapped: {df_api_mapping['Second Group Code'].notna().sum()}")
        print(f"{Fore.WHITE}  • Unmapped: {df_api_mapping['Second Group Code'].isna().sum()}")


def save_dataframes_to_excel(filepath: str):
    """Save DataFrames to Excel file with parameters tracked"""
    global df_api_call, df_api_mapping
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Save API Call data
            df_api_call.to_excel(writer, sheet_name='API_Calls', index=False)
            
            # Save Mapping data
            df_api_mapping.to_excel(writer, sheet_name='Mappings', index=False)
            
            # Create parameters sheet
            params_df = pd.DataFrame([
                ['Model', Config.model],
                ['Temperature', Config.temperature],
                ['Top P', Config.top_p],
                ['Max Tokens', Config.max_tokens],
                ['Max Batch Size', Config.max_batch_size],
                ['Wait Between Batches', f"{Config.wait_between_batches}s"],
                ['Threshold', Config.threshold],
                ['Use Compact JSON', Config.use_compact_json],
                ['Abbreviate Keys', Config.abbreviate_keys]
            ], columns=['Parameter', 'Value'])
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Total API Calls',
                    'Total Unique Mappings',
                    'Mapped Items',
                    'Unmapped Items',
                    'Average Similarity Score',
                    'Total Input Tokens',
                    'Total Output Tokens',
                    'Total Tokens Used',
                    'Average Latency'
                ],
                'Value': [
                    len(df_api_call),
                    len(df_api_mapping),
                    df_api_mapping['Second Group Code'].notna().sum(),
                    df_api_mapping['Second Group Code'].isna().sum(),
                    df_api_mapping['Similarity Score'].mean() if not df_api_mapping.empty else 0,
                    df_api_call['Input Tokens'].sum() if not df_api_call.empty else 0,
                    df_api_call['Output Tokens'].sum() if not df_api_call.empty else 0,
                    df_api_call['Total Tokens'].sum() if not df_api_call.empty else 0,
                    df_api_call['Latency'].mean() if not df_api_call.empty else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"{Fore.GREEN}✓ DataFrames saved to: {filepath}")
        print(f"{Fore.WHITE}  • Sheets: API_Calls, Mappings, Parameters, Summary")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗ Error saving to Excel: {str(e)}")
        return False