# input_handler.py
import json
import pandas as pd
from typing import Optional, List, Dict
from pathlib import Path
from colorama import Fore

from config import Config
from optimization_utils import create_compact_item
from batch_dispatcher import Dispatcher  # NEW: Import Dispatcher instead of direct API mapping


def SendInputParts(excel_path: str = None, prompt_path: str = None, verbose: bool = True):
    """
    Opens Excel file, reads First Group and Second Group sheets,
    creates JSON lists, reads prompt text, and sends to Dispatcher for batch processing.
    
    Args:
        excel_path: Path to Excel file (uses default if None)
        prompt_path: Path to prompt text file (uses default if None)
        verbose: If True, prints detailed progress information
    
    Returns:
        Result from mapping function or None if error
    """
    
    # Use default paths if not provided
    excel_path = excel_path or Config.excel_path
    prompt_path = prompt_path or Config.prompt_path
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}Starting SendInputParts Function")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    # Initialize variables
    first_group_list = []
    second_group_list = []
    first_group_compact = []  # Compact version for API
    second_group_compact = []  # Compact version for API
    prompt_text = ""
    
    # Variables to count rows
    first_group_count = 0
    second_group_count = 0
    
    # ===== Step 1: Open and Read Excel File =====
    print(f"{Fore.YELLOW}[Step 1] Opening Excel file...")
    print(f"{Fore.WHITE}Path: {excel_path}")
    
    try:
        # Check if file exists
        if not Path(excel_path).exists():
            print(f"{Fore.RED}✗ Error: Excel file not found at {excel_path}")
            return None
        
        # Load Excel file
        excel_data = pd.ExcelFile(excel_path)
        print(f"{Fore.GREEN}✓ Excel file opened successfully")
        print(f"{Fore.WHITE}Available sheets: {excel_data.sheet_names}")
        
    except Exception as e:
        print(f"{Fore.RED}✗ Error opening Excel file: {str(e)}")
        return None
    
    # ===== Step 2: Read First Group Sheet =====
    print(f"\n{Fore.YELLOW}[Step 2] Reading 'First Group' sheet...")
    
    try:
        if 'First Group' not in excel_data.sheet_names:
            print(f"{Fore.RED}✗ Error: 'First Group' sheet not found")
            print(f"{Fore.WHITE}Available sheets: {excel_data.sheet_names}")
            return None
        
        # Read First Group sheet
        df_first = pd.read_excel(excel_data, sheet_name='First Group', header=None)
        print(f"{Fore.GREEN}✓ First Group sheet loaded")
        print(f"{Fore.WHITE}Shape: {df_first.shape[0]} rows × {df_first.shape[1]} columns")
        
        # Process First Group data
        for index, row in df_first.iterrows():
            # Skip empty rows
            if pd.isna(row[0]) and pd.isna(row[1]):
                continue
                
            first_code = str(row[0]) if not pd.isna(row[0]) else ""
            first_name = str(row[1]) if not pd.isna(row[1]) else ""
            
            # Full format for display
            item = {
                "First Group Code": first_code,
                "First Group Name": first_name
            }
            first_group_list.append(item)
            
            # Compact format for API
            compact_item = create_compact_item(first_code, first_name, "first")
            first_group_compact.append(compact_item)
            
            # Increment counter
            first_group_count += 1
        
        print(f"{Fore.GREEN}✓ Processed {first_group_count} items from First Group")
        
        # Print sample of First Group data
        if verbose and first_group_list:
            print(f"\n{Fore.CYAN}First Group Sample (first 3 items - Display Format):")
            for item in first_group_list[:3]:
                print(f"{Fore.WHITE}  {json.dumps(item, ensure_ascii=False)}")
            if len(first_group_list) > 3:
                print(f"{Fore.WHITE}  ... and {len(first_group_list) - 3} more items")
    
    except Exception as e:
        print(f"{Fore.RED}✗ Error reading First Group sheet: {str(e)}")
        return None
    
    # ===== Step 3: Read Second Group Sheet =====
    print(f"\n{Fore.YELLOW}[Step 3] Reading 'Second Group' sheet...")
    
    try:
        if 'Second Group' not in excel_data.sheet_names:
            print(f"{Fore.RED}✗ Error: 'Second Group' sheet not found")
            print(f"{Fore.WHITE}Available sheets: {excel_data.sheet_names}")
            return None
        
        # Read Second Group sheet
        df_second = pd.read_excel(excel_data, sheet_name='Second Group', header=None)
        print(f"{Fore.GREEN}✓ Second Group sheet loaded")
        print(f"{Fore.WHITE}Shape: {df_second.shape[0]} rows × {df_second.shape[1]} columns")
        
        # Process Second Group data
        for index, row in df_second.iterrows():
            # Skip empty rows
            if pd.isna(row[0]) and pd.isna(row[1]):
                continue
                
            second_code = str(row[0]) if not pd.isna(row[0]) else ""
            second_name = str(row[1]) if not pd.isna(row[1]) else ""
            
            # Full format for display
            item = {
                "Second Group Code": second_code,
                "Second Group Name": second_name
            }
            second_group_list.append(item)
            
            # Compact format for API
            compact_item = create_compact_item(second_code, second_name, "second")
            second_group_compact.append(compact_item)
            
            # Increment counter
            second_group_count += 1
        
        print(f"{Fore.GREEN}✓ Processed {second_group_count} items from Second Group")
        
        # Print sample of Second Group data
        if verbose and second_group_list:
            print(f"\n{Fore.CYAN}Second Group Sample (first 3 items - Display Format):")
            for item in second_group_list[:3]:
                print(f"{Fore.WHITE}  {json.dumps(item, ensure_ascii=False)}")
            if len(second_group_list) > 3:
                print(f"{Fore.WHITE}  ... and {len(second_group_list) - 3} more items")
    
    except Exception as e:
        print(f"{Fore.RED}✗ Error reading Second Group sheet: {str(e)}")
        return None
    
    # ===== Step 4: Read Prompt Text File =====
    print(f"\n{Fore.YELLOW}[Step 4] Reading prompt text file...")
    print(f"{Fore.WHITE}Path: {prompt_path}")
    
    try:
        if not Path(prompt_path).exists():
            print(f"{Fore.RED}✗ Error: Prompt file not found at {prompt_path}")
            return None
        
        with open(prompt_path, 'r', encoding='utf-8-sig') as f:
            prompt_text = f.read().strip()
        
        print(f"{Fore.GREEN}✓ Prompt file read successfully")
        print(f"{Fore.WHITE}Prompt length: {len(prompt_text)} characters")
        
        # Print prompt preview
        if verbose:
            preview_length = 200
            if len(prompt_text) > preview_length:
                print(f"\n{Fore.CYAN}Prompt Preview (first {preview_length} chars):")
                print(f"{Fore.WHITE}{prompt_text[:preview_length]}...")
            else:
                print(f"\n{Fore.CYAN}Prompt Text:")
                print(f"{Fore.WHITE}{prompt_text}")
    
    except Exception as e:
        print(f"{Fore.RED}✗ Error reading prompt file: {str(e)}")
        return None
    
    # ===== Step 5: Validate Data =====
    print(f"\n{Fore.YELLOW}[Step 5] Validating data...")
    
    if not first_group_list:
        print(f"{Fore.RED}✗ Error: First Group list is empty")
        return None
    
    if not second_group_list:
        print(f"{Fore.RED}✗ Error: Second Group list is empty")
        return None
    
    if not prompt_text:
        print(f"{Fore.RED}✗ Error: Prompt text is empty")
        return None
    
    print(f"{Fore.GREEN}✓ All data validated successfully")
    
    # Print row counts
    print(f"\n{Fore.CYAN}Data Statistics:")
    print(f"{Fore.WHITE}  • First Group: {first_group_count} rows")
    print(f"{Fore.WHITE}  • Second Group: {second_group_count} rows")
    print(f"{Fore.WHITE}  • Total rows: {first_group_count + second_group_count}")
    
    # ===== Step 6: Send to Dispatcher =====
    print(f"\n{Fore.YELLOW}[Step 6] Sending data to Dispatcher for batch processing...")
    print(f"{Fore.WHITE}Parameters being sent:")
    print(f"  • First Group: {first_group_count} items")
    print(f"  • Second Group: {second_group_count} items")
    print(f"  • Prompt: {len(prompt_text)} characters")
    print(f"  • Using Compact JSON: {Config.use_compact_json}")
    
    try:
        # NEW: Call Dispatcher instead of direct PerformMapping
        result = Dispatcher(
            first_group_list=first_group_list,
            second_group_list=second_group_list,
            first_group_compact=first_group_compact,
            second_group_compact=second_group_compact,
            prompt=prompt_text,
            n1=first_group_count,
            n2=second_group_count,
            verbose=verbose
        )
        
        if result:
            print(f"\n{Fore.GREEN}✓ Batch processing completed successfully")
            return result
        else:
            print(f"\n{Fore.RED}✗ Batch processing failed")
            return None
            
    except Exception as e:
        print(f"\n{Fore.RED}✗ Error during batch processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


def SaveResults(results: Dict, output_path: str = None) -> bool:
    """
    Save mapping results to JSON file and DataFrames to Excel.
    
    Args:
        results: Results dictionary to save
        output_path: Path for output file (uses default if None)
    
    Returns:
        True if saved successfully, False otherwise
    """
    import time
    from result_processor import save_dataframes_to_excel
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results (without dataframes)
    json_output_path = output_path or f"mapping_results_{timestamp}.json"
    
    try:
        # Create a copy without dataframes for JSON
        json_results = {k: v for k, v in results.items() if k != "dataframes"}
        
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{Fore.GREEN}✓ JSON results saved to: {json_output_path}")
        
        # Also save DataFrames to Excel
        excel_output_path = f"mapping_results_{timestamp}.xlsx"
        save_dataframes_to_excel(excel_output_path)
        
        return True
        
    except Exception as e:
        print(f"\n{Fore.RED}✗ Error saving results: {str(e)}")
        return False