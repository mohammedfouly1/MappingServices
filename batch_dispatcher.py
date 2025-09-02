# batch_dispatcher.py
import json
import math
import time
from typing import List, Dict, Optional, Tuple
from colorama import Fore

from config import Config
from api_mapping import PerformMapping
from result_processor import ProcessMappingResults


def calculate_optimal_batch_split(n1: int, n2: int, max_batch_size: int = 200) -> Dict:
    """
    Calculate optimal batch splitting strategy to minimize total batches.
    
    Args:
        n1: Number of rows in first group
        n2: Number of rows in second group
        max_batch_size: Maximum rows per batch (default 200)
    
    Returns:
        Dictionary with optimal batching plan
    """
    
    best_f, best_s = 0, 0
    min_batches = float('inf')
    
    # Try all possible splits where f + s = max_batch_size
    for f in range(1, max_batch_size):
        s = max_batch_size - f
        if s < 1:
            continue
            
        b1 = math.ceil(n1 / f)
        b2 = math.ceil(n2 / s)
        total_batches = b1 * b2
        
        if total_batches < min_batches:
            min_batches = total_batches
            best_f = f
            best_s = s
        elif total_batches == min_batches:
            # Tie-breaking rules
            current_diff = abs(best_f - best_s)
            new_diff = abs(f - s)
            
            if new_diff < current_diff:
                # Prefer more balanced split
                best_f = f
                best_s = s
            elif new_diff == current_diff:
                # Prefer smaller remainders
                current_remainder = (n1 % best_f) + (n2 % best_s)
                new_remainder = (n1 % f) + (n2 % s)
                
                if new_remainder < current_remainder:
                    best_f = f
                    best_s = s
                elif new_remainder == current_remainder and s > best_s:
                    # Prefer larger s
                    best_f = f
                    best_s = s
    
    # Calculate blocks
    b1 = math.ceil(n1 / best_f)
    b2 = math.ceil(n2 / best_s)
    
    # Create block ranges
    first_blocks = []
    for i in range(b1):
        start = i * best_f
        end = min((i + 1) * best_f, n1)
        first_blocks.append({
            "index": i + 1,
            "start": start + 1,  # 1-indexed
            "end": end
        })
    
    second_blocks = []
    for j in range(b2):
        start = j * best_s
        end = min((j + 1) * best_s, n2)
        second_blocks.append({
            "index": j + 1,
            "start": start + 1,  # 1-indexed
            "end": end
        })
    
    # Create batch plan
    batches = []
    batch_index = 1
    for i in range(b1):
        for j in range(b2):
            batch = {
                "batch_index": batch_index,
                "first_block_index": i + 1,
                "second_block_index": j + 1,
                "first_range": [first_blocks[i]["start"], first_blocks[i]["end"]],
                "second_range": [second_blocks[j]["start"], second_blocks[j]["end"]]
            }
            batches.append(batch)
            batch_index += 1
    
    return {
        "n1": n1,
        "n2": n2,
        "f": best_f,
        "s": best_s,
        "b1": b1,
        "b2": b2,
        "total_batches": min_batches,
        "first_blocks": first_blocks,
        "second_blocks": second_blocks,
        "batches": batches
    }


def Dispatcher(first_group_list: List[Dict],
               second_group_list: List[Dict],
               first_group_compact: List[Dict],
               second_group_compact: List[Dict],
               prompt: str,
               n1: int,
               n2: int,
               verbose: bool = True,
               max_batch_size: int = None,
               wait_between_batches: int = None) -> Optional[Dict]:
    """
    Dispatcher function that handles batching of large datasets.
    
    Args:
        first_group_list: Full format first group data
        second_group_list: Full format second group data
        first_group_compact: Compact format first group data
        second_group_compact: Compact format second group data
        prompt: Prompt text
        n1: Number of rows in first group
        n2: Number of rows in second group
        verbose: Print detailed information
        max_batch_size: Maximum rows per batch (uses Config if None)
        wait_between_batches: Seconds to wait between batches (uses Config if None)
    
    Returns:
        Combined results from all batches
    """
    
    # Use Config values if not provided
    max_batch_size = max_batch_size or Config.max_batch_size
    wait_between_batches = wait_between_batches or Config.wait_between_batches
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}DISPATCHER: Batch Processing Handler")
    print(f"{Fore.MAGENTA}Using Parameters:")
    print(f"{Fore.WHITE}  • Model: {Config.model}")
    print(f"{Fore.WHITE}  • Temperature: {Config.temperature}")
    print(f"{Fore.WHITE}  • Top P: {Config.top_p}")
    print(f"{Fore.WHITE}  • Max Tokens: {Config.max_tokens}")
    print(f"{Fore.WHITE}  • Max Batch Size: {max_batch_size}")
    print(f"{Fore.WHITE}  • Wait Between Batches: {wait_between_batches}s")
    print(f"{Fore.WHITE}  • Threshold: {Config.threshold}")
    print(f"{Fore.MAGENTA}{'='*60}\n")
    
    # Check if batching is needed
    total_rows = n1 + n2
    
    if total_rows <= max_batch_size:
        # No batching needed
        print(f"{Fore.GREEN}✓ Total rows ({total_rows}) ≤ max batch size ({max_batch_size})")
        print(f"{Fore.YELLOW}Proceeding without batching...")
        
        # Call PerformMapping directly
        if Config.use_compact_json:
            api_result = PerformMapping(
                first_group=first_group_compact,
                second_group=second_group_compact,
                prompt=prompt,
                verbose=verbose,
                use_compact=True,
                full_format_first=first_group_list,
                full_format_second=second_group_list
            )
        else:
            api_result = PerformMapping(
                first_group=first_group_list,
                second_group=second_group_list,
                prompt=prompt,
                verbose=verbose,
                use_compact=False
            )
        
        if api_result is None:
            return None
        
        # Process results
        return ProcessMappingResults(
            mappings=api_result["mappings"],
            response=api_result["response"],
            elapsed_time=api_result["elapsed_time"],
            verbose=verbose
        )
    
    # Batching is needed
    print(f"{Fore.YELLOW}⚠ Total rows ({total_rows}) > max batch size ({max_batch_size})")
    print(f"{Fore.CYAN}Calculating optimal batch strategy...")
    
    # Calculate optimal batching plan
    batch_plan = calculate_optimal_batch_split(n1, n2, max_batch_size)
    
    # Display batch plan
    print(f"\n{Fore.CYAN}Batch Plan:")
    print(f"{Fore.WHITE}  • First group rows (n1): {batch_plan['n1']}")
    print(f"{Fore.WHITE}  • Second group rows (n2): {batch_plan['n2']}")
    print(f"{Fore.WHITE}  • Rows per batch from first group (f): {batch_plan['f']}")
    print(f"{Fore.WHITE}  • Rows per batch from second group (s): {batch_plan['s']}")
    print(f"{Fore.WHITE}  • Number of first group blocks (b1): {batch_plan['b1']}")
    print(f"{Fore.WHITE}  • Number of second group blocks (b2): {batch_plan['b2']}")
    print(f"{Fore.WHITE}  • Total batches: {batch_plan['total_batches']}")
    print(f"{Fore.WHITE}  • Wait between batches: {wait_between_batches} seconds")
    print(f"{Fore.WHITE}  • Estimated total time: ~{batch_plan['total_batches'] * (wait_between_batches + 10) / 60:.1f} minutes")
    
    if verbose:
        print(f"\n{Fore.CYAN}Batch Details (first 5):")
        for batch in batch_plan['batches'][:5]:
            print(f"{Fore.WHITE}  Batch {batch['batch_index']}: "
                  f"First[{batch['first_range'][0]}-{batch['first_range'][1]}] × "
                  f"Second[{batch['second_range'][0]}-{batch['second_range'][1]}]")
        if len(batch_plan['batches']) > 5:
            print(f"{Fore.WHITE}  ... and {len(batch_plan['batches']) - 5} more batches")
    
    # Process batches
    print(f"\n{Fore.YELLOW}Starting batch processing...")
    all_results = []
    
    for i, batch in enumerate(batch_plan['batches'], 1):
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}Processing Batch {i}/{batch_plan['total_batches']}")
        print(f"{Fore.CYAN}Using Parameters:")
        print(f"{Fore.WHITE}  • Model: {Config.model}")
        print(f"{Fore.WHITE}  • Temperature: {Config.temperature}")
        print(f"{Fore.WHITE}  • Top P: {Config.top_p}")
        print(f"{Fore.CYAN}{'='*50}")
        
        # Extract batch data (using 0-indexed for list slicing)
        first_start = batch['first_range'][0] - 1
        first_end = batch['first_range'][1]
        second_start = batch['second_range'][0] - 1
        second_end = batch['second_range'][1]
        
        # Get batch subsets
        batch_first_list = first_group_list[first_start:first_end]
        batch_second_list = second_group_list[second_start:second_end]
        batch_first_compact = first_group_compact[first_start:first_end]
        batch_second_compact = second_group_compact[second_start:second_end]
        
        print(f"{Fore.WHITE}  • First group: rows {batch['first_range'][0]}-{batch['first_range'][1]} ({len(batch_first_list)} items)")
        print(f"{Fore.WHITE}  • Second group: rows {batch['second_range'][0]}-{batch['second_range'][1]} ({len(batch_second_list)} items)")
        
        # Call PerformMapping for this batch
        if Config.use_compact_json:
            api_result = PerformMapping(
                first_group=batch_first_compact,
                second_group=batch_second_compact,
                prompt=prompt,
                verbose=False,  # Less verbose for individual batches
                use_compact=True,
                full_format_first=batch_first_list,
                full_format_second=batch_second_list
            )
        else:
            api_result = PerformMapping(
                first_group=batch_first_list,
                second_group=batch_second_list,
                prompt=prompt,
                verbose=False,
                use_compact=False
            )
        
        if api_result is None:
            print(f"{Fore.RED}✗ Batch {i} failed")
            continue
        
        # Process batch results
        batch_result = ProcessMappingResults(
            mappings=api_result["mappings"],
            response=api_result["response"],
            elapsed_time=api_result["elapsed_time"],
            verbose=False,
            reset_before_processing=False  # Don't reset, accumulate results
        )
        
        if batch_result:
            all_results.append(batch_result)
            print(f"{Fore.GREEN}✓ Batch {i} completed successfully")
            print(f"{Fore.WHITE}  • Mappings in this batch: {len(api_result['mappings'])}")
        
        # Wait between batches (except for the last one)
        if i < batch_plan['total_batches']:
            print(f"\n{Fore.YELLOW}Waiting {wait_between_batches} seconds before next batch...")
            for j in range(wait_between_batches, 0, -30):
                remaining = min(j, 30)
                print(f"{Fore.WHITE}  {j} seconds remaining...", end='\r')
                time.sleep(remaining)
            print(f"{Fore.WHITE}  Ready to continue...                    ")
    
    # Combine all results
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}Combining results from all batches...")
    print(f"{Fore.CYAN}Parameters used throughout:")
    print(f"{Fore.WHITE}  • Model: {Config.model}")
    print(f"{Fore.WHITE}  • Temperature: {Config.temperature}")
    print(f"{Fore.WHITE}  • Top P: {Config.top_p}")
    print(f"{Fore.WHITE}  • Max Batch Size: {max_batch_size}")
    print(f"{Fore.WHITE}  • Wait Between Batches: {wait_between_batches}s")
    print(f"{Fore.CYAN}{'='*60}")
    
    if not all_results:
        print(f"{Fore.RED}✗ No successful batches")
        return None
    
    # Return the last batch result (which contains accumulated DataFrames)
    final_result = all_results[-1] if all_results else None
    
    if final_result:
        # Update summary statistics
        total_mappings = sum(len(r.get("mappings", [])) for r in all_results)
        print(f"\n{Fore.GREEN}✓ Batch processing completed")
        print(f"{Fore.WHITE}  • Total batches processed: {len(all_results)}/{batch_plan['total_batches']}")
        print(f"{Fore.WHITE}  • Total mappings: {total_mappings}")
        
        # Add batch processing metadata
        final_result["batch_metadata"] = {
            "total_batches": batch_plan['total_batches'],
            "batches_processed": len(all_results),
            "batch_plan": batch_plan,
            "parameters_used": {
                "model": Config.model,
                "temperature": Config.temperature,
                "top_p": Config.top_p,
                "max_batch_size": max_batch_size,
                "wait_between_batches": wait_between_batches
            }
        }
    
    return final_result