# api_mapping.py
import json
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from colorama import Fore
import traceback

from config import Config
from optimization_utils import create_compact_item, expand_compact_result

def PerformMapping(first_group: List[Dict], 
                   second_group: List[Dict], 
                   prompt: str,
                   verbose: bool = True,
                   use_compact: bool = True,
                   full_format_first: List[Dict] = None,
                   full_format_second: List[Dict] = None) -> Optional[Dict]:
    """
    Performs the actual mapping using OpenAI API with optimized token usage.
    Uses parameters from Config which are set by the user in Streamlit.
    
    Args:
        first_group: List of First Group items (compact or full format)
        second_group: List of Second Group items (compact or full format)
        prompt: Prompt text for the mapping
        verbose: If True, prints detailed information
        use_compact: If True, uses compact JSON format
        full_format_first: Full format of first group for result processing
        full_format_second: Full format of second group for result processing
    
    Returns:
        Dictionary with mapping results or None if error
    """
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}Starting Mapping Process (Optimized)")
    print(f"{Fore.MAGENTA}Using User-Defined Parameters:")
    print(f"{Fore.WHITE}  • Model: {Config.model}")
    print(f"{Fore.WHITE}  • Temperature: {Config.temperature}")
    print(f"{Fore.WHITE}  • Top P: {Config.top_p}")
    print(f"{Fore.WHITE}  • Max Tokens: {Config.max_tokens}")
    print(f"{Fore.WHITE}  • Threshold: {Config.threshold}")
    print(f"{Fore.MAGENTA}{'='*60}\n")
    
    # Check API key
    if not Config.api_key:
        print(f"{Fore.RED}✗ Error: OpenAI API key not found")
        print(f"{Fore.WHITE}Please set OPENAI_API_KEY environment variable")
        return None
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=Config.api_key)
        print(f"{Fore.GREEN}✓ OpenAI client initialized")
        
        # Prepare optimized prompt based on format
        if use_compact:
            # Create ultra-compact prompt
            optimized_prompt = f"""Map items from Group1 to Group2. Each item has 'c'(code) and 'n'(name).

Return JSON object with 'mappings' array. Each mapping:
{{"fc":"<first_code>","fn":"<first_name>","sc":"<second_code>","sn":"<second_name>","s":<score_1-100>,"r":"<reason>"}}

If no match: sc and sn should be null, s should be <{Config.threshold}.

Group1:
{json.dumps(first_group, ensure_ascii=False, separators=(',', ':'))}

Group2:
{json.dumps(second_group, ensure_ascii=False, separators=(',', ':'))}

Map ALL Group1 items. Return only JSON."""
        else:
            # Standard format prompt
            optimized_prompt = f"""
{prompt}

FIRST_GROUP:
{json.dumps(first_group, ensure_ascii=False, separators=(',', ':'))}

SECOND_GROUP:
{json.dumps(second_group, ensure_ascii=False, separators=(',', ':'))}

Return JSON object with 'mappings' array containing all mappings. Use threshold: {Config.threshold}"""
        
        if verbose:
            print(f"\n{Fore.CYAN}Optimized Prompt Preview (first 500 chars):")
            print(f"{Fore.WHITE}{optimized_prompt[:500]}...")
            print(f"\n{Fore.WHITE}Optimized prompt length: {len(optimized_prompt)} characters")
            print(f"{Fore.WHITE}Estimated tokens: ~{len(optimized_prompt)//4}")
        
        # Prepare API call
        print(f"\n{Fore.YELLOW}Calling OpenAI API...")
        print(f"{Fore.WHITE}Model: {Config.model}")
        print(f"{Fore.WHITE}Max Tokens: {Config.max_tokens}")
        print(f"{Fore.WHITE}Temperature: {Config.temperature}")
        print(f"{Fore.WHITE}Top P: {Config.top_p}")
        print(f"{Fore.WHITE}Optimization: {'COMPACT MODE' if use_compact else 'STANDARD MODE'}")
        
        start_time = time.time()
        
        # System message based on mode
        if use_compact:
            system_msg = f"You are a laboratory mapping expert. Use the exact abbreviated JSON format specified. Be concise. Apply threshold {Config.threshold} for similarity scores."
        else:
            system_msg = f"You are a world-class Laboratory Mapping expert. Return valid JSON with all mappings. Apply threshold {Config.threshold} for similarity scores."
        
        # Make API call with user-defined parameters
        try:
            response = client.chat.completions.create(
                model=Config.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": optimized_prompt
                    }
                ],
                max_tokens=Config.max_tokens,
                temperature=Config.temperature,
                top_p=Config.top_p,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            # Fallback without response_format
            print(f"{Fore.YELLOW}Note: JSON format enforcement not supported, using text mode")
            response = client.chat.completions.create(
                model=Config.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": optimized_prompt
                    }
                ],
                max_tokens=Config.max_tokens,
                temperature=Config.temperature,
                top_p=Config.top_p
            )
        
        elapsed_time = time.time() - start_time
        
        print(f"{Fore.GREEN}✓ API call completed in {elapsed_time:.2f} seconds")
        print(f"{Fore.WHITE}  • Model used: {Config.model}")
        print(f"{Fore.WHITE}  • Temperature used: {Config.temperature}")
        print(f"{Fore.WHITE}  • Top P used: {Config.top_p}")
        
        # Extract response
        response_text = response.choices[0].message.content
        
        if verbose:
            print(f"\n{Fore.CYAN}Response preview (first 500 chars):")
            print(f"{Fore.WHITE}{response_text[:500]}...")
        
        # Parse JSON response
        mapping_results = parse_optimized_response(response_text, use_compact, verbose)
        
        if mapping_results is None:
            return None
        
        print(f"{Fore.GREEN}✓ Successfully parsed {len(mapping_results)} mappings")
        
        # If using compact format, expand results
        if use_compact:
            expanded_results = []
            for item in mapping_results:
                expanded = expand_compact_result(item, "mapping")
                expanded_results.append(expanded)
            mapping_results = expanded_results
        
        # Return raw data for processing in another module
        return {
            "mappings": mapping_results,
            "response": response,
            "elapsed_time": elapsed_time,
            "response_text": response_text,
            "parameters_used": {
                "model": Config.model,
                "temperature": Config.temperature,
                "top_p": Config.top_p,
                "max_tokens": Config.max_tokens,
                "threshold": Config.threshold
            }
        }
            
    except Exception as e:
        print(f"{Fore.RED}✗ Error during API call: {str(e)}")
        print(f"{Fore.WHITE}  • Model attempted: {Config.model}")
        print(f"{Fore.WHITE}  • Temperature: {Config.temperature}")
        print(f"{Fore.WHITE}  • Top P: {Config.top_p}")
        if verbose:
            print(f"{Fore.YELLOW}Traceback:")
            print(traceback.format_exc())
        return None


def parse_optimized_response(response_text: str, is_compact: bool, verbose: bool) -> Optional[List[Dict]]:
    """Parse response based on format (compact or standard)"""
    
    mapping_results = None
    
    try:
        # Parse JSON
        parsed_json = json.loads(response_text)
        
        # Extract mappings
        if isinstance(parsed_json, dict) and "mappings" in parsed_json:
            mapping_results = parsed_json["mappings"]
            print(f"{Fore.GREEN}✓ Found mappings in JSON object")
        elif isinstance(parsed_json, list):
            mapping_results = parsed_json
            print(f"{Fore.GREEN}✓ Response is a direct JSON array")
        
    except json.JSONDecodeError as e:
        print(f"{Fore.YELLOW}JSON parse error, attempting extraction...")
        
        import re
        # Try to extract JSON
        obj_match = re.search(r'\{[^{}]*"mappings"\s*:\s*\[[^\]]*\][^{}]*\}', response_text, re.DOTALL)
        if obj_match:
            try:
                parsed_obj = json.loads(obj_match.group())
                mapping_results = parsed_obj.get("mappings", [])
                print(f"{Fore.GREEN}✓ Extracted mappings from text")
            except:
                pass
    
    if mapping_results is None:
        print(f"{Fore.RED}✗ Could not parse mapping results")
        if verbose:
            print(f"\n{Fore.YELLOW}Response (first 1000 chars):")
            print(f"{Fore.WHITE}{response_text[:1000]}")
        return None
    
    return mapping_results