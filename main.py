# main.py
from colorama import init, Fore
from config import Config
from input_handler import SendInputParts, SaveResults
from result_processor import display_dataframe_summary, reset_dataframes

# Initialize colorama for colored output
init(autoreset=True)

def main():
    """Main execution function"""
    
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}Laboratory Mapping Service - DataFrame Version")
    print(f"{Fore.CYAN}{'='*60}")
    
    # Display optimization settings
    print(f"\n{Fore.YELLOW}Optimization Settings:")
    print(f"{Fore.WHITE}  • Compact JSON: {Config.use_compact_json}")
    print(f"{Fore.WHITE}  • Abbreviated Keys: {Config.abbreviate_keys}")
    print(f"{Fore.WHITE}  • Model: {Config.model}")
    print(f"{Fore.WHITE}  • Max Tokens: {Config.max_tokens}")
    
    try:
        # Reset DataFrames at start
        reset_dataframes()
        
        # Call SendInputParts which handles everything
        results = SendInputParts(verbose=True)
        
        if results:
            # Display DataFrame summary
            display_dataframe_summary()
            
            # Save results to files
            SaveResults(results)
            
            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"{Fore.GREEN}Process completed successfully!")
            print(f"{Fore.GREEN}{'='*60}")
        else:
            print(f"\n{Fore.RED}{'='*60}")
            print(f"{Fore.RED}Process failed!")
            print(f"{Fore.RED}{'='*60}")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Process interrupted by user")
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()