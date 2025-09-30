#!/usr/bin/env python3
"""
Interactive Module Interface
===========================

Wrapper for the existing interactive interface to be called from CLI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the interactive interface"""
    try:
        # Import the existing interactive module
        import tier1_interactive
        
        # Check if it has a main function, otherwise just run the module
        if hasattr(tier1_interactive, 'main'):
            tier1_interactive.main()
        else:
            # If no main function, the module likely runs on import
            print("Interactive interface loaded successfully!")
    except Exception as e:
        print(f"Error launching interactive interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()