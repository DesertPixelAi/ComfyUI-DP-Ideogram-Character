"""
ComfyUI Ideogram Character Node
A custom node for generating consistent character images using Ideogram API v3
"""

# Import the node class with multiple fallback strategies
try:
    # First try relative import (works when loaded as package)
    from .nodes.ideogram_character import DP_IdeogramCharacter
except ImportError:
    try:
        # Try absolute import (works when running standalone)
        from nodes.ideogram_character import DP_IdeogramCharacter
    except ImportError:
        try:
            # Try with current directory in path
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from nodes.ideogram_character import DP_IdeogramCharacter
        except ImportError:
            # Final fallback - direct file import
            import sys
            import os
            import importlib.util
            
            # Get the path to the node file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            node_file = os.path.join(current_dir, 'nodes', 'ideogram_character.py')
            
            # Load the module directly
            spec = importlib.util.spec_from_file_location("ideogram_character", node_file)
            ideogram_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ideogram_module)
            
            DP_IdeogramCharacter = ideogram_module.DP_IdeogramCharacter

NODE_CLASS_MAPPINGS = {
    "DP_IdeogramCharacter": DP_IdeogramCharacter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DP_IdeogramCharacter": "DP Ideogram Character"
}

# Version info
__version__ = "1.0.0"
__author__ = "DP"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']