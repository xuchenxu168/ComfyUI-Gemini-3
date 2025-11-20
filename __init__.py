"""
ComfyUI-Gemini-3-2: Full-featured Google Gemini 3 Integration for ComfyUI
Implements all Gemini 3 capabilities including:
- Dynamic/High/Low Thinking Levels
- Media Resolution Control (Low/Medium/High)
- Thought Signatures for reasoning context
- Function Calling with strict validation
- Structured Outputs with tools
- Multimodal inputs (text, image, video, audio, PDF)
- Context Caching
- Batch API support
"""

# Patch asyncio to allow nested event loops, which is necessary for ComfyUI's async execution.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("nest_asyncio not found. Please install it with 'pip install nest_asyncio' if you encounter event loop errors.")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

