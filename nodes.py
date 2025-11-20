"""
ComfyUI Gemini 3 Nodes - Complete Implementation
Implements all Gemini 3 capabilities with high performance
"""

import torch
import os
import json
import asyncio
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image
import io
import base64
import numpy as np

from .gemini_client import (
    GeminiClient, Content, Part, GenerationConfig, Tool, SafetySetting,
    encode_image_tensor, encode_video, encode_audio, encode_pdf, get_api_key, encode_audio_tensor
)


def run_async(coro):
    """Helper to run async coroutines - compatible with ComfyUI's event loop"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)

def _extract_video_path(video: Any) -> str:
    """Extracts the video file path from various VIDEO object types."""
    if not video:
        return ""

    if isinstance(video, str):
        return video

    # Common attributes for file paths from different video loader nodes
    path_attributes = ['file_path', 'filepath', 'saved_path', 'filename', 'file', 'path']
    for attr in path_attributes:
        if hasattr(video, attr):
            value = getattr(video, attr)
            if isinstance(value, str) and value.strip():
                # Check if it's a full path
                if os.path.isabs(value) and os.path.exists(value):
                    return value

    # Handle dictionary-based VIDEO objects (e.g., from VHS)
    if isinstance(video, dict) and 'filename' in video:
        filename = video['filename']
        subfolder = video.get('subfolder', '')
        video_type = video.get('type', 'output')
        try:
            import folder_paths
            if video_type == 'input':
                base_dir = folder_paths.get_input_directory()
            elif video_type == 'temp':
                base_dir = folder_paths.get_temp_directory()
            else: # output
                base_dir = folder_paths.get_output_directory()

            full_path = os.path.join(base_dir, subfolder, filename) if subfolder else os.path.join(base_dir, filename)
            if os.path.exists(full_path):
                return full_path
        except ImportError:
            print("Warning: folder_paths module not found. Could not construct full path for video object.")

    # Fallback for list-based VIDEO objects
    if isinstance(video, (list, tuple)) and len(video) > 0 and isinstance(video[0], str):
        if os.path.exists(video[0]):
            return video[0]

    print(f"Warning: Could not extract a valid file path from VIDEO object of type {type(video)}")
    return ""




# Load config data
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.json")
API_PROVIDERS = ["google"]  # Default value
DEFAULT_PROVIDER = "google"
ALL_MODELS = ["gemini-3-pro-preview"] # Default value
PROVIDER_MODELS = {}

if os.path.exists(CONFIG_FILE_PATH):
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if "api_providers" in config and isinstance(config["api_providers"], dict):
                API_PROVIDERS = list(config["api_providers"].keys())
                # Create a set of all unique models and a mapping for validation
                all_models_set = set()
                for provider, details in config["api_providers"].items():
                    if "models" in details and isinstance(details["models"], list):
                        PROVIDER_MODELS[provider] = details["models"]
                        all_models_set.update(details["models"])
                if all_models_set:
                    ALL_MODELS = sorted(list(all_models_set))

            if "default_provider" in config and config["default_provider"] in API_PROVIDERS:
                DEFAULT_PROVIDER = config["default_provider"]
    except Exception as e:
        print(f"[Gemini 3] Warning: Could not load API providers and models from config.json: {e}")

class Gemini3TextGeneration:
    """
    Gemini 3 Text Generation Node

    Full Features:
    - Dynamic thinking levels (low/high)
    - Media resolution control for images, videos, PDFs
    - Multimodal inputs: text, images, video, audio, PDF
    - Thought signatures for reasoning context
    - System instructions
    - Token usage tracking
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Explain quantum computing in simple terms.",
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "video": ("VIDEO", ),
                "audio": ("AUDIO",),
                "pdf_path": ("STRING", {"default": ""}),
                "media_resolution": (["Auto", "media_resolution_low", "media_resolution_medium", "media_resolution_high"],
                                    {"default": "Auto"}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 16, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thought_signature", "usage_metadata")
    FUNCTION = "generate"
    CATEGORY = "Gemini3"

    def generate(
        self,
        prompt: str,
        api_provider: str,
        api_key: str, # This is the override from the UI
        model: str,
        thinking_level: str,
        images: Optional[torch.Tensor] = None,
        video: Optional[Any] = None,
        audio: Optional[Any] = None,
        pdf_path: str = "",
        media_resolution: str = "media_resolution_high",
        max_output_tokens: int = 8192,
        temperature: float = 1.0,
        system_instruction: str = "",
    ) -> Tuple[str, str, str]:
        """Generate text with Gemini 3"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "", "")

        try:
            # Extract video path from VIDEO object
            video_path = _extract_video_path(video)

            # Build parts list
            parts = [Part(text=prompt)]

            # Determine if we need alpha API (for media_resolution)
            use_alpha = False

            # Add images with media resolution
            if images is not None:
                use_alpha = True
                for i in range(images.shape[0]):
                    img_data = encode_image_tensor(images[i])
                    part_kwargs = {
                        "inline_data": {
                            "mimeType": "image/png",
                            "data": img_data
                        }
                    }
                    if media_resolution != "Auto":
                        part_kwargs["media_resolution"] = {"level": media_resolution}
                    parts.append(Part(**part_kwargs))

            # Add video
            if video_path and os.path.exists(video_path):
                use_alpha = True
                video_data = encode_video(video_path)
                part_kwargs = {
                    "inline_data": {
                        "mimeType": "video/mp4",
                        "data": video_data
                    }
                }
                if media_resolution != "Auto":
                    part_kwargs["media_resolution"] = {"level": media_resolution}
                parts.append(Part(**part_kwargs))

            # Add audio
            if audio is not None:
                encoded_audio = None
                mime_type = "audio/wav"  # Default to wav for tensor encoding

                # Prioritize direct tensor processing
                if isinstance(audio, dict) and 'waveform' in audio and 'sample_rate' in audio:
                    print("[Gemini 3 Debug] Audio is a tensor dict. Encoding directly.")  # DEBUG LOG
                    try:
                        encoded_audio = encode_audio_tensor(audio)
                    except Exception as e:
                        print(f"[Gemini 3 Debug] Failed to encode audio tensor: {e}")  # DEBUG LOG

                # Fallback to path-based processing
                if encoded_audio is None:
                    audio_path = _extract_media_path(audio)
                    print(f"[Gemini 3 Debug] Extracted audio path: {audio_path}")  # DEBUG LOG
                    if audio_path and os.path.exists(audio_path):
                        print(f"[Gemini 3 Debug] Audio file found. Encoding from path: {audio_path}")  # DEBUG LOG
                        try:
                            encoded_audio = encode_audio(audio_path)
                            mime_type = "audio/mp3" if audio_path.lower().endswith(".mp3") else "audio/wav"
                        except Exception as e:
                            print(f"[Gemini 3 Debug] Failed to encode audio from path: {e}")  # DEBUG LOG

                if encoded_audio:
                    parts.append(Part(inline_data={
                        "mimeType": mime_type,
                        "data": encoded_audio
                    }))

            # Add PDF
            if pdf_path and os.path.exists(pdf_path):
                use_alpha = True
                pdf_data = encode_pdf(pdf_path)
                part_kwargs = {
                    "inline_data": {
                        "mimeType": "application/pdf",
                        "data": pdf_data
                    }
                }
                if media_resolution != "Auto":
                    part_kwargs["media_resolution"] = {"level": media_resolution}
                parts.append(Part(**part_kwargs))


            # Build contents
            contents = [Content(role='user', parts=parts)]

            # Generation config
            gen_config = GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )

            # System instruction
            sys_instruction = None
            if system_instruction:
                sys_instruction = Content(
                    role='user',
                    parts=[Part(text=system_instruction)]
                )

            # Call API
            async def _generate():
                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        thinking_level=thinking_level,
                        generation_config=gen_config,
                        system_instruction=sys_instruction,
                        use_alpha_api=(media_resolution != "Auto")
                    )
                    return response

            response = run_async(_generate())

            # Extract results
            if not response.get('candidates'):
                error_msg = response.get('promptFeedback', {}).get('blockReasonMessage', 'No response')
                return (f'Error: {error_msg}', '', json.dumps(response, indent=2))

            candidate = response['candidates'][0]
            content = candidate.get('content', {})
            parts_list = content.get('parts', [])

            # Extract text
            text_parts = [p.get('text', '') for p in parts_list if 'text' in p]
            output_text = '\n'.join(text_parts) if text_parts else 'No text output'

            # Extract thought signature
            thought_sig = ''
            for p in parts_list:
                if 'thoughtSignature' in p:
                    thought_sig = p['thoughtSignature']
                    break

            # Extract usage metadata
            usage = response.get('usageMetadata', {})
            usage_info = {
                'promptTokens': usage.get('promptTokenCount', 0),
                'candidatesTokens': usage.get('candidatesTokenCount', 0),
                'totalTokens': usage.get('totalTokenCount', 0),
                'cachedTokens': usage.get('cachedContentTokenCount', 0)
            }
            usage_str = json.dumps(usage_info, indent=2)

            return (output_text, thought_sig, usage_str)

        except Exception as e:
            return (f'Error: {str(e)}', '', '')


# 导入高级节点
from .nodes_advanced import Gemini3FunctionCalling, Gemini3StructuredOutput, Gemini3Chat
from .nodes_streaming import Gemini3StreamingGeneration
from .nodes_cache_batch import Gemini3ContextCache, Gemini3UseCachedContent
from .nodes_tools import Gemini3WithGoogleSearch, Gemini3WithCodeExecution, Gemini3SafetySettings
from .nodes_multimodal import Gemini3MultiModalAnalysis, _extract_media_path
from .nodes_image_editing import Gemini3ImageEditorPrompt



# Node mappings
NODE_CLASS_MAPPINGS = {
    # 基础节点
    'Gemini3TextGeneration': Gemini3TextGeneration,
    'Gemini3MultiModalAnalysis': Gemini3MultiModalAnalysis,

    # 高级节点
    'Gemini3FunctionCalling': Gemini3FunctionCalling,
    'Gemini3StructuredOutput': Gemini3StructuredOutput,
    'Gemini3Chat': Gemini3Chat,

    # 流式节点
    'Gemini3StreamingGeneration': Gemini3StreamingGeneration,

    # 优化节点
    'Gemini3ContextCache': Gemini3ContextCache,
    'Gemini3UseCachedContent': Gemini3UseCachedContent,

    # 工具节点
    'Gemini3WithGoogleSearch': Gemini3WithGoogleSearch,
    'Gemini3WithCodeExecution': Gemini3WithCodeExecution,
    'Gemini3SafetySettings': Gemini3SafetySettings,

    # 实验性节点
    'Gemini3ImageEditorPrompt': Gemini3ImageEditorPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 基础节点
    'Gemini3TextGeneration': 'Gemini 3 文本生成',
    'Gemini3MultiModalAnalysis': 'Gemini 3 多模态分析',

    # 高级节点
    'Gemini3FunctionCalling': 'Gemini 3 函数调用',
    'Gemini3StructuredOutput': 'Gemini 3 结构化输出',
    'Gemini3Chat': 'Gemini 3 聊天',

    # 流式节点
    'Gemini3StreamingGeneration': 'Gemini 3 流式生成',

    # 优化节点
    'Gemini3ContextCache': 'Gemini 3 创建缓存',
    'Gemini3UseCachedContent': 'Gemini 3 使用缓存',

    # 工具节点
    'Gemini3WithGoogleSearch': 'Gemini 3 + Google搜索',
    'Gemini3WithCodeExecution': 'Gemini 3 + 代码执行',
    'Gemini3SafetySettings': 'Gemini 3 安全设置',

    # 实验性节点
    'Gemini3ImageEditorPrompt': 'Gemini 3 图像提示词编辑',
}
