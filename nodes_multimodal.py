"""
ComfyUI-Gemini-3-2: Multimodal Analysis Node
Author: Ken

This file implements the unified multimodal analysis node for Gemini 3.
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
    GeminiClient, Content, Part, GenerationConfig,
    encode_image_tensor, encode_video, encode_audio, get_api_key, encode_audio_tensor
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

def _extract_media_path(media_input: Any) -> str:
    """Extracts the media file path from various ComfyUI loader node types."""
    if not media_input:
        return ""

    if isinstance(media_input, str):
        return media_input

    # Common attributes for file paths from different loader nodes
    path_attributes = ['file_path', 'filepath', 'saved_path', 'filename', 'file', 'path']
    for attr in path_attributes:
        if hasattr(media_input, attr):
            path = getattr(media_input, attr)
            if isinstance(path, str):
                return path

    # Fallback for list-based outputs (e.g., some video loaders)
    if isinstance(media_input, (list, tuple)) and media_input:
        return str(media_input[0])

    return str(media_input)



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

class Gemini3MultiModalAnalysis:
    """
    Gemini 3 统一多模态分析节点

    功能:
    - 支持最多4张图片、1个视频、1个音频的混合输入
    - 根据输入组合进行单模态分析、多图对比或混合媒体分析
    - 所有媒体输入均为可选，但至少需要一个
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "请对所有提供的媒体进行详细的综合分析。\n1. 如果有图片，请描述图片内容。如果有多张图片，请详细对比它们的异同之处。\n2. 如果有视频，请描述视频中的主要内容和场景变化。\n3. 如果有音频，请描述音频中的主要内容（例如语音转录、声音事件等）。\n4. 总结所有媒体表达的共同主题或情感（如果有的话）。"
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
                "media_resolution": (["Auto", "media_resolution_low", "media_resolution_medium", "media_resolution_high"], {"default": "Auto"}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "image_3": ("IMAGE", ),
                "image_4": ("IMAGE", ),
                "video": ("VIDEO", ),
                "audio": ("AUDIO", ),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "thought_signature", "usage_info")
    FUNCTION = "analyze_multimodal"
    CATEGORY = "Gemini3/分析"

    def analyze_multimodal(
        self,
        prompt: str,
        api_provider: str,
        api_key: str, # This is the override from the UI
        model: str,
        thinking_level: str,
        media_resolution: str,
        max_output_tokens: int,
        temperature: float,
        image_1: Optional[torch.Tensor] = None,
        image_2: Optional[torch.Tensor] = None,
        image_3: Optional[torch.Tensor] = None,
        image_4: Optional[torch.Tensor] = None,
        video: Optional[Any] = None,
        audio: Optional[Any] = None,
        system_instruction: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Performs multimodal analysis based on the provided inputs."""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "", "")

        parts = [Part(text=prompt)]
        media_provided = False

        # Image processing
        images = [image_1, image_2, image_3, image_4]
        for img_tensor in images:
            if img_tensor is not None:
                media_provided = True
                try:
                    # Ensure we are working with a single image from the batch
                    if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
                        img_tensor = img_tensor.squeeze(0)

                    encoded_data = encode_image_tensor(img_tensor)
                    part_kwargs = {
                        "inline_data": {
                            "mimeType": "image/png",
                            "data": encoded_data
                        }
                    }
                    if media_resolution != "Auto":
                        part_kwargs["media_resolution"] = {"level": media_resolution}
                    parts.append(Part(**part_kwargs))
                except Exception as e:
                    return (f"Error encoding image: {e}", "", "")

        # Video processing
        if video is not None:
            media_provided = True
            video_path = _extract_media_path(video)
            if video_path and os.path.exists(video_path):
                try:
                    encoded_video = encode_video(video_path)
                    part_kwargs = {
                        "inline_data": {
                            "mimeType": "video/mp4",
                            "data": encoded_video
                        }
                    }
                    if media_resolution != "Auto":
                        part_kwargs["media_resolution"] = {"level": media_resolution}
                    parts.append(Part(**part_kwargs))
                except Exception as e:
                    return (f"Error encoding video: {e}", "", "")
            else:
                return (f"Error: Video path '{video_path}' not found or could not be extracted.", "", "")

        # Audio processing
        if audio is not None:
            media_provided = True
            encoded_audio = None
            mime_type = "audio/wav"  # Default to wav for tensor encoding

            # Prioritize direct tensor processing
            if isinstance(audio, dict) and 'waveform' in audio and 'sample_rate' in audio:
                try:
                    encoded_audio = encode_audio_tensor(audio)
                except Exception as e:
                    return (f"Error encoding audio tensor: {e}", "", "")

            # Fallback to path-based processing
            if encoded_audio is None:
                audio_path = _extract_media_path(audio)
                if audio_path and os.path.exists(audio_path):
                    try:
                        encoded_audio = encode_audio(audio_path)
                        # Determine MIME type based on file extension
                        if audio_path.lower().endswith(".mp3"):
                            mime_type = "audio/mp3"
                        elif audio_path.lower().endswith(".wav"):
                            mime_type = "audio/wav"
                        elif audio_path.lower().endswith(".aiff"):
                            mime_type = "audio/aiff"
                        elif audio_path.lower().endswith(".aac"):
                            mime_type = "audio/aac"
                        elif audio_path.lower().endswith(".ogg"):
                            mime_type = "audio/ogg"
                        elif audio_path.lower().endswith(".flac"):
                            mime_type = "audio/flac"
                        else:
                            mime_type = "audio/wav"  # Default fallback
                    except Exception as e:
                        return (f"Error encoding audio from path: {e}", "", "")

            if encoded_audio:
                parts.append(Part(inline_data={
                    "mimeType": mime_type,
                    "data": encoded_audio
                }))
            elif audio is not None: # If audio was provided but failed to encode
                return ("Error: Failed to process the provided audio input. Check debug logs.", "", "")

        if not media_provided:
            return ("Error: At least one media input (image, video, or audio) is required.", "", "")

        contents = [Content(role="user", parts=parts)]
        gen_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

        system_content = Content(parts=[Part(text=system_instruction)]) if system_instruction and system_instruction.strip() else None

        try:
            async def _generate():
                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        thinking_level=thinking_level,
                        generation_config=gen_config,
                        system_instruction=system_content,
                        use_alpha_api=(media_resolution != "Auto")
                    )
                    return response

            response = run_async(_generate())

            if not response.get("candidates"):
                return (f"API Error: {response.get('error', {}).get('message', 'No candidates returned.')}", "", "")

            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])

            output_text = "".join(p.get("text", "") for p in parts_list if "text" in p)
            thought_sig = "".join(p.get("thoughtSignature", "") for p in parts_list if "thoughtSignature" in p)

            usage = response.get('usageMetadata', {})
            usage_info = json.dumps(usage, indent=2)

            return (output_text, thought_sig, usage_info)

        except Exception as e:
            return (f'Error: {str(e)}', '', '')

