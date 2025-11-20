"""
Gemini 3 Image Editing Node
"""

import torch
import os
import json
import asyncio
from typing import Optional, Tuple, List, Dict, Any

from .gemini_client import (
    GeminiClient, Content, Part, GenerationConfig, get_api_key, encode_image_tensor
)

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



class Gemini3ImageEditorPrompt:
    """
    Uses Gemini to generate a new image prompt based on an input image and an editing instruction.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "将天空变成紫色。",
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("new_prompt", "analysis_text")
    FUNCTION = "edit_image_prompt"
    CATEGORY = "Gemini3/实验性"

    def edit_image_prompt(self, image: torch.Tensor, prompt: str, api_provider: str, api_key: str, model: str) -> Tuple[str, str]:
        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'.", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required.", "")

        async def _execute():
            try:
                system_prompt = (
                    "You are a creative and experienced prompt engineer for a powerful text-to-image AI model. "
                    "Your task is to analyze a user-provided image and an editing instruction. "
                    "First, describe the original image in detail, covering its subject, composition, style, lighting, and any other relevant visual elements. "
                    "Then, explain how you will incorporate the user's editing instruction to create a new prompt. "
                    "Finally, generate a single, highly detailed, and coherent text prompt that the text-to-image model can use to generate the modified image. "
                    "Separate your analysis from the final prompt with the exact line '---PROMPT---'."
                )

                image_part = Part(inline_data={"mimeType": "image/png", "data": encode_image_tensor(image[0])})
                text_part = Part(text=f"Editing Instruction: {prompt}")
                contents = [Content(role="user", parts=[image_part, text_part])]
                sys_instruction = Content(role='user', parts=[Part(text=system_prompt)])

                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        system_instruction=sys_instruction,
                        generation_config=GenerationConfig(temperature=0.8)
                    )

                if not response.get('candidates'):
                    error_msg = response.get('promptFeedback', {}).get('blockReasonMessage', 'No response from API.')
                    return (f"Error: {error_msg}", "")

                full_response_text = response['candidates'][0]['content']['parts'][0]['text']

                # Parse the response
                if '---PROMPT---' in full_response_text:
                    parts = full_response_text.split('---PROMPT---', 1)
                    analysis_text = parts[0].strip()
                    new_prompt = parts[1].strip()
                else:
                    analysis_text = "(Warning: Separator '---PROMPT---' not found in response. Returning full text as prompt.)"
                    new_prompt = full_response_text.strip()

                return (new_prompt, analysis_text)

            except Exception as e:
                return (f"Error: {str(e)}", "")

        # The run_async helper executes the async function and returns the result.
        return run_async(_execute())

