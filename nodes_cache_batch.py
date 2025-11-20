"""
Gemini 3 上下文缓存节点
"""

import torch
import os
import json
import asyncio
import aiohttp
from typing import Optional, Tuple, List, Dict, Any

from .gemini_client import (
    GeminiClient, Content, Part, GenerationConfig, get_api_key
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



class Gemini3ContextCache:
    """
    Gemini 3 上下文缓存节点
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_to_cache": ("STRING", {"multiline": True, "default": "这是要缓存的长文本内容..."}),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
            },
            "optional": {
                "ttl_minutes": ("INT", {"default": 60, "min": 5, "max": 1440}),
                "cache_name": ("STRING", {"default": "my_cache"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("cache_name", "cache_info", "status")
    FUNCTION = "create_cache"
    CATEGORY = "Gemini3/优化"

    def create_cache(self, content_to_cache: str, api_provider: str, api_key: str, model: str, ttl_minutes: int = 60, cache_name: str = "my_cache") -> Tuple[str, str, str]:
        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return ("", "", f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("", "", "Error: API key is required.")
        try:
            # Get base_url from config
            base_url = "https://generativelanguage.googleapis.com" # Default
            if os.path.exists(CONFIG_FILE_PATH):
                try:
                    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        if 'api_providers' in config and api_provider in config['api_providers']:
                            provider_config = config['api_providers'][api_provider]
                            base_url = provider_config.get('base_url', base_url)
                except Exception as e:
                    print(f"[Gemini 3] Warning: Could not read base_url from config: {e}")

            async def _create_cache():
                url = f"{base_url}/v1beta/cachedContents"
                payload = {
                    "model": f"models/{model}",
                    "contents": [{"role": "user", "parts": [{"text": content_to_cache}]}],
                    "ttl": f"{ttl_minutes * 60}s",
                    "displayName": cache_name
                }
                headers = {"Content-Type": "application/json", "x-goog-api-key": final_api_key}
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"缓存创建失败 {response.status}: {error_text}")
                        return await response.json()
            cache_response = run_async(_create_cache())
            cache_name_result = cache_response.get("name", "")
            cache_info = {
                "name": cache_name_result,
                "model": cache_response.get("model", ""),
                "createTime": cache_response.get("createTime", ""),
                "updateTime": cache_response.get("updateTime", ""),
                "expireTime": cache_response.get("expireTime", ""),
                "usageMetadata": cache_response.get("usageMetadata", {})
            }
            cache_info_str = json.dumps(cache_info, indent=2, ensure_ascii=False)
            return (cache_name_result, cache_info_str, "缓存创建成功")
        except Exception as e:
            return ("", "", f"错误：{str(e)}")

class Gemini3UseCachedContent:
    """
    Gemini 3 使用缓存内容节点
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "基于缓存的内容回答问题..."}),
                "cache_name": ("STRING", {"default": ""}),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "usage_metadata")
    FUNCTION = "generate_with_cache"
    CATEGORY = "Gemini3/优化"

    def generate_with_cache(self, prompt: str, cache_name: str, api_provider: str, api_key: str, model: str) -> Tuple[str, str]:
        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required.", "")
        try:
            contents = [Content(role="user", parts=[Part(text=prompt)])]
            gen_config = GenerationConfig(temperature=1.0)
            async def _generate():
                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        generation_config=gen_config,
                        cached_content_name=cache_name,
                        use_alpha_api=False
                    )
                    return response
            response = run_async(_generate())
            if not response.get("candidates"):
                return ("错误：无响应", "")
            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])
            output_text = "\n".join([p.get("text", "") for p in parts_list if "text" in p])
            usage = response.get("usageMetadata", {})
            usage_info = {
                "promptTokens": usage.get("promptTokenCount", 0),
                "cachedContentTokens": usage.get("cachedContentTokenCount", 0),
                "candidatesTokens": usage.get("candidatesTokenCount", 0),
                "totalTokens": usage.get("totalTokenCount", 0),
            }
            usage_str = json.dumps(usage_info, indent=2, ensure_ascii=False)
            return (output_text, usage_str)
        except Exception as e:
            return (f"错误：{str(e)}", "")
