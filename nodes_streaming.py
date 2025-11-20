"""
Gemini 3 流式输出和实时处理节点
"""

import torch
import os
import json
import asyncio
from typing import Optional, Tuple, List, Dict, Any

from .gemini_client import (
    GeminiClient, Content, Part, GenerationConfig, Tool, get_api_key
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




class Gemini3StreamingGeneration:
    """
    Gemini 3 流式生成节点

    功能：
    - 实时流式输出
    - 逐步接收响应
    - 支持所有 Gemini 3 功能
    - 捕获 thought signatures
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "写一篇关于人工智能的文章",
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
            },
            "optional": {
                "max_output_tokens": ("INT", {"default": 8192, "min": 16, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("full_text", "thought_signature", "stream_chunks")
    FUNCTION = "generate_stream"
    CATEGORY = "Gemini3/流式"

    def generate_stream(
        self,
        prompt: str,
        api_provider: str,
        api_key: str,
        model: str,
        thinking_level: str,
        max_output_tokens: int = 8192,
        temperature: float = 1.0,
        system_instruction: str = "",
    ) -> Tuple[str, str, str]:
        """流式生成内容"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "", "")

        try:
            # 构建内容
            contents = [Content(role="user", parts=[Part(text=prompt)])]

            # 系统指令
            sys_instruction = None
            if system_instruction:
                sys_instruction = Content(
                    role="user",
                    parts=[Part(text=system_instruction)]
                )

            # 生成配置
            gen_config = GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )

            # 调用流式 API
            async def _generate_stream():
                chunks = []
                full_text = ""
                thought_sig = ""

                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    async for chunk in client.generate_content_stream(
                        model=model,
                        contents=contents,
                        thinking_level=thinking_level,
                        generation_config=gen_config,
                        system_instruction=sys_instruction,
                        use_alpha_api=False
                    ):
                        chunks.append(chunk)

                        # 提取文本
                        if "candidates" in chunk:
                            for candidate in chunk["candidates"]:
                                content = candidate.get("content", {})
                                for part in content.get("parts", []):
                                    if "text" in part:
                                        full_text += part["text"]
                                    if "thoughtSignature" in part and not thought_sig:
                                        thought_sig = part["thoughtSignature"]

                return full_text, thought_sig, chunks

            full_text, thought_sig, chunks = run_async(_generate_stream())

            # 格式化 chunks 信息
            chunks_info = json.dumps({
                "total_chunks": len(chunks),
                "chunks": chunks
            }, indent=2, ensure_ascii=False)

            return (full_text, thought_sig, chunks_info)

        except Exception as e:
            return (f"错误：{str(e)}", "", "")

