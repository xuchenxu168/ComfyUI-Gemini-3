"""
Gemini 3 工具集成节点 - Google搜索、代码执行、URL上下文等
"""

import torch
import os
import json
import asyncio
from typing import Optional, Tuple, List, Dict, Any

from .gemini_client import (
    GeminiClient, Content, Part, GenerationConfig, Tool, SafetySetting, get_api_key
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




class Gemini3WithGoogleSearch:
    """
    Gemini 3 + Google 搜索节点

    功能：
    - 实时网络搜索
    - 自动grounding
    - 获取最新信息
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "搜索最新的AI新闻",
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
            },
            "optional": {
                "max_output_tokens": ("INT", {"default": 8192, "min": 16, "max": 8192}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "grounding_metadata", "usage_metadata")
    FUNCTION = "search_and_generate"
    CATEGORY = "Gemini3/工具"

    def search_and_generate(
        self,
        prompt: str,
        api_provider: str,
        api_key: str,
        model: str,
        thinking_level: str,
        max_output_tokens: int = 8192,
    ) -> Tuple[str, str, str]:
        """使用Google搜索生成内容"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "", "")

        try:
            # 构建内容
            contents = [Content(role="user", parts=[Part(text=prompt)])]

            # 启用Google搜索工具
            tools = [Tool(google_search={})]

            # 生成配置
            gen_config = GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=1.0
            )

            # 调用 API
            async def _generate():
                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        thinking_level=thinking_level,
                        generation_config=gen_config,
                        tools=tools,
                        use_alpha_api=False
                    )
                    return response

            response = run_async(_generate())

            # 提取结果
            if not response.get("candidates"):
                return ("错误：无响应", "", "")

            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])

            # 提取文本
            text_parts = [p.get("text", "") for p in parts_list if "text" in p]
            output_text = "\n".join(text_parts)

            # 提取grounding元数据
            grounding_metadata = candidate.get("groundingMetadata", {})
            grounding_str = json.dumps(grounding_metadata, indent=2, ensure_ascii=False)

            # 提取使用元数据
            usage = response.get("usageMetadata", {})
            usage_info = {
                "promptTokens": usage.get("promptTokenCount", 0),
                "candidatesTokens": usage.get("candidatesTokenCount", 0),
                "totalTokens": usage.get("totalTokenCount", 0),
            }
            usage_str = json.dumps(usage_info, indent=2, ensure_ascii=False)

            return (output_text, grounding_str, usage_str)

        except Exception as e:
            return (f"错误：{str(e)}", "", "")


class Gemini3WithCodeExecution:
    """
    Gemini 3 + 代码执行节点

    功能：
    - 自动执行Python代码
    - 数学计算
    - 数据分析
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "计算斐波那契数列的前20项",
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "execution_result", "usage_metadata")
    FUNCTION = "execute_code"
    CATEGORY = "Gemini3/工具"

    def execute_code(
        self,
        prompt: str,
        api_provider: str,
        api_key: str,
        model: str,
        thinking_level: str,
    ) -> Tuple[str, str, str]:
        """使用代码执行生成内容"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "", "")

        try:
            # 构建内容
            contents = [Content(role="user", parts=[Part(text=prompt)])]

            # 启用代码执行工具
            tools = [Tool(code_execution={})]

            # 生成配置
            gen_config = GenerationConfig(
                temperature=1.0
            )

            # 调用 API
            async def _generate():
                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        thinking_level=thinking_level,
                        generation_config=gen_config,
                        tools=tools,
                        use_alpha_api=False
                    )
                    return response

            response = run_async(_generate())

            # 提取结果
            if not response.get("candidates"):
                return ("错误：无响应", "", "")

            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])

            # 提取文本和执行结果
            text_parts = []
            execution_results = []

            for p in parts_list:
                if "text" in p:
                    text_parts.append(p["text"])
                if "executableCode" in p:
                    execution_results.append({
                        "code": p.get("executableCode", {}),
                        "result": p.get("codeExecutionResult", {})
                    })

            output_text = "\n".join(text_parts)
            execution_str = json.dumps(execution_results, indent=2, ensure_ascii=False)

            # 提取使用元数据
            usage = response.get("usageMetadata", {})
            usage_info = {
                "promptTokens": usage.get("promptTokenCount", 0),
                "candidatesTokens": usage.get("candidatesTokenCount", 0),
                "totalTokens": usage.get("totalTokenCount", 0),
            }
            usage_str = json.dumps(usage_info, indent=2, ensure_ascii=False)

            return (output_text, execution_str, usage_str)

        except Exception as e:
            return (f"错误：{str(e)}", "", "")


class Gemini3SafetySettings:
    """
    Gemini 3 安全设置节点

    功能：
    - 配置内容安全阈值
    - 控制有害内容过滤
    - 自定义安全级别
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "生成内容...",
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),

                # 安全设置
                "harassment_threshold": ([
                    "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
                ], {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
                "hate_speech_threshold": ([
                    "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
                ], {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
                "sexually_explicit_threshold": ([
                    "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
                ], {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
                "dangerous_content_threshold": ([
                    "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
                ], {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "safety_ratings")
    FUNCTION = "generate_with_safety"
    CATEGORY = "Gemini3/安全"

    def generate_with_safety(
        self,
        prompt: str,
        api_provider: str,
        api_key: str,
        model: str,
        thinking_level: str,
        harassment_threshold: str,
        hate_speech_threshold: str,
        sexually_explicit_threshold: str,
        dangerous_content_threshold: str,
    ) -> Tuple[str, str]:
        """使用安全设置生成内容"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "")

        try:
            # 构建内容
            contents = [Content(role="user", parts=[Part(text=prompt)])]

            # 安全设置
            safety_settings = [
                SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=harassment_threshold),
                SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=hate_speech_threshold),
                SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=sexually_explicit_threshold),
                SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=dangerous_content_threshold),
            ]

            # 生成配置
            gen_config = GenerationConfig(
                temperature=1.0
            )

            # 调用 API
            async def _generate():
                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        thinking_level=thinking_level,
                        generation_config=gen_config,
                        safety_settings=safety_settings,
                        use_alpha_api=False
                    )
                    return response

            response = run_async(_generate())

            # 提取结果
            if not response.get("candidates"):
                # 检查是否被安全过滤阻止
                prompt_feedback = response.get("promptFeedback", {})
                block_reason = prompt_feedback.get("blockReason", "UNKNOWN")
                safety_ratings = prompt_feedback.get("safetyRatings", [])

                return (
                    f"内容被阻止：{block_reason}",
                    json.dumps(safety_ratings, indent=2, ensure_ascii=False)
                )

            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])

            # 提取文本
            text_parts = [p.get("text", "") for p in parts_list if "text" in p]
            output_text = "\n".join(text_parts)

            # 提取安全评级
            safety_ratings = candidate.get("safetyRatings", [])
            safety_str = json.dumps(safety_ratings, indent=2, ensure_ascii=False)

            return (output_text, safety_str)

        except Exception as e:
            return (f"错误：{str(e)}", "")

