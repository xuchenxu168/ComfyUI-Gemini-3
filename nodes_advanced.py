"""
Gemini 3 高级节点 - 函数调用、结构化输出、聊天等
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




class Gemini3FunctionCalling:
    """
    Gemini 3 函数调用节点

    功能：
    - 严格验证的函数调用
    - 支持并行和顺序函数调用
    - Thought signatures 保持推理上下文
    - 多步骤函数执行
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "查询巴黎和伦敦的天气",
                }),
                "function_declarations": ("STRING", {
                    "multiline": True,
                    "default": json.dumps([{
                        "name": "get_weather",
                        "description": "获取城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "城市名称"}
                            },
                            "required": ["city"]
                        }
                    }], indent=2, ensure_ascii=False)
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
            },
            "optional": {
                "thought_signature": ("STRING", {"default": ""}),
                "function_responses": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("function_calls", "thought_signature", "text_response")
    FUNCTION = "call_functions"
    CATEGORY = "Gemini3/高级"

    def call_functions(
        self,
        prompt: str,
        function_declarations: str,
        api_provider: str,
        api_key: str,
        model: str,
        thinking_level: str,
        thought_signature: str = "",
        function_responses: str = "",
    ) -> Tuple[str, str, str]:
        """执行函数调用"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "", "")

        try:
            # 解析函数声明
            func_decls = json.loads(function_declarations)

            # 构建内容
            parts = [Part(text=prompt)]
            contents = [Content(role="user", parts=parts)]

            # 添加之前的函数响应
            if function_responses:
                try:
                    responses = json.loads(function_responses)
                    if thought_signature:
                        # 添加模型的函数调用（带 thought signature）
                        func_call_parts = []
                        for idx, resp in enumerate(responses):
                            func_call_parts.append(Part(
                                function_call=resp.get("function_call"),
                                thought_signature=thought_signature if idx == 0 else None
                            ))
                        contents.append(Content(role="model", parts=func_call_parts))

                        # 添加用户的函数响应
                        func_resp_parts = [Part(function_response=resp) for resp in responses]
                        contents.append(Content(role="user", parts=func_resp_parts))
                except json.JSONDecodeError:
                    pass

            # 工具配置
            tools = [Tool(function_declarations=func_decls)]

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

            # 提取函数调用
            function_calls = []
            new_thought_sig = ""
            text_response = ""

            for p in parts_list:
                if "functionCall" in p:
                    function_calls.append(p["functionCall"])
                    if "thoughtSignature" in p and not new_thought_sig:
                        new_thought_sig = p["thoughtSignature"]
                if "text" in p:
                    text_response += p["text"]

            func_calls_str = json.dumps(function_calls, indent=2, ensure_ascii=False)

            return (func_calls_str, new_thought_sig, text_response)

        except Exception as e:
            return (f"错误：{str(e)}", "", "")


class Gemini3StructuredOutput:
    """
    Gemini 3 结构化输出节点

    功能：
    - JSON schema 验证
    - 与工具集成（搜索、代码执行、URL上下文）
    - 严格的类型检查
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "搜索最新的欧洲杯比赛结果",
                }),
                "json_schema": ("STRING", {
                    "multiline": True,
                    "default": json.dumps({
                        "type": "object",
                        "properties": {
                            "winner": {"type": "string", "description": "获胜者名称"},
                            "final_score": {"type": "string", "description": "最终比分"},
                            "scorers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "进球者名单"
                            }
                        },
                        "required": ["winner", "final_score", "scorers"]
                    }, indent=2, ensure_ascii=False)
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
            },
            "optional": {
                "enable_google_search": ("BOOLEAN", {"default": False}),
                "enable_code_execution": ("BOOLEAN", {"default": False}),
                "enable_url_context": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json_output", "usage_metadata")
    FUNCTION = "generate_structured"
    CATEGORY = "Gemini3/高级"

    def generate_structured(
        self,
        prompt: str,
        json_schema: str,
        api_provider: str,
        api_key: str,
        model: str,
        thinking_level: str,
        enable_google_search: bool = False,
        enable_code_execution: bool = False,
        enable_url_context: bool = False,
    ) -> Tuple[str, str]:
        """生成结构化JSON输出"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", "")

        try:
            # 解析 JSON schema
            schema = json.loads(json_schema)

            # 构建内容
            contents = [Content(role="user", parts=[Part(text=prompt)])]

            # 构建工具列表
            tools = []
            if enable_google_search:
                tools.append(Tool(google_search={}))
            if enable_code_execution:
                tools.append(Tool(code_execution={}))
            if enable_url_context:
                tools.append(Tool(url_context={}))

            # 生成配置
            gen_config = GenerationConfig(
                temperature=1.0,
                response_mime_type="application/json",
                response_json_schema=schema
            )

            # 调用 API
            async def _generate():
                async with GeminiClient(api_key=final_api_key, api_provider=api_provider) as client:
                    response = await client.generate_content(
                        model=model,
                        contents=contents,
                        thinking_level=thinking_level,
                        generation_config=gen_config,
                        tools=tools if tools else None,
                        use_alpha_api=False
                    )
                    return response

            response = run_async(_generate())

            # 提取结果
            if not response.get("candidates"):
                return ("错误：无响应", "")

            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])

            # 提取文本（应该是JSON）
            text_parts = [p.get("text", "") for p in parts_list if "text" in p]
            json_output = "".join(text_parts)

            # 验证JSON
            try:
                json.loads(json_output)
            except json.JSONDecodeError:
                return (f"错误：输出不是有效的JSON\n{json_output}", "")

            # 提取使用元数据
            usage = response.get("usageMetadata", {})
            usage_info = {
                "promptTokens": usage.get("promptTokenCount", 0),
                "candidatesTokens": usage.get("candidatesTokenCount", 0),
                "totalTokens": usage.get("totalTokenCount", 0),
            }
            usage_str = json.dumps(usage_info, indent=2, ensure_ascii=False)

            return (json_output, usage_str)

        except Exception as e:
            return (f"错误：{str(e)}", "")



class Gemini3Chat:
    """
    Gemini 3 聊天节点

    功能：
    - 多轮对话
    - 自动管理聊天历史
    - Thought signatures 保持推理上下文
    - 支持系统指令
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {
                    "multiline": True,
                    "default": "你好，请介绍一下你自己",
                }),
                "api_provider": (API_PROVIDERS, {"default": DEFAULT_PROVIDER}),
                "api_key": ("STRING", {"multiline": False, "placeholder": "Override API Key (Optional)"}),
                "model": (ALL_MODELS, {"default": "gemini-3-pro-preview" if "gemini-3-pro-preview" in ALL_MODELS else ALL_MODELS[0]}),
                "thinking_level": (["high", "low"], {"default": "high"}),
            },
            "optional": {
                "chat_history": ("STRING", {"multiline": True, "default": ""}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 16, "max": 8192}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "updated_history", "usage_metadata")
    FUNCTION = "chat"
    CATEGORY = "Gemini3/高级"

    def chat(
        self,
        message: str,
        api_provider: str,
        api_key: str,
        model: str,
        thinking_level: str,
        chat_history: str = "",
        system_instruction: str = "",
        max_output_tokens: int = 8192,
    ) -> Tuple[str, str, str]:
        """进行聊天对话"""

        # Validate model for the selected provider
        if api_provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[api_provider]:
            return (f"Error: Model '{model}' is not supported by provider '{api_provider}'. Please select a different model or provider.", chat_history, "")

        final_api_key = get_api_key(api_provider, api_key)
        if not final_api_key:
            return ("Error: API key is required. Please provide it in the node, in config.json, or as an environment variable.", chat_history, "")

        try:
            # 解析聊天历史
            contents = []
            if chat_history:
                try:
                    history = json.loads(chat_history)
                    for msg in history:
                        parts = []
                        for p in msg.get("parts", []):
                            part = Part(**p)
                            parts.append(part)
                        contents.append(Content(role=msg["role"], parts=parts))
                except json.JSONDecodeError:
                    pass

            # 添加新消息
            contents.append(Content(role="user", parts=[Part(text=message)]))

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
                        system_instruction=sys_instruction,
                        use_alpha_api=False
                    )
                    return response

            response = run_async(_generate())

            # 提取结果
            if not response.get("candidates"):
                return ("错误：无响应", chat_history, "")

            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])

            # 提取响应文本
            text_parts = [p.get("text", "") for p in parts_list if "text" in p]
            response_text = "\n".join(text_parts)

            # 更新聊天历史
            history_list = []
            if chat_history:
                try:
                    history_list = json.loads(chat_history)
                except json.JSONDecodeError:
                    pass

            # 添加用户消息
            history_list.append({
                "role": "user",
                "parts": [{"text": message}]
            })

            # 添加模型响应（保留 thought signature）
            model_parts = []
            for p in parts_list:
                part_dict = {}
                if "text" in p:
                    part_dict["text"] = p["text"]
                if "thoughtSignature" in p:
                    part_dict["thoughtSignature"] = p["thoughtSignature"]
                if part_dict:
                    model_parts.append(part_dict)

            history_list.append({
                "role": "model",
                "parts": model_parts
            })

            updated_history = json.dumps(history_list, indent=2, ensure_ascii=False)

            # 提取使用元数据
            usage = response.get("usageMetadata", {})
            usage_info = {
                "promptTokens": usage.get("promptTokenCount", 0),
                "candidatesTokens": usage.get("candidatesTokenCount", 0),
                "totalTokens": usage.get("totalTokenCount", 0),
            }
            usage_str = json.dumps(usage_info, indent=2, ensure_ascii=False)

            return (response_text, updated_history, usage_str)

        except Exception as e:
            return (f"错误：{str(e)}", chat_history, "")
