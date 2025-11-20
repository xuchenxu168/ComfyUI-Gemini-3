"""
Gemini 3 API Client
High-performance client for Google Gemini 3 API with full feature support
"""

import os
import base64
import json
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import aiohttp
import ssl
import certifi
from pydantic import BaseModel, Field


class ThinkingLevel(str, Enum):
    """Thinking level controls the depth of reasoning"""
    LOW = "low"
    HIGH = "high"
    # MEDIUM coming soon - not supported at launch


class MediaResolution(str, Enum):
    """Media resolution controls token allocation per media item"""
    LOW = "media_resolution_low"
    MEDIUM = "media_resolution_medium"
    HIGH = "media_resolution_high"


class GeminiModel(str, Enum):
    """Gemini 3 model variants"""
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"


class Part(BaseModel):
    """Content part - can be text, inline data, or function call/response"""
    text: Optional[str] = None
    inline_data: Optional[Dict[str, Any]] = Field(None, alias="inlineData")
    media_resolution: Optional[Dict[str, str]] = Field(None, alias="mediaResolution")
    thought_signature: Optional[str] = Field(None, alias="thoughtSignature")
    function_call: Optional[Dict[str, Any]] = Field(None, alias="functionCall")
    function_response: Optional[Dict[str, Any]] = Field(None, alias="functionResponse")

    class Config:
        populate_by_name = True


class Content(BaseModel):
    """Message content with role and parts"""
    role: str
    parts: List[Part]


class GenerationConfig(BaseModel):
    """Configuration for content generation"""
    temperature: Optional[float] = 1.0  # Keep at 1.0 for Gemini 3
    max_output_tokens: Optional[int] = Field(None, alias="maxOutputTokens")
    top_p: Optional[float] = Field(None, alias="topP")
    top_k: Optional[int] = Field(None, alias="topK")
    response_mime_type: Optional[str] = Field(None, alias="responseMimeType")
    response_json_schema: Optional[Dict[str, Any]] = Field(None, alias="responseJsonSchema")

    class Config:
        populate_by_name = True


class SafetySetting(BaseModel):
    """Safety configuration"""
    category: str
    threshold: str


class Tool(BaseModel):
    """Tool definition for function calling or built-in tools"""
    function_declarations: Optional[List[Dict[str, Any]]] = Field(None, alias="functionDeclarations")
    google_search: Optional[Dict] = Field(None, alias="googleSearch")
    code_execution: Optional[Dict] = Field(None, alias="codeExecution")
    url_context: Optional[Dict] = Field(None, alias="urlContext")

    class Config:
        populate_by_name = True


class GeminiClient:
    """High-performance async client for Gemini 3 API"""

    API_VERSION = "v1beta"  # v1alpha for media_resolution
    TIMEOUT = 120 # seconds

    def __init__(self, api_key: str, api_provider: str):
        self.api_key = api_key
        self.api_provider = api_provider
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "https://generativelanguage.googleapis.com"  # Default

        # Load provider config
        config = {}
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"[Gemini 3] Warning: Could not read or parse config.json: {e}")

        if 'api_providers' in config and self.api_provider in config['api_providers']:
            provider_config = config['api_providers'][self.api_provider]
            self.base_url = provider_config.get('base_url', self.base_url)

    def _create_session(self) -> aiohttp.ClientSession:
        """Creates an aiohttp session with proper SSL context using certifi."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        return aiohttp.ClientSession(connector=connector)

    async def __aenter__(self):
        self.session = self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_endpoint(self, model: str, use_alpha: bool = False) -> str:
        """Get API endpoint URL"""
        version = self.API_VERSION  # Default to v1beta

        # Google is the only provider that needs v1alpha for this feature
        if self.api_provider == 'google' and use_alpha:
            version = "v1alpha"

        # If base_url already contains a version path, don't add another one.
        if self.base_url.rstrip('/').endswith(('v1', 'v1beta')):
            return f"{self.base_url.rstrip('/')}/models/{model}:generateContent"
        else:
            return f"{self.base_url.rstrip('/')}/{version}/models/{model}:generateContent"

    async def generate_content(
        self,
        model: str,
        contents: List[Content],
        thinking_level: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None,
        tools: Optional[List[Tool]] = None,
        system_instruction: Optional[Content] = None,
        use_alpha_api: bool = False,
        cached_content_name: Optional[str] = None, # Added for cache support
    ) -> Dict[str, Any]:
        """Generate content with Gemini 3"""
        if not self.session:
            self.session = self._create_session()

        url = self._get_endpoint(model, use_alpha_api)

        # Build request payload
        payload = {
            "contents": [c.model_dump(by_alias=True, exclude_none=True) for c in contents]
        }

        # Add cached content if provided
        if cached_content_name:
            payload["cachedContent"] = cached_content_name

        # Build generationConfig with thinkingConfig if needed
        gen_config_dict = {}
        if generation_config:
            gen_config_dict = generation_config.model_dump(by_alias=True, exclude_none=True)

        if thinking_level:
            # thinkingLevel goes inside generationConfig.thinkingConfig
            gen_config_dict["thinkingConfig"] = {"thinkingLevel": thinking_level}

        if gen_config_dict:
            payload["generationConfig"] = gen_config_dict

        if safety_settings:
            payload["safetySettings"] = [s.model_dump(by_alias=True) for s in safety_settings]

        if tools:
            payload["tools"] = [t.model_dump(by_alias=True, exclude_none=True) for t in tools]

        if system_instruction:
            payload["systemInstruction"] = system_instruction.model_dump(by_alias=True, exclude_none=True)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        async with self.session.post(url, json=payload, headers=headers, timeout=self.TIMEOUT) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Gemini API error {response.status}: {error_text}")
            return await response.json()

    async def generate_content_stream(
        self,
        model: str,
        contents: List[Content],
        thinking_level: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None,
        tools: Optional[List[Tool]] = None,
        system_instruction: Optional[Content] = None,
        use_alpha_api: bool = False,
    ):
        """Stream content generation with Gemini 3"""
        if not self.session:
            self.session = self._create_session()

        url = self._get_endpoint(model, use_alpha_api).replace(":generateContent", ":streamGenerateContent?alt=sse")

        # Build request payload (same as generate_content)
        payload = {
            "contents": [c.model_dump(by_alias=True, exclude_none=True) for c in contents]
        }

        # Build generationConfig with thinkingConfig if needed
        gen_config_dict = {}
        if generation_config:
            gen_config_dict = generation_config.model_dump(by_alias=True, exclude_none=True)

        if thinking_level:
            # thinkingLevel goes inside generationConfig.thinkingConfig
            gen_config_dict["thinkingConfig"] = {"thinkingLevel": thinking_level}

        if gen_config_dict:
            payload["generationConfig"] = gen_config_dict

        if safety_settings:
            payload["safetySettings"] = [s.model_dump(by_alias=True) for s in safety_settings]

        if tools:
            payload["tools"] = [t.model_dump(by_alias=True, exclude_none=True) for t in tools]

        if system_instruction:
            payload["systemInstruction"] = system_instruction.model_dump(by_alias=True, exclude_none=True)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        async with self.session.post(url, json=payload, headers=headers, timeout=self.TIMEOUT) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Gemini API error {response.status}: {error_text}")

            async for line in response.content:
                if line:
                    try:
                        # Parse SSE format
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data = json.loads(line_str[6:])
                            yield data
                    except Exception:
                        continue


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_image_tensor(image_tensor) -> str:
    """Encode image tensor to base64 PNG"""
    from PIL import Image
    import numpy as np
    import io

    # Convert tensor to numpy array (assuming [H, W, C] format with values 0-1)
    if hasattr(image_tensor, 'cpu'):
        image_np = image_tensor.cpu().numpy()
    else:
        image_np = np.array(image_tensor)

    # Convert to 0-255 range
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    # Create PIL Image
    img = Image.fromarray(image_np)

    # Encode to PNG
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_video(video_path: str) -> str:
    """Encode video to base64"""
    with open(video_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')



def encode_audio_tensor(audio_data: dict) -> str:
    """Encode audio tensor from dictionary to base64 WAV."""
    import scipy.io.wavfile
    import numpy as np
    import io

    waveform = audio_data.get('waveform')
    sample_rate = audio_data.get('sample_rate')

    if waveform is None or sample_rate is None:
        raise ValueError("Audio data dictionary must contain 'waveform' and 'sample_rate'")

    # Convert to numpy array and ensure correct format
    if hasattr(waveform, 'cpu'):
        waveform = waveform.cpu().numpy()
    waveform = np.squeeze(waveform) # Remove batch/channel dims

    # Normalize to 16-bit PCM
    if np.issubdtype(waveform.dtype, np.floating):
        waveform = (waveform * 32767).astype(np.int16)

    # Write to in-memory WAV file
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, waveform)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')

def encode_audio(audio_path: str) -> str:
    """Encode audio to base64"""
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_pdf(pdf_path: str) -> str:
    """Encode PDF to base64"""
    with open(pdf_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')




CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def get_api_key(api_provider: str, api_key_override: str) -> Optional[str]:
    """
    Gets the API key with a clear priority: UI Override > Config File > Environment Variable.
    """
    # 1. From Node Input (UI Override)
    if api_key_override and api_key_override.strip():
        return api_key_override.strip()

    # 2. From config.json
    config = {}
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"[Gemini 3] Warning: Could not read or parse config.json: {e}")

    if 'api_providers' in config and api_provider in config['api_providers']:
        provider_config = config['api_providers'][api_provider]
        api_key_from_config = provider_config.get('api_key')
        if api_key_from_config and api_key_from_config.strip():
            return api_key_from_config.strip()

    # 3. From Environment Variable (Fallback)
    env_var_map = {
        "google": "GEMINI_API_KEY",
        "comet": "COMET_API_KEY"
    }
    env_var_name = env_var_map.get(api_provider.lower())
    if env_var_name:
        api_key_from_env = os.environ.get(env_var_name)
        if api_key_from_env and api_key_from_env.strip():
            return api_key_from_env.strip()

    return None
