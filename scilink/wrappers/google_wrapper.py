"""
GenAI SDK Wrapper for Backward Compatibility

This module wraps the new `google-genai` SDK to provide the same interface
as the legacy `google.generativeai` SDK. This allows existing code that uses
the old SDK patterns to work without modification.

Usage:
    # Instead of:
    # import google.generativeai as genai
    # genai.configure(api_key=...)
    # model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Use:
    from wrappers.genai_wrapper import GenAIClient
    client = GenAIClient(api_key=...)
    model = client.GenerativeModel('gemini-2.0-flash')
    
    # Or directly:
    from wrappers.genai_wrapper import GenAIAsLegacyGenerativeModel
    model = GenAIAsLegacyGenerativeModel('gemini-2.0-flash', api_key=...)
    
    # Then use exactly as before:
    response = model.generate_content("Hello!")
    print(response.text)
"""

import io
import base64
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union, Iterator

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


class LegacyPart:
    """
    Mimics the structure of a Part object from the legacy SDK.
    Can contain text, function_call, or other content types.
    """
    def __init__(self, text: str = None, function_call: Any = None, 
                 inline_data: Any = None, executable_code: Any = None,
                 code_execution_result: Any = None):
        self.text = text
        self.function_call = function_call
        self.inline_data = inline_data
        self.executable_code = executable_code
        self.code_execution_result = code_execution_result


class LegacyContent:
    """
    Mimics the structure of a Content object from the legacy SDK.
    Contains a list of parts and a role.
    """
    def __init__(self, parts: List[LegacyPart], role: str = "model"):
        self.parts = parts
        self.role = role


class LegacyCandidate:
    """
    Mimics the structure of a Candidate object from the legacy SDK.
    """
    def __init__(self, content: LegacyContent, finish_reason: int, 
                 safety_ratings: List = None, index: int = 0):
        self.content = content
        self.finish_reason = finish_reason
        self.safety_ratings = safety_ratings or []
        self.index = index


class LegacyGenerateContentResponse:
    """
    Mimics the structure of a GenerateContentResponse from the legacy SDK.
    
    Provides:
    - response.text: The concatenated text from all parts
    - response.candidates: List of candidate responses
    - response.parts: Direct access to parts (convenience)
    - response.prompt_feedback: Feedback about the prompt
    """
    def __init__(self, new_sdk_response, is_stream_chunk: bool = False):
        self._raw_response = new_sdk_response
        self._is_stream_chunk = is_stream_chunk
        self.candidates = self._convert_candidates(new_sdk_response)
        self.prompt_feedback = self._extract_prompt_feedback(new_sdk_response)
        self.usage_metadata = self._extract_usage_metadata(new_sdk_response)
    
    def _convert_candidates(self, response) -> List[LegacyCandidate]:
        """Convert new SDK candidates to legacy format."""
        candidates = []
        
        # Handle the response structure from new SDK
        raw_candidates = getattr(response, 'candidates', None) or []
        
        for idx, candidate in enumerate(raw_candidates):
            parts = []
            content = getattr(candidate, 'content', None)
            
            if content:
                raw_parts = getattr(content, 'parts', None) or []
                for part in raw_parts:
                    legacy_part = self._convert_part(part)
                    if legacy_part:
                        parts.append(legacy_part)
            
            # Convert finish reason
            finish_reason = self._convert_finish_reason(
                getattr(candidate, 'finish_reason', None)
            )
            
            # Get safety ratings
            safety_ratings = getattr(candidate, 'safety_ratings', [])
            
            role = getattr(content, 'role', 'model') if content else 'model'
            legacy_content = LegacyContent(parts=parts, role=role)
            
            candidates.append(LegacyCandidate(
                content=legacy_content,
                finish_reason=finish_reason,
                safety_ratings=safety_ratings,
                index=idx
            ))
        
        return candidates
    
    def _convert_part(self, part) -> Optional[LegacyPart]:
        """Convert a new SDK part to legacy format."""
        # Handle different part types
        
        # Text part
        text = getattr(part, 'text', None)
        if text is not None:
            return LegacyPart(text=text)
        
        # Function call part
        function_call = getattr(part, 'function_call', None)
        if function_call:
            # Convert to legacy function call format
            fc = SimpleNamespace(
                name=getattr(function_call, 'name', ''),
                args=dict(getattr(function_call, 'args', {}))
            )
            return LegacyPart(function_call=fc)
        
        # Inline data (images, etc.)
        inline_data = getattr(part, 'inline_data', None)
        if inline_data:
            return LegacyPart(inline_data=inline_data)
        
        # Executable code
        executable_code = getattr(part, 'executable_code', None)
        if executable_code:
            return LegacyPart(executable_code=executable_code)
        
        # Code execution result
        code_result = getattr(part, 'code_execution_result', None)
        if code_result:
            return LegacyPart(code_execution_result=code_result)
        
        return None
    
    def _convert_finish_reason(self, reason) -> int:
        """
        Convert new SDK finish reason to legacy integer format.
        
        Legacy mapping:
        0 = FINISH_REASON_UNSPECIFIED
        1 = STOP (normal completion)
        2 = MAX_TOKENS
        3 = SAFETY
        4 = RECITATION
        5 = OTHER
        """
        if reason is None:
            return 0
        
        # Handle string or enum
        reason_str = str(reason).upper()
        
        mapping = {
            'STOP': 1,
            'FINISH_REASON_STOP': 1,
            'MAX_TOKENS': 2,
            'FINISH_REASON_MAX_TOKENS': 2,
            'SAFETY': 3,
            'FINISH_REASON_SAFETY': 3,
            'RECITATION': 4,
            'FINISH_REASON_RECITATION': 4,
            'OTHER': 5,
            'FINISH_REASON_OTHER': 5,
            'TOOL_CALL': 1,  # Treat tool calls as normal stop
        }
        
        for key, value in mapping.items():
            if key in reason_str:
                return value
        
        return 0
    
    def _extract_prompt_feedback(self, response) -> Optional[Any]:
        """Extract prompt feedback from response."""
        return getattr(response, 'prompt_feedback', None)
    
    def _extract_usage_metadata(self, response) -> Optional[Any]:
        """Extract usage metadata from response."""
        return getattr(response, 'usage_metadata', None)
    
    @property
    def text(self) -> str:
        """
        Get the concatenated text from all parts of the first candidate.
        This mimics the legacy SDK's response.text property.
        """
        if not self.candidates:
            return ""
        
        texts = []
        for part in self.candidates[0].content.parts:
            if part.text:
                texts.append(part.text)
        
        return "".join(texts)
    
    @property
    def parts(self) -> List[LegacyPart]:
        """
        Direct access to parts of the first candidate.
        Convenience property matching legacy SDK behavior.
        """
        if not self.candidates:
            return []
        return self.candidates[0].content.parts


class LegacyChatSession:
    """
    Mimics the ChatSession from the legacy SDK.
    
    Wraps the new SDK's chat functionality to provide the same interface.
    """
    def __init__(self, client: 'genai.Client', model_name: str, 
                 history: List = None, generation_config: Any = None,
                 safety_settings: Any = None, tools: Any = None,
                 enable_automatic_function_calling: bool = False,
                 system_instruction: Any = None):
        self._client = client
        self._model_name = model_name
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._enable_afc = enable_automatic_function_calling
        self._system_instruction = system_instruction
        
        # Convert legacy history format to new format if needed
        self._history = self._convert_history(history) if history else []
        
        # Create the underlying chat session
        self._chat = None
        self._init_chat()
    
    def _init_chat(self):
        """Initialize the underlying new SDK chat."""
        try:
            config = {}
            if self._history:
                config['history'] = self._history
            
            # NOTE: In the new SDK, tools are NOT passed at chat creation time.
            # They are passed with each send_message() call instead.
            # We store them in self._tools and use them in send_message().
            
            self._chat = self._client.chats.create(
                model=self._model_name,
                config=config if config else None
            )
            logging.debug(f"Chat session created for model: {self._model_name}")
            if self._tools:
                logging.debug(f"Tools will be passed with each message ({len(self._tools) if isinstance(self._tools, list) else 1} tools)")
        except Exception as e:
            logging.warning(f"Failed to create chat session: {e}")
            self._chat = None
    
    def _convert_history(self, legacy_history: List) -> List:
        """Convert legacy history format to new SDK format."""
        new_history = []
        
        for entry in legacy_history:
            if isinstance(entry, dict):
                role = entry.get('role', 'user')
                parts = entry.get('parts', [])
            else:
                role = getattr(entry, 'role', 'user')
                parts = getattr(entry, 'parts', [])
            
            # Convert parts
            new_parts = []
            for part in parts:
                if isinstance(part, str):
                    new_parts.append({'text': part})
                elif isinstance(part, dict):
                    new_parts.append(part)
                elif hasattr(part, 'text'):
                    new_parts.append({'text': part.text})
                else:
                    new_parts.append({'text': str(part)})
            
            new_history.append({
                'role': role,
                'parts': new_parts
            })
        
        return new_history
    
    @property
    def history(self) -> List:
        """Return the conversation history."""
        if self._chat and hasattr(self._chat, 'history'):
            return self._chat.history
        return self._history
    
    def send_message(self, content: Union[str, List], 
                     generation_config: Any = None,
                     safety_settings: Any = None,
                     stream: bool = False) -> LegacyGenerateContentResponse:
        """
        Send a message in the chat session.
        
        Args:
            content: The message to send (string or list of parts)
            generation_config: Optional generation configuration
            safety_settings: Optional safety settings
            stream: If True, return a streaming response
            
        Returns:
            LegacyGenerateContentResponse mimicking the legacy SDK response
        """
        if self._chat is None:
            raise RuntimeError("Chat session not initialized")
        
        # Convert content format
        if isinstance(content, str):
            message = content
        elif isinstance(content, list):
            # Handle list of parts
            message = self._convert_parts_to_new_format(content)
        else:
            message = str(content)
        
        try:
            if stream:
                return self._send_message_stream(message)
            else:
                # Build GenerateContentConfig if we have tools, generation config, or system instruction
                config = self._build_send_message_config(generation_config, safety_settings)
                
                if config:
                    response = self._chat.send_message(message=message, config=config)
                else:
                    response = self._chat.send_message(message=message)
                return LegacyGenerateContentResponse(response)
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            raise
    
    def _build_send_message_config(self, generation_config: Any = None, 
                                    safety_settings: Any = None) -> Any:
        """Build a GenerateContentConfig for send_message if needed."""
        # Check if we need a config at all
        has_tools = bool(self._tools)
        has_gen_config = bool(generation_config or self._generation_config)
        has_safety = bool(safety_settings or self._safety_settings)
        has_system = bool(self._system_instruction)
        
        if not any([has_tools, has_gen_config, has_safety, has_system]):
            return None
        
        config_kwargs = {}
        
        # Add tools
        if self._tools:
            config_kwargs['tools'] = self._tools
        
        # Add system instruction
        if self._system_instruction:
            config_kwargs['system_instruction'] = self._system_instruction
        
        # Add generation config parameters
        gen_cfg = generation_config or self._generation_config
        if gen_cfg:
            if isinstance(gen_cfg, dict):
                for key in ['temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'response_mime_type', 'response_schema', 'stop_sequences']:
                    if key in gen_cfg:
                        config_kwargs[key] = gen_cfg[key]
            else:
                # Object with attributes
                for attr in ['temperature', 'top_p', 'top_k', 'max_output_tokens',
                            'response_mime_type', 'response_schema', 'stop_sequences']:
                    val = getattr(gen_cfg, attr, None)
                    if val is not None:
                        config_kwargs[attr] = val
        
        # Add safety settings
        safety = safety_settings or self._safety_settings
        if safety:
            # Convert to new format if needed
            config_kwargs['safety_settings'] = self._convert_safety_for_config(safety)
        
        # Create the config object
        try:
            return types.GenerateContentConfig(**config_kwargs)
        except Exception as e:
            logging.warning(f"Failed to create GenerateContentConfig: {e}")
            # Fall back to dict-based config
            return config_kwargs
    
    def _convert_safety_for_config(self, safety_settings: Any) -> List:
        """Convert safety settings for GenerateContentConfig."""
        if isinstance(safety_settings, list):
            result = []
            for setting in safety_settings:
                if isinstance(setting, dict):
                    result.append(types.SafetySetting(
                        category=setting.get('category'),
                        threshold=setting.get('threshold')
                    ))
                else:
                    result.append(setting)
            return result
        return safety_settings
    
    def _send_message_stream(self, message) -> Iterator[LegacyGenerateContentResponse]:
        """Handle streaming message send."""
        # Build config for streaming too
        config = self._build_send_message_config(None, None)
        
        if config:
            response = self._chat.send_message(message=message, config=config)
        else:
            response = self._chat.send_message(message=message)
        yield LegacyGenerateContentResponse(response, is_stream_chunk=True)
    
    def _convert_parts_to_new_format(self, parts: List) -> List:
        """Convert legacy parts to new SDK format."""
        converted = []
        for part in parts:
            if isinstance(part, str):
                converted.append({'text': part})
            elif isinstance(part, dict):
                converted.append(part)
            elif hasattr(part, 'text'):
                converted.append({'text': part.text})
            else:
                converted.append({'text': str(part)})
        return converted


class GenAIAsLegacyGenerativeModel:
    """
    Wraps the new google-genai SDK to provide the same interface as
    the legacy google.generativeai.GenerativeModel class.
    
    This allows existing code to work without modification:
    
        # Legacy code:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello!")
        
        # With wrapper:
        model = GenAIAsLegacyGenerativeModel('gemini-2.0-flash', api_key=...)
        response = model.generate_content("Hello!")  # Same interface!
    """
    
    def __init__(self, model_name: str, 
                 api_key: str = None,
                 generation_config: Any = None,
                 safety_settings: Any = None,
                 tools: Any = None,
                 tool_config: Any = None,
                 system_instruction: Union[str, List] = None):
        """
        Initialize the wrapper.
        
        Args:
            model_name: Name of the model (e.g., 'gemini-2.0-flash')
            api_key: API key (or set GEMINI_API_KEY env var)
            generation_config: Default generation configuration
            safety_settings: Default safety settings
            tools: Tools/functions for function calling
            tool_config: Tool configuration
            system_instruction: System instruction for the model
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai SDK not installed. "
                "Install with: pip install google-genai"
            )
        
        # Create the client
        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()
        
        self._model_name = model_name
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._tool_config = tool_config
        self._system_instruction = system_instruction
    
    def generate_content(self, 
                         contents: Union[str, List, Dict],
                         generation_config: Any = None,
                         safety_settings: Any = None,
                         stream: bool = False,
                         tools: Any = None,
                         tool_config: Any = None) -> LegacyGenerateContentResponse:
        """
        Generate content using the model.
        
        This method signature matches the legacy SDK exactly.
        
        Args:
            contents: The prompt (string, list of parts, or content dict)
            generation_config: Generation configuration (temperature, etc.)
            safety_settings: Safety filter settings
            stream: If True, return a streaming iterator
            tools: Tools for function calling
            tool_config: Tool configuration
            
        Returns:
            LegacyGenerateContentResponse that behaves like the legacy response
        """
        # Convert contents to new SDK format
        converted_contents = self._convert_contents(contents)
        
        # Build configuration
        config = self._build_config(
            generation_config or self._generation_config,
            safety_settings or self._safety_settings,
            tools or self._tools,
            tool_config or self._tool_config
        )
        
        try:
            if stream:
                return self._generate_content_stream(converted_contents, config)
            else:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=converted_contents,
                    config=config
                )
                return LegacyGenerateContentResponse(response)
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            raise
    
    def _generate_content_stream(self, contents, config) -> Iterator[LegacyGenerateContentResponse]:
        """Handle streaming generation."""
        try:
            stream = self._client.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=config
            )
            for chunk in stream:
                yield LegacyGenerateContentResponse(chunk, is_stream_chunk=True)
        except Exception as e:
            logging.error(f"Error in streaming generation: {e}")
            raise
    
    def generate_content_async(self, 
                               contents: Union[str, List, Dict],
                               generation_config: Any = None,
                               safety_settings: Any = None,
                               stream: bool = False,
                               tools: Any = None,
                               tool_config: Any = None):
        """
        Async version of generate_content.
        
        Note: The new SDK uses client.aio for async operations.
        """
        # For async, we'd need to use client.aio.models.generate_content
        # This is a placeholder that falls back to sync for now
        logging.warning("Async generation falling back to sync. Use client.aio directly for true async.")
        return self.generate_content(
            contents, generation_config, safety_settings, stream, tools, tool_config
        )
    
    def start_chat(self, 
                   history: List = None,
                   enable_automatic_function_calling: bool = False) -> LegacyChatSession:
        """
        Start a chat session.
        
        Args:
            history: Optional conversation history
            enable_automatic_function_calling: Enable auto function calling
            
        Returns:
            LegacyChatSession that behaves like the legacy ChatSession
        """
        return LegacyChatSession(
            client=self._client,
            model_name=self._model_name,
            history=history,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=self._tools,
            enable_automatic_function_calling=enable_automatic_function_calling,
            system_instruction=self._system_instruction
        )
    
    def count_tokens(self, contents: Union[str, List, Dict]) -> SimpleNamespace:
        """
        Count tokens in the content.
        
        Args:
            contents: The content to count tokens for
            
        Returns:
            Object with total_tokens attribute
        """
        converted_contents = self._convert_contents(contents)
        
        try:
            response = self._client.models.count_tokens(
                model=self._model_name,
                contents=converted_contents
            )
            # Return in legacy format
            return SimpleNamespace(
                total_tokens=getattr(response, 'total_tokens', 0)
            )
        except Exception as e:
            logging.error(f"Error counting tokens: {e}")
            raise
    
    def _convert_contents(self, contents: Union[str, List, Dict, Any]) -> Any:
        """
        Convert legacy content format to new SDK format.
        
        Handles:
        - Plain strings
        - Lists of parts (text, images, etc.)
        - PIL Images (passed through - new SDK handles natively)
        - Dicts with mime_type/data
        - Content objects from legacy SDK
        """
        if contents is None:
            return None
        
        # Simple string - pass through
        if isinstance(contents, str):
            return contents
        
        # PIL Image - pass through directly!
        # The new SDK handles PIL Images natively (same as legacy SDK)
        # See migration guide: "PIL.Image objects are automatically converted"
        if Image and isinstance(contents, Image.Image):
            return contents
        
        # List of parts
        if isinstance(contents, list):
            converted_parts = []
            for part in contents:
                converted = self._convert_single_part(part)
                if converted is not None:
                    if isinstance(converted, list):
                        converted_parts.extend(converted)
                    else:
                        converted_parts.append(converted)
            return converted_parts
        
        # Dict (could be content dict or part dict)
        if isinstance(contents, dict):
            return self._convert_dict_content(contents)
        
        # Fallback: convert to string
        return str(contents)
    
    def _convert_single_part(self, part: Any) -> Any:
        """Convert a single part to new SDK format."""
        # String - pass through
        if isinstance(part, str):
            return part
        
        # PIL Image - pass through directly!
        # The new SDK handles PIL Images natively (same as legacy SDK)
        if Image and isinstance(part, Image.Image):
            return part
        
        # Dict with mime_type and data (inline data)
        if isinstance(part, dict):
            mime_type = part.get('mime_type', '')
            data = part.get('data')
            
            if mime_type and data:
                # This is inline data (like an image)
                if isinstance(data, bytes):
                    return types.Part.from_bytes(data=data, mime_type=mime_type)
                elif isinstance(data, str):
                    # Might be base64 encoded
                    try:
                        decoded = base64.b64decode(data)
                        return types.Part.from_bytes(data=decoded, mime_type=mime_type)
                    except Exception:
                        pass
            
            # Text part
            if 'text' in part:
                return part['text']
            
            # Return as-is for other dict types
            return part
        
        # Object with text attribute
        if hasattr(part, 'text'):
            return part.text
        
        # Fallback
        return str(part)
    
    def _convert_dict_content(self, content: Dict) -> Any:
        """Convert a dict content to new SDK format."""
        # Check if it's a content object with role and parts
        if 'role' in content and 'parts' in content:
            parts = [self._convert_single_part(p) for p in content['parts']]
            return {
                'role': content['role'],
                'parts': parts
            }
        
        # Check if it's inline data
        if 'mime_type' in content and 'data' in content:
            return self._convert_single_part(content)
        
        # Check if it's a text part
        if 'text' in content:
            return content['text']
        
        return content
    
    def _build_config(self, 
                      generation_config: Any = None,
                      safety_settings: Any = None,
                      tools: Any = None,
                      tool_config: Any = None) -> Optional[Any]:
        """
        Build a GenerateContentConfig from legacy parameters.
        """
        if not any([generation_config, safety_settings, tools, 
                    tool_config, self._system_instruction]):
            return None
        
        config_dict = {}
        
        # Convert generation config
        if generation_config:
            config_dict.update(self._convert_generation_config(generation_config))
        
        # Convert safety settings
        if safety_settings:
            config_dict['safety_settings'] = self._convert_safety_settings(safety_settings)
        
        # Convert tools
        if tools:
            config_dict['tools'] = self._convert_tools(tools)
        
        # Add system instruction
        if self._system_instruction:
            config_dict['system_instruction'] = self._system_instruction
        
        if not config_dict:
            return None
        
        try:
            return types.GenerateContentConfig(**config_dict)
        except Exception as e:
            logging.warning(f"Error building config: {e}. Using dict instead.")
            return config_dict
    
    def _convert_generation_config(self, config: Any) -> Dict:
        """Convert legacy GenerationConfig to dict format."""
        if config is None:
            return {}
        
        # If it's already a dict, use it
        if isinstance(config, dict):
            return self._normalize_config_keys(config)
        
        # Extract attributes from object
        result = {}
        
        attr_mapping = {
            'temperature': 'temperature',
            'top_p': 'top_p',
            'top_k': 'top_k',
            'max_output_tokens': 'max_output_tokens',
            'stop_sequences': 'stop_sequences',
            'response_mime_type': 'response_mime_type',
            'response_schema': 'response_schema',
            'candidate_count': 'candidate_count',
            'presence_penalty': 'presence_penalty',
            'frequency_penalty': 'frequency_penalty',
            'seed': 'seed',
        }
        
        for old_key, new_key in attr_mapping.items():
            value = getattr(config, old_key, None)
            if value is not None:
                result[new_key] = value
        
        return result
    
    def _normalize_config_keys(self, config: Dict) -> Dict:
        """Normalize configuration keys to new SDK format."""
        # The new SDK uses the same key names, but let's ensure consistency
        normalized = {}
        
        key_mapping = {
            'maxOutputTokens': 'max_output_tokens',
            'topP': 'top_p',
            'topK': 'top_k',
            'stopSequences': 'stop_sequences',
            'responseMimeType': 'response_mime_type',
            'responseSchema': 'response_schema',
            'candidateCount': 'candidate_count',
            'presencePenalty': 'presence_penalty',
            'frequencyPenalty': 'frequency_penalty',
        }
        
        for key, value in config.items():
            # Convert camelCase to snake_case if needed
            new_key = key_mapping.get(key, key)
            normalized[new_key] = value
        
        return normalized
    
    def _convert_safety_settings(self, settings: Any) -> List:
        """Convert legacy safety settings to new SDK format."""
        if settings is None:
            return []
        
        # If it's a dict mapping category to threshold
        if isinstance(settings, dict):
            converted = []
            for category, threshold in settings.items():
                converted.append(
                    types.SafetySetting(
                        category=self._normalize_safety_category(category),
                        threshold=self._normalize_safety_threshold(threshold)
                    )
                )
            return converted
        
        # If it's already a list
        if isinstance(settings, list):
            converted = []
            for setting in settings:
                if isinstance(setting, dict):
                    converted.append(
                        types.SafetySetting(
                            category=self._normalize_safety_category(setting.get('category', '')),
                            threshold=self._normalize_safety_threshold(setting.get('threshold', ''))
                        )
                    )
                else:
                    # Assume it's already in correct format
                    converted.append(setting)
            return converted
        
        return []
    
    def _normalize_safety_category(self, category: str) -> str:
        """Normalize safety category string."""
        category = str(category).upper()
        
        # Add prefix if missing
        if not category.startswith('HARM_CATEGORY_'):
            category = f'HARM_CATEGORY_{category}'
        
        return category
    
    def _normalize_safety_threshold(self, threshold: str) -> str:
        """Normalize safety threshold string."""
        threshold = str(threshold).upper()
        
        # Common mappings
        threshold_map = {
            'BLOCK_ONLY_HIGH': 'BLOCK_ONLY_HIGH',
            'BLOCK_MEDIUM_AND_ABOVE': 'BLOCK_MEDIUM_AND_ABOVE',
            'BLOCK_LOW_AND_ABOVE': 'BLOCK_LOW_AND_ABOVE',
            'BLOCK_NONE': 'BLOCK_NONE',
        }
        
        return threshold_map.get(threshold, threshold)
    
    def _convert_tools(self, tools: Any) -> List:
        """Convert legacy tools to new SDK format."""
        if tools is None:
            return []
        
        # Handle special string tools like 'code_execution' or 'google_search_retrieval'
        if isinstance(tools, str):
            if tools == 'code_execution':
                return [types.Tool(code_execution=types.ToolCodeExecution())]
            elif tools == 'google_search_retrieval':
                return [types.Tool(google_search=types.GoogleSearch())]
            else:
                logging.warning(f"Unknown tool string: {tools}")
                return []
        
        # Handle list of tools
        if isinstance(tools, list):
            converted = []
            for tool in tools:
                if callable(tool):
                    # It's a function - the new SDK should handle this
                    converted.append(tool)
                elif isinstance(tool, dict):
                    converted.append(tool)
                else:
                    converted.append(tool)
            return converted
        
        # Single callable
        if callable(tools):
            return [tools]
        
        return [tools]


class GenAIClient:
    """
    A client class that mimics the legacy genai module-level interface.
    
    Usage:
        # Instead of:
        # import google.generativeai as genai
        # genai.configure(api_key=...)
        # model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Use:
        client = GenAIClient(api_key=...)
        model = client.GenerativeModel('gemini-2.0-flash')
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the client.
        
        Args:
            api_key: API key (or set GEMINI_API_KEY env var)
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai SDK not installed. "
                "Install with: pip install google-genai"
            )
        
        self._api_key = api_key
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
    
    def GenerativeModel(self, model_name: str, **kwargs) -> GenAIAsLegacyGenerativeModel:
        """
        Create a GenerativeModel instance.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional arguments (generation_config, safety_settings, etc.)
            
        Returns:
            GenAIAsLegacyGenerativeModel instance
        """
        return GenAIAsLegacyGenerativeModel(
            model_name=model_name,
            api_key=self._api_key,
            **kwargs
        )
    
    def embed_content(self, model: str, content: Any, 
                      task_type: str = None, title: str = None) -> Dict:
        """
        Generate embeddings for content.
        
        Args:
            model: Model name
            content: Content to embed
            task_type: Task type (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            title: Optional title
            
        Returns:
            Dict with 'embedding' key
        """
        # The new SDK's embed_content
        response = self._client.models.embed_content(
            model=model,
            contents=content
        )
        
        # Convert to legacy format
        embeddings = getattr(response, 'embeddings', [])
        if embeddings:
            # Return the embedding vectors
            if hasattr(embeddings[0], 'values'):
                return {'embedding': [e.values for e in embeddings]}
            else:
                return {'embedding': embeddings}
        
        return {'embedding': []}
    
    def upload_file(self, path: str, **kwargs) -> Any:
        """Upload a file."""
        return self._client.files.upload(file=path, **kwargs)
    
    def list_files(self) -> Any:
        """List uploaded files."""
        return self._client.files.list()
    
    def get_file(self, name: str) -> Any:
        """Get a file by name."""
        return self._client.files.get(name=name)
    
    def delete_file(self, name: str) -> Any:
        """Delete a file."""
        return self._client.files.delete(name=name)


# Convenience function to create a model directly
def create_model(model_name: str, api_key: str = None, **kwargs) -> GenAIAsLegacyGenerativeModel:
    """
    Convenience function to create a legacy-compatible model.
    
    Args:
        model_name: Name of the model
        api_key: API key
        **kwargs: Additional model arguments
        
    Returns:
        GenAIAsLegacyGenerativeModel instance
    """
    return GenAIAsLegacyGenerativeModel(model_name, api_key=api_key, **kwargs)