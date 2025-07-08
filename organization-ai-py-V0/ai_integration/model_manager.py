"""
AI Model Manager Module

This module provides comprehensive management of AI models supporting multiple providers
including OpenAI, Anthropic, Groq, HuggingFace, and Ollama. It handles model loading,
inference execution, and real-time streaming responses.

Key Features:
- Multi-provider AI model support (API and local models)
- Asynchronous inference with progress tracking
- Secure API key management
- Local model caching and optimization
- Real-time streaming responses
- Comprehensive error handling and logging

Supported Providers:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic (Claude 3 family)
- Groq (Mixtral, Llama 2, etc.)
- HuggingFace (Local transformer models)
- Ollama (Local model serving)
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, AsyncGenerator, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import logging

# Configure module-level logging
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """
    Enumeration of supported AI model types.
    
    This enum categorizes models by their deployment and access method,
    enabling appropriate handling for each model type.
    """
    API_OPENAI = "openai"          # OpenAI API models (GPT family)
    API_ANTHROPIC = "anthropic"    # Anthropic API models (Claude family)
    API_GROQ = "groq"              # Groq API models (fast inference)
    LOCAL_HUGGINGFACE = "huggingface"  # Local HuggingFace transformer models
    LOCAL_OLLAMA = "ollama"        # Local Ollama served models


@dataclass
class ModelConfig:
    """
    Configuration class for AI models.
    
    Contains all necessary parameters for model initialization and operation,
    supporting both API-based and locally hosted models.
    
    Attributes:
        id (str): Unique identifier for the model
        name (str): Human-readable model name
        type (ModelType): Model deployment type
        endpoint (str, optional): API endpoint URL for remote models
        api_key (str, optional): Authentication key for API models
        model_path (str, optional): Model path/identifier for loading
        max_tokens (int): Maximum tokens to generate in response
        temperature (float): Sampling temperature for response generation
        context_length (int): Maximum context window size
        description (str): Model description and capabilities
        tags (List[str]): Tags for model categorization and filtering
        capabilities (List[str]): List of model capabilities
    """
    id: str
    name: str
    type: ModelType
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    context_length: int = 4096
    description: str = ""
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate required fields based on model type
        if self.type in [ModelType.API_OPENAI, ModelType.API_ANTHROPIC, ModelType.API_GROQ]:
            if not self.model_path:
                raise ValueError(f"model_path required for {self.type.value} models")
        elif self.type in [ModelType.LOCAL_HUGGINGFACE, ModelType.LOCAL_OLLAMA]:
            if not self.model_path:
                raise ValueError(f"model_path required for {self.type.value} models")
                
        # Set default capabilities based on model type
        if not self.capabilities:
            self.capabilities = self._get_default_capabilities()
            
    def _get_default_capabilities(self) -> List[str]:
        """Get default capabilities based on model type."""
        if self.type == ModelType.API_OPENAI:
            return ["chat", "completion", "function_calling"]
        elif self.type == ModelType.API_ANTHROPIC:
            return ["chat", "completion", "long_context"]
        elif self.type == ModelType.API_GROQ:
            return ["chat", "completion", "fast_inference"]
        elif self.type == ModelType.LOCAL_HUGGINGFACE:
            return ["chat", "completion", "offline"]
        elif self.type == ModelType.LOCAL_OLLAMA:
            return ["chat", "completion", "offline", "local_hosting"]
        return []


@dataclass
class ChatMessage:
    """
    Represents a chat message in a conversation.
    
    Supports different message roles and includes metadata for tracking
    and debugging conversation flow.
    
    Attributes:
        role (str): Message role ("system", "user", "assistant", "function")
        content (str): Message content/text
        timestamp (str, optional): ISO timestamp of message creation
        metadata (dict, optional): Additional message metadata
    """
    role: str  # "system", "user", "assistant", "function"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        valid_roles = ["system", "user", "assistant", "function"]
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of: {valid_roles}")
            
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

class AIModelManager(QObject):
    """
    Comprehensive AI model management system.
    
    This class provides a unified interface for managing AI models across multiple
    providers, handling everything from model loading and configuration to 
    inference execution and streaming responses.
    
    Key Features:
    - Multi-provider support (OpenAI, Anthropic, Groq, HuggingFace, Ollama)
    - Asynchronous inference with real-time progress tracking
    - Secure API key management and storage
    - Local model caching and optimization
    - Thread-safe operations for GUI applications
    - Comprehensive error handling and recovery
    
    Signals:
        model_loaded(str): Emitted when a model is successfully loaded
        model_error(str, str): Emitted when model loading fails (model_id, error)
        inference_complete(str, str): Emitted when inference completes (request_id, response)
        inference_progress(str, str): Emitted during streaming (request_id, partial_response)
        model_status_changed(str, str): Emitted when model status changes (model_id, status)
    """
    
    # Define Qt signals for async communication
    model_loaded = pyqtSignal(str)  # model_id
    model_error = pyqtSignal(str, str)  # model_id, error_message
    inference_complete = pyqtSignal(str, str)  # request_id, response
    inference_progress = pyqtSignal(str, str)  # request_id, partial_response
    model_status_changed = pyqtSignal(str, str)  # model_id, status
    
    def __init__(self):
        """Initialize the AI model manager."""
        super().__init__()
        
        # Core data structures
        self.models: Dict[str, ModelConfig] = {}  # Available model configurations
        self.loaded_models: Dict[str, any] = {}   # Currently loaded model instances
        self.model_status: Dict[str, str] = {}    # Model loading/status tracking
        
        # Configuration and security
        self.api_keys = self._load_api_keys()     # Secure API key storage
        self.cache_dir = self._get_cache_directory()  # Local model cache location
        
        # Performance and monitoring
        self.request_history: List[Dict] = []     # Request history for analytics
        self.performance_metrics: Dict = {}       # Performance tracking
        
        # Initialize logging and setup
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_cache_directory()
        self._initialize_torch_settings()
        
        self.logger.info("AI Model Manager initialized successfully")
        
    def _get_cache_directory(self) -> str:
        """
        Get or create the model cache directory.
        
        Returns:
            str: Path to the model cache directory
        """
        cache_dir = os.path.join(os.path.expanduser('~'), '.organization_ai', 'models')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
        
    def _setup_cache_directory(self):
        """Setup the model cache directory with proper structure."""
        try:
            # Create subdirectories for different model types
            subdirs = ['huggingface', 'ollama', 'temp', 'logs']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
            self.logger.debug(f"Cache directory setup completed: {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to setup cache directory: {e}")
            
    def _initialize_torch_settings(self):
        """Initialize PyTorch settings for optimal performance."""
        try:
            # Set multiprocessing start method for compatibility
            if hasattr(torch.multiprocessing, 'set_start_method'):
                try:
                    torch.multiprocessing.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass  # Already set
                    
            # Log available compute resources
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                self.logger.info(f"CUDA available: {device_count} devices, current: {device_name}")
            else:
                self.logger.info("CUDA not available, using CPU for local models")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize PyTorch settings: {e}")
        
    def setup_logging(self):
        """
        DEPRECATED: Use module-level logger instead.
        
        This method is maintained for backward compatibility.
        """
        self.logger.warning("setup_logging() is deprecated, using module-level logger")
        
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from multiple sources.
        
        Attempts to load API keys from environment variables first,
        then from local configuration file as fallback.
        
        Returns:
            Dict[str, str]: Dictionary of provider -> API key mappings
        """
        api_keys = {}
        
        # Load from environment variables (preferred method)
        env_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY', 
            'groq': 'GROQ_API_KEY'
        }
        
        for provider, env_var in env_keys.items():
            key = os.getenv(env_var)
            if key:
                api_keys[provider] = key
                self.logger.debug(f"Loaded {provider} API key from environment")
        
        # Load from configuration file as fallback
        config_path = os.path.join(os.path.expanduser('~'), '.organization_ai', 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    file_keys = config.get('api_keys', {})
                    
                    # Only use file keys if not already loaded from environment
                    for provider, key in file_keys.items():
                        if provider not in api_keys and key:
                            api_keys[provider] = key
                            self.logger.debug(f"Loaded {provider} API key from config file")
                            
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}")
        
        # Log summary (without exposing actual keys)
        loaded_providers = list(api_keys.keys())
        self.logger.info(f"API keys loaded for providers: {loaded_providers}")
        
        return api_keys
    
    def save_api_keys(self, keys: Dict[str, str]):
        """Save API keys to config file"""
        config_dir = os.path.join(os.path.expanduser('~'), '.organization_ai')
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, 'config.json')
        config = {'api_keys': keys}
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.api_keys.update(keys)
        except Exception as e:
            self.logger.error(f"Could not save config file: {e}")
    
    def register_model(self, config: ModelConfig):
        """Register a new model configuration"""
        self.models[config.id] = config
        self.logger.info(f"Registered model: {config.name}")
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available models"""
        return list(self.models.values())
    
    def load_model(self, model_id: str) -> bool:
        """Load a model for inference"""
        if model_id not in self.models:
            self.model_error.emit(model_id, "Model not found")
            return False
            
        config = self.models[model_id]
        
        try:
            if config.type == ModelType.LOCAL_HUGGINGFACE:
                self._load_huggingface_model(config)
            elif config.type == ModelType.LOCAL_OLLAMA:
                self._load_ollama_model(config)
            # API models don't need loading
            elif config.type in [ModelType.API_OPENAI, ModelType.API_ANTHROPIC, ModelType.API_GROQ]:
                self.loaded_models[model_id] = config
                
            self.model_loaded.emit(model_id)
            return True
            
        except Exception as e:
            self.model_error.emit(model_id, str(e))
            return False
    
    def _load_huggingface_model(self, config: ModelConfig):
        """Load a HuggingFace model"""
        self.logger.info(f"Loading HuggingFace model: {config.model_path}")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            do_sample=True
        )
        
        self.loaded_models[config.id] = {
            'pipeline': pipe,
            'tokenizer': tokenizer,
            'config': config
        }
    
    def _load_ollama_model(self, config: ModelConfig):
        """Load an Ollama model"""
        # Ollama models are loaded on-demand
        self.loaded_models[config.id] = config
    
    async def generate_response(self, model_id: str, messages: List[ChatMessage], 
                              request_id: str = None) -> str:
        """Generate a response using the specified model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
            
        config = self.models[model_id]
        
        if config.type == ModelType.API_OPENAI:
            return await self._generate_openai(config, messages, request_id)
        elif config.type == ModelType.API_ANTHROPIC:
            return await self._generate_anthropic(config, messages, request_id)
        elif config.type == ModelType.API_GROQ:
            return await self._generate_groq(config, messages, request_id)
        elif config.type == ModelType.LOCAL_HUGGINGFACE:
            return await self._generate_huggingface(config, messages, request_id)
        elif config.type == ModelType.LOCAL_OLLAMA:
            return await self._generate_ollama(config, messages, request_id)
        else:
            raise ValueError(f"Unsupported model type: {config.type}")
    
    async def _generate_openai(self, config: ModelConfig, messages: List[ChatMessage], 
                              request_id: str) -> str:
        """Generate response using OpenAI API"""
        if not self.api_keys.get('openai'):
            raise ValueError("OpenAI API key not configured")
            
        headers = {
            'Authorization': f'Bearer {self.api_keys["openai"]}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': config.model_path or 'gpt-3.5-turbo',
            'messages': [{'role': msg.role, 'content': msg.content} for msg in messages],
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")
                
                full_response = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    if request_id:
                                        self.inference_progress.emit(request_id, full_response)
                        except json.JSONDecodeError:
                            continue
                
                return full_response
    
    async def _generate_anthropic(self, config: ModelConfig, messages: List[ChatMessage], 
                                 request_id: str) -> str:
        """Generate response using Anthropic API"""
        if not self.api_keys.get('anthropic'):
            raise ValueError("Anthropic API key not configured")
            
        headers = {
            'x-api-key': self.api_keys['anthropic'],
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        # Convert messages format for Anthropic
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                user_messages.append({'role': msg.role, 'content': msg.content})
        
        payload = {
            'model': config.model_path or 'claude-3-sonnet-20240229',
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'messages': user_messages,
            'stream': True
        }
        
        if system_message:
            payload['system'] = system_message
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {error_text}")
                
                full_response = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        try:
                            chunk = json.loads(data)
                            if chunk.get('type') == 'content_block_delta':
                                delta = chunk.get('delta', {})
                                if 'text' in delta:
                                    content = delta['text']
                                    full_response += content
                                    if request_id:
                                        self.inference_progress.emit(request_id, full_response)
                        except json.JSONDecodeError:
                            continue
                
                return full_response
    
    async def _generate_groq(self, config: ModelConfig, messages: List[ChatMessage], 
                            request_id: str) -> str:
        """Generate response using Groq API"""
        if not self.api_keys.get('groq'):
            raise ValueError("Groq API key not configured")
            
        headers = {
            'Authorization': f'Bearer {self.api_keys["groq"]}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': config.model_path or 'mixtral-8x7b-32768',
            'messages': [{'role': msg.role, 'content': msg.content} for msg in messages],
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API error: {error_text}")
                
                full_response = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    if request_id:
                                        self.inference_progress.emit(request_id, full_response)
                        except json.JSONDecodeError:
                            continue
                
                return full_response
    
    async def _generate_huggingface(self, config: ModelConfig, messages: List[ChatMessage], 
                                   request_id: str) -> str:
        """Generate response using local HuggingFace model"""
        if config.id not in self.loaded_models:
            raise ValueError(f"Model {config.id} not loaded")
            
        model_data = self.loaded_models[config.id]
        pipeline = model_data['pipeline']
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate response
        result = pipeline(
            prompt,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated_text = result[0]['generated_text']
        response = generated_text[len(prompt):].strip()
        
        if request_id:
            self.inference_progress.emit(request_id, response)
        
        return response
    
    async def _generate_ollama(self, config: ModelConfig, messages: List[ChatMessage], 
                              request_id: str) -> str:
        """Generate response using Ollama"""
        import aiohttp
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            'model': config.model_path,
            'prompt': prompt,
            'stream': True,
            'options': {
                'temperature': config.temperature,
                'num_predict': config.max_tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{config.endpoint or "http://localhost:11434"}/api/generate',
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {error_text}")
                
                full_response = ""
                async for line in response.content:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            content = chunk['response']
                            full_response += content
                            if request_id:
                                self.inference_progress.emit(request_id, full_response)
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                
                return full_response
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string"""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

# Initialize default models
def get_default_models() -> List[ModelConfig]:
    """Get default model configurations"""
    return [
        # OpenAI Models
        ModelConfig(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            type=ModelType.API_OPENAI,
            model_path="gpt-3.5-turbo",
            max_tokens=4096,
            context_length=16384
        ),
        ModelConfig(
            id="gpt-4",
            name="GPT-4",
            type=ModelType.API_OPENAI,
            model_path="gpt-4",
            max_tokens=8192,
            context_length=32768
        ),
        
        # Anthropic Models
        ModelConfig(
            id="claude-3-sonnet",
            name="Claude 3 Sonnet",
            type=ModelType.API_ANTHROPIC,
            model_path="claude-3-sonnet-20240229",
            max_tokens=4096,
            context_length=200000
        ),
        ModelConfig(
            id="claude-3-haiku",
            name="Claude 3 Haiku",
            type=ModelType.API_ANTHROPIC,
            model_path="claude-3-haiku-20240307",
            max_tokens=4096,
            context_length=200000
        ),
        
        # Groq Models
        ModelConfig(
            id="mixtral-8x7b",
            name="Mixtral 8x7B",
            type=ModelType.API_GROQ,
            model_path="mixtral-8x7b-32768",
            max_tokens=32768,
            context_length=32768
        ),
        ModelConfig(
            id="llama2-70b",
            name="Llama 2 70B",
            type=ModelType.API_GROQ,
            model_path="llama2-70b-4096",
            max_tokens=4096,
            context_length=4096
        ),
        
        # Local HuggingFace Models
        ModelConfig(
            id="microsoft-dialoGPT",
            name="DialoGPT Medium",
            type=ModelType.LOCAL_HUGGINGFACE,
            model_path="microsoft/DialoGPT-medium",
            max_tokens=1024,
            context_length=1024
        ),
        ModelConfig(
            id="gpt2",
            name="GPT-2",
            type=ModelType.LOCAL_HUGGINGFACE,
            model_path="gpt2",
            max_tokens=1024,
            context_length=1024
        ),
        
        # Ollama Models
        ModelConfig(
            id="llama2-ollama",
            name="Llama 2 (Ollama)",
            type=ModelType.LOCAL_OLLAMA,
            model_path="llama2",
            endpoint="http://localhost:11434",
            max_tokens=2048,
            context_length=4096
        ),
        ModelConfig(
            id="mistral-ollama",
            name="Mistral (Ollama)",
            type=ModelType.LOCAL_OLLAMA,
            model_path="mistral",
            endpoint="http://localhost:11434",
            max_tokens=2048,
            context_length=8192
        )
    ]
