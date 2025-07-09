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

# Import data models
from core.models.data_models import ModelType, ModelConfig, ChatMessage

# Configure module-level logging
logger = logging.getLogger(__name__)

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
        
        # Initialize logging first
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
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
        
        # Initialize setup methods
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
        """
        Save API keys to secure local configuration.
        
        Stores API keys in a local configuration file with appropriate permissions
        and updates the in-memory key storage for immediate use.
        
        Args:
            keys (Dict[str, str]): Dictionary of provider -> API key mappings
            
        Raises:
            PermissionError: If unable to create config directory or file
            JSONEncodeError: If keys cannot be serialized to JSON
        """
        try:
            # Ensure configuration directory exists
            config_dir = os.path.join(os.path.expanduser('~'), '.organization_ai')
            os.makedirs(config_dir, exist_ok=True)
            
            config_path = os.path.join(config_dir, 'config.json')
            
            # Load existing configuration or create new
            config = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self.logger.warning("Existing config file corrupted, creating new one")
                    config = {}
            
            # Update API keys section
            config['api_keys'] = keys
            
            # Save configuration with restrictive permissions
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            # Set restrictive file permissions (read/write for owner only)
            os.chmod(config_path, 0o600)
            
            # Update in-memory keys
            self.api_keys.update(keys)
            
            # Log success (without exposing actual keys)
            saved_providers = [k for k, v in keys.items() if v]
            self.logger.info(f"API keys saved for providers: {saved_providers}")
            
        except Exception as e:
            self.logger.error(f"Failed to save API keys: {e}")
            raise
    
    def register_model(self, config: ModelConfig):
        """
        Register a new AI model configuration.
        
        Adds a model configuration to the available models registry,
        enabling it for loading and inference operations.
        
        Args:
            config (ModelConfig): Model configuration to register
            
        Raises:
            ValueError: If model configuration is invalid
            KeyError: If model ID already exists and overwrite not allowed
        """
        try:
            # Validate configuration
            if not config.id:
                raise ValueError("Model ID cannot be empty")
            if not config.name:
                raise ValueError("Model name cannot be empty")
                
            # Check for duplicate IDs
            if config.id in self.models:
                self.logger.warning(f"Overwriting existing model configuration: {config.id}")
            
            # Register the model
            self.models[config.id] = config
            self.model_status[config.id] = "registered"
            
            self.logger.info(f"Registered model: {config.name} (ID: {config.id}, Type: {config.type.value})")
            
            # Emit status change signal
            self.model_status_changed.emit(config.id, "registered")
            
        except Exception as e:
            self.logger.error(f"Failed to register model {config.id}: {e}")
            raise
    
    def get_available_models(self) -> List[ModelConfig]:
        """
        Get list of all available model configurations.
        
        Returns:
            List[ModelConfig]: List of all registered model configurations
        """
        return list(self.models.values())
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get a specific model configuration by ID.
        
        Args:
            model_id (str): Unique model identifier
            
        Returns:
            Optional[ModelConfig]: Model configuration if found, None otherwise
        """
        return self.models.get(model_id)
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelConfig]:
        """
        Get all models of a specific type.
        
        Args:
            model_type (ModelType): Type of models to retrieve
            
        Returns:
            List[ModelConfig]: List of models matching the specified type
        """
        return [model for model in self.models.values() if model.type == model_type]
    
    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model IDs.
        
        Returns:
            List[str]: List of model IDs that are currently loaded
        """
        return list(self.loaded_models.keys())
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a specific model is currently loaded.
        
        Args:
            model_id (str): Model ID to check
            
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return model_id in self.loaded_models
    
    def get_model_status(self, model_id: str) -> str:
        """
        Get the current status of a model.
        
        Args:
            model_id (str): Model ID to check
            
        Returns:
            str: Current model status ("registered", "loading", "loaded", "error", "unloaded")
        """
        return self.model_status.get(model_id, "unknown")
    
    def load_model(self, model_id: str) -> bool:
        """
        Load a model for inference operations.
        
        Initializes and prepares a model for inference, handling different model types
        appropriately. For local models, this involves loading weights and creating
        inference pipelines. For API models, this validates configuration.
        
        Args:
            model_id (str): Unique identifier of the model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            ValueError: If model ID is not found
            RuntimeError: If model loading fails due to system constraints
        """
        if model_id not in self.models:
            error_msg = f"Model not found: {model_id}"
            self.logger.error(error_msg)
            self.model_error.emit(model_id, error_msg)
            return False
            
        config = self.models[model_id]
        
        try:
            # Update status to loading
            self.model_status[model_id] = "loading"
            self.model_status_changed.emit(model_id, "loading")
            
            self.logger.info(f"Loading model: {config.name} (Type: {config.type.value})")
            
            # Handle different model types
            if config.type == ModelType.LOCAL_HUGGINGFACE:
                success = self._load_huggingface_model(config)
            elif config.type == ModelType.LOCAL_OLLAMA:
                success = self._load_ollama_model(config)
            elif config.type in [ModelType.API_OPENAI, ModelType.API_ANTHROPIC, ModelType.API_GROQ]:
                success = self._validate_api_model(config)
            else:
                raise ValueError(f"Unsupported model type: {config.type}")
            
            if success:
                self.model_status[model_id] = "loaded"
                self.model_status_changed.emit(model_id, "loaded")
                self.model_loaded.emit(model_id)
                self.logger.info(f"Model loaded successfully: {config.name}")
                return True
            else:
                self.model_status[model_id] = "error"
                self.model_status_changed.emit(model_id, "error")
                return False
                
        except Exception as e:
            error_msg = f"Failed to load model {model_id}: {str(e)}"
            self.logger.error(error_msg)
            self.model_status[model_id] = "error"
            self.model_status_changed.emit(model_id, "error")
            self.model_error.emit(model_id, error_msg)
            return False
    
    def _validate_api_model(self, config: ModelConfig) -> bool:
        """
        Validate API model configuration.
        
        Args:
            config (ModelConfig): API model configuration
            
        Returns:
            bool: True if configuration is valid
        """
        # Check if API key is available
        provider_key = None
        if config.type == ModelType.API_OPENAI:
            provider_key = self.api_keys.get('openai')
        elif config.type == ModelType.API_ANTHROPIC:
            provider_key = self.api_keys.get('anthropic')
        elif config.type == ModelType.API_GROQ:
            provider_key = self.api_keys.get('groq')
            
        if not provider_key:
            raise ValueError(f"API key not configured for {config.type.value}")
        
        # Store the configuration (API models don't need actual loading)
        self.loaded_models[config.id] = config
        return True
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model to free resources.
        
        Args:
            model_id (str): Model ID to unload
            
        Returns:
            bool: True if successfully unloaded, False otherwise
        """
        try:
            if model_id not in self.loaded_models:
                self.logger.warning(f"Model {model_id} is not loaded")
                return False
            
            # Get model data before removal
            model_data = self.loaded_models[model_id]
            
            # Handle cleanup for different model types
            if isinstance(model_data, dict) and 'pipeline' in model_data:
                # HuggingFace model cleanup
                del model_data['pipeline']
                if 'model' in model_data:
                    del model_data['model']
                if 'tokenizer' in model_data:
                    del model_data['tokenizer']
            
            # Remove from loaded models
            del self.loaded_models[model_id]
            
            # Update status
            self.model_status[model_id] = "unloaded"
            self.model_status_changed.emit(model_id, "unloaded")
            
            # Force garbage collection for memory cleanup
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info(f"Model unloaded successfully: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    def _load_huggingface_model(self, config: ModelConfig) -> bool:
        """
        Load a HuggingFace transformer model.
        
        Downloads, caches, and initializes a HuggingFace model for local inference.
        Handles device placement, memory optimization, and error recovery.
        
        Args:
            config (ModelConfig): HuggingFace model configuration
            
        Returns:
            bool: True if model loaded successfully
            
        Raises:
            RuntimeError: If model loading fails due to memory or compatibility issues
            ValueError: If model path is invalid or model not found
        """
        try:
            self.logger.info(f"Loading HuggingFace model: {config.model_path}")
            
            # Determine optimal device and settings
            device, dtype = self._get_optimal_device_settings()
            
            # Create cache directory for this model
            model_cache_dir = os.path.join(self.cache_dir, 'huggingface', 
                                         config.model_path.replace('/', '_'))
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # Load tokenizer with error handling
            self.logger.debug(f"Loading tokenizer for {config.model_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_path,
                    cache_dir=model_cache_dir,
                    trust_remote_code=False,  # Security: don't execute remote code
                    local_files_only=False    # Allow downloads if needed
                )
                
                # Ensure tokenizer has required special tokens
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {e}")
            
            # Load model with optimizations
            self.logger.debug(f"Loading model weights for {config.model_path}")
            try:
                model_kwargs = {
                    'cache_dir': model_cache_dir,
                    'trust_remote_code': False,
                    'local_files_only': False,
                    'torch_dtype': dtype,
                    'low_cpu_mem_usage': True,  # Optimize memory usage
                }
                
                # Configure device placement
                if device == "cuda":
                    model_kwargs['device_map'] = "auto"
                    model_kwargs['max_memory'] = self._get_max_memory_config()
                
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    **model_kwargs
                )
                
                # Move to device if not using device_map
                if device != "cuda":
                    model = model.to(device)
                
                # Set to evaluation mode
                model.eval()
                
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights: {e}")
            
            # Create optimized inference pipeline
            self.logger.debug(f"Creating inference pipeline for {config.model_path}")
            try:
                pipeline_kwargs = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'max_new_tokens': config.max_tokens,
                    'temperature': config.temperature,
                    'do_sample': True,
                    'return_full_text': False,  # Only return generated text
                    'clean_up_tokenization_spaces': True
                }
                
                # Set device for pipeline
                if device == "cuda":
                    pipeline_kwargs['device'] = 0
                else:
                    pipeline_kwargs['device'] = -1
                
                pipe = pipeline(
                    "text-generation",
                    **pipeline_kwargs
                )
                
            except Exception as e:
                raise RuntimeError(f"Failed to create inference pipeline: {e}")
            
            # Store model components
            self.loaded_models[config.id] = {
                'pipeline': pipe,
                'model': model,
                'tokenizer': tokenizer,
                'config': config,
                'device': device,
                'dtype': str(dtype),
                'cache_dir': model_cache_dir
            }
            
            # Log success with resource info
            model_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"HuggingFace model loaded successfully: {config.model_path}")
            self.logger.info(f"Model parameters: {model_params:,} ({model_params/1e6:.1f}M)")
            self.logger.info(f"Device: {device}, Dtype: {dtype}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model {config.model_path}: {e}")
            # Cleanup any partially loaded components
            self._cleanup_failed_model_load(config.id)
            raise
    
    def _get_optimal_device_settings(self):
        """
        Determine optimal device and dtype settings for model loading.
        
        Returns:
            Tuple[str, torch.dtype]: Device name and optimal data type
        """
        if torch.cuda.is_available():
            # Check available VRAM
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / 1024**3
                
                if gpu_memory_gb >= 8:  # Sufficient VRAM for float16
                    return "cuda", torch.float16
                elif gpu_memory_gb >= 4:  # Limited VRAM, use smaller models only
                    return "cuda", torch.float16
                else:
                    self.logger.warning(f"Limited GPU memory ({gpu_memory_gb:.1f}GB), using CPU")
                    return "cpu", torch.float32
                    
            except Exception as e:
                self.logger.warning(f"Failed to query GPU memory, using CPU: {e}")
                return "cpu", torch.float32
        else:
            return "cpu", torch.float32
    
    def _get_max_memory_config(self):
        """
        Get maximum memory configuration for multi-GPU setups.
        
        Returns:
            Dict: Memory configuration for device placement
        """
        if not torch.cuda.is_available():
            return None
            
        try:
            device_count = torch.cuda.device_count()
            max_memory = {}
            
            for i in range(device_count):
                # Reserve some memory for system operations
                total_memory = torch.cuda.get_device_properties(i).total_memory
                usable_memory = int(total_memory * 0.9)  # Use 90% of available memory
                max_memory[i] = f"{usable_memory // 1024**2}MB"
            
            return max_memory
            
        except Exception as e:
            self.logger.warning(f"Failed to configure memory limits: {e}")
            return None
    
    def _cleanup_failed_model_load(self, model_id: str):
        """
        Cleanup resources from a failed model load attempt.
        
        Args:
            model_id (str): ID of the model that failed to load
        """
        try:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup after model load failure: {e}")
    
    def _load_ollama_model(self, config: ModelConfig) -> bool:
        """
        Load an Ollama model configuration.
        
        Ollama models are loaded on-demand when inference is requested,
        so this method validates the configuration and connectivity.
        
        Args:
            config (ModelConfig): Ollama model configuration
            
        Returns:
            bool: True if configuration is valid
        """
        try:
            self.logger.info(f"Validating Ollama model: {config.model_path}")
            
            # Test Ollama connectivity
            endpoint = config.endpoint or "http://localhost:11434"
            
            # Try to ping Ollama server
            try:
                import requests
                response = requests.get(f"{endpoint}/api/tags", timeout=5)
                if response.status_code == 200:
                    available_models = response.json().get('models', [])
                    model_names = [m['name'] for m in available_models]
                    
                    if config.model_path not in model_names:
                        self.logger.warning(f"Model {config.model_path} not found in Ollama. "
                                          f"Available: {model_names}")
                else:
                    self.logger.warning(f"Ollama server responded with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Cannot connect to Ollama server: {e}")
                # Don't fail completely - model might become available later
            
            # Store configuration
            self.loaded_models[config.id] = {
                'config': config,
                'endpoint': endpoint,
                'type': 'ollama'
            }
            
            self.logger.info(f"Ollama model configured: {config.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure Ollama model {config.model_path}: {e}")
            raise
    
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
