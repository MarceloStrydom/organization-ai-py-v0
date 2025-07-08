import asyncio
import json
import os
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import aiohttp
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import logging

class ModelType(Enum):
    API_OPENAI = "openai"
    API_ANTHROPIC = "anthropic"
    API_GROQ = "groq"
    LOCAL_HUGGINGFACE = "huggingface"
    LOCAL_OLLAMA = "ollama"

@dataclass
class ModelConfig:
    id: str
    name: str
    type: ModelType
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    context_length: int = 4096

@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: Optional[str] = None

class AIModelManager(QObject):
    """Manages AI model connections and inference"""
    
    model_loaded = pyqtSignal(str)  # model_id
    model_error = pyqtSignal(str, str)  # model_id, error_message
    inference_complete = pyqtSignal(str, str)  # request_id, response
    inference_progress = pyqtSignal(str, str)  # request_id, partial_response
    
    def __init__(self):
        super().__init__()
        self.models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, any] = {}
        self.api_keys = self.load_api_keys()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or config file"""
        api_keys = {}
        
        # Try environment variables first
        api_keys['openai'] = os.getenv('OPENAI_API_KEY', '')
        api_keys['anthropic'] = os.getenv('ANTHROPIC_API_KEY', '')
        api_keys['groq'] = os.getenv('GROQ_API_KEY', '')
        
        # Try loading from config file
        config_path = os.path.join(os.path.expanduser('~'), '.organization_ai', 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_keys.update(config.get('api_keys', {}))
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}")
                
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
