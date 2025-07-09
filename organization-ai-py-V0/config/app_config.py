"""
Configuration Management for Organization AI

This module handles application configuration, settings persistence,
and environment-specific configurations.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class AppConfig:
    """
    Application configuration manager.
    
    Handles loading, saving, and managing application settings
    including user preferences, API keys, and system configurations.
    """
    
    def __init__(self):
        self.config_dir = self._get_config_directory()
        self.config_file = os.path.join(self.config_dir, 'config.json')
        self.api_keys_file = os.path.join(self.config_dir, 'api_keys.json')
        
        # Default configuration
        self.default_config = {
            "ui": {
                "theme": "dark",
                "window_geometry": {
                    "width": 1400,
                    "height": 900,
                    "x": 100,
                    "y": 100
                },
                "sidebar_width": 280,
                "splitter_sizes": [280, 1120]
            },
            "ai": {
                "default_model": "gpt-3.5-turbo",
                "max_tokens": 2048,
                "temperature": 0.7,
                "timeout": 30
            },
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "console_logging": True
            },
            "performance": {
                "cache_enabled": True,
                "auto_save": True,
                "max_cache_size_mb": 1024
            }
        }
        
        self._config = self.load_config()
        
    def _get_config_directory(self) -> str:
        """Get or create the configuration directory."""
        config_dir = os.path.join(os.path.expanduser('~'), '.organization_ai')
        os.makedirs(config_dir, exist_ok=True)
        return config_dir
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.default_config, loaded_config)
            else:
                # Create default config file
                self.save_config(self.default_config)
                return self.default_config.copy()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config.copy()
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file."""
        try:
            config_to_save = config or self._config
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., 'ui.theme')
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value
            value: Value to set
        """
        keys = key_path.split('.')
        target = self._config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # Set the final value
        target[keys[-1]] = value
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result


# Global configuration instance
app_config = AppConfig()


def get_config() -> AppConfig:
    """Get the global application configuration instance."""
    return app_config


def load_config() -> Dict[str, Any]:
    """Load configuration from the global config instance."""
    return app_config._config


def save_config(config: Optional[Dict[str, Any]] = None) -> bool:
    """Save configuration using the global config instance."""
    return app_config.save_config(config)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation."""
    return app_config.get(key_path, default)


def set_config_value(key_path: str, value: Any) -> None:
    """Set a configuration value using dot notation."""
    app_config.set(key_path, value)
