"""
Core Utility Functions for Organization AI

This module provides common utility functions used throughout the application,
including logging setup, file operations, and data validation.
"""

import os
import logging
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup application-wide logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("organization_ai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory(path: Union[str, Path]) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        str: Absolute path to the directory
    """
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return str(path_obj)


def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON from a file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if loading fails
        
    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.getLogger(__name__).warning(f"Failed to load JSON from {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        ensure_directory(Path(file_path).parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (PermissionError, OSError) as e:
        logging.getLogger(__name__).error(f"Failed to save JSON to {file_path}: {e}")
        return False


def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate model configuration dictionary.
    
    Args:
        config: Model configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    required_fields = ['id', 'name', 'type']
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
        elif not config[field]:
            errors.append(f"Empty required field: {field}")
    
    # Validate specific field types
    if 'max_tokens' in config:
        try:
            max_tokens = int(config['max_tokens'])
            if max_tokens <= 0:
                errors.append("max_tokens must be positive")
        except (ValueError, TypeError):
            errors.append("max_tokens must be a valid integer")
    
    if 'temperature' in config:
        try:
            temp = float(config['temperature'])
            if not 0.0 <= temp <= 2.0:
                errors.append("temperature must be between 0.0 and 2.0")
        except (ValueError, TypeError):
            errors.append("temperature must be a valid float")
    
    return errors


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_app_data_dir() -> str:
    """
    Get the application data directory.
    
    Returns:
        str: Path to application data directory
    """
    return ensure_directory(
        Path.home() / '.organization_ai'
    )


def get_timestamp(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem
    """
    # Characters not allowed in filenames
    invalid_chars = '<>:"/\\|?*'
    
    # Replace invalid characters with underscore
    sanitized = ''.join('_' if c in invalid_chars else c for c in filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure we have something left
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized
