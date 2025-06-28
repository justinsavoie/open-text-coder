# text_classifier/config.py
"""
Configuration utilities for the text classifier system.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union, List, Optional


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration parameters are present.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required parameters are missing
    """
    required = ["file_path", "text_column", "id_column"]
    missing = [key for key in required if key not in config]
    
    if missing:
        raise ValueError(f"Missing required config parameters: {missing}")
    
    # Validate file exists
    if not Path(config["file_path"]).exists():
        raise FileNotFoundError(f"Input file not found: {config['file_path']}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base: Base configuration
        override: Configuration values to override
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    result.update(override)
    return result


def parse_categories(config: Dict[str, Any]) -> Optional[List[str]]:
    """
    Parse categories from config, handling both string and list formats.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of categories or None
    """
    raw = config.get("categories")
    if raw is None:
        return None
    
    if isinstance(raw, str):
        categories = [c.strip() for c in raw.split(",") if c.strip()]
    elif isinstance(raw, list):
        categories = [str(c).strip() for c in raw if str(c).strip()]
    else:
        raise ValueError("'categories' must be a list or comma-separated string")
    
    return categories if categories else None


# Default configuration values
DEFAULT_CONFIG = {
    "classifier_model": "gemma3n:latest",
    "category_model": None,  # Will use classifier_model if not specified
    "judge_model": "gemma3n:latest",
    "backend": "ollama",
    "multiclass": False,
    "n_samples": 100,
    "question_context": "",
    "validation_samples": None  # Validate all by default
}


def get_config_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values to a configuration dictionary.
    
    Args:
        config: User-provided configuration
        
    Returns:
        Configuration with defaults applied
    """
    return merge_configs(DEFAULT_CONFIG, config)