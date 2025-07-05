# text_classifier/config.py
"""
Configuration utilities for the text classifier system.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

# --- Default configuration values ---
# This is now defined at the top to be used by other functions.
DEFAULT_CONFIG = {
    "classifier_model": "gemma3n:latest",
    "category_model": None,
    "judge_model": "gemma3n:latest",
    "classifier_backend": "ollama",
    "category_backend": None,
    "judge_backend": None, # Fixed: This key is now present.
    "multiclass": False,
    "n_samples": 100,
    "question_context": "",
    "validation_samples": None,
    "max_retries": 3,
    "retry_delay": 1,
    "error_category": "Classification Error",
    "fallback_categories": ["Positive", "Negative", "Neutral", "Other"]
}


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        user_config = json.load(f)
    
    return get_config_with_defaults(user_config)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration parameters are present and valid.
    """
    required = ["file_path", "text_column", "id_column"]
    missing = [key for key in required if key not in config]
    
    if missing:
        raise ValueError(f"Missing required config parameters: {missing}")
    
    file_path = Path(config["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    valid_backends = {"ollama", "openai"}
    backend_keys = ["classifier_backend", "category_backend", "judge_backend"]
    
    for key in backend_keys:
        if key in config and config[key] is not None:
            if config[key] not in valid_backends:
                raise ValueError(
                    f"Invalid {key}: '{config[key]}'. Must be one of: {valid_backends}"
                )


def parse_categories(config: Dict[str, Any]) -> Optional[List[str]]:
    """
    Parse categories from config, handling both string and list formats.
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


def get_config_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values to configuration.
    """
    result = DEFAULT_CONFIG.copy()
    result.update(config)
    
    # Apply cascading defaults for backend and model settings
    if result.get("category_backend") is None:
        result["category_backend"] = result["classifier_backend"]
    
    if result.get("judge_backend") is None:
        result["judge_backend"] = result["classifier_backend"]
    
    if result.get("category_model") is None:
        result["category_model"] = result["classifier_model"]
    
    return result