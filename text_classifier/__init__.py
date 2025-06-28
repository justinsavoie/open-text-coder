# text_classifier/__init__.py
"""
Text Classification System

A flexible system for classifying open-text survey responses using LLMs.
"""

# Import main API functions
from .api import (
    classify_texts,
    validate_classification,
    load_classification_results,
    load_validation_results,
    compare_runs,
    list_all_runs,
    get_run_info,
    load_config
)

# Import classes for advanced usage
from .classifier import TextClassifier
from .validator import ClassificationValidator
from .storage import RunStorage
from .models import ClassificationRun, ValidationRun

__version__ = "0.1.0"

__all__ = [
    # Main API
    "classify_texts",
    "validate_classification",
    "load_classification_results",
    "load_validation_results",
    "compare_runs",
    "list_all_runs",
    "get_run_info",
    "load_config",
    
    # Classes
    "TextClassifier",
    "ClassificationValidator",
    "RunStorage",
    "ClassificationRun",
    "ValidationRun"
]