# text_classifier/api.py
"""
High-level API for text classification system.
This module provides the main functions users should interact with.
"""

from pathlib import Path
from datetime import datetime
import uuid
import json
from typing import Dict, Any, Optional, List
import pandas as pd

from .models import ClassificationRun, ValidationRun
from .storage import RunStorage
from .classifier import TextClassifier
from .validator import ClassificationValidator
from .config import (
    load_config as _load_config,
    validate_config,
    get_config_with_defaults,
    parse_categories
)


def classify_texts(
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    storage_dir: Path = Path("./runs")
) -> str:
    """
    Run text classification with proper storage.
    
    Args:
        config: Configuration dictionary with keys:
            - file_path: Path to input CSV
            - text_column: Column containing text
            - id_column: Column with unique IDs
            - categories: Optional list/string of categories
            - classifier_model: Model name (default: "gemma3n:latest")
            - backend: "ollama" or "openai" (default: "ollama")
            - multiclass: Enable multi-label (default: False)
            - n_samples: Samples for category generation (default: 100)
            - question_context: Context for category generation
            - category_model: Model for generating categories
        run_id: Optional run ID (auto-generated if None)
        storage_dir: Directory to store runs
        
    Returns:
        run_id: Unique identifier for this classification run
    """
    # Apply defaults and validate
    config = get_config_with_defaults(config)
    validate_config(config)
    
    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    
    # Initialize storage
    storage = RunStorage(storage_dir)
    
    # Load data
    df = pd.read_csv(config["file_path"],keep_default_na=False)
    
    # Parse categories from config
    categories = parse_categories(config)
    
    # Initialize classifier
    classifier = TextClassifier(
        config.get("classifier_model", "gemma3n:latest"),
        config.get("classifier_backend", config.get("backend", "ollama"))
    )
    
    # Run classification
    classified_df, final_categories, metrics = classifier.run_classification(
        df=df,
        text_column=config["text_column"],
        id_column=config["id_column"],
        categories=categories,
        multiclass=config.get("multiclass", False),
        n_samples=config.get("n_samples", 100),
        question_context=config.get("question_context", ""),
        category_model=config.get("category_model"),
        category_backend=config.get("category_backend", config.get("classifier_backend", config.get("backend", "ollama")))
    )
    
    # Create run record
    run = ClassificationRun(
        run_id=run_id,
        timestamp=datetime.now(),
        config=config,
        input_file=Path(config["file_path"]),
        categories=final_categories,
        metrics=metrics
    )
    
    # Save results
    output_file = storage.save_classification_run(run, classified_df)
    print(f"[*] Classification complete. Run ID: {run_id}")
    print(f"[*] Results saved to: {output_file}")
    
    return run_id


def validate_classification(
    classification_run_id: str,
    config: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = None,
    storage_dir: Path = Path("./runs")
) -> str:
    """
    Validate a classification run using LLM-as-judge.
    
    Args:
        classification_run_id: ID of the classification run to validate
        config: Optional config to override validation settings
        sample_size: Number of samples to validate (None = all)
        storage_dir: Directory where runs are stored
        
    Returns:
        validation_id: Unique identifier for this validation run
    """
    # Initialize storage
    storage = RunStorage(storage_dir)
    
    # Load classification run
    class_run = storage.get_classification_run(classification_run_id)
    if not class_run:
        raise ValueError(f"Classification run {classification_run_id} not found")
    
    # Load classification data
    classified_df = storage.load_classification_data(classification_run_id)
    if classified_df is None:
        raise ValueError(f"Could not load data for run {classification_run_id}")
    
    # Use config from classification run if not provided
    if config is None:
        config = class_run.config.copy()
    
    # Override sample size if provided
    if sample_size is not None:
        config["validation_samples"] = sample_size
    
    # Initialize validator
    validator = ClassificationValidator(
        config.get("judge_model", "gemma3n:latest"),
        config.get("backend", "ollama")
    )
    
    # Determine category column
    if class_run.config.get("multiclass", False):
        # For multiclass, we'd need to validate each category separately
        # For now, let's skip multiclass validation
        raise NotImplementedError("Multiclass validation not yet implemented")
    else:
        category_column = "category"
    
    # Run validation
    validated_df, metrics = validator.validate_classification_run(
        df=classified_df,
        text_column=class_run.config["text_column"],
        category_column=category_column,
        categories=class_run.categories,
        sample_size=config.get("validation_samples")
    )
    
    # Generate validation ID
    val_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    
    # Save validation results
    val_dir = storage.base_dir / f"validation_{val_id}"
    val_dir.mkdir(exist_ok=True)
    
    output_file = val_dir / f"validation_{val_id}.csv"
    validated_df.to_csv(output_file, index=False)
    
    # Save validation metadata
    val_run = {
        "validation_id": val_id,
        "classification_run_id": classification_run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results_file": str(output_file),
        "metrics": metrics
    }
    
    storage.metadata["validation_runs"][val_id] = val_run
    storage._save_metadata()
    
    print(f"[*] Validation complete. ID: {val_id}")
    print(f"[*] Average quality score: {metrics['average_score']:.2f}")
    print(f"[*] Results saved to: {output_file}")
    
    return val_id


def load_classification_results(
    run_id: str,
    storage_dir: Path = Path("./runs")
) -> pd.DataFrame:
    """
    Load classification results for a given run.
    
    Args:
        run_id: Classification run ID
        storage_dir: Directory where runs are stored
        
    Returns:
        DataFrame with classification results
    """
    storage = RunStorage(storage_dir)
    df = storage.load_classification_data(run_id)
    if df is None:
        raise ValueError(f"Could not load results for run {run_id}")
    return df


def load_validation_results(
    validation_id: str,
    storage_dir: Path = Path("./runs")
) -> pd.DataFrame:
    """
    Load validation results.
    
    Args:
        validation_id: Validation run ID
        storage_dir: Directory where runs are stored
        
    Returns:
        DataFrame with validation results
    """
    storage = RunStorage(storage_dir)
    val_run = storage.metadata.get("validation_runs", {}).get(validation_id)
    if not val_run:
        raise ValueError(f"Validation run {validation_id} not found")
    
    results_file = Path(val_run["results_file"])
    if not results_file.exists():
        raise ValueError(f"Results file not found: {results_file}")
    
    return pd.read_csv(results_file)


def compare_runs(
    run_ids: List[str],
    storage_dir: Path = Path("./runs")
) -> pd.DataFrame:
    """
    Compare multiple classification runs.
    
    Args:
        run_ids: List of classification run IDs to compare
        storage_dir: Directory where runs are stored
        
    Returns:
        DataFrame with comparison metrics
    """
    storage = RunStorage(storage_dir)
    
    comparisons = []
    for run_id in run_ids:
        run = storage.get_classification_run(run_id)
        if run:
            comparison = {
                "run_id": run_id,
                "timestamp": run.timestamp,
                "model": run.config.get("classifier_model"),
                "backend": run.config.get("backend"),
                "multiclass": run.config.get("multiclass", False),
                "num_categories": len(run.categories),
                "total_rows": run.metrics.get("total_rows", 0),
                "classified_rows": run.metrics.get("classified_rows", 0)
            }
            
            # Add validation metrics if available
            validations = [v for v in storage.metadata.get("validation_runs", {}).values()
                          if v["classification_run_id"] == run_id]
            if validations:
                latest_val = max(validations, key=lambda x: x["timestamp"])
                comparison["validation_score"] = latest_val["metrics"]["average_score"]
                comparison["validated_samples"] = latest_val["metrics"]["total_validated"]
            
            comparisons.append(comparison)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparisons)
    
    print("\n=== Run Comparison ===")
    print(comparison_df.to_string())
    
    return comparison_df


def list_all_runs(
    run_type: str = "classification",
    storage_dir: Path = Path("./runs")
) -> List[Dict[str, Any]]:
    """
    List all runs of a given type.
    
    Args:
        run_type: "classification" or "validation"
        storage_dir: Directory where runs are stored
        
    Returns:
        List of run metadata dictionaries
    """
    storage = RunStorage(storage_dir)
    runs = storage.list_runs(run_type)
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return runs


def get_run_info(
    run_id: str,
    storage_dir: Path = Path("./runs")
) -> Dict[str, Any]:
    """
    Get detailed information about a specific run.
    
    Args:
        run_id: Classification or validation run ID
        storage_dir: Directory where runs are stored
        
    Returns:
        Dictionary with run information
    """
    storage = RunStorage(storage_dir)
    
    # Check classification runs
    run = storage.get_classification_run(run_id)
    if run:
        return {
            "type": "classification",
            "run": run.to_dict(),
            "validations": [v for v in storage.metadata.get("validation_runs", {}).values()
                           if v["classification_run_id"] == run_id]
        }
    
    # Check validation runs
    val_run = storage.metadata.get("validation_runs", {}).get(run_id)
    if val_run:
        return {
            "type": "validation",
            "run": val_run,
            "classification_run": storage.get_classification_run(
                val_run["classification_run_id"]
            ).to_dict() if val_run.get("classification_run_id") else None
        }
    
    raise ValueError(f"Run {run_id} not found")


# Convenience function for loading config from file
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return _load_config(config_path)