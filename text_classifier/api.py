# text_classifier/api.py
"""
High-level API for text classification system.
This module provides the main functions users should interact with.
"""

from pathlib import Path
from datetime import datetime
import uuid
import json
from typing import Dict, Any, Optional, List, Union, Tuple  # Added Union and Tuple
import pandas as pd

from .models import ClassificationRun, ValidationRun
from .storage import RunStorage
from .classifier import TextClassifier
from .validator import ClassificationValidator
from .setfit_classifier import SetFitClassifier, HybridClassifier, SETFIT_AVAILABLE
from .translator import translate_dataset, TRANSLATION_AVAILABLE
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
    """
    config = get_config_with_defaults(config)
    validate_config(config)
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    
    storage = RunStorage(storage_dir)
    
    df = pd.read_csv(config["file_path"],keep_default_na=False)

    original_count = len(df)
    df = df[df[config["text_column"]].str.len() >= 1]
    df = df.reset_index(drop=True) # Reset index here as well for consistency
    dropped_count = original_count - len(df)
    
    if dropped_count > 0:
        print(f"[*] Dropped {dropped_count} rows with empty text")

    categories = parse_categories(config)
    
    classifier = TextClassifier(
        config["classifier_model"],
        config["classifier_backend"]
    )
    
    classified_df, final_categories, metrics = classifier.run_classification(
        df=df,
        text_column=config["text_column"],
        id_column=config["id_column"],
        categories=categories,
        multiclass=config["multiclass"],
        n_samples=config["n_samples"],
        question_context=config["question_context"],
        category_model=config["category_model"],
        category_backend=config["category_backend"]
    )

    metrics["dropped_empty_rows"] = dropped_count
    metrics["original_rows"] = original_count    
    
    run = ClassificationRun(
        run_id=run_id,
        timestamp=datetime.now(),
        config=config,
        input_file=Path(config["file_path"]),
        categories=final_categories,
        metrics=metrics
    )
    
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
    Supports both single category and multiclass validation.
    
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
        config["judge_model"],
        config["judge_backend"]  # Changed from config.get("backend", "ollama")
    )
    
    # Determine if multiclass
    multiclass = class_run.config.get("multiclass", False)
    
    # Run validation
    if multiclass:
        # For multiclass, we don't have a single category column
        validated_df, metrics = validator.validate_classification_run(
            df=classified_df,
            text_column=class_run.config["text_column"],
            category_column=None,
            categories=class_run.categories,
            sample_size=config.get("validation_samples"),
            multiclass=True
        )
    else:
        # Single category validation
        validated_df, metrics = validator.validate_classification_run(
            df=classified_df,
            text_column=class_run.config["text_column"],
            category_column="category",
            categories=class_run.categories,
            sample_size=config.get("validation_samples"),
            multiclass=False
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
        "metrics": metrics,
        "multiclass": multiclass
    }
    
    storage.metadata["validation_runs"][val_id] = val_run
    storage._save_metadata()
    
    print(f"[*] Validation complete. ID: {val_id}")
    print(f"[*] Average quality score: {metrics['average_score']:.2f}")
    
    if multiclass and "category_scores" in metrics:
        print(f"[*] Per-category scores:")
        for cat, score in metrics["category_scores"].items():
            print(f"    - {cat}: {score:.2f}")
    
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

def generate_categories_only(
    config: Dict[str, Any],
    save_to_file: bool = True,
    storage_dir: Path = Path("./runs")
) -> List[str]:
    """
    Generate categories from data without running classification.
    
    Args:
        config: Configuration dictionary with keys:
            - file_path: Path to input CSV
            - text_column: Column containing text
            - n_samples: Number of samples for category generation (default: 100)
            - question_context: Context for category generation
            - category_model: Model for generating categories (default: classifier_model)
            - category_backend: Backend for category generation (default: classifier_backend)
        save_to_file: Whether to save categories to a JSON file
        storage_dir: Directory to save categories file
        
    Returns:
        List of generated categories
        
    Example:
        >>> config = {
        ...     "file_path": "survey.csv",
        ...     "text_column": "response",
        ...     "question_context": "What features would you like to see?"
        ... }
        >>> categories = generate_categories_only(config)
        >>> print(categories)
        ['Feature Request', 'Bug Report', 'Positive Feedback', ...]
    """
    # Apply defaults but only validate required fields for category generation
    config_with_defaults = get_config_with_defaults(config)
    
    # Minimal validation - we don't need id_column for category generation
    required = ["file_path", "text_column"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config parameters: {missing}")
    
    # Validate file exists
    file_path = Path(config_with_defaults["file_path"])  # Use config_with_defaults
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Load data
    print(f"[*] Loading data from {file_path}")
    df = pd.read_csv(file_path, keep_default_na=False)
    
    # Drop empty rows
    original_count = len(df)
    df = df[df[config_with_defaults["text_column"]].str.len() >= 1]  # Use config_with_defaults
    dropped_count = original_count - len(df)
    
    if dropped_count > 0:
        print(f"[*] Dropped {dropped_count} rows with empty text")
    
    print(f"[*] Total rows available: {len(df)}")
    
    # Initialize classifier for category generation
    classifier = TextClassifier(
        config_with_defaults.get("category_model") or config_with_defaults["classifier_model"],
        config_with_defaults.get("category_backend") or config_with_defaults["classifier_backend"]
    )
    
    # Generate categories
    categories = classifier.generate_categories(
        df=df,
        text_column=config_with_defaults["text_column"],  # Use config_with_defaults
        n_samples=config_with_defaults.get("n_samples", 100),
        question_context=config_with_defaults.get("question_context", "")
    )
    
    print(f"\n[*] Generated {len(categories)} categories:")
    for i, cat in enumerate(categories, 1):
        print(f"    {i}. {cat}")
    
    # Save categories if requested
    if save_to_file:
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cat_file = storage_dir / f"categories_{timestamp}.json"
        
        save_data = {
            "categories": categories,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_file": str(file_path),
                "text_column": config_with_defaults["text_column"],  # Use config_with_defaults
                "n_samples": config_with_defaults.get("n_samples", 100),
                "question_context": config_with_defaults.get("question_context", ""),
                "model": config_with_defaults.get("category_model") or config_with_defaults["classifier_model"],
                "backend": config_with_defaults.get("category_backend") or config_with_defaults["classifier_backend"],
                "total_rows": len(df),
                "dropped_empty_rows": dropped_count
            }
        }
        
        with open(cat_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n[*] Categories saved to: {cat_file}")
    
    return categories

def load_saved_categories(
    categories_file: Union[str, Path]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Load categories from a previously saved categories file.
    
    Args:
        categories_file: Path to the categories JSON file
        
    Returns:
        Tuple of (categories list, metadata dict)
        
    Example:
        >>> categories, metadata = load_saved_categories("./runs/categories_20231230_143022.json")
        >>> print(f"Loaded {len(categories)} categories generated on {metadata['timestamp']}")
    """
    categories_file = Path(categories_file)
    
    if not categories_file.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_file}")
    
    with open(categories_file) as f:
        data = json.load(f)
    
    return data["categories"], data.get("metadata", {})

# Convenience function for loading config from file
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return _load_config(config_path)

# Add to imports at the top of api.py
from .setfit_classifier import SetFitClassifier, HybridClassifier, SETFIT_AVAILABLE

# Add new function after the existing ones
def classify_texts_hybrid(
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    storage_dir: Path = Path("./runs"),
    train_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Run text classification using hybrid LLM + SetFit approach.
    Supports both single-label and multi-label classification.
    """
    if not SETFIT_AVAILABLE:
        raise ImportError("SetFit not available. Run: pip install setfit sentence-transformers")
    
    config = get_config_with_defaults(config)
    validate_config(config)
    
    is_multiclass = config.get("multiclass", False)
    
    setfit_defaults = {
        "use_setfit": True,
        "setfit_model": "sentence-transformers/paraphrase-mpnet-base-v2",
        "confidence_threshold": 0.5,
        "max_llm_samples": 200,
        "min_samples_per_category": 10
    }
    
    for key, value in setfit_defaults.items():
        if key not in config:
            config[key] = value
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    
    storage = RunStorage(storage_dir)
    run_dir = storage.base_dir / f"classification_{run_id}"
    run_dir.mkdir(exist_ok=True)
    
    df = pd.read_csv(config["file_path"], keep_default_na=False)
    original_count = len(df)
    df = df[df[config["text_column"]].str.len() >= 1]
    
    # *** THE FIX IS HERE ***
    # Reset the index after filtering to ensure it's a clean, sequential 0, 1, 2...
    df = df.reset_index(drop=True)
    
    dropped_count = original_count - len(df)
    
    if dropped_count > 0:
        print(f"[*] Dropped {dropped_count} rows with empty text")
    
    categories = parse_categories(config)
    
    llm_classifier = TextClassifier(
        config["classifier_model"],
        config["classifier_backend"]
    )
    
    if categories is None:
        cat_classifier = TextClassifier(
            model_name=config.get("category_model") or config["classifier_model"],
            backend=config.get("category_backend") or config["classifier_backend"]
        )
        categories = cat_classifier.generate_categories(
            df, config["text_column"], config["n_samples"], config["question_context"]
        )
        print(f"[*] Generated categories: {', '.join(categories)}")
    
    hybrid = HybridClassifier(
        llm_classifier,
        config["setfit_model"],
        config["confidence_threshold"],
        config["min_samples_per_category"],
        config["max_llm_samples"]
    )
    
    print("\n=== Phase 1: Collecting training data with LLM ===")
    training_df, training_metrics = hybrid.collect_training_data(
        df.sample(frac=1, random_state=42),
        config["text_column"],
        config["id_column"],
        categories,
        config["question_context"],
        multiclass=is_multiclass
    )
    
    training_file = run_dir / "training_data.csv"
    training_df.to_csv(training_file, index=False)
    
    print("\n=== Phase 2: Training SetFit model ===")
    setfit_metrics = hybrid.train_setfit(multiclass=is_multiclass)
    
    setfit_dir = run_dir / "setfit_model"
    hybrid.setfit.save(setfit_dir)
    
    print("\n=== Phase 3: Hybrid classification ===")
    classified_df, classification_metrics = hybrid.classify_hybrid(
        df,
        config["text_column"],
        config["id_column"],
        config["question_context"],
        multiclass=is_multiclass
    )
    
    metrics = {
        "total_rows": len(df),
        "dropped_empty_rows": dropped_count,
        "original_rows": original_count,
        "training_metrics": training_metrics,
        "setfit_metrics": setfit_metrics,
        "classification_metrics": classification_metrics,
        "hybrid_mode": True,
        "num_categories": len(categories)
    }
    
    run = ClassificationRun(
        run_id=run_id,
        timestamp=datetime.now(),
        config=config,
        input_file=Path(config["file_path"]),
        categories=categories,
        metrics=metrics
    )
    
    output_file = storage.save_classification_run(run, classified_df)
    
    print(f"\n[*] Hybrid classification complete. Run ID: {run_id}")
    print(f"[*] Results saved to: {output_file}")
    print(f"[*] LLM used for: {classification_metrics['llm_percentage']:.1f}% of classifications")
    
    return run_id

def load_setfit_model(
    run_id: str,
    storage_dir: Path = Path("./runs")
) -> SetFitClassifier:
    """
    Load a trained SetFit model from a previous run.
    
    Args:
        run_id: Classification run ID that used SetFit
        storage_dir: Directory where runs are stored
        
    Returns:
        Loaded SetFitClassifier instance
    """
    run_dir = storage_dir / f"classification_{run_id}" / "setfit_model"
    
    if not run_dir.exists():
        raise ValueError(f"No SetFit model found for run {run_id}")
    
    classifier = SetFitClassifier()
    classifier.load(run_dir)
    
    return classifier