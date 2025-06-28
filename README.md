# Text Classification System

A flexible system for classifying open-text survey responses using LLMs, with built-in validation and experiment tracking.

## Features

- **Multiple LLM backends**: Support for Ollama and OpenAI models
- **Automatic category generation**: Can generate categories from sample responses
- **Single and multi-label classification**: Assign one or multiple categories per response
- **Built-in validation**: Use LLM-as-judge to validate classification quality
- **Experiment tracking**: All runs are saved with metadata for comparison
- **Flexible configuration**: Override any parameter programmatically

## Installation

```bash
# Clone the repository
git clone <your-repo>
cd text-classifier

# Install dependencies
pip install pandas tqdm ollama

# Optional: For OpenAI support
pip install openai
export OPENAI_API_KEY="your-key-here"
```

## Project Structure

```
your_project/
├── text_classifier/
│   ├── __init__.py
│   ├── models.py          # Data models (ClassificationRun, ValidationRun)
│   ├── storage.py         # Data storage and retrieval (RunStorage)
│   ├── classifier.py      # Classification logic (TextClassifier)
│   ├── validator.py       # Validation logic (ClassificationValidator)
│   ├── config.py          # Configuration utilities
│   └── api.py            # High-level API functions
├── analysis.py        # Your analysis scripts
├── config.json        # Default configuration
├── README.md         # This file
├── data-cps21.csv    # Your data files
└── runs/             # Storage directory (created automatically)
```

## Quick Start

### 1. Basic Classification

```python
from text_classifier.api import classify_texts, validate_classification

# Option 1: Use config file
config = {
    "file_path": "data-cps21.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "classifier_model": "gemma3n:latest",
    "backend": "ollama"
}

# Run classification
run_id = classify_texts(config)
print(f"Classification complete! Run ID: {run_id}")

# Validate the results
val_id = validate_classification(run_id)
```

### 2. Generate Categories Automatically

```python
config = {
    "file_path": "survey_responses.csv",
    "text_column": "response",
    "id_column": "id",
    "n_samples": 100,  # Sample size for category generation
    "question_context": "What is most important to you?",
    "category_model": "cogito:14b",  # Model for generating categories
    "classifier_model": "gemma3n:latest"  # Model for classification
}

run_id = classify_texts(config)
```

### 3. Use Specific Categories

```python
config = {
    "file_path": "data.csv",
    "text_column": "response",
    "id_column": "id",
    "categories": ["Healthcare", "Economy", "Education", "Environment", "Other"],
    "classifier_model": "gpt-3.5-turbo",
    "backend": "openai"
}

run_id = classify_texts(config)
```

### 4. Multi-label Classification

```python
config = {
    "file_path": "data.csv",
    "text_column": "response", 
    "id_column": "id",
    "multiclass": True,  # Enable multi-label
    "categories": ["Urgent", "Important", "Complex", "Political", "Personal"]
}

run_id = classify_texts(config)
# Output will have columns for each category with "yes"/"no" values
```

### 5. Compare Multiple Runs

```python
from text_classifier import compare_runs, classify_texts

# Try different models
models = ["gemma3n:latest", "llama2", "mistral"]
run_ids = []

for model in models:
    config["classifier_model"] = model
    run_id = classify_texts(config)
    run_ids.append(run_id)

# Compare results
compare_runs(run_ids)
```

### 6. Advanced Analysis Script

```python
# analysis.py
from text_classifier import (
    classify_texts, 
    validate_classification,
    load_classification_results,
    list_all_runs
)
import json

# Load base config
with open("config.json") as f:
    base_config = json.load(f)

# Experiment 1: Compare backends
for backend in ["ollama", "openai"]:
    config = base_config.copy()
    config["backend"] = backend
    config["classifier_model"] = "gpt-3.5-turbo" if backend == "openai" else "gemma3n:latest"
    
    run_id = classify_texts(config)
    validate_classification(run_id, sample_size=50)

# Experiment 2: Test category generation
config = base_config.copy()
config.pop("categories", None)  # Remove predefined categories
config["n_samples"] = 200  # Use more samples

run_id = classify_texts(config)

# Load and analyze results
df = load_classification_results(run_id)
print(df["category"].value_counts())

# List all runs
all_runs = list_all_runs()
for run in all_runs:
    print(f"{run['run_id']}: {run['timestamp']} - {run['metrics']}")
```

## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `file_path` | str | Path to input CSV file | Required |
| `text_column` | str | Column containing text to classify | Required |
| `id_column` | str | Column with unique identifiers | Required |
| `categories` | list/str | Predefined categories (comma-separated string or list) | None (auto-generate) |
| `multiclass` | bool | Enable multi-label classification | False |
| `classifier_model` | str | Model for classification | "gemma3n:latest" |
| `category_model` | str | Model for generating categories | Same as classifier_model |
| `backend` | str | LLM backend ("ollama" or "openai") | "ollama" |
| `n_samples` | int | Sample size for category generation | 100 |
| `question_context` | str | Survey question for context | "" |
| `validation_samples` | int | Sample size for validation | None (validate all) |
| `judge_model` | str | Model for validation | "gemma3n:latest" |

## API Reference

### Main Functions

```python
# Classify texts
run_id = classify_texts(
    config: dict,
    run_id: str = None,  # Optionally specify run ID
    storage_dir: Path = Path("./runs")
) -> str

# Validate classification
val_id = validate_classification(
    classification_run_id: str,
    config: dict = None,  # Override validation settings
    sample_size: int = None,
    storage_dir: Path = Path("./runs")
) -> str

# Load results
df = load_classification_results(
    run_id: str,
    storage_dir: Path = Path("./runs")
) -> pd.DataFrame

# Compare runs
comparison_df = compare_runs(
    run_ids: List[str],
    storage_dir: Path = Path("./runs")
) -> pd.DataFrame

# List all runs
runs = list_all_runs(
    run_type: str = "classification",  # or "validation"
    storage_dir: Path = Path("./runs")
) -> List[dict]
```

### Lower-Level Access

```python
from text_classifier.classifier import TextClassifier
from text_classifier.storage import RunStorage

# Direct classifier usage
classifier = TextClassifier("gemma3n:latest", "ollama")
category = classifier.classify_single("This is about healthcare", ["Health", "Economy", "Other"])

# Direct storage access
storage = RunStorage()
run = storage.get_classification_run("20240115_143022_a1b2c3d4")
df = storage.load_classification_data(run.run_id)
```

## Output Structure

Each classification run creates:
```
runs/
├── metadata.json                                    # Global run registry
├── classification_20240115_143022_a1b2c3d4/
│   ├── config.json                                 # Run configuration
│   └── classified_20240115_143022_a1b2c3d4.csv   # Results
└── validation_20240115_144512_e5f6g7h8/
    └── validation_20240115_144512_e5f6g7h8.csv    # Validation results
```

Classification output columns:
- Single-class: `id_column`, `text_column`, `category`
- Multi-class: `id_column`, `text_column`, `category1`, `category2`, ...

Validation output columns:
- All classification columns plus: `quality_score` (1-5), `explanation`

## Tips

1. **Start with small samples**: Test with `validation_samples=20` to quickly iterate
2. **Compare models**: Always try 2-3 different models to see which works best
3. **Refine categories**: If auto-generated categories aren't good, manually specify them
4. **Check validation scores**: Average scores below 3.5 suggest classification issues
5. **Use appropriate models**: 
   - Larger models (70B+) for category generation
   - Smaller, faster models for classification
   - High-quality models for validation

## Troubleshooting

**Import errors**: Make sure the text_classifier package is in your Python path:
```python
# If running from the project root
import sys
sys.path.append('.')  # Add current directory to path
```

**Ollama connection errors**: Ensure Ollama is running:
```bash
ollama serve
```

**OpenAI errors**: Check your API key:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

**Memory issues**: Reduce sample sizes or process in batches:
```python
# Process large files in chunks
chunk_size = 1000
for i in range(0, len(df), chunk_size):
    chunk_df = df.iloc[i:i+chunk_size]
    # Process chunk...
```

**Package not found errors**: Install missing packages:
```bash
pip install pandas tqdm ollama
# Optional: pip install openai
```