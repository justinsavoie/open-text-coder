# Text Classification System

A flexible system for classifying open-text survey responses using LLMs, with built-in validation and experiment tracking.

## Features

- **Multiple LLM backends**: Support for Ollama and OpenAI models
- **Automatic category generation**: Let the LLM discover categories from your data
- **Manual category specification**: Use predefined categories for consistent classification
- **Multi-label classification**: Assign multiple categories to a single response
- **Built-in validation**: Use LLM-as-judge to assess classification quality
- **Experiment tracking**: Automatic storage and versioning of all runs
- **Robust error handling**: Retry logic and graceful degradation
- **Empty response handling**: Automatic filtering of empty/invalid responses

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/text-classifier.git
cd text-classifier

# Install dependencies
pip install pandas tqdm ollama openai

# For OpenAI support, set your API key
export OPENAI_API_KEY="your-api-key-here"
```

## Project Structure

```
text_classifier/
├── __init__.py          # Package initialization and exports
├── api.py               # High-level API functions
├── classifier.py        # Core classification logic
├── validator.py         # Classification validation using LLM-as-judge
├── config.py            # Configuration management
├── models.py            # Data models for runs and validations
└── storage.py           # Run storage and retrieval

runs/                    # Created automatically to store results
├── classification_*/    # Classification run folders
├── validation_*/        # Validation run folders
└── categories_*.json    # Generated category files
```

## Quick Start

### 1. Basic Classification with Auto-Generated Categories

```python
from text_classifier import classify_texts

# Minimal configuration - let the system figure out categories
config = {
    "file_path": "survey_responses.csv",
    "text_column": "response",
    "id_column": "respondent_id"
}

# Run classification
run_id = classify_texts(config)
print(f"Classification complete! Run ID: {run_id}")
```

### 2. Classification with Predefined Categories

```python
# Use your own categories
config = {
    "file_path": "support_tickets.csv",
    "text_column": "ticket_text",
    "id_column": "ticket_id",
    "categories": ["Technical Issue", "Billing", "Feature Request", "Other"],
    "question_context": "Customer support ticket"
}

run_id = classify_texts(config)
```

### 3. Multi-label Classification

```python
# Each response can belong to multiple categories
config = {
    "file_path": "product_reviews.csv",
    "text_column": "review_text",
    "id_column": "review_id",
    "multiclass": True,
    "categories": ["Quality Issues", "Shipping Problems", "Good Value", "Easy to Use"],
    "question_context": "Product review"
}

run_id = classify_texts(config)

# Load results
from text_classifier import load_classification_results
results = load_classification_results(run_id)
# Results will have columns: review_id, review_text, Quality Issues, Shipping Problems, etc.
# Each category column contains "yes" or "no"
```

### 4. Generate Categories Without Classification

```python
from text_classifier import generate_categories_only

# Explore what categories the LLM finds in your data
config = {
    "file_path": "survey.csv",
    "text_column": "comments",
    "n_samples": 200,  # Use 200 samples to generate categories
    "question_context": "What features would you like to see in our app?"
}

categories = generate_categories_only(config)
print("Discovered categories:", categories)

# Load saved categories later
from text_classifier import load_saved_categories
categories, metadata = load_saved_categories("./runs/categories_20231230_143022.json")
```

### 5. Validate Classification Quality

```python
from text_classifier import validate_classification

# Validate a previous classification run
validation_id = validate_classification(
    classification_run_id=run_id,
    sample_size=50  # Validate 50 random samples
)

# Load validation results
from text_classifier import load_validation_results
val_results = load_validation_results(validation_id)
print(f"Average quality score: {val_results['quality_score'].mean():.2f}/5.0")
```

### 6. Using Different Models

```python
# Use GPT-4 for category generation, GPT-3.5 for classification
config = {
    "file_path": "data.csv",
    "text_column": "text",
    "id_column": "id",
    "category_model": "gpt-4",
    "category_backend": "openai",
    "classifier_model": "gpt-3.5-turbo",
    "classifier_backend": "openai",
    "judge_model": "gpt-4",  # For validation
    "judge_backend": "openai"
}

run_id = classify_texts(config)
```

### 7. Compare Multiple Runs

```python
from text_classifier import compare_runs, list_all_runs

# List all classification runs
runs = list_all_runs("classification")
for run in runs[:5]:  # Show last 5 runs
    print(f"{run['run_id']}: {run['timestamp']}")

# Compare specific runs
comparison = compare_runs([
    "20231230_143022_abc123",
    "20231230_150000_def456"
])
```

### 8. Load Configuration from File

```python
# config.json
{
    "file_path": "responses.csv",
    "text_column": "answer",
    "id_column": "id",
    "categories": ["Satisfied", "Neutral", "Dissatisfied"],
    "question_context": "How satisfied are you with our service?",
    "classifier_model": "gemma2:latest",
    "n_samples": 100
}
```

```python
from text_classifier import load_config, classify_texts

config = load_config("config.json")
run_id = classify_texts(config)
```

### 9. Hybrid classification using SetFit

```python
# Example 1: Basic hybrid classification
from text_classifier import classify_texts_hybrid

config = {
    "file_path": "survey_responses.csv",
    "text_column": "response",
    "id_column": "id",
    "question_context": "What do you think about our product?",
    # SetFit specific
    "max_llm_samples": 300,  # Use LLM for up to 300 training samples
    "confidence_threshold": 0.9,  # High confidence required for SetFit
}

run_id = classify_texts_hybrid(config)

# Example 2: With predefined categories
config = {
    "file_path": "large_dataset.csv",
    "text_column": "feedback",
    "id_column": "id",
    "categories": ["Positive", "Negative", "Neutral", "Feature Request", "Bug Report"],
    "max_llm_samples": 500,  # More training data for 5 categories
    "min_samples_per_category": 20,  # Ensure good coverage
}

run_id = classify_texts_hybrid(config)

# Example 3: Load and reuse trained model
from text_classifier import load_setfit_model, load_classification_results

# Load the trained model
setfit_model = load_setfit_model(run_id)

# Use it directly for new data
new_texts = ["This product is amazing!", "Terrible experience"]
predictions = setfit_model.predict_batch(new_texts, return_proba=True)
for text, (category, confidence) in zip(new_texts, predictions):
    print(f"{text} -> {category} (confidence: {confidence:.2f})")

# Example 4: Check which samples used LLM vs SetFit
results = load_classification_results(run_id)
print("Classification sources:")
print(results['source'].value_counts())
print("\nLow confidence samples that used LLM:")
low_conf = results[results['source'] == 'llm']
print(low_conf[['text', 'category', 'confidence']].head())
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **file_path** | str | *required* | Path to input CSV file |
| **text_column** | str | *required* | Column name containing text to classify |
| **id_column** | str | *required* | Column name with unique identifiers |
| **categories** | list/str | None | Predefined categories (auto-generated if None) |
| **classifier_model** | str | "gemma3n:latest" | Model for classification |
| **classifier_backend** | str | "ollama" | Backend: "ollama" or "openai" |
| **category_model** | str | *classifier_model* | Model for category generation |
| **category_backend** | str | *classifier_backend* | Backend for category generation |
| **judge_model** | str | "gemma3n:latest" | Model for validation |
| **judge_backend** | str | *classifier_backend* | Backend for validation |
| **multiclass** | bool | False | Enable multi-label classification |
| **n_samples** | int | 100 | Samples to use for category generation |
| **question_context** | str | "" | Original survey question for context |
| **validation_samples** | int | None | Number of samples to validate (None = all) |
| **max_retries** | int | 3 | Maximum retries for API calls |

## API Reference

### Core Functions

#### `classify_texts(config, run_id=None, storage_dir="./runs")`
Run text classification on a dataset.

**Returns:** `str` - Unique run identifier

#### `validate_classification(classification_run_id, config=None, sample_size=None, storage_dir="./runs")`
Validate classification quality using LLM-as-judge.

**Returns:** `str` - Unique validation identifier

#### `generate_categories_only(config, save_to_file=True, storage_dir="./runs")`
Generate categories without running classification.

**Returns:** `List[str]` - List of generated categories

### Data Loading Functions

#### `load_classification_results(run_id, storage_dir="./runs")`
Load classification results as a pandas DataFrame.

#### `load_validation_results(validation_id, storage_dir="./runs")`
Load validation results as a pandas DataFrame.

#### `load_saved_categories(categories_file)`
Load previously generated categories from JSON file.

**Returns:** `Tuple[List[str], Dict]` - (categories, metadata)

### Analysis Functions

#### `compare_runs(run_ids, storage_dir="./runs")`
Compare metrics across multiple classification runs.

**Returns:** `pd.DataFrame` - Comparison table

#### `list_all_runs(run_type="classification", storage_dir="./runs")`
List all runs of specified type.

**Returns:** `List[Dict]` - List of run metadata

#### `get_run_info(run_id, storage_dir="./runs")`
Get detailed information about a specific run.

**Returns:** `Dict` - Complete run information

### Utility Functions

#### `load_config(config_path="config.json")`
Load configuration from JSON file.

**Returns:** `Dict` - Configuration dictionary

## Advanced Usage

### Custom Validation Logic

```python
# Run validation with custom configuration
val_config = {
    "judge_model": "gpt-4",
    "judge_backend": "openai",
    "validation_samples": 100
}

validation_id = validate_classification(
    run_id,
    config=val_config
)
```

### Batch Processing

```python
import glob

# Process multiple files with same configuration
base_config = {
    "text_column": "feedback",
    "id_column": "id",
    "categories": ["Positive", "Negative", "Neutral", "Bug Report", "Feature Request"]
}

for file in glob.glob("data/*.csv"):
    config = base_config.copy()
    config["file_path"] = file
    
    run_id = classify_texts(config)
    print(f"Processed {file}: {run_id}")
```

### Export Results

```python
# Combine classification and validation results
run_id = "20231230_143022_abc123"
class_results = load_classification_results(run_id)
val_results = load_validation_results(validation_id)

# Merge on ID column
combined = class_results.merge(
    val_results[['id', 'quality_score', 'explanation']], 
    on='id'
)

# Export high-quality classifications
high_quality = combined[combined['quality_score'] >= 4]
high_quality.to_csv("high_quality_classifications.csv", index=False)
```

## Tips and Best Practices

1. **Start with category generation**: Use `generate_categories_only()` to explore your data before running full classification.

2. **Provide context**: Always include `question_context` - it significantly improves classification accuracy.

3. **Validate samples**: For large datasets, validate a representative sample rather than all results.

4. **Use appropriate models**: 
   - Larger models (GPT-4, Claude) for category generation
   - Smaller, faster models for classification
   - Best available model for validation

5. **Monitor quality**: Check validation scores and adjust categories or models as needed.

6. **Handle edge cases**: The system automatically filters empty responses, but review the `dropped_empty_rows` metric.

## License

MIT License - see LICENSE file for details.