# example_run.py

from text_classifier.api import (
    classify_texts,
    validate_classification,
    load_classification_results,
    load_validation_results
)

# 1. Create a sample CSV file for testing
import pandas as pd

# Sample survey data
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'response': [
        "The product quality is excellent and shipping was fast",
        "Terrible customer service, very disappointed",
        "Average product, nothing special but works fine",
        "Love it! Best purchase I've made this year",
        "Broken on arrival, had to return it",
        "Good value for money, would recommend",
        "Not sure how I feel about it yet",
        "The color is different from the picture",
        "Amazing! Exceeded my expectations",
        "Waste of money, complete garbage"
    ]
}

df = pd.DataFrame(data)
df.to_csv('sample_responses.csv', index=False)

# 2. Set up configuration
config = {
    "file_path": "sample_responses.csv",
    "text_column": "response",
    "id_column": "id",
    "classifier_model": "gemma3n:latest",
    "category_model": "gemma3n:latest",  # For category generation
    "judge_model": "gemma3n:latest",     # For validation
    "classifier_backend": "ollama",
    "question_context": "What do you think about the product you purchased?"
}

# 3. Run classification
print("=== Running Classification ===")
run_id = classify_texts(config)
print(f"\nClassification completed with run ID: {run_id}")

# 4. Load and display results
results_df = load_classification_results(run_id)
print("\n=== Classification Results ===")
print(results_df[['id', 'response', 'category']].to_string())

# 5. Run validation on the classification
print("\n\n=== Running Validation ===")
validation_id = validate_classification(
    classification_run_id=run_id,
    sample_size=5  # Validate 5 samples for speed
)
print(f"\nValidation completed with ID: {validation_id}")

# 6. Load and display validation results
val_results = load_validation_results(validation_id)
print("\n=== Validation Results ===")
print(val_results[['response', 'category', 'quality_score']].to_string())
print(f"\nAverage quality score: {val_results['quality_score'].mean():.2f}")