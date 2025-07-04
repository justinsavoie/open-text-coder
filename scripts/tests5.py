from text_classifier import classify_texts

# Minimal configuration - let the system figure out categories
config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "n_samples": 10
}

# Run classification
run_id = classify_texts(config)
print(f"Classification complete! Run ID: {run_id}")

# Use your own categories
config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "categories": ["Economy", "Environment", "Healthcare", "Other"],
    "n_samples": 10
}

run_id = classify_texts(config)

config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "categories": ["Economy", "Environment", "Healthcare", "Other"],
    "n_samples": 10,
    "classifier_backend" : 'openai',
    "classifier_model": 'gpt-4.1'
}

run_id = classify_texts(config)

config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "categories": ["Economy", "Environment", "Healthcare", "Other"],
    "n_samples": 10,
    "multiclass": True,
    "classifier_model": 'granite3.3:8b'
}

run_id = classify_texts(config)

# Load results
from text_classifier import load_classification_results
results = load_classification_results(run_id)
# Results will have columns: review_id, review_text, Quality Issues, Shipping Problems, etc.
# Each category column contains "yes" or "no"