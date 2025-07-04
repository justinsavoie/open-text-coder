# 1. Basic Classification with Auto-Generated Categories

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

# 2. Classification with Predefined Categories

# Use your own categories - of course these one are non-sensical
# Will still code non-sensically
config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "categories": ["Technical Issue", "Billing", "Feature Request", "Other"],
    "question_context": "Customer support ticket"
}

run_id = classify_texts(config)

config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "categories": ["Environment", "Health", "Economy", "Culture and multiculturalism","Other"],
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
}

run_id = classify_texts(config)

# 3. Multi-label Classification

# Each response can belong to multiple categories
config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "multiclass": True,
    "categories": ["Environment", "Health", "Economy", "Culture and multiculturalism","Other"],
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
}

run_id = classify_texts(config)

# Load results
from text_classifier import load_classification_results
results = load_classification_results(run_id)

# 4. Generate Categories Without Classification

from text_classifier import generate_categories_only

# Explore what categories the LLM finds in your data
config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "n_samples": 10,  # Use 200 samples to generate categories
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
}

categories = generate_categories_only(config)
print("Discovered categories:", categories)

# Load saved categories later
from text_classifier import load_saved_categories
categories, metadata = load_saved_categories("runs/categories_20250703_224833.json")

# 5. Validate Classification Quality

config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "n_samples": 10,  # Use 200 samples to generate categories
    "judge_model": "gpt-4o",
    "judge_backend": "openai",
}

from text_classifier import validate_classification

# Validate a previous classification run
validation_id = validate_classification(
    classification_run_id='20250702_220000_f9bee455',
    sample_size=10,
    config=config,
)

# Load validation results
from text_classifier import load_validation_results
val_results = load_validation_results(validation_id)
print(f"Average quality score: {val_results['quality_score'].mean():.2f}/5.0")

# 6. Using Different Models

config = {
    "file_path": "data/data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "n_samples": 10,
    "category_model": "gpt-4o",
    "category_backend": "openai",
    "classifier_model": "gemma3n:latest",
    "classifier_backend": "ollama"
}

run_id = classify_texts(config)

# 7. Compare Multiple Runs

from text_classifier import compare_runs, list_all_runs

# List all classification runs
runs = list_all_runs("classification")
for run in runs[:5]:  # Show last 5 runs
    print(f"{run['run_id']}: {run['timestamp']}")

# Compare specific runs
comparison = compare_runs([
    "20250703_225707_9b61aa81",
    "20250703_224627_db64ffea"
])

# 8. Load Configuration from File
# config.json
#{
#    "file_path": "responses.csv",
#    "text_column": "answer",
#    "id_column": "id",
#    "categories": ["Satisfied", "Neutral", "Dissatisfied"],
#    "question_context": "How satisfied are you with our service?",
#    "classifier_model": "gemma3:latest",
#    "n_samples": 10
#}
#from text_classifier import load_config, classify_texts#
#config = load_config("config.json")
#run_id = classify_texts(config)

# 9. Hybrid classification using SetFit

# Example 1: Basic hybrid classification
from text_classifier import classify_texts_hybrid

config = {
    "file_path": "data/data-cps21-40.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "max_llm_samples": 30,  # Use LLM for up to 300 training samples
    "confidence_threshold": 0.9,  # High confidence required for SetFit
    "categories": [
    "Economy, Jobs & Inflation",
    "Healthcare & Senior Care",
    "Climate Change & Environment",
    "COVID-19 Pandemic Response & Preparedness",
    "Housing Affordability & Homelessness",
    "Indigenous Peoples & Reconciliation",
    "Government Leadership, Ethics & Accountability",
    "National Unity, Quebec & Regional Issues",
    "Law, Order & Public Safety",
    "Social Programs, Education & Childcare",
    "Immigration & Multiculturalism",
    "Foreign Policy, Defence & International Relations",
    "No Opinion / Uncertain"]
}

run_id = classify_texts_hybrid(config)

# Example 2: Load and reuse trained model
from text_classifier import load_setfit_model, load_classification_results

# Load the trained model
setfit_model = load_setfit_model(run_id)

# Use it directly for new data
new_texts = ["This product is amazing!", "Terrible experience"]
predictions = setfit_model.predict_batch(new_texts, return_proba=True)
for text, (category, confidence) in zip(new_texts, predictions):
    print(f"{text} -> {category} (confidence: {confidence:.2f})")

# Example 3: Check which samples used LLM vs SetFit
results = load_classification_results(run_id)
print("Classification sources:")
print(results['source'].value_counts())
print("\nLow confidence samples that used LLM:")
low_conf = results[results['source'] == 'llm']
print(low_conf.head())
high_conf = results[results['source'] == 'setfit']
print(high_conf.head())


