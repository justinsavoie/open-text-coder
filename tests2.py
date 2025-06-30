from text_classifier.api import (
    classify_texts,
    validate_classification,
    load_validation_results
)

# --- Classification Configs ---
base_config = {
    "file_path": "data-cps21.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "classifier_model": "cogito:14b",
    "classifier_backend": "ollama",
    "categories": ["Healthcare & Pandemic Management", "Economy Jobs & Cost of Living", "Environment & Climate Change", "Government Leadership & Accountability", "Democratic & Electoral Reform", "Housing Affordability & Availability", "Indigenous Relations & Reconciliation", "Social Cohesion & Inclusion", "Foreign Policy & National Unity", "Education & Skills Training", "Don’t Know / Uncertain"],
}

# Classify with local model
gemma_config = base_config.copy()
run_id_gemma = classify_texts(gemma_config)

from text_classifier.api import (
    classify_texts,
    validate_classification,
    load_validation_results
)

# --- Classification Configs ---
base_config = {
    "file_path": "data-cps21.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "classifier_model": "cogito:14b",
    "category_model": "deepseek-r1:14b",
    "classifier_backend": "ollama",
    "category_backend": "ollama",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)"
}

# Classify with local model
gemma_config = base_config.copy()
run_id_gemma = classify_texts(gemma_config)
