from text_classifier import classify_texts

# Minimal configuration - let the system figure out categories
config = {
    "file_path": "data-cps21-10.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "n_samples": 10
}

# Run classification
run_id = classify_texts(config)

config = {
    "file_path": "data-cps21.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
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
    "No Opinion / Uncertain"],
    "n_samples": 200,
    "classifier_backend" : 'openai',
    "classifier_model": 'gpt-4.1'
}
run_id = classify_texts(config)