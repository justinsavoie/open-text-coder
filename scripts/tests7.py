from text_classifier import classify_texts

# Minimal configuration - let the system figure out categories
config = {
    "file_path": "data/data-cps21-GOLD-raw.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "n_samples": 200,
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
    "category_model": "gpt-4o",
    "category_backend": "openai",
    "classifier_model": "gpt-4o",
    "classifier_backend": "openai",
    "multiclass": True,
}

# Run classification
run_id = classify_texts(config)
print(f"Classification complete! Run ID: {run_id}")

###################
from text_classifier import classify_texts_hybrid

config = {
    "file_path": "data/data-cps21-FULL.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "max_llm_samples": 150,  # Use LLM for up to 300 training samples
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
    "No Opinion / Uncertain"],
    "classifier_model": "gpt-4.1-mini",
    "classifier_backend": "openai"
}

run_id = classify_texts_hybrid(config)

results_df = run_evaluation(
    run_id=run_id,
    test_file="data/golden_test_set.csv",
    text_column="cps21_imp_iss",
    text_label_column="cps21_imp_iss_label"
    
)
# 3. Analyze the results DataFrame
if not results_df.empty:
    print("\n--- Analysis of Incorrect Predictions ---")
    error_df = results_df[results_df['is_correct'] == False]
    print(f"Total errors: {len(error_df)}")
    # You can now work with error_df directly in your script
    print(error_df.head())