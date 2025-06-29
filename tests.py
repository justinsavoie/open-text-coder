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
    "classifier_model": "gemma3n:latest",
    "category_model": "gemma3n:latest",
    "classifier_backend": "ollama",
    "category_backend": "ollama",
}

# Classify with local model
gemma_config = base_config.copy()
run_id_gemma = classify_texts(gemma_config)

# Classify with API model
gpt_config = base_config.copy()
gpt_config["classifier_model"] = "gpt-4.1"
gpt_config["category_model"] = "gpt-4.1"
gpt_config["classifier_backend"] = "openai"
gpt_config["category_backend"] = "openai"
run_id_gpt = classify_texts(gpt_config)

# --- Validation Config ---
judge_config = {
    "backend": "openai",
    "judge_model": "gpt-4.1"
}

val_id_gemma = validate_classification(run_id_gemma, config=judge_config)
val_id_gpt = validate_classification(run_id_gpt, config=judge_config)

# --- Results Summary ---
df_val_gemma = load_validation_results(val_id_gemma)
df_val_gpt = load_validation_results(val_id_gpt)

print("\n=== Average Quality Scores ===")
print(f"gemma3n:latest → {df_val_gemma['quality_score'].mean():.2f}")
print(f"gpt-4.1 → {df_val_gpt['quality_score'].mean():.2f}")

print("\n=== % High Quality (Score ≥ 4) ===")
print(f"gemma3n:latest → {(df_val_gemma['quality_score'] >= 4).mean() * 100:.1f}%")
print(f"gpt-4.1 → {(df_val_gpt['quality_score'] >= 4).mean() * 100:.1f}%")
