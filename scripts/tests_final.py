from text_classifier.api import (
    classify_texts,
    validate_classification,
    load_validation_results
)
import pandas as pd
from scipy.stats import pearsonr

# --- Base Config ---
base_config = {
    "file_path": "data-cps21.csv",
    "text_column": "cps21_imp_iss",
    "id_column": "cps21_ResponseId",
    "question_context": "What is the most important issue to you personally in this federal election? (answer in English or French)",
    "n_samples": 200
}

# --- Models to Test ---
model_pairs = [
    ("gemma3n:latest", "gemma3n:latest"),
    ("gpt-4.1", "gpt-4.1"),
    ("cogito:14b", "cogito:14b"),
    ("deepseek-r1:14b", "deepseek-r1:14b"),
    ("qwen3:14b", "qwen3:14b")
]

# --- Run Classifications ---
run_ids = {}
for category_model, classifier_model in model_pairs:
    cfg = base_config.copy()
    cfg["category_model"] = category_model
    cfg["classifier_model"] = classifier_model
    cfg["category_backend"] = "openai" if "gpt" in category_model else "ollama"
    cfg["classifier_backend"] = "openai" if "gpt" in classifier_model else "ollama"
    
    run_id = classify_texts(cfg)
    run_ids[f"{classifier_model}"] = run_id

# --- Validate with gpt-4.1 ---
val_ids_gpt = {}
for model, run_id in run_ids.items():
    val_id = validate_classification(
        run_id,
        config={"backend": "openai", "judge_model": "gpt-4.1"}
    )
    val_ids_gpt[model] = val_id

# --- Validate with cogito:14b ---
val_ids_cogito = {}
for model, run_id in run_ids.items():
    val_id = validate_classification(
        run_id,
        config={"backend": "ollama", "judge_model": "cogito:14b"}
    )
    val_ids_cogito[model] = val_id

# --- Load All Results ---
results = []
for model in run_ids:
    df_gpt = load_validation_results(val_ids_gpt[model])
    df_cog = load_validation_results(val_ids_cogito[model])
    avg_gpt = df_gpt["quality_score"].mean()
    avg_cog = df_cog["quality_score"].mean()
    results.append({
        "Model": model,
        "GPT‑4.1 Validation Score": avg_gpt,
        "Cogito:14b Validation Score": avg_cog
    })

df_scores = pd.DataFrame(results)
print("\n=== Validation Scores by Model ===")
print(df_scores.to_string(index=False))

# --- Correlation Analysis ---
print("\n=== Correlation Between Validators ===")
gpt_scores = []
cog_scores = []
for model in run_ids:
    df_gpt = load_validation_results(val_ids_gpt[model])
    df_cog = load_validation_results(val_ids_cogito[model])
    df_merged = df_gpt[["cps21_ResponseId", "quality_score"]].merge(
        df_cog[["cps21_ResponseId", "quality_score"]], on="cps21_ResponseId", suffixes=("_gpt", "_cog")
    )
    corr, _ = pearsonr(df_merged["quality_score_gpt"], df_merged["quality_score_cog"])
    print(f"{model}: Pearson r = {corr:.2f}")
