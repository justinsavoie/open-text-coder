# analysis.py
"""
Example analysis script showing how to use the text_classifier package.
This is YOUR file where you do your actual work.
"""

from text_classifier import (
    classify_texts,
    validate_classification,
    load_classification_results,
    compare_runs,
    list_all_runs,
    load_config
)
import pandas as pd


def run_basic_classification():
    """Basic classification example"""
    print("=== Running Basic Classification ===")
    
    # Load config from file
    config = load_config("config.json")
    
    # Run classification
    run_id = classify_texts(config)
    
    # Load and examine results
    df = load_classification_results(run_id)
    print(f"\nClassified {len(df)} responses")
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    return run_id


def experiment_with_models():
    """Try different models and compare"""
    print("\n=== Model Comparison Experiment ===")
    
    base_config = load_config("config.json")
    run_ids = []
    
    # Test different models
    models = [
        ("gemma3n:latest", "ollama"),
        ("llama2", "ollama"),
        # ("gpt-3.5-turbo", "openai"),  # Uncomment if you have OpenAI key
    ]
    
    for model, backend in models:
        print(f"\nTesting {model} on {backend}...")
        config = base_config.copy()
        config.update({
            "classifier_model": model,
            "backend": backend
        })
        
        try:
            run_id = classify_texts(config)
            run_ids.append(run_id)
            
            # Validate a sample
            val_id = validate_classification(run_id, sample_size=50)
            
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    # Compare results
    if run_ids:
        compare_runs(run_ids)
    
    return run_ids


def test_category_generation():
    """Test automatic category generation"""
    print("\n=== Testing Category Generation ===")
    
    config = load_config("config.json")
    
    # Remove predefined categories to force generation
    if "categories" in config:
        del config["categories"]
    
    # Try different sample sizes
    for n_samples in [50, 100, 200]:
        print(f"\nGenerating categories with {n_samples} samples...")
        config["n_samples"] = n_samples
        config["category_model"] = "cogito:14b"  # Use a good model for generation
        
        run_id = classify_texts(config)
        
        # Show generated categories
        runs = list_all_runs()
        latest_run = next(r for r in runs if r['run_id'] == run_id)
        print(f"Generated categories: {latest_run['categories']}")


def analyze_validation_results():
    """Analyze validation scores to find problem areas"""
    print("\n=== Analyzing Validation Results ===")
    
    # Get the latest classification run
    runs = list_all_runs("classification")
    if not runs:
        print("No classification runs found. Run classification first.")
        return
    
    latest_run = runs[0]  # Most recent
    print(f"Analyzing run: {latest_run['run_id']}")
    
    # Validate if not already done
    validations = [v for v in list_all_runs("validation") 
                   if v.get('classification_run_id') == latest_run['run_id']]
    
    if not validations:
        print("Running validation...")
        val_id = validate_classification(latest_run['run_id'], sample_size=100)
    else:
        val_id = validations[0]['validation_id']
    
    # Load validation results
    from text_classifier import load_validation_results
    val_df = load_validation_results(val_id)
    
    # Analyze problem classifications (score <= 3)
    problems = val_df[val_df['quality_score'] <= 3]
    print(f"\nFound {len(problems)} problematic classifications out of {len(val_df)}")
    
    if len(problems) > 0:
        print("\nProblem categories:")
        print(problems['category'].value_counts())
        
        print("\nExample problems:")
        for _, row in problems.head(3).iterrows():
            print(f"\nText: {row[config['text_column']][:100]}...")
            print(f"Category: {row['category']}")
            print(f"Score: {row['quality_score']}")
            print(f"Explanation: {row['explanation'][:200]}...")


def custom_categories_example():
    """Example using custom categories"""
    print("\n=== Custom Categories Example ===")
    
    config = {
        "file_path": "data-cps21.csv",
        "text_column": "cps21_imp_iss",
        "id_column": "cps21_ResponseId",
        "categories": [
            "Healthcare and COVID-19",
            "Economy and Jobs", 
            "Climate and Environment",
            "Housing and Cost of Living",
            "Education",
            "Indigenous Issues",
            "Immigration",
            "Senior Care",
            "Other/Uncertain"
        ],
        "classifier_model": "gemma3n:latest"
    }
    
    run_id = classify_texts(config)
    
    # Show distribution
    df = load_classification_results(run_id)
    print("\nCategory distribution:")
    print(df['category'].value_counts().to_string())


def production_pipeline():
    """Full production pipeline with best practices"""
    print("\n=== Production Pipeline ===")
    
    # 1. Load and validate config
    config = load_config("config.json")
    
    # 2. First pass: Generate categories with a large model
    print("Step 1: Generating categories...")
    category_config = config.copy()
    category_config.update({
        "n_samples": 200,
        "category_model": "cogito:14b",  # Best model for understanding
        "classifier_model": "gemma3n:latest"  # Fast model for classification
    })
    if "categories" in category_config:
        del category_config["categories"]
    
    run1_id = classify_texts(category_config)
    
    # 3. Validate categories
    print("\nStep 2: Validating category quality...")
    val_id = validate_classification(run1_id, sample_size=50)
    
    # 4. If validation scores are good, run full classification
    from text_classifier import load_validation_results
    val_df = load_validation_results(val_id)
    avg_score = val_df['quality_score'].mean()
    
    print(f"\nAverage validation score: {avg_score:.2f}")
    
    if avg_score >= 4.0:
        print("Categories look good! Running full classification...")
        
        # Get the generated categories
        from text_classifier import get_run_info
        run_info = get_run_info(run1_id)
        categories = run_info['run']['categories']
        
        # Run with those categories on full dataset
        final_config = config.copy()
        final_config['categories'] = categories
        final_config['classifier_model'] = "gemma3n:latest"  # Or try gpt-3.5-turbo
        
        final_run_id = classify_texts(final_config)
        
        # Final validation
        final_val_id = validate_classification(final_run_id, sample_size=100)
        
        print(f"\nPipeline complete! Final run ID: {final_run_id}")
    else:
        print(f"Categories need refinement (score: {avg_score:.2f}). Consider:")
        print("- Manually defining categories")
        print("- Using a better model")
        print("- Providing more context")


if __name__ == "__main__":
    # Choose what to run
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "basic":
            run_basic_classification()
        elif sys.argv[1] == "models":
            experiment_with_models()
        elif sys.argv[1] == "categories":
            test_category_generation()
        elif sys.argv[1] == "validate":
            analyze_validation_results()
        elif sys.argv[1] == "custom":
            custom_categories_example()
        elif sys.argv[1] == "production":
            production_pipeline()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Options: basic, models, categories, validate, custom, production")
    else:
        # Run basic example by default
        print("Running basic classification (use 'python analysis.py [command]' for other options)")
        run_basic_classification()