# scripts/evaluate_model.py
"""
A script to evaluate a trained SetFit model against a golden test set.

This script can be run from the terminal or imported and used in other
Python scripts (e.g., Jupyter notebooks).

=========================
1. Terminal Usage
=========================

# Single-label evaluation
python scripts/evaluate_model.py \
    --run-id "20250704_110000_abcdef12" \
    --test-file "data/golden_test_set_single.csv" \
    --text-column "text" \
    --label-column "verified_category"

# Multi-label evaluation
python scripts/evaluate_model.py \
    --run-id "20250704_120000_fedcba21" \
    --test-file "data/golden_test_set_multi.csv" \
    --text-column "text"

=========================
2. Programmatic Usage
=========================

from scripts.evaluate_model import run_evaluation

# Train a model first to get a run_id
# run_id = classify_texts_hybrid(...) 

results_df = run_evaluation(
    run_id="YOUR_RUN_ID",
    test_file="data/golden_test_set.csv",
    text_column="text",
    label_column="verified_category" # Omit for multi-label
)
print("Evaluation complete. See the returned DataFrame for results.")
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
from typing import Optional

# Add the root directory to the Python path to allow importing the text_classifier module
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import MultiLabelBinarizer
    from text_classifier.api import load_setfit_model
except ImportError:
    print("Error: Make sure you have scikit-learn installed (`pip install scikit-learn`)")
    sys.exit(1)


def evaluate_single_label(y_true, y_pred, labels):
    """Calculates and prints metrics for single-label classification."""
    print("="*50)
    print(" " * 15, "SINGLE-LABEL EVALUATION")
    print("="*50)
    
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    print(classification_report(y_true, y_pred, labels=all_labels, digits=3, zero_division=0))
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    print("="*50)


def evaluate_multi_label(y_true, y_pred, classes):
    """Calculates and prints metrics for multi-label classification."""
    print("="*50)
    print(" " * 15, "MULTI-LABEL EVALUATION")
    print("="*50)

    mlb = MultiLabelBinarizer(classes=classes)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    print(classification_report(y_true_bin, y_pred_bin, target_names=classes, digits=3, zero_division=0))
    
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    print(f"\nExact Match Ratio (Accuracy): {accuracy:.3f}")
    print("(This is the percentage of samples where all labels were predicted correctly)")
    print("="*50)


def run_evaluation(
    run_id: str, 
    test_file: str, 
    text_column: str, 
    label_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads a trained SetFit model and evaluates it against a golden test set.
    
    Args:
        run_id: The run ID of the trained SetFit model.
        test_file: Path to the golden test set CSV file.
        text_column: Name of the column containing the text.
        label_column: Name of the column with true labels (for single-label only).

    Returns:
        A pandas DataFrame with the test data, predictions, and correctness check.
    """
    print(f"[*] Loading SetFit model from run: {run_id}")
    try:
        model = load_setfit_model(run_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        return pd.DataFrame()

    print(f"[*] Loading golden test set from: {test_file}")
    test_df = pd.read_csv(test_file)

    print("[*] Generating predictions for the test set...")
    texts_to_predict = test_df[text_column].tolist()
    predictions = model.predict_batch(texts_to_predict)

    if model.multiclass:
        true_labels = []
        for _, row in test_df.iterrows():
            labels = [cat for cat in model.categories if row.get(cat) in [1, '1', True, 'yes']]
            true_labels.append(labels)
        
        evaluate_multi_label(true_labels, predictions, model.categories)
        
        test_df['predicted_labels'] = predictions
        test_df['true_labels'] = true_labels
        test_df['is_correct'] = [set(p) == set(t) for p, t in zip(predictions, true_labels)]
        
    else:
        if not label_column:
            print("Error: `label_column` is required for single-label evaluation.")
            return pd.DataFrame()
        true_labels = test_df[label_column].tolist()
        
        evaluate_single_label(true_labels, predictions, model.categories)
        
        test_df['predicted_label'] = predictions
        test_df['is_correct'] = (test_df[label_column] == test_df['predicted_label'])

    error_df = test_df[test_df['is_correct'] == False]
    output_file = Path(f"evaluation_report_{run_id}.csv")
    test_df.to_csv(output_file, index=False)
    
    print(f"\n[*] Full evaluation report saved to: {output_file}")
    print(f"[*] Found {len(error_df)} incorrect predictions out of {len(test_df)} total samples.")
    
    return test_df


def main():
    """Parses command-line arguments and runs the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SetFit model.")
    parser.add_argument("--run-id", required=True, help="The run ID of the trained SetFit model.")
    parser.add_argument("--test-file", required=True, help="Path to the golden test set CSV file.")
    parser.add_argument("--text-column", required=True, help="Name of the column containing the text.")
    parser.add_argument("--label-column", help="Name of the column with true labels (for single-label).")
    
    args = parser.parse_args()
    
    run_evaluation(
        run_id=args.run_id,
        test_file=args.test_file,
        text_column=args.text_column,
        label_column=args.label_column
    )


if __name__ == "__main__":
    main()
