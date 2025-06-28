# text_classifier/validator.py
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from tqdm import tqdm

from .classifier import TextClassifier


class ClassificationValidator:
    """Handles validation of classifications"""
    
    def __init__(self, model_name: str, backend: str = "ollama"):
        self.classifier = TextClassifier(model_name, backend)
    
    def validate_single(
        self, 
        text: str, 
        predicted_category: str, 
        categories: List[str]
    ) -> Tuple[int, str]:
        """Validate a single classification"""
        categories_str = ", ".join(sorted(categories))
        
        system_prompt = "You are evaluating text classifications. Rate classification quality (1-5) and explain whether you agree."
        
        user_prompt = f"""For this text, I'll show you:
1. The original text
2. The predicted category
3. Available categories

Original text: \"{text}\"
Predicted category: \"{predicted_category}\"
Available categories: {categories_str}

Rate the classification quality (1-5) and explain whether you agree in one sentence."""
        
        result = self.classifier.send_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])['message']['content']
        
        # Parse score
        lines = result.strip().split('\n')
        score_line = next((l for l in lines if any(ch.isdigit() for ch in l)), None)
        score = int(next((ch for ch in score_line if ch.isdigit()), '0')) if score_line else None
        
        return score, result
    
    def validate_classification_run(
        self,
        df: pd.DataFrame,
        text_column: str,
        category_column: str,
        categories: List[str],
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate a classification run"""
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Validate each row
        results_df = df.copy()
        results_df['quality_score'] = None
        results_df['explanation'] = None
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
            score, explanation = self.validate_single(
                row[text_column],
                row[category_column],
                categories
            )
            results_df.at[idx, 'quality_score'] = score
            results_df.at[idx, 'explanation'] = explanation
        
        # Calculate metrics
        metrics = {
            "total_validated": len(results_df),
            "average_score": results_df['quality_score'].astype(float).mean(),
            "score_distribution": results_df['quality_score'].value_counts().to_dict()
        }
        
        return results_df, metrics