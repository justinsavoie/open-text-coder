# text_classifier/validator.py
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from tqdm import tqdm
import re

from .classifier import TextClassifier


class ClassificationValidator:
    """Handles validation of classifications"""
    
    def __init__(self, model_name: str, backend: str = "ollama"):
        self.classifier = TextClassifier(model_name, backend)

    def validate_single_multiclass(
        self, 
        text: str, 
        predictions: Dict[str, str], 
        categories: List[str]
    ) -> Tuple[Dict[str, int], str]:
        """Validate multiclass predictions for a single text"""
        
        # Build the predictions summary
        pred_summary = "\n".join([
            f"- {cat}: {predictions.get(cat, 'no')}" 
            for cat in categories
        ])
        
        system_prompt = """You are evaluating multiclass text classifications.
For each category prediction, rate its accuracy from 1-5.
Format your response as:
Category scores:
- [Category name]: [1-5] - [brief reason]
Overall: [1-5] - [overall assessment]"""
        
        user_prompt = f"""Original text: "{text}"

Predictions:
{pred_summary}

For each category, evaluate if the yes/no prediction is accurate.
Consider:
1. Does a "yes" prediction correctly identify that the text relates to this category?
2. Does a "no" prediction correctly identify that the text does NOT relate to this category?
3. Are there any missed categories (should be "yes" but marked "no")?
4. Are there any false positives (marked "yes" but shouldn't be)?

Rate each category prediction 1-5 where:
1 = completely wrong
3 = partially correct  
5 = perfectly accurate"""
        
        try:
            result = self.classifier.send_chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])['message']['content']
            
            # Parse individual category scores
            scores = self._parse_multiclass_scores(result, categories)
            
            return scores, result
            
        except Exception as e:
            print(f"Error in validate_single_multiclass: {e}")
            # Return middle scores as default
            return {cat: 3 for cat in categories}, f"Validation error: {str(e)}"
    
    def _parse_multiclass_scores(self, text: str, categories: List[str]) -> Dict[str, int]:
        """Parse scores for each category from validation response"""
        import re
        
        scores = {}
        
        # Try to find scores for each category
        for cat in categories:
            # Look for patterns like "CategoryName: 4" or "CategoryName: 4/5"
            # Make pattern flexible to handle variations
            escaped_cat = re.escape(cat)
            patterns = [
                rf'{escaped_cat}\s*:\s*(\d)',  # Category: 4
                rf'{escaped_cat}.*?:\s*(\d)',   # Category blah blah: 4
                rf'{escaped_cat}.*?(\d)\s*/',   # Category 4/5
                rf'{escaped_cat}.*?(\d)\s*out', # Category 4 out of 5
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    if 1 <= score <= 5:
                        scores[cat] = score
                        break
            
            # Default to 3 if not found
            if cat not in scores:
                scores[cat] = 3
                
        # Try to find overall score
        overall_patterns = [
            r'[Oo]verall\s*:\s*(\d)',
            r'[Oo]verall.*?(\d)\s*/',
            r'[Aa]verage\s*:\s*(\d)',
        ]
        
        for pattern in overall_patterns:
            match = re.search(pattern, text)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    scores['_overall'] = score
                    break
                    
        return scores

    def validate_single(
        self, 
        text: str, 
        predicted_category: str, 
        categories: List[str]
    ) -> Tuple[int, str]:
        """Validate a single classification with robust score parsing"""
        categories_str = ", ".join(sorted(categories))
        
        system_prompt = """You are evaluating text classifications. 
Rate classification quality from 1 to 5 and explain.
Format your response as:
Score: [number]
Explanation: [your explanation]"""
        
        user_prompt = f"""Original text: \"{text}\"
Predicted category: \"{predicted_category}\"
Available categories: {categories_str}

Rate overall classification quality from 1 (very poor) to 5 (excellent)."""
        
        try:
            result = self.classifier.send_chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])['message']['content']
            
            # More robust score parsing
            score = self._parse_score(result)
            
            if score is None:
                print(f"Warning: Could not parse score from: {result[:100]}...")
                score = 3  # Default middle score
                
            return score, result
            
        except Exception as e:
            print(f"Error in validate_single: {e}")
            return 3, f"Validation error: {str(e)}"

    def _parse_score(self, text: str) -> Optional[int]:
        """Parse score from text with multiple strategies"""
        # Strategy 1: Look for "Score: X" pattern
        score_match = re.search(r'[Ss]core\s*:\s*(\d)', text)
        if score_match:
            score = int(score_match.group(1))
            if 1 <= score <= 5:
                return score
        
        # Strategy 2: Look for "X/5" or "X out of 5" pattern
        out_of_five = re.search(r'(\d)\s*(?:/|out of)\s*5', text)
        if out_of_five:
            score = int(out_of_five.group(1))
            if 1 <= score <= 5:
                return score
        
        # Strategy 3: Find first standalone digit 1-5
        for match in re.finditer(r'\b([1-5])\b', text):
            return int(match.group(1))
        
        # Strategy 4: Look for written numbers
        word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
        for word, num in word_to_num.items():
            if word in text.lower():
                return num
        
        return None

    def validate_classification_run(
        self,
        df: pd.DataFrame,
        text_column: str,
        category_column: Optional[str] = None,  # Make optional for multiclass
        categories: List[str] = None,
        sample_size: Optional[int] = None,
        multiclass: bool = None  # Auto-detect if not specified
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate a classification run (both single and multiclass)"""
        
        # Auto-detect multiclass if not specified
        if multiclass is None:
            # Check if we have category columns (multiclass) or a single category column
            multiclass = all(cat in df.columns for cat in categories) if categories else False
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Prepare results dataframe
        results_df = df.copy()
        
        if multiclass:
            # Multiclass validation
            results_df['quality_scores'] = None
            results_df['explanation'] = None
            results_df['average_score'] = None
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating multiclass"):
                # Get predictions for this row
                predictions = {cat: row.get(cat, 'no') for cat in categories}
                
                scores, explanation = self.validate_single_multiclass(
                    row[text_column],
                    predictions,
                    categories
                )
                
                # Store results
                results_df.at[idx, 'quality_scores'] = scores
                results_df.at[idx, 'explanation'] = explanation
                
                # Calculate average score for this row
                cat_scores = [s for c, s in scores.items() if c != '_overall']
                avg_score = sum(cat_scores) / len(cat_scores) if cat_scores else 3
                results_df.at[idx, 'average_score'] = avg_score
                
        else:
            # Single category validation (existing code)
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
        if multiclass:
            metrics = {
                "total_validated": len(results_df),
                "multiclass": True,
                "average_score": results_df['average_score'].mean(),
                "score_distribution": results_df['average_score'].value_counts().to_dict()
            }
            
            # Add per-category average scores
            category_scores = {}
            for cat in categories:
                cat_scores = [
                    scores.get(cat, 3) 
                    for scores in results_df['quality_scores'] 
                    if scores and cat in scores
                ]
                if cat_scores:
                    category_scores[cat] = sum(cat_scores) / len(cat_scores)
            metrics["category_scores"] = category_scores
            
        else:
            metrics = {
                "total_validated": len(results_df),
                "multiclass": False,
                "average_score": results_df['quality_score'].astype(float).mean(),
                "score_distribution": results_df['quality_score'].value_counts().to_dict()
            }
        
        return results_df, metrics