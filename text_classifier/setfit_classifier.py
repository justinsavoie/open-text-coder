# text_classifier/setfit_classifier.py
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

try:
    from setfit import SetFitModel
    from sentence_transformers import SentenceTransformer
    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False
    SetFitModel = None
    SentenceTransformer = None


class SetFitClassifier:
    """SetFit-based text classifier for fast, few-shot learning"""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        device: str = None
    ):
        if not SETFIT_AVAILABLE:
            raise ImportError(
                "SetFit not installed. Run: pip install setfit sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self.categories = None
        self.is_trained = False
        
    def train(
        self,
        texts: List[str],
        labels: List[str],
        categories: List[str],
        validation_texts: Optional[List[str]] = None,
        validation_labels: Optional[List[str]] = None,
        num_epochs: int = 1,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """Train SetFit model on labeled data"""
        print(f"[*] Training SetFit model with {len(texts)} samples...")
        
        # Store categories for later use
        self.categories = categories
        
        # Convert string labels to numeric
        label_to_id = {label: idx for idx, label in enumerate(categories)}
        numeric_labels = [label_to_id[label] for label in labels]
        
        # Initialize model
        self.model = SetFitModel.from_pretrained(
            self.model_name,
            labels=categories,
            device=self.device
        )
        
        # Train
        if validation_texts and validation_labels:
            val_numeric_labels = [label_to_id[label] for label in validation_labels]
            self.model.train(
                texts,
                numeric_labels,
                eval_dataset=(validation_texts, val_numeric_labels),
                num_epochs=num_epochs,
                batch_size=batch_size
            )
        else:
            self.model.train(
                texts,
                numeric_labels,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
        
        self.is_trained = True
        
        # Calculate training metrics
        metrics = {
            "num_training_samples": len(texts),
            "num_categories": len(categories),
            "model_name": self.model_name,
            "num_epochs": num_epochs
        }
        
        # If validation data provided, calculate accuracy
        if validation_texts and validation_labels:
            predictions = self.predict_batch(validation_texts)
            accuracy = sum(p == l for p, l in zip(predictions, validation_labels)) / len(validation_labels)
            metrics["validation_accuracy"] = accuracy
            
        return metrics
    
    def predict(self, text: str, return_proba: bool = False) -> Union[str, Tuple[str, float]]:
        """Predict category for a single text"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get prediction
        prediction = self.model.predict([text])[0]
        
        if return_proba:
            # Get probabilities
            proba = self.model.predict_proba([text])[0]
            confidence = float(max(proba))
            return self.categories[prediction], confidence
        
        return self.categories[prediction]
    
    def predict_batch(
        self, 
        texts: List[str], 
        return_proba: bool = False
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Predict categories for multiple texts"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(texts)
        
        if return_proba:
            probas = self.model.predict_proba(texts)
            results = []
            for pred, proba in zip(predictions, probas):
                confidence = float(max(proba))
                results.append((self.categories[pred], confidence))
            return results
        
        return [self.categories[pred] for pred in predictions]
    
    def save(self, save_dir: Union[str, Path]):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        self.model.save_pretrained(str(save_dir))
        
        # Save metadata
        metadata = {
            "categories": self.categories,
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(save_dir / "setfit_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[*] SetFit model saved to: {save_dir}")
    
    def load(self, load_dir: Union[str, Path]):
        """Load trained model"""
        load_dir = Path(load_dir)
        
        if not (load_dir / "setfit_metadata.json").exists():
            raise ValueError(f"No SetFit model found in {load_dir}")
        
        # Load metadata
        with open(load_dir / "setfit_metadata.json") as f:
            metadata = json.load(f)
        
        self.categories = metadata["categories"]
        self.model_name = metadata["model_name"]
        self.is_trained = metadata["is_trained"]
        
        # Load model
        self.model = SetFitModel.from_pretrained(str(load_dir))
        
        print(f"[*] SetFit model loaded from: {load_dir}")


class HybridClassifier:
    """Hybrid classifier that uses LLM for training data and SetFit for bulk classification"""
    
    def __init__(
        self,
        llm_classifier,  # TextClassifier instance
        setfit_model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        confidence_threshold: float = 0.85,
        min_samples_per_category: int = 10,
        max_llm_samples: int = 200
    ):
        self.llm_classifier = llm_classifier
        self.setfit = SetFitClassifier(setfit_model_name)
        self.confidence_threshold = confidence_threshold
        self.min_samples_per_category = min_samples_per_category
        self.max_llm_samples = max_llm_samples
        self.training_data = []
        self.categories = None
        
    def collect_training_data(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: str,
        categories: List[str],
        question_context: str = ""
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Use LLM to classify initial samples for training"""
        print(f"[*] Collecting training data using LLM (max {self.max_llm_samples} samples)...")
        
        self.categories = categories
        
        # Sample data for training
        sample_size = min(self.max_llm_samples, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        texts = []
        labels = []
        ids = []
        
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="LLM Classification"):
            text = row[text_column]
            label = self.llm_classifier.classify_single(text, categories, question_context)
            
            texts.append(text)
            labels.append(label)
            ids.append(row[id_column])
        
        # Create training dataframe
        training_df = pd.DataFrame({
            id_column: ids,
            text_column: texts,
            'category': labels
        })
        
        # Store for later use
        self.training_data = list(zip(texts, labels))
        
        # Check if we have enough samples per category
        category_counts = training_df['category'].value_counts()
        metrics = {
            "total_training_samples": len(training_df),
            "category_distribution": category_counts.to_dict(),
            "categories_with_insufficient_samples": [
                cat for cat in categories 
                if category_counts.get(cat, 0) < self.min_samples_per_category
            ]
        }
        
        return training_df, metrics
    
    def train_setfit(self, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train SetFit model on collected data"""
        if not self.training_data:
            raise ValueError("No training data collected. Run collect_training_data first.")
        
        texts, labels = zip(*self.training_data)
        texts, labels = list(texts), list(labels)
        
        # Split for validation
        if validation_split > 0:
            split_idx = int(len(texts) * (1 - validation_split))
            train_texts, val_texts = texts[:split_idx], texts[split_idx:]
            train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        else:
            train_texts, val_texts = texts, None
            train_labels, val_labels = labels, None
        
        # Train SetFit
        metrics = self.setfit.train(
            train_texts,
            train_labels,
            self.categories,
            val_texts,
            val_labels
        )
        
        return metrics
    
    def classify_hybrid(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: str,
        question_context: str = "",
        use_active_learning: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Classify using hybrid approach"""
        if not self.setfit.is_trained:
            raise ValueError("SetFit model not trained. Run train_setfit first.")
        
        print(f"[*] Running hybrid classification on {len(df)} samples...")
        
        results = []
        llm_count = 0
        setfit_count = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Hybrid Classification"):
            text = row[text_column]
            
            if use_active_learning:
                # Get prediction with confidence
                category, confidence = self.setfit.predict(text, return_proba=True)
                
                if confidence < self.confidence_threshold:
                    # Low confidence - use LLM
                    category = self.llm_classifier.classify_single(
                        text, self.categories, question_context
                    )
                    llm_count += 1
                    source = "llm"
                else:
                    # High confidence - use SetFit prediction
                    setfit_count += 1
                    source = "setfit"
            else:
                # Just use SetFit
                category = self.setfit.predict(text)
                setfit_count += 1
                source = "setfit"
                confidence = None
            
            result = {
                id_column: row[id_column],
                text_column: text,
                'category': category,
                'source': source
            }
            
            if confidence is not None:
                result['confidence'] = confidence
                
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        metrics = {
            "total_classified": len(results_df),
            "llm_classifications": llm_count,
            "setfit_classifications": setfit_count,
            "llm_percentage": (llm_count / len(results_df)) * 100 if len(results_df) > 0 else 0,
            "category_distribution": results_df['category'].value_counts().to_dict()
        }
        
        if 'confidence' in results_df.columns:
            metrics["average_confidence"] = results_df['confidence'].mean()
            metrics["low_confidence_ratio"] = (results_df['confidence'] < self.confidence_threshold).mean()
        
        return results_df, metrics