# text_classifier/setfit_classifier.py
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

try:
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sentence_transformers import SentenceTransformer
    from datasets import Dataset
    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False
    # Define dummy classes if dependencies are not installed
    SetFitModel, Trainer, TrainingArguments, SentenceTransformer, Dataset = (None, None, None, None, None)


class SetFitClassifier:
    """SetFit-based text classifier for fast, few-shot learning, with multi-label support."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        device: str = None
    ):
        if not SETFIT_AVAILABLE:
            raise ImportError(
                "SetFit not installed. Run: pip install setfit sentence-transformers datasets"
            )
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self.categories = None
        self.is_trained = False
        self.multiclass = False

    def train(
        self,
        texts: List[str],
        labels: Union[List[str], List[List[str]]], # Can be single or multi-label
        categories: List[str],
        multiclass: bool = False,
        validation_texts: Optional[List[str]] = None,
        validation_labels: Optional[Union[List[str], List[List[str]]]] = None,
        num_epochs: int = 1,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """Train SetFit model on labeled data (supports single and multi-label)."""
        print(f"[*] Training SetFit model with {len(texts)} samples...")
        
        self.categories = categories
        self.multiclass = multiclass
        
        # Convert labels to numeric format
        if multiclass:
            # Multi-label: multi-hot encode the labels
            label_to_id = {label: idx for idx, label in enumerate(categories)}
            numeric_labels = []
            for sample_labels in labels:
                multi_hot = np.zeros(len(categories), dtype=np.float32)
                for label in sample_labels:
                    if label in label_to_id:
                        multi_hot[label_to_id[label]] = 1.0
                numeric_labels.append(multi_hot)
        else:
            # Single-label: Filter out texts and labels that are not valid
            label_to_id = {label: idx for idx, label in enumerate(categories)}
            
            # Pair up texts and labels to filter them in unison
            paired_data = list(zip(texts, labels))
            valid_pairs = [(text, label) for text, label in paired_data if label in label_to_id]

            if len(valid_pairs) < len(paired_data):
                print(f"[!] Warning: Filtered out {len(paired_data) - len(valid_pairs)} samples with invalid labels (e.g., 'Uncategorized').")

            if not valid_pairs:
                print("[!] Error: No valid training data left after filtering. Aborting training.")
                return {"error": "No valid training data."}

            # Unzip back into separate lists, now guaranteed to be the same length
            texts, labels = zip(*valid_pairs)
            
            # Convert valid string labels to numeric IDs
            numeric_labels = [label_to_id[label] for label in labels]
        
        # Initialize model
        self.model = SetFitModel.from_pretrained(
            self.model_name,
            labels=list(range(len(categories))) if not multiclass else None,
            multi_target_strategy="one-vs-rest" if multiclass else None,
            device=self.device
        )
        
        # Create HuggingFace Dataset objects
        train_dataset = Dataset.from_dict({"text": list(texts), "label": numeric_labels})
        
        eval_dataset = None
        # (Validation logic for multiclass would need similar multi-hot encoding)

        args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            column_mapping={"text": "text", "label": "label"}
        )
        
        trainer.train()
        self.is_trained = True
        
        metrics = {
            "num_training_samples": len(texts),
            "num_categories": len(categories),
            "multiclass": self.multiclass
        }
        return metrics            
    
    def predict_batch(
        self, 
        texts: List[str], 
        threshold: float = 0.5,
        return_confidence: bool = False  # Add this parameter
    ) -> Union[List[str], List[List[str]], List[Tuple[List[str], float]]]:
        """Predict categories for multiple texts with optional confidence scores."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.multiclass:
            # For multi-label, predict_proba gives probabilities for each class
            probas = self.model.predict_proba(texts)
            results = []
            
            for p_array in probas:
                # Get labels where probability is above the threshold
                predicted_labels = [
                    self.categories[i] for i, prob in enumerate(p_array) if prob >= threshold
                ]
                
                if return_confidence:
                    # For multiclass, confidence is the average of selected probabilities
                    # or the max probability if no labels pass threshold
                    if predicted_labels:
                        selected_probs = [prob for prob in p_array if prob >= threshold]
                        avg_confidence = sum(selected_probs) / len(selected_probs)
                    else:
                        avg_confidence = max(p_array)  # Max prob even if below threshold
                    results.append((predicted_labels, avg_confidence))
                else:
                    results.append(predicted_labels)
                    
            return results
        else:
            # For single-label classification
            predictions = self.model.predict(texts)
            
            if return_confidence:
                probas = self.model.predict_proba(texts)
                results = []
                for pred, proba in zip(predictions, probas):
                    confidence = float(max(proba))
                    results.append((self.categories[pred], confidence))
                return results
            else:
                return [self.categories[pred] for pred in predictions]
    
    def save(self, save_dir: Union[str, Path]):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(str(save_dir))
        
        metadata = {
            "categories": self.categories,
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "multiclass": self.multiclass, # Save multiclass status
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
        
        with open(load_dir / "setfit_metadata.json") as f:
            metadata = json.load(f)
        
        self.categories = metadata["categories"]
        self.model_name = metadata["model_name"]
        self.is_trained = metadata["is_trained"]
        self.multiclass = metadata.get("multiclass", False) # Load multiclass status
        
        self.model = SetFitModel.from_pretrained(str(load_dir))
        print(f"[*] SetFit model loaded from: {load_dir}")


class HybridClassifier:
    """Hybrid classifier that uses LLM for training data and SetFit for bulk classification"""
    
    def __init__(
        self,
        llm_classifier,
        setfit_model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        confidence_threshold: float = 0.5, # For multiclass, this is the prediction threshold
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
        question_context: str = "",
        multiclass: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Use LLM to classify initial samples for training"""
        print(f"[*] Collecting training data using LLM (max {self.max_llm_samples} samples)...")
        self.categories = categories
        
        sample_size = min(self.max_llm_samples, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        rows = []
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="LLM Classification"):
            text = row[text_column]
            record = {id_column: row[id_column], text_column: text}
            
            if multiclass:
                # Get {'cat1': 'yes', 'cat2': 'no'} from LLM
                labels_dict = self.llm_classifier.classify_single_multiclass(text, categories, question_context)
                # Convert to list of applicable categories: ['cat1']
                labels_list = [cat for cat, answer in labels_dict.items() if answer == 'yes']
                record['labels'] = labels_list # Store as a list
                # Also store the dict for the DataFrame
                record.update(labels_dict)

            else:
                label = self.llm_classifier.classify_single(text, categories, question_context)
                record['category'] = label
            
            rows.append(record)

        training_df = pd.DataFrame(rows)
        
        # Store for training
        if multiclass:
            self.training_data = list(zip(training_df[text_column], training_df['labels']))
        else:
            self.training_data = list(zip(training_df[text_column], training_df['category']))

        metrics = {"total_training_samples": len(training_df)}
        return training_df, metrics
    
    def train_setfit(self, multiclass: bool = False) -> Dict[str, Any]:
        """Train SetFit model on collected data"""
        if not self.training_data:
            raise ValueError("No training data collected. Run collect_training_data first.")
        
        texts, labels = zip(*self.training_data)
        
        # Train SetFit
        metrics = self.setfit.train(
            list(texts),
            list(labels),
            self.categories,
            multiclass=multiclass
        )
        return metrics
    
    def classify_hybrid(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: str,
        question_context: str = "",
        multiclass: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Classify using hybrid approach with proper confidence handling"""
        if not self.setfit.is_trained:
            raise ValueError("SetFit model not trained. Run train_setfit first.")
        
        print(f"[*] Running hybrid classification on {len(df)} samples...")
        
        # Get all SetFit predictions with confidence scores
        if multiclass:
            # For multiclass, get predictions with confidence
            setfit_predictions_with_conf = self.setfit.predict_batch(
                df[text_column].tolist(), 
                threshold=self.confidence_threshold,
                return_confidence=True
            )
            setfit_predictions = [pred for pred, _ in setfit_predictions_with_conf]
            setfit_confidences = [conf for _, conf in setfit_predictions_with_conf]
        else:
            # For single-label, also get confidence
            setfit_predictions_with_conf = self.setfit.predict_batch(
                df[text_column].tolist(),
                return_confidence=True
            )
            setfit_predictions = [pred for pred, _ in setfit_predictions_with_conf]
            setfit_confidences = [conf for _, conf in setfit_predictions_with_conf]
        
        results = []
        llm_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Hybrid Classification"):
            text = row[text_column]
            record = {id_column: row[id_column], text_column: text}
            
            # Determine if this is a low confidence prediction
            if multiclass:
                # For multiclass: low confidence if:
                # 1. No labels predicted (empty list), OR
                # 2. Average confidence is below threshold
                is_low_confidence = (
                    not setfit_predictions[idx] or 
                    setfit_confidences[idx] < self.confidence_threshold
                )
            else:
                # For single-label: low confidence if below threshold
                is_low_confidence = setfit_confidences[idx] < self.confidence_threshold
    
            if is_low_confidence:
                # Low confidence - use LLM
                llm_count += 1
                source = "llm"
                if multiclass:
                    labels_dict = self.llm_classifier.classify_single_multiclass(
                        text, self.categories, question_context
                    )
                    record.update(labels_dict)
                else:
                    label = self.llm_classifier.classify_single(
                        text, self.categories, question_context
                    )
                    record['category'] = label
            else:
                # High confidence - use SetFit prediction
                source = "setfit"
                if multiclass:
                    # Convert list from SetFit to yes/no dictionary
                    labels_dict = {
                        cat: ('yes' if cat in setfit_predictions[idx] else 'no') 
                        for cat in self.categories
                    }
                    record.update(labels_dict)
                else:
                    record['category'] = setfit_predictions[idx]
    
            record['source'] = source
            record['confidence'] = setfit_confidences[idx]  # Store confidence for analysis
            results.append(record)
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics with more detail
        metrics = {
            "total_classified": len(results_df),
            "llm_classifications": llm_count,
            "setfit_classifications": len(df) - llm_count,
            "llm_percentage": (llm_count / len(df)) * 100 if len(df) > 0 else 0,
            "avg_confidence": results_df['confidence'].mean(),
            "low_confidence_count": (results_df['confidence'] < self.confidence_threshold).sum()
        }
        
        return results_df, metrics