# classifier.py
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

from llm_backends import create_backend


class TextClassifier:
    def __init__(self, backend="openai", model=None):
        """
        Initialize classifier with LLM backend.
        
        Args:
            backend: "openai" or "ollama"
            model: Model name (defaults: gpt-3.5-turbo for openai, llama2 for ollama)
        """
        self.llm = create_backend(backend, model)
        self.embedder = None
        self.classifier = None
        self.categories = None
        self.mlb = None  # For multi-label
        
    def suggest_categories(self, filepath: str, text_column: str, n_samples: int = 100, context: str = "") -> List[str]:
        """Generate category suggestions from sample data."""
        df = pd.read_csv(filepath)
        samples = df[text_column].dropna().sample(min(n_samples, len(df)), random_state=42)
        
        prompt = f"""Context: {context}

Sample responses:
{samples.to_list()[:20]}  # Just show first 20

Based on these samples, suggest 5-15 categories as a Python list.
Return ONLY a Python list like: ["Category 1", "Category 2", ...]"""

        response = self.llm.complete(prompt)
        
        # Try to extract list from response
        try:
            # Find list in response
            import ast
            start = response.find('[')
            end = response.rfind(']') + 1
            categories = ast.literal_eval(response[start:end])
            return categories
        except:
            print(f"Failed to parse categories. LLM response:\n{response}")
            return ["Category 1", "Category 2", "Category 3", "Other"]
    
    def classify_hybrid(
        self, 
        filepath: str, 
        text_column: str,
        categories: List[str],
        n_llm_samples: int = 1000,
        multiclass: bool = False,
        context: str = "",
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Classify using hybrid approach: LLM for training, ML for bulk prediction.
        
        Args:
            filepath: CSV file path
            text_column: Column with text to classify
            categories: List of categories
            n_llm_samples: Number of samples to label with LLM
            multiclass: If True, allow multiple categories per text
            context: Additional context for classification
            output_file: If provided, save results to this CSV file
        """
        # Load data
        df = pd.read_csv(filepath)
        df = df[df[text_column].notna()].copy()
        
        print(f"Loaded {len(df)} texts")
        
        # Store categories
        self.categories = categories
        
        # Phase 1: Label samples with LLM
        print(f"\nPhase 1: Labeling {n_llm_samples} samples with LLM...")
        sample_size = min(n_llm_samples, len(df))
        sample_indices = df.sample(sample_size, random_state=42).index
        
        if multiclass:
            labeled_data = []
            for idx in sample_indices:
                text = df.loc[idx, text_column]
                labels = self._classify_multi(text, categories, context)
                labeled_data.append((text, labels))
                if len(labeled_data) % 100 == 0:
                    print(f"  Labeled {len(labeled_data)} samples...")
        else:
            labeled_data = []
            for idx in sample_indices:
                text = df.loc[idx, text_column]
                label = self._classify_single(text, categories, context)
                labeled_data.append((text, label))
                if len(labeled_data) % 100 == 0:
                    print(f"  Labeled {len(labeled_data)} samples...")
        
        # Phase 2: Train classifier
        print(f"\nPhase 2: Training classifier on {len(labeled_data)} labeled samples...")
        self._train_classifier(labeled_data, multiclass)
        
        # Phase 3: Predict on all data
        print(f"\nPhase 3: Predicting on all {len(df)} texts...")
        predictions = self._predict_batch(df[text_column].tolist(), multiclass)
        
        # Add predictions to dataframe
        if multiclass:
            # Add binary columns for each category
            for cat in categories:
                df[cat] = [cat in pred for pred in predictions]
        else:
            df['category'] = predictions
        
        # Save if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nSaved results to {output_file}")
        
        return df
    
    def _classify_single(self, text: str, categories: List[str], context: str) -> str:
        """Classify single text into one category using LLM."""
        prompt = f"""Context: {context}

Text: "{text}"

Categories: {', '.join(categories)}

Which category best fits this text? Return ONLY the category name."""

        response = self.llm.complete(prompt)
        
        # Find best matching category
        response_lower = response.lower().strip()
        for cat in categories:
            if cat.lower() in response_lower:
                return cat
        
        # Default to first category if no match
        return categories[0]
    
    def _classify_multi(self, text: str, categories: List[str], context: str) -> List[str]:
        """Classify text into multiple categories using LLM."""
        prompt = f"""Context: {context}

Text: "{text}"

Categories: {', '.join(categories)}

Which categories apply to this text? Return ONLY a Python list of applicable category names.
Example: ["Category 1", "Category 3"]"""

        response = self.llm.complete(prompt)
        
        # Try to extract list
        try:
            import ast
            start = response.find('[')
            end = response.rfind(']') + 1
            labels = ast.literal_eval(response[start:end])
            # Validate labels
            return [l for l in labels if l in categories]
        except:
            # Fallback: check which categories are mentioned
            response_lower = response.lower()
            return [cat for cat in categories if cat.lower() in response_lower]
    
    def _train_classifier(self, labeled_data: List[Tuple[str, any]], multiclass: bool):
        """Train LogisticRegression on labeled data."""
        # Initialize sentence transformer
        print("  Loading sentence transformer...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embeddings
        texts, labels = zip(*labeled_data)
        print("  Computing embeddings...")
        embeddings = self.embedder.encode(list(texts), show_progress_bar=False)
        
        # Train classifier
        if multiclass:
            # Convert labels to binary matrix
            self.mlb = MultiLabelBinarizer()
            y = self.mlb.fit_transform(labels)
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            # Convert labels to indices
            self.label_to_idx = {label: idx for idx, label in enumerate(self.categories)}
            y = [self.label_to_idx.get(label, 0) for label in labels]
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        
        print("  Training LogisticRegression...")
        self.classifier.fit(embeddings, y)
        print("  Training complete!")
    
    def _predict_batch(self, texts: List[str], multiclass: bool) -> List:
        """Predict categories for batch of texts."""
        # Get embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Predict
        if multiclass:
            # Get probabilities and threshold
            probas = self.classifier.predict_proba(embeddings)
            predictions = []
            threshold = 0.5
            
            for proba in probas:
                # Get labels above threshold
                indices = np.where(proba > threshold)[0]
                if len(indices) == 0:
                    # If none above threshold, take highest
                    indices = [np.argmax(proba)]
                labels = self.mlb.inverse_transform(np.eye(len(self.mlb.classes_))[indices])
                predictions.append(list(labels[0]) if len(labels) > 0 else [])
            
            return predictions
        else:
            # Single label prediction
            predictions = self.classifier.predict(embeddings)
            idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            return [idx_to_label[pred] for pred in predictions]