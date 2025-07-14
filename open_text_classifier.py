# open_text_classifier.py
"""
A lightweight text classification system using LLMs and machine learning.
Combines LLM backends, translation, and hybrid classification in one file.
"""

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LLM BACKENDS
# ============================================================================

class LLMBackend(ABC):
    """Base class for LLM backends."""
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and return response."""
        pass


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI()
        self.model = model
    
    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama2"):
        try:
            import ollama
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
        
        self.client = ollama
        self.model = model
        
        # Test connection
        try:
            self.client.list()
        except:
            raise ConnectionError("Cannot connect to Ollama. Is it running?")
    
    def complete(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']


def create_backend(backend_type: str, model: Optional[str] = None) -> LLMBackend:
    """
    Factory function to create LLM backends.
    
    Args:
        backend_type: "openai" or "ollama"
        model: Model name (optional, uses defaults if not provided)
    """
    backends = {
        "openai": OpenAIBackend,
        "ollama": OllamaBackend
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}")
    
    if model:
        return backends[backend_type](model)
    else:
        return backends[backend_type]()


# ============================================================================
# TRANSLATOR
# ============================================================================

class Translator:
    def __init__(self):
        """Initialize translator with Helsinki NLP models."""
        self.models = {}
        self.tokenizers = {}
        
    def translate_csv(
        self,
        filepath: str,
        text_column: str,
        language_column: str,
        filter_value: str,
        model_name: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Translate text in CSV for rows matching specified filter value.
        
        Args:
            filepath: Path to CSV file
            text_column: Column containing text to translate
            language_column: Column containing language codes
            filter_value: Value to filter for in language_column (e.g., "FR-CA", "es")
            model_name: Helsinki model name (e.g., "Helsinki-NLP/opus-mt-fr-en")
            output_file: Optional path to save translated CSV
            
        Returns:
            DataFrame with translated column added as text_column + "_tr"
        """
        # Load data
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        
        # Filter for specified value
        mask = df[language_column] == filter_value
        filtered_df = df[mask].copy()
        print(f"Found {len(filtered_df)} rows with {language_column}='{filter_value}'")
        
        if len(filtered_df) == 0:
            print(f"No rows found with {language_column}='{filter_value}'")
            return df
        
        # Load translation model
        model, tokenizer = self._get_model(model_name)
        
        # Translate texts
        translated_column = f"{text_column}_tr"
        translations = []
        
        print(f"Translating {len(filtered_df)} texts using {model_name}...")
        
        # Process in batches for efficiency
        batch_size = 32
        total_batches = (len(filtered_df) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(filtered_df), batch_size), 
                      total=total_batches, 
                      desc="Translating batches"):
            batch = filtered_df.iloc[i:i+batch_size]
            batch_texts = batch[text_column].fillna("").tolist()
            
            # Skip empty texts
            non_empty_indices = [j for j, text in enumerate(batch_texts) if text.strip()]
            if not non_empty_indices:
                translations.extend([""] * len(batch_texts))
                continue
            
            # Translate non-empty texts
            non_empty_texts = [batch_texts[j] for j in non_empty_indices]
            batch_translations = self._translate_batch(non_empty_texts, model, tokenizer, model_name)
            
            # Reconstruct full batch with empty strings preserved
            full_batch_translations = [""] * len(batch_texts)
            for j, trans in zip(non_empty_indices, batch_translations):
                full_batch_translations[j] = trans
            
            translations.extend(full_batch_translations)
        
        # Add translations back to filtered dataframe
        filtered_df[translated_column] = translations
        
        # Coalesce the translated text with the original text from other languages
        print("Merging translations back into the main dataframe...")
        
        # 1. Initialize the new column with a direct copy of the original text
        df[translated_column] = df[text_column]
        
        # 2. Overwrite only the rows for the translated language with the new translations
        df.loc[mask, translated_column] = filtered_df[translated_column].values
        
        print(f"Translation complete! Coalesced translations into column '{translated_column}'")
        
        # Save if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        
        return df
    
    def _get_model(self, model_name: str):
        """Load or retrieve a translation model and tokenizer."""
        if model_name in self.models:
            return self.models[model_name], self.tokenizers[model_name]
    
        print(f"Loading translation model: {model_name}")
    
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch sentencepiece")
    
        try:
            # Use AutoTokenizer and AutoModelForSeq2SeqLM for broader compatibility
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
            # Cache for reuse
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
    
            return model, tokenizer
        except Exception as e:
            raise ValueError(f"Could not load model '{model_name}'. Error: {e}")    

    def _translate_batch(self, texts, model, tokenizer, model_name):
        """
        Translate a batch of texts, adapting the strategy based on the model type.
        """
        # --- Generic settings for any model ---
        generation_params = {
            "num_beams": 4,
            "early_stopping": True,
            "max_new_tokens": 512
        }
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # --- Model-specific adjustments ---
        if "nllb" in model_name.lower():
            # For NLLB, specify the source and target languages
            tokenizer.src_lang = "fra_Latn"
            generation_params["forced_bos_token_id"] = tokenizer.vocab["eng_Latn"]
            # NLLB requires re-tokenizing after setting src_lang
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # --- Generate translation ---
        translated_tokens = model.generate(**inputs, **generation_params)
        translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return translations


# ============================================================================
# TEXT CLASSIFIER
# ============================================================================

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
{samples.to_list()}  

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
        sample_size = min(n_llm_samples, len(df))
        
        # Skip hybrid approach for small datasets
        if len(df) <= n_llm_samples:
            print(f"\nDataset has {len(df)} texts - using LLM for all (no ML training needed)")
        else:
            print(f"\nPhase 1: Labeling {sample_size} samples with LLM...")
        
        sample_indices = df.sample(sample_size, random_state=42).index
        
        if multiclass:
            labeled_data = []
            for idx in sample_indices:
                text = df.loc[idx, text_column]
                labels = self._classify_multi(text, categories, context)
                labeled_data.append((text, labels))
                if len(labeled_data) % 10 == 0:
                    print(f"  Labeled {len(labeled_data)} samples...")
        else:
            labeled_data = []
            for idx in sample_indices:
                text = df.loc[idx, text_column]
                label = self._classify_single(text, categories, context)
                labeled_data.append((text, label))
                if len(labeled_data) % 10 == 0:
                    print(f"  Labeled {len(labeled_data)} samples...")
        
        # Phase 2: Train classifier
        if len(df) <= n_llm_samples:
            # Skip training - we'll just return LLM labels
            print("\nSkipping ML training - all texts already labeled by LLM")
            # Create predictions from labeled data
            if multiclass:
                for cat in categories:
                    df[cat] = False
                for idx, (text, labels) in enumerate(labeled_data):
                    row_idx = sample_indices[idx]
                    for cat in categories:
                        df.loc[row_idx, cat] = cat in labels
            else:
                df['category'] = ''
                for idx, (text, label) in enumerate(labeled_data):
                    row_idx = sample_indices[idx]
                    df.loc[row_idx, 'category'] = label
        else:
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Suggest categories
    clf = TextClassifier(backend="openai", model="gpt-4")
    
    categories = clf.suggest_categories(
        "data.csv",
        text_column="response",
        n_samples=100,
        context="Survey responses about important political issues"
    )
    print("Suggested categories:", categories)
    
    # Example: Classify texts
    results = clf.classify_hybrid(
        "data.csv",
        text_column="response",
        categories=categories,
        n_llm_samples=500,
        multiclass=False,
        output_file="classified_data.csv"
    )
    
    print("\nCategory distribution:")
    print(results['category'].value_counts())