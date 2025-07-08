# translator.py
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


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
        source_language: str,
        target_language: str = "en",
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Translate text in CSV for rows matching specified language.
        
        Args:
            filepath: Path to CSV file
            text_column: Column containing text to translate
            language_column: Column containing language codes
            source_language: Language to filter for (e.g., "fr", "es")
            target_language: Target language (default: "en")
            output_file: Optional path to save translated CSV
            
        Returns:
            DataFrame with translated column added as text_column + "_tr"
        """
        # Load data
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        
        # Filter for specified language
        mask = df[language_column] == source_language
        filtered_df = df[mask].copy()
        print(f"Found {len(filtered_df)} rows in {source_language}")
        
        if len(filtered_df) == 0:
            print(f"No rows found with language '{source_language}'")
            return df
        
        # Load translation model
        model, tokenizer = self._get_model(source_language, target_language)
        
        # Translate texts
        translated_column = f"{text_column}_tr"
        translations = []
        
        print(f"Translating {len(filtered_df)} texts from {source_language} to {target_language}...")
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(filtered_df), batch_size):
            batch = filtered_df.iloc[i:i+batch_size]
            batch_texts = batch[text_column].fillna("").tolist()
            
            # Skip empty texts
            non_empty_indices = [j for j, text in enumerate(batch_texts) if text.strip()]
            if not non_empty_indices:
                translations.extend([""] * len(batch_texts))
                continue
            
            # Translate non-empty texts
            non_empty_texts = [batch_texts[j] for j in non_empty_indices]
            batch_translations = self._translate_batch(non_empty_texts, model, tokenizer)
            
            # Reconstruct full batch with empty strings preserved
            full_batch_translations = [""] * len(batch_texts)
            for j, trans in zip(non_empty_indices, batch_translations):
                full_batch_translations[j] = trans
            
            translations.extend(full_batch_translations)
            
            if (i + batch_size) % 320 == 0:
                print(f"  Translated {min(i + batch_size, len(filtered_df))} texts...")
        
        # Add translations back to filtered dataframe
        filtered_df[translated_column] = translations
        
        # Merge back with original dataframe
        df = df.copy()
        df[translated_column] = ""  # Initialize column
        df.loc[mask, translated_column] = filtered_df[translated_column].values
        
        print(f"Translation complete! Added column '{translated_column}'")
        
        # Save if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        
        return df
    
    def _get_model(self, source_lang: str, target_lang: str):
        """Load or retrieve Helsinki NLP translation model."""
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        if model_name in self.models:
            return self.models[model_name], self.tokenizers[model_name]
        
        print(f"Loading translation model: {model_name}")
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            # Cache for reuse
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            return model, tokenizer
        except Exception as e:
            # Try reverse direction or multi-language model
            alt_model_names = [
                f"Helsinki-NLP/opus-mt-{target_lang}-{source_lang}",  # Reverse
                f"Helsinki-NLP/opus-mt-{source_lang[:2]}-{target_lang[:2]}",  # 2-letter codes
                "Helsinki-NLP/opus-mt-tc-big-en-pt",  # Multi-language fallback
            ]
            
            for alt_name in alt_model_names:
                try:
                    print(f"  Trying alternative model: {alt_name}")
                    tokenizer = MarianTokenizer.from_pretrained(alt_name)
                    model = MarianMTModel.from_pretrained(alt_name)
                    self.models[model_name] = model
                    self.tokenizers[model_name] = tokenizer
                    return model, tokenizer
                except:
                    continue
            
            raise ValueError(f"Could not find translation model for {source_lang} to {target_lang}")
    
    def _translate_batch(self, texts, model, tokenizer):
        """Translate a batch of texts."""
        # Tokenize
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate translations
        translated = model.generate(**inputs)
        
        # Decode
        translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return translations