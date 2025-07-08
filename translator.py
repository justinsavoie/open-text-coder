# translator.py
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


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