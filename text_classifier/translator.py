# text_classifier/translator.py
"""
Translation utilities for the text classifier system.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

try:
    from transformers import pipeline
    import torch
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    pipeline = None
    torch = None


def translate_dataset(
    file_path: str,
    text_column: str,
    lang_column: str,
    lang_to_translate: str = "FR-CA",
    target_lang_code: str = "en",
    model_name: str = "Helsinki-NLP/opus-mt-fr-en",
    batch_size: int = 32
) -> str:
    """
    Translates a specific language in a dataset to a target language.

    Args:
        file_path (str): Path to the input CSV file.
        text_column (str): The column containing text to translate.
        lang_column (str): The column that specifies the language of the text.
        lang_to_translate (str): The language code to filter and translate (e.g., "FR-CA").
        target_lang_code (str): The target language code (e.g., "en").
        model_name (str): The Hugging Face model to use for translation.
        batch_size (int): The number of texts to translate at once.

    Returns:
        str: The path to the new, translated CSV file.
    """
    if not TRANSLATION_AVAILABLE:
        raise ImportError(
            "Translation dependencies not installed. Run: pip install transformers torch sentencepiece"
        )

    print(f"[*] Loading dataset from: {file_path}")
    p = Path(file_path)
    df = pd.read_csv(p)

    # Filter the DataFrame to get only the rows that need translation
    to_translate_df = df[df[lang_column] == lang_to_translate]
    
    if to_translate_df.empty:
        print(f"[*] No rows found with language '{lang_to_translate}'. No translation needed.")
        return file_path

    print(f"[*] Found {len(to_translate_df)} rows to translate from '{lang_to_translate}' to '{target_lang_code}'.")
    
    # Load the translation pipeline
    print(f"[*] Loading translation model: {model_name}")
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    translator = pipeline(
        "translation",
        model=model_name,
        device=device
    )

    # Extract the texts to translate
    texts_to_translate = to_translate_df[text_column].tolist()

    # Translate in batches for efficiency and progress tracking
    print("[*] Translating texts...")
    translated_texts = []
    for i in tqdm(range(0, len(texts_to_translate), batch_size), desc="Translating"):
        batch = texts_to_translate[i:i+batch_size]
        translated_batch = translator(batch)
        translated_texts.extend([item['translation_text'] for item in translated_batch])

    # Overwrite the original text column with the translated text
    # Use .loc to ensure we are modifying the original DataFrame correctly
    df.loc[to_translate_df.index, text_column] = translated_texts
    
    # Create the new filename
    output_filename = f"{p.stem}_translated.csv"
    output_path = p.parent / output_filename
    
    # Save the modified DataFrame
    print(f"[*] Saving translated dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    
    return str(output_path)