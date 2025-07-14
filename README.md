# open-text-classifier

A lightweight text classification system using LLMs and machine learning. All functionality is contained in a single Python file for easy deployment and use.

## Features

- **Hybrid Classification**: Uses LLMs to label a sample, then trains a fast ML classifier
- **Multiple LLM Backends**: Supports OpenAI and Ollama (local models)
- **Translation Support**: Built-in translation for multilingual datasets
- **Category Generation**: Automatically suggest categories from your data
- **Multi-label Support**: Classify texts into multiple categories
- **Cost-Effective**: Label only a sample with LLMs, predict the rest with ML

## Installation

```bash
pip install pandas scikit-learn sentence-transformers openai ollama transformers torch tqdm
```

For OpenAI backend:
```bash
export OPENAI_API_KEY="your-key-here"
```

For Ollama backend, make sure Ollama is running locally:
```bash
ollama serve
```

## Quick Start

Download the single Python file: `open_text_classifier.py`

```python
from open_text_classifier import TextClassifier, Translator
```

### 1. Generate Categories from Your Data

```python
# Use OpenAI
clf = TextClassifier(backend="openai", model="gpt-4")

# Or use local Ollama
clf = TextClassifier(backend="ollama", model="llama2")

# Generate categories from your data
categories = clf.suggest_categories(
    "data.csv",
    text_column="response",
    n_samples=200,
    context="Survey question: What is the most important issue?"
)
print(categories)
# Output: ['Healthcare', 'Economy', 'Education', 'Environment', 'Other']
```

### 2. Classify Texts (Single Label)

```python
# Define your categories
categories = [
    'Healthcare',
    'Economy', 
    'Environment',
    'Education',
    'Housing',
    'Other'
]

# Classify using hybrid approach
results = clf.classify_hybrid(
    "data.csv",
    text_column="response",
    categories=categories,
    n_llm_samples=500,      # Label 500 samples with LLM
    multiclass=False,       # One category per text
    context="Survey responses about important political issues",
    output_file="classified_data.csv"
)

# Check distribution
print(results['category'].value_counts())
```

### 3. Multi-Label Classification

```python
# Allow multiple categories per text
results = clf.classify_hybrid(
    "data.csv",
    text_column="response",
    categories=categories,
    n_llm_samples=500,
    multiclass=True,        # Multiple categories allowed
    context="Survey responses about important political issues",
    output_file="classified_multilabel.csv"
)

# Check which categories were assigned
for cat in categories:
    print(f"{cat}: {results[cat].sum()} texts")
```

### 4. Translate Multilingual Data

```python
translator = Translator()

# Translate French responses to English
df = translator.translate_csv(
    "multilingual_data.csv",
    text_column="response",
    language_column="language",
    filter_value="FR",
    model_name="Helsinki-NLP/opus-mt-fr-en",
    output_file="translated_data.csv"
)

# Original column: 'response'
# Translated column: 'response_tr'
```

### 5. Full Pipeline: Translate Then Classify

```python
# Step 1: Translate non-English responses
translator = Translator()
df = translator.translate_csv(
    "multilingual_survey.csv",
    text_column="response",
    language_column="language",
    filter_value="FR",
    model_name="Helsinki-NLP/opus-mt-fr-en",
    output_file="survey_translated.csv"
)

# Step 2: Get categories from translated data
clf = TextClassifier(backend="openai", model="gpt-4")
categories = clf.suggest_categories(
    "survey_translated.csv",
    text_column="response_tr",
    n_samples=200,
    context="Political survey responses"
)

# Step 3: Classify all responses
results = clf.classify_hybrid(
    "survey_translated.csv",
    text_column="response_tr",
    categories=categories,
    n_llm_samples=500,
    output_file="survey_classified.csv"
)
```

## API Reference

### TextClassifier

```python
TextClassifier(backend="openai", model=None)
```

**Parameters:**
- `backend`: "openai" or "ollama"
- `model`: Model name (defaults: "gpt-3.5-turbo" for OpenAI, "llama2" for Ollama)

**Methods:**

`suggest_categories(filepath, text_column, n_samples=100, context="")`
- Generate category suggestions from your data
- Returns list of suggested categories

`classify_hybrid(filepath, text_column, categories, n_llm_samples=1000, multiclass=False, context="", output_file=None)`
- Classify texts using hybrid LLM/ML approach
- Returns DataFrame with classifications

### Translator

```python
Translator()
```

**Methods:**

`translate_csv(filepath, text_column, language_column, filter_value, model_name, output_file=None)`
- Translate texts in specified language
- Returns DataFrame with translated column added

## Supported Models

### LLM Backends
- **OpenAI**: gpt-3.5-turbo, gpt-4, gpt-4-turbo, etc.
- **Ollama**: llama2, mistral, mixtral, gemma, etc.

### Translation Models
- **French to English**: Helsinki-NLP/opus-mt-fr-en
- **Spanish to English**: Helsinki-NLP/opus-mt-es-en
- **German to English**: Helsinki-NLP/opus-mt-de-en
- **Multilingual**: facebook/nllb-200-distilled-600M

## How It Works

1. **LLM labels a sample** of your data (e.g., 500-1000 texts)
2. **Trains a fast classifier** (LogisticRegression) on these labels
3. **Predicts remaining texts** using the trained classifier
4. **Saves results** with classifications

This hybrid approach gives you LLM-quality classifications at a fraction of the cost and time.

## Tips

- **Start with `suggest_categories()`** to understand your data
- **Use 500-1000 samples** for `n_llm_samples` - more samples = better accuracy
- **Always provide `context`** - it significantly improves results
- **For multilingual datasets**, translate first, then classify
- **Multi-label mode** is great when texts cover multiple topics
- **Small datasets** (<1000 texts) will use LLM for all texts automatically

## Performance Considerations

- **LLM costs**: Only the sample size (n_llm_samples) incurs API costs
- **Speed**: After initial labeling, classification is very fast
- **Memory**: Translation models are loaded on-demand and cached
- **Accuracy**: Typically 85-95% agreement with full LLM classification

## Example Datasets

The classifier works well with:
- Survey responses
- Customer feedback
- Support tickets
- Social media posts
- News articles
- Product reviews
- Any text that needs categorization

## Troubleshooting

**OpenAI errors**: Check your API key is set correctly
```bash
export OPENAI_API_KEY="sk-..."
```

**Ollama errors**: Ensure Ollama is running
```bash
ollama serve
```

**Memory issues**: Reduce batch size for translation or use smaller models

**Poor classifications**: 
- Increase `n_llm_samples`
- Improve category definitions
- Add more specific `context`

## License

MIT License - feel free to use in your projects!

## Citation

If you use this tool in research, please cite:
```
@software{open_text_classifier,
  title = {Open Text Classifier: Hybrid LLM/ML Text Classification},
  year = {2024},
  url = {https://github.com/yourusername/open-text-classifier}
}
```