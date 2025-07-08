# Minimal Text Classifier

A lightweight text classification system using LLMs and machine learning. No frameworks, no complexity - just clean functions that work.

## Installation

```bash
pip install pandas scikit-learn sentence-transformers openai ollama transformers torch ipython
```

For OpenAI:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Quick Start

### 1. Generate Categories from Your Data

```python
from classifier import TextClassifier

# Use OpenAI
clf = TextClassifier(backend="openai", model="gpt-3.5-turbo")

# Or use local Ollama
clf = TextClassifier(backend="ollama", model="llama2")

# Generate categories from your data
categories = clf.suggest_categories(
    "survey.csv",
    text_column="response",
    n_samples=100,
    context="What is the most important issue?"
)
print(categories)
# Output: ['Healthcare', 'Economy', 'Education', 'Environment', 'Other']
```

### 2. Classify Texts (Single Label)

```python
# Define your categories
categories = ['Healthcare', 'Economy', 'Education', 'Environment', 'Other']

# Classify using hybrid approach
results = clf.classify_hybrid(
    "survey.csv",
    text_column="response",
    categories=categories,
    n_llm_samples=500,      # Label 500 samples with LLM
    multiclass=False,       # One category per text
    context="What is the most important issue?",
    output_file="classified.csv"
)

# Check distribution
print(results['category'].value_counts())
```

### 3. Multi-Label Classification

```python
# Allow multiple categories per text
results = clf.classify_hybrid(
    "survey.csv",
    text_column="response", 
    categories=categories,
    n_llm_samples=500,
    multiclass=True,        # Multiple categories allowed
    context="What issues matter to you?",
    output_file="multi_classified.csv"
)

# Check which categories were assigned
for cat in categories:
    print(f"{cat}: {results[cat].sum()} texts")
```

### 4. Translate Multilingual Data

```python
from translator import Translator

translator = Translator()

# Translate French responses to English
df = translator.translate_csv(
    "multilingual_survey.csv",
    text_column="response",
    language_column="lang",
    source_language="fr",
    target_language="en",
    output_file="translated.csv"
)

# Original column: 'response'
# Translated column: 'response_tr'
```

### 5. Full Pipeline: Translate Then Classify

```python
from classifier import TextClassifier
from translator import Translator

# Step 1: Translate non-English responses
translator = Translator()
df = translator.translate_csv("survey.csv", "response", "language", "fr", "en")
df = translator.translate_csv("survey_fr_translated.csv", "response", "language", "es", "en")

# Step 2: Save translated data
df.to_csv("survey_translated.csv", index=False)

# Step 3: Classify all responses
clf = TextClassifier(backend="openai")
results = clf.classify_hybrid(
    "survey_translated.csv",
    text_column="response_tr",  # Use translated column
    categories=['Positive', 'Negative', 'Neutral'],
    n_llm_samples=500,
    output_file="final_classified.csv"
)
```

## Parameters

### TextClassifier

- `backend`: "openai" or "ollama"
- `model`: Model name (default: "gpt-3.5-turbo" or "llama2")

### classify_hybrid()

- `filepath`: CSV file path
- `text_column`: Column containing text
- `categories`: List of category names  
- `n_llm_samples`: Number of samples to label with LLM (default: 1000)
- `multiclass`: Enable multi-label classification (default: False)
- `context`: Question/context for better classification
- `output_file`: Save results to CSV

### Translator

- `filepath`: CSV file path
- `text_column`: Column to translate
- `language_column`: Column with language codes
- `source_language`: Language to translate from (e.g., "fr", "es")
- `target_language`: Language to translate to (default: "en")
- `output_file`: Save results to CSV

## How It Works

1. **LLM labels a sample** of your data (e.g., 500-1000 texts)
2. **Trains a fast classifier** (LogisticRegression) on these labels
3. **Predicts remaining texts** using the trained classifier
4. **Saves results** with classifications

This hybrid approach gives you LLM-quality classifications at a fraction of the cost and time.

## Tips

- Start with `suggest_categories()` to understand your data
- Use 500-1000 samples for `n_llm_samples` - more samples = better accuracy
- Always provide `context` - it significantly improves results
- For large multilingual datasets, translate first, then classify
- Multi-label mode is great when texts cover multiple topics