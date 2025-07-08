# The Text Classifier

A relatively lightweight text classification system using LLMs and machine learning.

## Installation

```bash
pip install pandas scikit-learn sentence-transformers openai ollama transformers torch ipython umap-learn matplotlib sentencepiece
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
clf = TextClassifier(backend="openai", model="gpt-4.1")

# Or use local Ollama
clf = TextClassifier(backend="ollama", model="gemma3n:latest")

# Generate categories from your data
categories = clf.suggest_categories(
    "data/data-cps21.csv",
    text_column="cps21_imp_iss",
    n_samples=200,
    context="The open text survey question is: What is the most important issue in the upcoming Canadian federal election? (answers in English or French)"
)
print(categories)
# Output: ['Healthcare', 'Economy', 'Education', 'Environment', 'Other']
```

### 2. Classify Texts (Single Label)

```python
# Define your categories
categories = ['COVID-19 / Pandemic', 'Healthcare', 'Economy / Economic Recovery', 'Environment / Climate Change', 'Taxes / Government Spending / Deficit', 'Housing / Affordability / Cost of Living', 'Government Leadership / Integrity / Transparency', 'Social Equality / Minority Rights', 'Education', 'Seniors / Senior Care', 'Child Care / Family Benefits', 'Immigration', 'National Unity / Canadian Values', 'Other / Miscellaneous']

# Classify using hybrid approach
results = clf.classify_hybrid(
    "data/data-cps21-40.csv",
    text_column="cps21_imp_iss",
    categories=categories,
    n_llm_samples=500,      # Label 500 samples with LLM
    multiclass=False,       # One category per text
    context="What is the most important issue in the upcoming Canadian federal election? (answers in English or French)",
    output_file="data/data-cps21-40-classified.csv",
)

# Check distribution
print(results['category'].value_counts())
```

### 3. Multi-Label Classification

```python
# Allow multiple categories per text
results = clf.classify_hybrid(
    "data/data-cps21-40.csv",
    text_column="cps21_imp_iss",
    categories=categories,
    n_llm_samples=500,
    multiclass=True,        # Multiple categories allowed
    context="What is the most important issue in the upcoming Canadian federal election? (answers in English or French)",
    output_file="data/data-cps21-40-classified.csv"
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
    "data/data-cps21-language.csv",
    text_column="cps21_imp_iss",
    language_column="Q_Language",
    filter_value="FR-CA",
    model_name="Helsinki-NLP/opus-mt-fr-en",  # Helsinki-NLP/opus-mt-fr-en; facebook/nllb-200-distilled-600M
    output_file="data/translated_helsinki.csv"
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
df = translator.translate_csv(
    "data/data-cps21-FULL.csv",
    text_column="cps21_imp_iss",
    language_column="Q_Language",
    filter_value="FR-CA",
    model_name="Helsinki-NLP/opus-mt-fr-en",  # Helsinki-NLP/opus-mt-fr-en; facebook/nllb-200-distilled-600M
    output_file="data/data-cps21-FULL-translated.csv"
)

# Step 2: Get categories

clf = TextClassifier(backend="ollama", model="deepseek-r1:14b")

# Generate categories from your data
categories = clf.suggest_categories(
    "data/data-cps21-FULL-translated.csv",
    text_column="cps21_imp_iss_tr",
    n_samples=400,
    context="The open text survey question is: What is the most important issue in the upcoming Canadian federal election? (answers in English or French)"
)
print(categories)

categories = ['Housing affordability', 'COVID-19 response', 'Economy', 'Climate change', 'Healthcare', 'Taxes', 'Indigenous rights', 'Education', 'Immigration', 'Social programs', 'Leadership and governance', 'Corruption and ethics', 'National unity', 'International relations', "Other, Uncertain, No answer"]

# Step 3: Classify all responses

clf = TextClassifier(backend="ollama", model="gemma3n:latest")

results = clf.classify_hybrid(
    "data/data-cps21-FULL-translated.csv",
    text_column="cps21_imp_iss_tr",  # Use translated column
    categories=categories,
    n_llm_samples=500,
    output_file="output/data-cps21-FULL-translated-classified.csv"
)

```

### 6. Latent space clustering

See latent_space_clustering.py for another approach that transforms text into numerical vectors using sentence embeddings, reduces them to 2D space with UMAP to reveal underlying structure, then applies HDBSCAN clustering to find natural groupings of similar responses. The key insight is that dimension reduction before clustering helps find meaningful patterns in survey responses by preserving semantic relationships while making the data geometrically clusterable.

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