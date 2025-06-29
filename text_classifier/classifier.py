# text_classifier/classifier.py
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import os

try:
    import ollama
except ImportError:
    ollama = None
    
try:
    import openai
except ImportError:
    openai = None


class TextClassifier:
    def __init__(self, model_name: str, backend: str = "ollama"):
        self.model_name = model_name
        self.backend = backend

        if backend == "openai":
            if openai is None:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI()  # <-- initialize once here
        elif backend == "ollama":
            if ollama is None:
                raise ImportError("Ollama package not installed. Run: pip install ollama")
    
    def send_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send chat request to LLM"""
        if self.backend == "openai":
            resp = self.client.chat.completions.create(model=self.model_name, messages=messages)
            content = resp.choices[0].message.content
            return {"message": {"content": content}}
        else:
            return ollama.chat(model=self.model_name, messages=messages)
    
    def generate_categories(
        self,
        df: pd.DataFrame,
        text_column: str,
        n_samples: int = 100,
        question_context: str = ""
    ) -> List[str]:
        """Generate categories from sample responses"""
        print(f"[*] Sampling {n_samples} responses to generate categories...")
        sample = df[text_column].dropna().sample(n=min(n_samples, len(df)))
        sample_text = "\n\n".join(sample)
        
        prompt = (
            "The following are sample survey responses to the survey question:"
            f" {question_context}\n\n"
            f"{sample_text}\n\n"
            "Based on these responses, generate 5-15 mutually exclusive categories "
            "that summarise the main themes. Include a Don't know or Uncertain category "
            "if appropriate. Return a comma-separated list only."
        )
        
        reply = self.send_chat([{"role": "system", "content": prompt}])
        text = reply["message"]["content"].strip()
        if "\n" in text:
            text = text.split("\n")[-1]
        return [c.strip() for c in text.split(",") if c.strip()]
    
    def classify_single(self, text: str, categories: List[str]) -> str:
        """Classify a single text"""
        prompt = (
            "You are a survey response classifier. "
            f"Choose the best category among: {', '.join(categories)}.\n\n"
            f"Response:\n\"{text}\"\n\nReturn the category name only."
        )
        reply = self.send_chat([{"role": "system", "content": prompt}])
        return reply["message"]["content"].strip()
    
    def classify_single_multiclass(self, text: str, categories: List[str]) -> Dict[str, str]:
        """Classify a single text into multiple categories"""
        category_list = "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(categories))
        prompt = (
            f"For the following response, determine if it belongs to each category. "
            "Answer with a list of yes/no for each category in order.\n\n"
            f"Categories:\n{category_list}\n\n"
            f"Response:\n\"{text}\"\n\n"
            f"Reply with exactly {len(categories)} answers separated by commas (e.g., \"yes, no, yes, no\"):"
        )
        
        reply = self.send_chat([{"role": "system", "content": prompt}])
        answer = reply["message"]["content"].strip().lower()
        responses = [resp.strip() for resp in answer.split(",")]
        
        result = {}
        for i, cat in enumerate(categories):
            result[cat] = "yes" if i < len(responses) and "yes" in responses[i] else "no"
        return result
    
    def run_classification(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: str,
        categories: Optional[List[str]] = None,
        multiclass: bool = False,
        n_samples: int = 100,
        question_context: str = "",
        category_model: Optional[str] = None,
        category_backend: Optional[str] = None  # NEW
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """Run full classification pipeline"""
        # Generate categories if needed
        if categories is None:
            cat_classifier = TextClassifier(
                model_name=category_model or self.model_name,
                backend=category_backend or self.backend
            )
            categories = cat_classifier.generate_categories(
                df, text_column, n_samples, question_context
            )
            print(f"[*] Generated categories: {', '.join(categories)}")
        else:
            print(f"[*] Using provided categories: {', '.join(categories)}")
        
        # Classify responses
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            txt = row[text_column]
            if pd.isna(txt):
                continue
            
            if multiclass:
                record = {
                    id_column: row[id_column],
                    text_column: txt,
                    **self.classify_single_multiclass(txt, categories)
                }
            else:
                record = {
                    id_column: row[id_column],
                    text_column: txt,
                    "category": self.classify_single(txt, categories)
                }
            rows.append(record)
        
        classified_df = pd.DataFrame(rows)
        
        # Calculate metrics
        metrics = {
            "total_rows": len(df),
            "classified_rows": len(classified_df),
            "multiclass": multiclass,
            "num_categories": len(categories)
        }
        
        if not multiclass:
            metrics["category_distribution"] = classified_df["category"].value_counts().to_dict()
        
        return classified_df, categories, metrics