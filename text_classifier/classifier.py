# text_classifier/classifier.py
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import os
import time
import re

try:
    import ollama
except ImportError:
    ollama = None
    
try:
    import openai
except ImportError:
    openai = None


class TextClassifier:
    def __init__(self, model_name: str, backend: str = "ollama", max_retries: int = 3):
        self.model_name = model_name
        self.backend = backend
        self.max_retries = max_retries

        if backend == "openai":
            if openai is None:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI()
        elif backend == "ollama":
            if ollama is None:
                raise ImportError("Ollama package not installed. Run: pip install ollama")
    
    def send_chat(self, messages: List[Dict[str, Any]], retry_count: int = 0) -> Dict[str, Any]:
        """Send chat request to LLM with retry logic"""
        try:
            if self.backend == "openai":
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                    )
                    content = resp.choices[0].message.content
                    return {"message": {"content": content}}
                except openai.RateLimitError as e:
                    if retry_count < self.max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        print(f"Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        return self.send_chat(messages, retry_count + 1)
                    raise
                except openai.APIError as e:
                    if retry_count < self.max_retries:
                        print(f"API error, retrying... ({retry_count + 1}/{self.max_retries})")
                        time.sleep(1)
                        return self.send_chat(messages, retry_count + 1)
                    raise
            else:  # ollama
                try:
                    return ollama.chat(model=self.model_name, messages=messages)
                except Exception as e:
                    if retry_count < self.max_retries:
                        print(f"Ollama error, retrying... ({retry_count + 1}/{self.max_retries})")
                        time.sleep(1)
                        return self.send_chat(messages, retry_count + 1)
                    raise
                    
        except Exception as e:
            print(f"Failed after {self.max_retries} retries: {e}")
            raise
    
    def generate_categories(
        self,
        df: pd.DataFrame,
        text_column: str,
        n_samples: int = 100,
        question_context: str = ""
    ) -> List[str]:
        """Generate categories from sample responses with robust parsing"""
        print(f"[*] Sampling {n_samples} responses to generate categories...")
        sample = df[text_column].dropna().sample(n=min(n_samples, len(df)))
        sample_text = "\n\n".join(sample)
        
        # More explicit prompt format
        prompt = f"""The following are sample survey responses to the survey question: {question_context}

{sample_text}

Based on these responses, generate 5-15 mutually exclusive categories that summarise the main themes.

IMPORTANT: Format your response as a numbered list, one category per line:
1. First Category
2. Second Category
3. Third Category
...

Include a "Don't know/Uncertain" or "Other" category if appropriate.
Each category should be concise (2-5 words) and clearly distinct from others."""
        
        try:
            reply = self.send_chat([{"role": "system", "content": prompt}])
            text = reply["message"]["content"].strip()
            
            # Parse categories with multiple strategies
            categories = self._parse_categories(text)
            
            if not categories:
                print("[!] Warning: Could not parse categories from LLM response")
                # Fallback to generic categories
                categories = ["Positive", "Negative", "Neutral", "Other"]
            
            # Validate and clean categories
            categories = self._validate_categories(categories)
            
            return categories
            
        except Exception as e:
            print(f"[!] Error generating categories: {e}")
            # Return sensible defaults
            return ["Positive", "Negative", "Neutral", "Other"]

    def _validate_categories(self, categories: List[str]) -> List[str]:
        """Validate and clean the parsed categories"""
        cleaned = []
        seen = set()
        
        for cat in categories:
            # Remove any remaining numbering or bullets
            cat = re.sub(r'^\d+[\.\)]\s*', '', cat)
            cat = re.sub(r'^[-*•]\s*', '', cat)
            
            # Remove quotes
            cat = cat.strip('"\'')
            
            # Remove trailing punctuation
            cat = cat.rstrip('.,;:')
            
            # Normalize whitespace
            cat = ' '.join(cat.split())
            
            # Skip if too short or too long
            if len(cat) < 2 or len(cat) > 100:
                continue
            
            # Skip duplicates (case-insensitive)
            cat_lower = cat.lower()
            if cat_lower in seen:
                continue
            seen.add(cat_lower)
            
            cleaned.append(cat)
        
        # Ensure we have at least some categories
        if len(cleaned) < 3:
            print(f"[!] Warning: Only {len(cleaned)} valid categories found")
            # Add some generic ones if needed
            generic = ["Other", "Uncertain", "Mixed"]
            for g in generic:
                if g.lower() not in seen:
                    cleaned.append(g)
                    if len(cleaned) >= 5:
                        break
        
        # Cap at reasonable number
        if len(cleaned) > 15:
            print(f"[!] Warning: {len(cleaned)} categories found, capping at 15")
            cleaned = cleaned[:15]
        
        return cleaned

    def _parse_categories(self, text: str) -> List[str]:
        """Parse categories from LLM response using multiple strategies"""
        categories = []
        
        # Strategy 1: Parse numbered list (1. Category, 2. Category, etc.)
        numbered_pattern = r'^\s*\d+[\.\)]\s*(.+)$'
        lines = text.strip().split('\n')
        
        for line in lines:
            match = re.match(numbered_pattern, line.strip())
            if match:
                categories.append(match.group(1).strip())
        
        if categories:
            return categories
        
        # Strategy 2: Parse bullet points (- Category, * Category, • Category)
        bullet_pattern = r'^\s*[-*•]\s*(.+)$'
        for line in lines:
            match = re.match(bullet_pattern, line.strip())
            if match:
                categories.append(match.group(1).strip())
        
        if categories:
            return categories
        
        # Strategy 3: Parse comma-separated list
        # Look for a line that contains multiple comma-separated items
        for line in lines:
            if ',' in line:
                # Clean up common prefixes
                line = re.sub(r'^(Categories:|The categories are:|Here are the categories:)', '', line, flags=re.IGNORECASE)
                line = line.strip()
                
                # Split by comma and clean each item
                potential_categories = [
                    item.strip().strip('"\'') 
                    for item in line.split(',') 
                    if item.strip()
                ]
                
                # Validate these look like categories (not too long)
                if all(len(cat) < 50 for cat in potential_categories) and len(potential_categories) >= 3:
                    categories = potential_categories
                    break
        
        if categories:
            return categories
        
        # Strategy 4: Each non-empty line as a category (if reasonable number)
        # Filter out common non-category lines
        exclude_patterns = [
            r'^(here|these|the|based on|categories|following)',
            r'^(include|should|must|note)',
            r'^\d+$',  # Just numbers
            r'^[a-z]$',  # Single lowercase letters
        ]
        
        potential_categories = []
        for line in lines:
            line = line.strip()
            if line and len(line) < 50:  # Reasonable length for category
                # Check if line should be excluded
                if not any(re.match(pattern, line.lower()) for pattern in exclude_patterns):
                    potential_categories.append(line)
        
        # If we got a reasonable number of categories this way
        if 3 <= len(potential_categories) <= 20:
            return potential_categories
        
        return categories

    def classify_single(self, text: str, categories: List[str], question_context: str = "") -> str:
        """Classify a single text with error handling"""
        prompt = (
            "You are a survey response classifier.\n"
            f"Survey question: {question_context}\n\n"
            f"Response:\n\"{text}\"\n\n"
            f"Choose the best category among:\n{', '.join(categories)}\n\n"
            "Return the category name only."
        )
        
        try:
            reply = self.send_chat([{"role": "system", "content": prompt}])
            response = reply["message"]["content"].strip()
            
            # Validate response is in categories
            if response not in categories:
                # Try case-insensitive match
                for cat in categories:
                    if cat.lower() == response.lower():
                        return cat
                
                # Try regex matching - check if response is contained in exactly one category
                response_lower = response.lower()
                matches = []
                
                for cat in categories:
                    if response_lower in cat.lower():
                        matches.append(cat)
                
                if len(matches) == 1:
                    # Found in exactly one category
                    print(f"Warning: LLM returned '{response}' - matched to category '{matches[0]}' via substring match.")
                    return matches[0]
                else:
                    # Not found or found in multiple categories
                    if len(matches) > 1:
                        print(f"Warning: LLM returned '{response}' which matches multiple categories: {matches}. Returning 'Uncategorized'.")
                    else:
                        print(f"Warning: LLM returned '{response}' which doesn't match any category. Returning 'Uncategorized'.")
                    return "Uncategorized"
            
            return response
            
        except Exception as e:
            print(f"Error in classify_single: {e}")
            # Return a default category or raise depending on your needs
            return "Classification Error"    

    def classify_single_multiclass(self, text: str, categories: List[str], question_context: str = "") -> Dict[str, str]:
        """Classify a single text into multiple categories with robust parsing"""
        category_list = "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(categories))
        
        # More explicit prompt format
        prompt = f"""You are a survey response classifier.
Survey question: {question_context}

Response: "{text}"

For each category below, indicate if the response applies (yes) or not (no).

Categories:
{category_list}

IMPORTANT: Reply with EXACTLY {len(categories)} answers in this format:
1. yes/no
2. yes/no
3. yes/no
(etc.)

Each line must start with a number followed by a period, then yes or no."""
        
        try:
            reply = self.send_chat([{"role": "system", "content": prompt}])
            answer = reply["message"]["content"].strip()
            
            # Parse numbered list format
            result = {}
            
            # Try to parse numbered format first (most reliable)
            numbered_pattern = r'(\d+)\.\s*(yes|no)'
            matches = re.findall(numbered_pattern, answer.lower())
            
            if matches:
                # Build result from numbered matches
                match_dict = {int(num): resp for num, resp in matches}
                for i, cat in enumerate(categories, 1):
                    if i in match_dict:
                        result[cat] = match_dict[i]
                    else:
                        result[cat] = "no"  # Default to no if missing
            else:
                # Fallback: try comma-separated or line-separated yes/no
                # Remove numbers, punctuation, extra whitespace
                clean_text = re.sub(r'[0-9\.\:\-\)\(]', ' ', answer.lower())
                # Find all yes/no occurrences
                responses = re.findall(r'\b(yes|no)\b', clean_text)
                
                for i, cat in enumerate(categories):
                    if i < len(responses):
                        result[cat] = responses[i]
                    else:
                        result[cat] = "no"
                        
            # Validate we have all categories
            for cat in categories:
                if cat not in result:
                    result[cat] = "no"
                    
            return result
            
        except Exception as e:
            print(f"Error in classify_single_multiclass: {e}")
            # Return all "no" as safe default
            return {cat: "no" for cat in categories}    

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
        category_backend: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """Run full classification pipeline with error handling"""
        
        # Generate categories if needed
        if categories is None:
            try:
                cat_classifier = TextClassifier(
                    model_name=category_model or self.model_name,
                    backend=category_backend or self.backend
                )
                categories = cat_classifier.generate_categories(
                    df, text_column, n_samples, question_context
                )
                print(f"[*] Generated categories: {', '.join(categories)}")
            except Exception as e:
                print(f"Error generating categories: {e}")
                # Fallback to generic categories
                categories = ["Positive", "Negative", "Neutral", "Other"]
                print(f"[!] Using fallback categories: {', '.join(categories)}")
        else:
            print(f"[*] Using provided categories: {', '.join(categories)}")
        
        # Classify responses
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            txt = row[text_column]
            # No need to check for empty/NaN - already filtered in api.py
            
            if multiclass:
                record = {
                    id_column: row[id_column],
                    text_column: txt,
                    **self.classify_single_multiclass(txt, categories, question_context)
                }
            else:
                record = {
                    id_column: row[id_column],
                    text_column: txt,
                    "category": self.classify_single(txt, categories, question_context)
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