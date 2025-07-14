# llm_backends.py
import os
from abc import ABC, abstractmethod
from typing import Optional


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