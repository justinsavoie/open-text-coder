# text_classifier/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class ClassificationRun:
    """Represents a single classification run with metadata"""
    run_id: str
    timestamp: datetime
    config: Dict[str, Any]
    input_file: Path
    output_file: Optional[Path] = None
    categories: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp.isoformat(),
            'config': self.config,
            'input_file': str(self.input_file),
            'output_file': str(self.output_file) if self.output_file else None,
            'categories': self.categories,
            'metrics': self.metrics
        }


@dataclass
class ValidationRun:
    """Represents a validation run"""
    validation_id: str
    classification_run_id: str
    timestamp: datetime
    config: Dict[str, Any]
    results_file: Optional[Path] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'validation_id': self.validation_id,
            'classification_run_id': self.classification_run_id,
            'timestamp': self.timestamp.isoformat(),
            'config': self.config,
            'results_file': str(self.results_file) if self.results_file else None,
            'metrics': self.metrics
        }