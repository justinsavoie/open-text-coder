# text_classifier/storage.py
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime


class RunStorage:
    """Manages storage and retrieval of classification/validation runs"""
    
    def __init__(self, base_dir: Path = Path("./runs")):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            self.metadata = json.loads(self.metadata_file.read_text())
        else:
            self.metadata = {"classification_runs": {}, "validation_runs": {}}
    
    def _save_metadata(self):
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))
    
    def save_classification_run(self, run: 'ClassificationRun', df: pd.DataFrame):
        """Save classification run data and metadata"""
        # Create run directory
        run_dir = self.base_dir / f"classification_{run.run_id}"
        run_dir.mkdir(exist_ok=True)
        
        # Save data
        output_file = run_dir / f"classified_{run.run_id}.csv"
        df.to_csv(output_file, index=False)
        run.output_file = output_file
        
        # Save config
        config_file = run_dir / "config.json"
        config_file.write_text(json.dumps(run.config, indent=2))
        
        # Update metadata
        self.metadata["classification_runs"][run.run_id] = run.to_dict()
        self._save_metadata()
        
        return output_file
    
    def get_classification_run(self, run_id: str) -> Optional['ClassificationRun']:
        """Retrieve classification run metadata"""
        if run_id in self.metadata["classification_runs"]:
            data = self.metadata["classification_runs"][run_id]
            # Import here to avoid circular import
            from .models import ClassificationRun
            return ClassificationRun(
                run_id=data['run_id'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                config=data['config'],
                input_file=Path(data['input_file']),
                output_file=Path(data['output_file']) if data['output_file'] else None,
                categories=data['categories'],
                metrics=data['metrics']
            )
        return None
    
    def load_classification_data(self, run_id: str) -> Optional[pd.DataFrame]:
        """Load classification results DataFrame"""
        run = self.get_classification_run(run_id)
        if run and run.output_file and run.output_file.exists():
            return pd.read_csv(run.output_file)
        return None
    
    def list_runs(self, run_type: str = "classification") -> List[Dict[str, Any]]:
        """List all runs of a given type"""
        key = f"{run_type}_runs"
        return list(self.metadata.get(key, {}).values())    