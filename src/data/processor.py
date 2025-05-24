from datasets import load_dataset, DatasetDict
from typing import Dict, Union
import numpy as np
from .utils import *

class DatasetProcessor:
    
    @staticmethod
    def load_dataset(name: str, quick_test: bool = False) -> DatasetDict:
        """Load dataset"""
        
        if name == "HC3":
            ds = load_dataset("Hello-SimpleAI/HC3", "all")
            ds = process_hc3(ds)
        elif name == "ai-human-gen":
            ds = (load_dataset("likhithasapu/ai-human-gen")
                  .rename_column("response", "text")
                  .map(lambda x: {"label": 1 - x["human-generated"]}))
        elif name == "AI-and-Human-Generated-Text":
            ds = (load_dataset("Ateeqq/AI-and-Human-Generated-Text")
                  .rename_column("abstract", "text"))
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        # Ensure test split exists
        if "test" not in ds:
            ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        
        # Quick test sampling
        if quick_test:
            ds = sample_dataset(ds)
    
        return ds