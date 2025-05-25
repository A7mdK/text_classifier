from datasets import load_dataset, DatasetDict
from typing import Dict, Union
import numpy as np
from .utils import *

class DatasetProcessor:
    
    @staticmethod
    def load_dataset(name: str, quick_test: bool = False) -> DatasetDict:
        """Load dataset"""
        
        # Human Written or AI Generated
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
            
        # Hate Speech and Offensive Language Detection
        elif name == "davidson_offensive":
            ds = load_dataset("tdavidson/hate_speech_offensive")
            ds = ds.rename_column("tweet", "text")
            ds = ds.rename_column("class", "label")
            if "test" not in ds:
                ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        elif name == "gender_hate":
            ds = load_dataset("ctoraman/gender-hate-speech")
            ds = ds.rename_column("Label", "label")
            ds = ds.rename_column("Text", "text")
            if "test" not in ds:
                ds = ds["train"].train_test_split(test_size=0.2, seed=42)
                ds = DatasetDict({
                    "train": ds["train"],
                    "test": ds["test"]
                })
        elif name == "bad_hate":
            ds = load_dataset("badmatr11x/hate-offensive-speech")
            # ds = ds.rename_column("Label", "label")
            ds = ds.rename_column("tweet", "text")


        elif name == "CNERG_hate":
            ds = load_dataset("Hate-speech-CNERG/hatexplain")
            ds = ds.rename_column("post", "text")
            ds = ds.rename_column("label", "label")
            if "test" not in ds:
                ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        elif name == "jigsaw-toxic-comment-classification":
            ds = load_dataset("jigsaw-toxic-comment-classification", split="train[:10%]")  # just a sample
            ds = ds.map(lambda x: {"text": x["comment_text"], "label": int(x["toxic"] > 0)})
            ds = ds.train_test_split(test_size=0.2, seed=42)
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        # Ensure test split exists
        if "test" not in ds:
            ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        
        # Quick test sampling
        if quick_test:
            ds = sample_dataset(ds)
    
        return ds