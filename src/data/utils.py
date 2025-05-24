from datasets import DatasetDict, Dataset
import numpy as np

def sample_dataset(ds: DatasetDict) -> DatasetDict:
    """Returns small portion of dataset"""
    return DatasetDict({
        "train": ds["train"].shuffle(seed=42).select(range(200)),
        "test": ds["test"].shuffle(seed=42).select(range(50))
    })


def process_hc3(ds: DatasetDict) -> DatasetDict:
    """Flattens HC3 dataset into text-label pairs for human (0) and ChatGPT (1) answers."""
    def flatten_examples(example):
        results = []
        for text in example.get("human_answers", []):
            results.append({"text": text, "label": 0})
        for text in example.get("chatgpt_answers", []):
            results.append({"text": text, "label": 1})
        return results
    
    # Flatten all splits
    processed = {}
    for split in ds:
        rows = []
        for example in ds[split]:
            rows.extend(flatten_examples(example))
        processed[split] = Dataset.from_list(rows)
    
    return DatasetDict(processed)
