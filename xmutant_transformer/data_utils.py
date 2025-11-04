import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
import json

from .config import ExperimentConfig
from .utils import save_jsonl, load_jsonl

def load_task_data(task_name: str, max_items: int = None) -> Tuple[Dataset, Dataset, Dataset]:
    if task_name.lower() == "sst2":
        dataset = load_dataset("glue", "sst2")
        train_data = dataset["train"]
        dev_data = dataset["validation"]
        test_data = dataset["test"]

    elif task_name.lower() == "qnli":
        dataset = load_dataset("glue", "qnli")
        train_data = dataset["train"]
        dev_data = dataset["validation"]
        test_data = dataset["test"]

    else:
        raise ValueError(f"Unsupported task: {task_name}")

    if max_items:
        train_data = train_data.select(range(min(max_items, len(train_data))))
        dev_data = dev_data.select(range(min(max_items // 10, len(dev_data))))
        test_data = test_data.select(range(min(max_items // 10, len(test_data))))

    return train_data, dev_data, test_data

def load_adversarial_data(task_name: str, max_items: int = None) -> Dataset:
    try:
        task_to_adv_config = {
            "sst2": "adv_sst2",
            "qnli": "adv_qnli",
            "mnli": "adv_mnli",
            "qqp": "adv_qqp",
            "rte": "adv_rte"
        }

        adv_config_name = task_to_adv_config.get(task_name.lower())
        if not adv_config_name:
            print(f"Warning: No adversarial config found for task {task_name}")
            return None

        adv_dataset = load_dataset("AI-Secure/adv_glue", adv_config_name)

        if "validation" in adv_dataset:
            adv_dev_data = adv_dataset["validation"]
        elif "test" in adv_dataset:
            adv_dev_data = adv_dataset["test"]
        else:
            raise ValueError(f"No suitable split found in adversarial dataset for {task_name}")

        if max_items:
            adv_dev_data = adv_dev_data.select(range(min(max_items // 10, len(adv_dev_data))))

        return adv_dev_data

    except Exception as e:
        print(f"Warning: Could not load adversarial data for {task_name}: {e}")
        return None

def split_dataset(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    results = {}

    for task in config.tasks:
        train_data, dev_data, test_data = load_task_data(
            task,
            max_items=config.max_items if config.debug else None
        )

        train_indices = list(range(len(train_data)))
        np.random.seed(config.seed)

        fit_indices, mining_indices = train_test_split(
            train_indices,
            test_size=config.mining_split_ratio,
            random_state=config.seed,
            stratify=[train_data[i]["label"] for i in train_indices]
        )

        fit_data = train_data.select(fit_indices)
        mining_data = train_data.select(mining_indices)

        fit_items = convert_to_standard_format(fit_data, task, fit_indices)
        mining_items = convert_to_standard_format(mining_data, task, mining_indices)
        dev_items = convert_to_standard_format(dev_data, task)
        test_items = convert_to_standard_format(test_data, task)

        task_dir = run_dir / "data" / task
        task_dir.mkdir(parents=True, exist_ok=True)

        save_jsonl(task_dir / "fit.jsonl", fit_items)
        save_jsonl(task_dir / "mining.jsonl", mining_items)
        save_jsonl(task_dir / "dev.jsonl", dev_items)
        save_jsonl(task_dir / "test.jsonl", test_items)

        def get_label_dist(items):
            labels = [item["label"] for item in items]
            unique, counts = np.unique(labels, return_counts=True)
            return dict(zip(unique.tolist(), counts.tolist()))
        
        results[task] = {
            "splits": {
                "fit": len(fit_items),
                "mining": len(mining_items),
                "dev": len(dev_items),
                "test": len(test_items)
            },
            "label_distributions": {
                "fit": get_label_dist(fit_items),
                "mining": get_label_dist(mining_items),
                "dev": get_label_dist(dev_items),
                "test": get_label_dist(test_items),
                "overall": get_label_dist(fit_items + mining_items + dev_items + test_items)
            }
        }

    with open(run_dir / "results" / "split_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def convert_to_standard_format(dataset: Dataset, task: str, indices: List[int] = None) -> List[Dict[str, Any]]:
    items = []

    for i, item in enumerate(dataset):
        original_idx = indices[i] if indices is not None else item.get("idx", i)
        
        if task.lower() == "sst2":
            standard_item = {
                "id": f"{task}_{original_idx}",
                "text_a": item["sentence"],
                "text_b": None,
                "label": item["label"]
            }
        elif task.lower() == "qnli":
            standard_item = {
                "id": f"{task}_{original_idx}",
                "text_a": item["question"],
                "text_b": item["sentence"],
                "label": item["label"]
            }
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        items.append(standard_item)
    
    return items

def load_split_data(run_dir: Path, task: str, split: str) -> List[Dict[str, Any]]:
    filepath = run_dir / "data" / task / f"{split}.jsonl"
    return load_jsonl(filepath)

def get_label_names(task: str) -> List[str]:
    if task.lower() == "sst2":
        return ["negative", "positive"]
    elif task.lower() == "qnli":
        return ["entailment", "not_entailment"]
    else:
        raise ValueError(f"Unsupported task: {task}")

def prepare_model_inputs(items: List[Dict[str, Any]], tokenizer, max_length: int = 512):
    texts_a = [item["text_a"] for item in items]
    texts_b = [item["text_b"] for item in items if item["text_b"] is not None]

    if len(texts_b) == 0:
        encodings = tokenizer(
            texts_a,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    else:
        encodings = tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    if 'input_ids' in encodings:
        encodings['input_ids'] = encodings['input_ids'].long()

    labels = [item["label"] for item in items]
    return encodings, labels