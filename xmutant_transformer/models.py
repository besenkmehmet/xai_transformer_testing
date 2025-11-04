import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from .config import ExperimentConfig
from .data_utils import load_split_data, prepare_model_inputs, get_label_names
from .utils import save_checkpoint, load_checkpoint

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() if torch.is_tensor(val[idx]) else torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model_name: str, train_dataset: Dataset, eval_dataset: Dataset,
                output_dir: Path, num_labels: int, learning_rate: float = 2e-5,
                batch_size: int = 16, num_epochs: int = 3, seed: int = 42,
                early_stopping_patience: int = 3) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    eval_results = trainer.evaluate()

    return {
        "model_path": str(output_dir),
        "eval_results": eval_results,
        "training_args": training_args.to_dict()
    }

def evaluate_model(model_path: Path, eval_items: List[Dict[str, Any]],
                   task: str, batch_size: int = 32) -> Dict[str, Any]:
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_predictions = []
    all_predicted_labels = []
    all_labels = []

    num_batches = (len(eval_items) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(eval_items), batch_size), total=num_batches, desc="Evaluating"):
        batch_items = eval_items[i:i+batch_size]
        encodings, labels = prepare_model_inputs(batch_items, tokenizer)

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)

        all_predictions.append(predictions.cpu().numpy())
        all_predicted_labels.append(predicted_labels.cpu().numpy())
        all_labels.extend(labels)

    predictions_np = np.vstack(all_predictions)
    predicted_labels_np = np.concatenate(all_predicted_labels)
    labels_np = np.array(all_labels)

    accuracy = accuracy_score(labels_np, predicted_labels_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, predicted_labels_np, average='weighted'
    )

    label_names = get_label_names(task)
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels_np, predicted_labels_np, average=None
    )

    per_class_metrics = {}
    for i, label_name in enumerate(label_names):
        per_class_metrics[label_name] = {
            "precision": per_class_precision[i],
            "recall": per_class_recall[i],
            "f1": per_class_f1[i]
        }

    cm = confusion_matrix(labels_np, predicted_labels_np)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions_np.tolist(),
        "predicted_labels": predicted_labels_np.tolist(),
        "true_labels": labels_np.tolist()
    }

def train_sft(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    results = {}

    for task in config.tasks:
        task_results = {}

        fit_items = load_split_data(run_dir, task, "fit")
        dev_items = load_split_data(run_dir, task, "dev")

        num_labels = len(get_label_names(task))

        for model_name in config.models:
            model_key = model_name.replace("/", "_").replace("-", "_")

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            train_encodings, train_labels = prepare_model_inputs(fit_items, tokenizer)
            dev_encodings, dev_labels = prepare_model_inputs(dev_items, tokenizer)

            train_dataset = TextDataset(train_encodings, train_labels)
            dev_dataset = TextDataset(dev_encodings, dev_labels)

            output_dir = run_dir / "models" / "sft" / task / model_key
            
            training_result = train_model(
                model_name=model_name,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                output_dir=output_dir,
                num_labels=num_labels,
                seed=config.seed
            )

            test_items = load_split_data(run_dir, task, "test")
            mining_items = load_split_data(run_dir, task, "mining")

            dev_metrics = evaluate_model(output_dir, dev_items, task)
            test_metrics = evaluate_model(output_dir, test_items, task)
            mining_metrics = evaluate_model(output_dir, mining_items, task)

            task_results[model_key] = {
                "training": training_result,
                "dev_metrics": dev_metrics,
                "test_metrics": test_metrics,
                "mining_metrics": mining_metrics
            }

        results[task] = task_results

    with open(run_dir / "results" / "sft_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def train_aft(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    results = {}

    for task in config.tasks:
        task_results = {}

        fit_items = load_split_data(run_dir, task, "fit")
        dev_items = load_split_data(run_dir, task, "dev")

        default_signal = config.signals[0]
        default_strategy = config.strategies[0]
        default_threshold = config.thresholds[2]

        edits_file = (run_dir / "probes" / task /
                     f"accepted_{default_signal}_{default_strategy}_{default_threshold}.jsonl")
        
        if not edits_file.exists():
            raise FileNotFoundError(f"Accepted edits file not found: {edits_file}")

        from .utils import load_jsonl
        accepted_edits = load_jsonl(edits_file)

        augmented_items = fit_items + accepted_edits

        num_labels = len(get_label_names(task))

        for model_name in config.models:
            model_key = model_name.replace("/", "_").replace("-", "_")

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            train_encodings, train_labels = prepare_model_inputs(augmented_items, tokenizer)
            dev_encodings, dev_labels = prepare_model_inputs(dev_items, tokenizer)

            train_dataset = TextDataset(train_encodings, train_labels)
            dev_dataset = TextDataset(dev_encodings, dev_labels)

            output_dir = run_dir / "models" / "aft" / task / model_key

            training_result = train_model(
                model_name=model_name,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                output_dir=output_dir,
                num_labels=num_labels,
                seed=config.seed
            )

            test_items = load_split_data(run_dir, task, "test")
            test_metrics = evaluate_model(output_dir, test_items, task)

            task_results[model_key] = {
                "training": training_result,
                "test_metrics": test_metrics,
                "augmentation_config": {
                    "signal": default_signal,
                    "strategy": default_strategy,
                    "threshold": default_threshold,
                    "num_augmented": len(accepted_edits)
                }
            }

        results[task] = task_results

    with open(run_dir / "results" / "aft_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results