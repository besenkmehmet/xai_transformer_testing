import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from tqdm import tqdm

from .config import ExperimentConfig
from .data_utils import load_split_data, prepare_model_inputs
from .utils import save_jsonl, load_jsonl, save_partial_results, load_partial_results

def get_attention_rollout(model, inputs, head_fusion="mean"):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

    num_layers = len(attentions)
    batch_size, num_heads, seq_len, _ = attentions[0].shape

    if head_fusion == "mean":
        attentions = [attn.mean(dim=1) for attn in attentions]
    elif head_fusion == "max":
        attentions = [attn.max(dim=1)[0] for attn in attentions]

    identity = torch.eye(seq_len).unsqueeze(0).to(attentions[0].device)
    attentions = [(attn + identity) / 2 for attn in attentions]

    attentions = [attn / attn.sum(dim=-1, keepdim=True) for attn in attentions]

    rollout = attentions[0]
    for i in range(1, num_layers):
        rollout = torch.matmul(attentions[i], rollout)

    cls_attention = rollout[:, 0, :]

    return cls_attention.cpu().numpy()

def get_integrated_gradients(model, tokenizer, inputs, target_class):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)
    embeddings.requires_grad_(True)

    def forward_func(embeddings):
        model_inputs = {"inputs_embeds": embeddings, "attention_mask": attention_mask}
        if "token_type_ids" in inputs:
            model_inputs["token_type_ids"] = inputs["token_type_ids"]
        return model(**model_inputs).logits
    
    ig = IntegratedGradients(forward_func)

    baseline = torch.zeros_like(embeddings)

    attributions = ig.attribute(embeddings, baseline, target=target_class, n_steps=50)

    token_attributions = attributions.sum(dim=-1).squeeze(0)
    scores = token_attributions.detach().cpu().numpy()

    if scores.ndim == 0:
        scores = [float(scores)]
    else:
        scores = scores.tolist()

    return scores

def mine_errors_for_task(config: ExperimentConfig, run_dir: Path, task: str,
                        model_name: str) -> Dict[str, Any]:
    model_key = model_name.replace("/", "_").replace("-", "_")
    model_path = run_dir / "models" / task / model_key

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.eval()

    mining_items = load_split_data(run_dir, task, "mining")

    partial_file = run_dir / "pools" / task / f"mining_partial_{model_key}.pkl"
    partial_data = load_partial_results(partial_file)
    
    if partial_data:
        start_idx = partial_data["last_processed"] + 1
        misclassified_items = partial_data["misclassified_items"]
        correctly_classified_items = partial_data.get("correctly_classified_items", [])
        print(f"Resuming from item {start_idx}")
    else:
        start_idx = 0
        misclassified_items = []
        correctly_classified_items = []

    for i in tqdm(range(start_idx, len(mining_items)), desc=f"Mining {task} {model_key}"):
        item = mining_items[i]

        encodings, labels = prepare_model_inputs([item], tokenizer)

        encodings = {k: v.long() if k == 'input_ids' else v for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()

        true_label = labels[0]

        if predicted_label != true_label:

            explanations = {}

            if "ig" in config.signals:
                try:
                    ig_attrs = get_integrated_gradients(model, tokenizer, encodings, predicted_label)
                    explanations["ig"] = ig_attrs
                except Exception as e:
                    print(f"IG failed for item {i}: {e}")
                    explanations["ig"] = [0.0] * len(encodings['input_ids'][0])

            if "attn" in config.signals:
                try:
                    attn_scores = get_attention_rollout(model, encodings)
                    explanations["attn"] = attn_scores[0].tolist()
                except Exception as e:
                    print(f"Attention failed for item {i}: {e}")
                    explanations["attn"] = [0.0] * len(encodings['input_ids'][0])

            tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])
            
            misclassified_item = {
                "id": item["id"],
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "prediction_confidence": predictions[0][predicted_label].item(),
                "tokens": tokens,
                "explanations": explanations
            }

            misclassified_items.append(misclassified_item)
        else:
            correctly_classified_item = {
                "id": item["id"],
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "label": item["label"],
                "predicted_label": predicted_label,
                "prediction_confidence": predictions[0][predicted_label].item()
            }
            correctly_classified_items.append(correctly_classified_item)

        if i % 100 == 99:
            save_partial_results(partial_file, {
                "last_processed": i,
                "misclassified_items": misclassified_items,
                "correctly_classified_items": correctly_classified_items
            })

    misclassified_file = run_dir / "pools" / task / f"mined_errors_{model_key}.jsonl"
    correctly_classified_file = run_dir / "pools" / task / f"correctly_classified_{model_key}.jsonl"
    misclassified_file.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(misclassified_file, misclassified_items)
    save_jsonl(correctly_classified_file, correctly_classified_items)

    if partial_file.exists():
        partial_file.unlink()

    return {
        "num_misclassified": len(misclassified_items),
        "num_correctly_classified": len(correctly_classified_items),
        "total_mining_items": len(mining_items),
        "error_rate": len(misclassified_items) / len(mining_items),
        "accuracy": len(correctly_classified_items) / len(mining_items),
        "misclassified_file": str(misclassified_file),
        "correctly_classified_file": str(correctly_classified_file)
    }

def tokens_to_words(tokens: List[str], scores: List[float]) -> List[Tuple[str, float, List[int]]]:
    words = []
    current_word = ""
    current_score = 0.0
    current_indices = []

    punctuation = {'.', ',', '?', '!', ';', ':', '-', '(', ')', '[', ']', '{', '}', "'", '"',
                   '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`'}

    for i, (token, score) in enumerate(zip(tokens, scores)):
        if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", "[UNK]"]:
            continue

        if token in punctuation:
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_score += abs(score)
            current_indices.append(i)
        else:
            if current_word:
                words.append((current_word, current_score, current_indices))

            current_word = token
            current_score = abs(score)
            current_indices = [i]

    if current_word:
        words.append((current_word, current_score, current_indices))

    return words

def build_candidate_pools(config: ExperimentConfig, run_dir: Path, task: str,
                         model_name: str) -> Dict[str, Any]:
    model_key = model_name.replace("/", "_").replace("-", "_")

    misclassified_file = run_dir / "pools" / task / f"mined_errors_{model_key}.jsonl"
    misclassified_items = load_jsonl(misclassified_file)

    pools = {}

    for signal in config.signals:
        candidates = []

        for item in misclassified_items:
            if signal not in item["explanations"]:
                continue

            explanations = item["explanations"][signal]
            tokens = item["tokens"]

            words = tokens_to_words(tokens, explanations)

            if words:
                top_word, top_score, top_indices = max(words, key=lambda x: x[1])

                candidate = {
                    "item_id": item["id"],
                    "word": top_word,
                    "word_indices": top_indices,
                    "attribution_score": top_score,
                    "original_tokens": [tokens[i] for i in top_indices],
                    "original_scores": [explanations[i] for i in top_indices]
                }

                candidates.append(candidate)
            else:
                print(f"Warning: No non-punctuation tokens found for {item['id']}, skipping...")

        pool_file = run_dir / "pools" / task / f"pool_{signal}_{model_key}.jsonl"
        save_jsonl(pool_file, candidates)

        all_words = [c["word"] for c in candidates]
        word_counts = Counter(all_words)
        unique_words = len(set(all_words))

        pools[signal] = {
            "pool_file": str(pool_file),
            "pool_size": len(candidates),
            "unique_words": unique_words,
            "top_10_words": word_counts.most_common(10),
            "word_counts": dict(word_counts)
        }

    return pools

def compute_pool_overlap(config: ExperimentConfig, run_dir: Path, task: str,
                        model_name: str) -> Dict[str, Any]:
    if len(config.signals) < 2:
        return {}

    model_key = model_name.replace("/", "_").replace("-", "_")

    pools_data = {}
    for signal in config.signals:
        pool_file = run_dir / "pools" / task / f"pool_{signal}_{model_key}.jsonl"
        pool_items = load_jsonl(pool_file)
        pools_data[signal] = set(item.get("word", item.get("token", "")) for item in pool_items)

    overlaps = {}
    signals = list(config.signals)
    
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            signal_a, signal_b = signals[i], signals[j]
            set_a, set_b = pools_data[signal_a], pools_data[signal_b]
            
            intersection = set_a & set_b
            union = set_a | set_b
            
            jaccard = len(intersection) / len(union) if union else 0
            
            overlaps[f"{signal_a}_{signal_b}"] = {
                "intersection_size": len(intersection),
                "jaccard_index": jaccard,
                "intersection_tokens": list(intersection)[:10]
            }

    return overlaps

def mine_errors(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "mining_results.json"

    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for task in config.tasks:
        task_results = {}

        for model_name in config.models:
            model_key = model_name.replace("/", "_").replace("-", "_")

            print(f"Mining errors for {task} with {model_name}")

            mining_results = mine_errors_for_task(config, run_dir, task, model_name)

            pool_results = build_candidate_pools(config, run_dir, task, model_name)

            overlap_results = compute_pool_overlap(config, run_dir, task, model_name)

            task_results[model_key] = {
                "mining": mining_results,
                "pools": pool_results,
                "overlaps": overlap_results
            }

        results[task] = task_results

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results