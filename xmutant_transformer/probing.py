import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import openai
import os

from .config import ExperimentConfig
from .data_utils import load_split_data, prepare_model_inputs
from .utils import save_jsonl, load_jsonl, save_partial_results, load_partial_results


class LMGuidedInjector:

    def __init__(self, model_name="openai/gpt-oss-20b", api_key=None, base_url="http://localhost:1234/v1"):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or "https://openrouter.ai/api/v1",
        )

    def batch_inject_word_into_sentences(self, candidate_word: str, items: List[Dict]) -> Dict[str, Dict]:
        sentences_list = []
        for i, item in enumerate(items):
            if item.get("text_b"):
                sentences_list.append(f'ID{i}: "{item["text_a"]}" ||| "{item["text_b"]}"')
            else:
                sentences_list.append(f'ID{i}: "{item["text_a"]}"')

        sentences_str = "\n".join(sentences_list)

        prompt = f"""You are given the word "{candidate_word}" and multiple sentences. For EACH sentence, try to insert the word naturally. Be creative and flexible - ACCEPT insertions even if they slightly modify the tone, as long as they are grammatically correct.

Word to insert: "{candidate_word}"

Sentences:
{sentences_str}

For each sentence, output one line in this EXACT format:
ID<number> | STATUS: <ACCEPTED or REJECTED> | RESULT: <edited sentence>

Rules:
- Be flexible and creative with insertions
- ACCEPT if the word can fit grammatically, even if it changes the tone slightly
- Only REJECT if the word makes no sense or creates nonsense
- For sentence pairs (with |||), insert into ONE sentence only
- Try to maximize ACCEPTED insertions
- Return ALL sentences in order

Example:
ID0 | STATUS: ACCEPTED | RESULT: The haunting film was great
ID1 | STATUS: ACCEPTED | RESULT: It was boring and lacking excitement
ID2 | STATUS: ACCEPTED | RESULT: A story about loss ||| It explores haunting
grief
ID3 | STATUS: REJECTED | RESULT:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise text editor. Follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                extra_body={"provider": {"only": ["groq"]}}
            )

            result_text = response.choices[0].message.content.strip()

            print("\n" + "="*60)
            print(f"LLM Response for word '{candidate_word}' (first 500 chars):")
            print("="*60)
            print(result_text[:500] + "..." if len(result_text) > 500 else result_text)
            print("="*60 + "\n")

            return self._parse_sentence_batch_response(result_text, items)

        except Exception as e:
            print(f"LLM batch injection failed: {e}, marking all as rejected")
            return {item["id"]: {"rejected": True, "edited_text_a": item["text_a"], "edited_text_b": item.get("text_b")} for item in items}

    def _parse_sentence_batch_response(self, response_text: str, items: List[Dict]) -> Dict[str, Dict]:
        results = {}
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or not line.startswith("ID"):
                continue

            try:
                parts = line.split(" | ")
                id_part = parts[0].strip()
                status_part = parts[1].replace("STATUS:", "").strip()
                result_part = parts[2].replace("RESULT:", "").strip()

                idx = int(id_part.replace("ID", ""))
                if idx >= len(items):
                    continue

                item = items[idx]
                item_id = item["id"]

                rejected = "REJECT" in status_part.upper()

                if rejected:
                    results[item_id] = {
                        "rejected": True,
                        "edited_text_a": item["text_a"],
                        "edited_text_b": item.get("text_b")
                    }
                else:
                    if " ||| " in result_part:
                        sent1, sent2 = result_part.split(" ||| ", 1)
                        results[item_id] = {
                            "rejected": False,
                            "edited_text_a": sent1.strip(),
                            "edited_text_b": sent2.strip()
                        }
                    else:
                        results[item_id] = {
                            "rejected": False,
                            "edited_text_a": result_part.strip(),
                            "edited_text_b": None
                        }
            except Exception as e:
                print(f"Failed to parse line: {line}, error: {e}")
                continue

        for item in items:
            if item["id"] not in results:
                results[item["id"]] = {
                    "rejected": True,
                    "edited_text_a": item["text_a"],
                    "edited_text_b": item.get("text_b")
                }

        return results


class NLIChecker:
    def __init__(self, model_name="roberta-large-mnli", device=None, use_fp16=False):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        if self.use_fp16:
            self.model = self.model.half()

        self.model.eval()

    def check_bidirectional_entailment(self, sentence1, sentence2, threshold=0.9):
        prob1_2 = self._get_entailment_probability(sentence1, sentence2)
        prob2_1 = self._get_entailment_probability(sentence2, sentence1)

        is_equivalent = prob1_2 >= threshold and prob2_1 >= threshold

        return {
            "is_equivalent": is_equivalent,
            "forward_score": prob1_2,
            "backward_score": prob2_1,
            "passes_gate": is_equivalent,
            "threshold_used": threshold,
        }

    def _get_entailment_probability(self, premise, hypothesis):
        with torch.no_grad():
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True
            ).to(self.device)
            logits = self.model(**inputs).logits

            probabilities = F.softmax(logits, dim=-1)[0]

            entailment_label_id = self.model.config.label2id.get("entailment", 2)

            return probabilities[entailment_label_id].item()

    def check_batch_bidirectional_entailment(self, sentence_pairs, threshold=0.9):
        if not sentence_pairs:
            return []

        forward_premises = [pair[0] for pair in sentence_pairs]
        forward_hypotheses = [pair[1] for pair in sentence_pairs]
        backward_premises = [pair[1] for pair in sentence_pairs]
        backward_hypotheses = [pair[0] for pair in sentence_pairs]

        with torch.no_grad():
            forward_inputs = self.tokenizer(
                forward_premises, forward_hypotheses,
                return_tensors="pt", truncation=True, padding=True
            ).to(self.device)

            backward_inputs = self.tokenizer(
                backward_premises, backward_hypotheses,
                return_tensors="pt", truncation=True, padding=True
            ).to(self.device)

            forward_logits = self.model(**forward_inputs).logits
            backward_logits = self.model(**backward_inputs).logits

            forward_probs = F.softmax(forward_logits, dim=-1)
            backward_probs = F.softmax(backward_logits, dim=-1)

            entailment_label_id = self.model.config.label2id.get("entailment", 2)

            forward_scores = forward_probs[:, entailment_label_id].cpu().tolist()
            backward_scores = backward_probs[:, entailment_label_id].cpu().tolist()

        results = []
        for fwd_score, bwd_score in zip(forward_scores, backward_scores):
            is_equivalent = fwd_score >= threshold and bwd_score >= threshold
            results.append({
                "is_equivalent": is_equivalent,
                "forward_score": fwd_score,
                "backward_score": bwd_score,
                "passes_gate": is_equivalent,
                "threshold_used": threshold,
            })

        return results

def apply_edit(text_a: str, text_b: Optional[str], candidate_token: str,
               strategy: str, position_hint: int = None, lm_injector=None) -> Tuple[str, Optional[str]]:
    if strategy == "prefix":
        edited_text_a = candidate_token + " " + text_a
        return edited_text_a, text_b

    elif strategy == "random":
        if text_b is not None:
            edit_text_a = random.random() < 0.5
            if edit_text_a:
                words = text_a.split()
                if len(words) == 0:
                    return candidate_token + " " + text_a, text_b
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, candidate_token)
                edited_text_a = " ".join(words)
                return edited_text_a, text_b
            else:
                words = text_b.split()
                if len(words) == 0:
                    return text_a, candidate_token + " " + text_b
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, candidate_token)
                edited_text_b = " ".join(words)
                return text_a, edited_text_b
        else:
            words = text_a.split()
            if len(words) == 0:
                return candidate_token + " " + text_a, text_b
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, candidate_token)
            edited_text_a = " ".join(words)
            return edited_text_a, text_b

    elif strategy == "lm":
        if lm_injector is None:
            raise ValueError("LM strategy requires lm_injector to be provided")
        return lm_injector.inject_word(text_a, candidate_token, text_b)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def check_nli_gate(original_text_a: str, original_text_b: Optional[str],
                  edited_text_a: str, edited_text_b: Optional[str],
                  nli_classifier, threshold: float) -> Dict[str, Any]:
    if original_text_b is not None:
        original_premise = original_text_a
        original_hypothesis = original_text_b
        edited_premise = edited_text_a
        edited_hypothesis = edited_text_b or original_text_b
    else:
        original_premise = original_text_a
        original_hypothesis = original_text_a
        edited_premise = edited_text_a
        edited_hypothesis = edited_text_a

    forward_input = f"{original_premise} [SEP] {edited_hypothesis}"
    forward_result = nli_classifier(forward_input)
    forward_entailment_score = next(
        (r["score"] for r in forward_result if r["label"] == "ENTAILMENT"), 0.0
    )

    backward_input = f"{edited_premise} [SEP] {original_hypothesis}"
    backward_result = nli_classifier(backward_input)
    backward_entailment_score = next(
        (r["score"] for r in backward_result if r["label"] == "ENTAILMENT"), 0.0
    )

    passes_gate = (forward_entailment_score >= threshold and
                   backward_entailment_score >= threshold)

    return {
        "forward_score": forward_entailment_score,
        "backward_score": backward_entailment_score,
        "passes_gate": passes_gate,
        "threshold_used": threshold
    }

class FlipTester:
    def __init__(self, model_path: Path, device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def test_flip(self, original_item: Dict[str, Any], edited_item: Dict[str, Any]) -> Dict[str, Any]:
        orig_encodings, orig_labels = prepare_model_inputs([original_item], self.tokenizer)
        edit_encodings, edit_labels = prepare_model_inputs([edited_item], self.tokenizer)

        orig_encodings = {k: v.to(self.device) for k, v in orig_encodings.items()}
        edit_encodings = {k: v.to(self.device) for k, v in edit_encodings.items()}

        with torch.no_grad():
            orig_outputs = self.model(**orig_encodings)
            orig_predictions = torch.nn.functional.softmax(orig_outputs.logits, dim=-1)
            orig_predicted_label = torch.argmax(orig_predictions, dim=-1).item()

            edit_outputs = self.model(**edit_encodings)
            edit_predictions = torch.nn.functional.softmax(edit_outputs.logits, dim=-1)
            edit_predicted_label = torch.argmax(edit_predictions, dim=-1).item()

        return {
            "original_prediction": orig_predicted_label,
            "edited_prediction": edit_predicted_label,
            "flipped": orig_predicted_label != edit_predicted_label,
            "original_confidence": orig_predictions[0][orig_predicted_label].item(),
            "edited_confidence": edit_predictions[0][edit_predicted_label].item()
        }

    def test_batch_flips(self, original_items: List[Dict[str, Any]],
                         edited_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not original_items:
            return []

        orig_encodings, orig_labels = prepare_model_inputs(original_items, self.tokenizer)
        edit_encodings, edit_labels = prepare_model_inputs(edited_items, self.tokenizer)

        orig_encodings = {k: v.to(self.device) for k, v in orig_encodings.items()}
        edit_encodings = {k: v.to(self.device) for k, v in edit_encodings.items()}

        with torch.no_grad():
            orig_outputs = self.model(**orig_encodings)
            orig_predictions = torch.nn.functional.softmax(orig_outputs.logits, dim=-1)
            orig_predicted_labels = torch.argmax(orig_predictions, dim=-1)

            edit_outputs = self.model(**edit_encodings)
            edit_predictions = torch.nn.functional.softmax(edit_outputs.logits, dim=-1)
            edit_predicted_labels = torch.argmax(edit_predictions, dim=-1)

        results = []
        for i in range(len(original_items)):
            orig_label = orig_predicted_labels[i].item()
            edit_label = edit_predicted_labels[i].item()

            results.append({
                "original_prediction": orig_label,
                "edited_prediction": edit_label,
                "flipped": orig_label != edit_label,
                "original_confidence": orig_predictions[i][orig_label].item(),
                "edited_confidence": edit_predictions[i][edit_label].item()
            })

        return results

def test_model_flip(model_path: Path, original_item: Dict[str, Any],
                   edited_item: Dict[str, Any]) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.eval()

    orig_encodings, orig_labels = prepare_model_inputs([original_item], tokenizer)
    with torch.no_grad():
        orig_outputs = model(**orig_encodings)
        orig_predictions = torch.nn.functional.softmax(orig_outputs.logits, dim=-1)
        orig_predicted_label = torch.argmax(orig_predictions, dim=-1).item()

    edit_encodings, edit_labels = prepare_model_inputs([edited_item], tokenizer)
    with torch.no_grad():
        edit_outputs = model(**edit_encodings)
        edit_predictions = torch.nn.functional.softmax(edit_outputs.logits, dim=-1)
        edit_predicted_label = torch.argmax(edit_predictions, dim=-1).item()

    return {
        "original_prediction": orig_predicted_label,
        "edited_prediction": edit_predicted_label,
        "flipped": orig_predicted_label != edit_predicted_label,
        "original_confidence": orig_predictions[0][orig_predicted_label].item(),
        "edited_confidence": edit_predictions[0][edit_predicted_label].item()
    }

def run_single_probe(config: ExperimentConfig, run_dir: Path, task: str,
                    model_name: str, signal: str, strategy: str,
                    threshold: float) -> Dict[str, Any]:
    model_key = model_name.replace("/", "_").replace("-", "_")
    model_path = run_dir / "models" / "sft" / task / model_key

    lm_injector = None
    if strategy == "lm":
        print("Initializing LM-guided injector with LM Studio...")
        lm_injector = LMGuidedInjector(
            model_name="openai/gpt-oss-20b",
            base_url="http://localhost:1234/v1"
        )

    pool_file = run_dir / "pools" / task / f"pool_{signal}_{model_key}.jsonl"
    candidates = load_jsonl(pool_file)

    mining_items = load_split_data(run_dir, task, "mining")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.eval()
    
    correct_items = []
    for item in mining_items:
        encodings, labels = prepare_model_inputs([item], tokenizer)

        with torch.no_grad():
            outputs = model(**encodings)
            predicted_label = torch.argmax(outputs.logits, dim=-1).item()

        true_label = labels[0]

        if predicted_label == true_label:
            correct_items.append(item)
    
    print(f"Found {len(correct_items)} correctly classified items out of {len(mining_items)} in mining split")
    
    random.seed(config.seed)

    attempts = []
    accepted_edits = []

    probe_id = f"{signal}_{strategy}_{threshold}"
    partial_file = run_dir / "probes" / task / f"probe_partial_{model_key}_{probe_id}.pkl"

    partial_data = load_partial_results(partial_file)
    if partial_data:
        start_idx = partial_data["last_processed"] + 1
        attempts = partial_data["attempts"]
        accepted_edits = partial_data["accepted_edits"]
        print(f"Resuming from item {start_idx}")
    else:
        start_idx = 0
    
    max_items = min(len(correct_items), config.max_items or len(correct_items))

    unique_candidate_words = []
    if strategy == "lm":
        seen_words = set()
        for cand in candidates:
            word = cand.get("word", cand.get("token", ""))
            if word and word not in seen_words:
                unique_candidate_words.append(word)
                seen_words.add(word)
        print(f"Found {len(unique_candidate_words)} unique candidate words for LM batching")

    for i in tqdm(range(start_idx, max_items), desc=f"Probing {probe_id}"):
        item = correct_items[i]

        if not candidates:
            continue

        try:
            if strategy == "lm":
                batch_results = lm_injector.batch_inject_words(
                    item["text_a"],
                    unique_candidate_words,
                    item["text_b"]
                )

                for candidate_word, result in batch_results.items():
                    if result["rejected"]:
                        attempt = {
                            "item_id": item["id"],
                            "candidate_word": candidate_word,
                            "signal": signal,
                            "strategy": strategy,
                            "threshold": threshold,
                            "generator_rejected": True,
                            "nli_scores": None,
                            "accepted": False,
                            "flipped": False
                        }
                        attempts.append(attempt)
                        continue

                    edited_text_a = result["edited_text_a"]
                    edited_text_b = result["edited_text_b"]

                    edited_item = {
                        "id": item["id"] + "_edited_" + candidate_word,
                        "text_a": edited_text_a,
                        "text_b": edited_text_b,
                        "label": item["label"]
                    }

                    nli_result = check_nli_gate(
                        item["text_a"], item["text_b"],
                        edited_text_a, edited_text_b,
                        "microsoft/deberta-large-mnli",
                        threshold
                    )

                    accepted = nli_result["passes_gate"]
                    flipped = False

                    if accepted:
                        flip_result = test_model_flip(model_path, item, edited_item)
                        flipped = flip_result["flipped"]

                        if flipped:
                            edit_pair = {
                                "original": item,
                                "edited": edited_item,
                                "candidate_word": candidate_word,
                                "signal": signal,
                                "strategy": strategy,
                                "threshold": threshold
                            }
                            accepted_edits.append(edit_pair)

                    attempt = {
                        "item_id": item["id"],
                        "candidate_word": candidate_word,
                        "signal": signal,
                        "strategy": strategy,
                        "threshold": threshold,
                        "generator_rejected": False,
                        "nli_scores": {
                            "forward": nli_result["forward_score"],
                            "backward": nli_result["backward_score"]
                        },
                        "accepted": accepted,
                        "flipped": flipped
                    }
                    attempts.append(attempt)

            else:
                candidate = random.choice(candidates)
                candidate_word = candidate.get("word", candidate.get("token", ""))

                edited_text_a, edited_text_b = apply_edit(
                    item["text_a"],
                    item["text_b"],
                    candidate_word,
                    strategy,
                    position_hint=candidate.get("word_indices", [0])[0] if candidate.get("word_indices") else None,
                    lm_injector=lm_injector
                )

                edited_item = {
                    "id": item["id"] + "_edited",
                    "text_a": edited_text_a,
                    "text_b": edited_text_b,
                    "label": item["label"]
                }

                nli_result = check_nli_gate(
                    item["text_a"], item["text_b"],
                    edited_text_a, edited_text_b,
                    "microsoft/deberta-large-mnli",
                    threshold
                )

                accepted = nli_result["passes_gate"]
                flipped = False

                if accepted:
                    flip_result = test_model_flip(model_path, item, edited_item)
                    flipped = flip_result["flipped"]

                    if flipped:
                        edit_pair = {
                            "original": item,
                            "edited": edited_item,
                            "candidate_word": candidate_word,
                            "signal": signal,
                            "strategy": strategy,
                            "threshold": threshold
                        }
                        accepted_edits.append(edit_pair)

                attempt = {
                    "item_id": item["id"],
                    "candidate_word": candidate_word,
                    "signal": signal,
                    "strategy": strategy,
                    "threshold": threshold,
                    "generator_rejected": False,
                    "nli_scores": {
                        "forward": nli_result["forward_score"],
                        "backward": nli_result["backward_score"]
                    },
                    "accepted": accepted,
                    "flipped": flipped
                }
                attempts.append(attempt)

        except Exception as e:
            print(f"Error processing item {item['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if i % 100 == 99:
            save_partial_results(partial_file, {
                "last_processed": i,
                "attempts": attempts,
                "accepted_edits": accepted_edits
            })

    results_dir = run_dir / "probes" / task
    results_dir.mkdir(parents=True, exist_ok=True)

    attempts_file = results_dir / f"attempts_{model_key}_{probe_id}.jsonl"
    accepted_file = results_dir / f"accepted_{model_key}_{probe_id}.jsonl"

    save_jsonl(attempts_file, attempts)
    save_jsonl(accepted_file, accepted_edits)

    if partial_file.exists():
        partial_file.unlink()

    total_proposals = len(attempts)
    generator_rejections = sum(1 for a in attempts if a["generator_rejected"])
    accepted_count = sum(1 for a in attempts if a["accepted"])
    flipped_count = sum(1 for a in attempts if a["flipped"])
    
    generator_rejection_rate = generator_rejections / total_proposals if total_proposals > 0 else 0
    acceptance_rate = accepted_count / total_proposals if total_proposals > 0 else 0
    flip_on_accepted_rate = flipped_count / accepted_count if accepted_count > 0 else 0
    overall_success_rate = flipped_count / total_proposals if total_proposals > 0 else 0

    valid_attempts = [a for a in attempts if not a["generator_rejected"] and a["nli_scores"]]
    if valid_attempts:
        forward_scores = [a["nli_scores"]["forward"] for a in valid_attempts]
        backward_scores = [a["nli_scores"]["backward"] for a in valid_attempts]
        
        nli_stats = {
            "forward": {
                "mean": np.mean(forward_scores),
                "median": np.median(forward_scores),
                "q25": np.percentile(forward_scores, 25),
                "q75": np.percentile(forward_scores, 75)
            },
            "backward": {
                "mean": np.mean(backward_scores),
                "median": np.median(backward_scores),
                "q25": np.percentile(backward_scores, 25),
                "q75": np.percentile(backward_scores, 75)
            }
        }
    else:
        nli_stats = None
    
    return {
        "probe_config": {
            "task": task,
            "model": model_key,
            "signal": signal,
            "strategy": strategy,
            "threshold": threshold
        },
        "results": {
            "total_proposals": total_proposals,
            "generator_rejections": generator_rejections,
            "generator_rejection_rate": generator_rejection_rate,
            "accepted_count": accepted_count,
            "acceptance_rate": acceptance_rate,
            "flipped_count": flipped_count,
            "flip_on_accepted_rate": flip_on_accepted_rate,
            "overall_success_rate": overall_success_rate,
            "nli_score_distributions": nli_stats
        },
        "files": {
            "attempts": str(attempts_file),
            "accepted": str(accepted_file)
        }
    }

def run_probes(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    results = {}

    probe_configs = []
    for task in config.tasks:
        for model_name in config.models:
            for signal in config.signals:
                for strategy in config.strategies:
                    for threshold in config.thresholds:
                        probe_configs.append((task, model_name, signal, strategy, threshold))

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []

        for task, model_name, signal, strategy, threshold in probe_configs:
            future = executor.submit(
                run_single_probe, config, run_dir, task, model_name,
                signal, strategy, threshold
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running probes"):
            try:
                result = future.result()
                
                task = result["probe_config"]["task"]
                model = result["probe_config"]["model"]
                signal = result["probe_config"]["signal"]
                strategy = result["probe_config"]["strategy"]
                threshold = result["probe_config"]["threshold"]
                
                if task not in results:
                    results[task] = {}
                if model not in results[task]:
                    results[task][model] = {}
                if signal not in results[task][model]:
                    results[task][model][signal] = {}
                if strategy not in results[task][model][signal]:
                    results[task][model][signal][strategy] = {}
                
                results[task][model][signal][strategy][threshold] = result
                
            except Exception as e:
                print(f"Probe failed: {e}")
                continue

    with open(run_dir / "results" / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def transfer_edits(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    if len(config.models) < 2:
        return {"message": "Transfer requires at least 2 models"}

    results = {}

    source_model = config.models[0]
    target_models = config.models[1:]
    
    for task in config.tasks:
        task_results = {}

        source_key = source_model.replace("/", "_").replace("-", "_")

        default_signal = config.signals[0]
        default_strategy = config.strategies[0]
        default_threshold = config.thresholds[2]

        accepted_file = (run_dir / "probes" / task /
                        f"accepted_{source_key}_{default_signal}_{default_strategy}_{default_threshold}.jsonl")
        
        if not accepted_file.exists():
            continue
            
        accepted_edits = load_jsonl(accepted_file)
        
        for target_model in target_models:
            target_key = target_model.replace("/", "_").replace("-", "_")
            target_model_path = run_dir / "models" / "sft" / task / target_key

            transfer_results = []

            for edit_pair in accepted_edits:
                original_item = edit_pair["original"]
                edited_item = edit_pair["edited"]

                nli_result = check_nli_gate(
                    original_item["text_a"], original_item["text_b"],
                    edited_item["text_a"], edited_item["text_b"],
                    "microsoft/deberta-large-mnli",
                    default_threshold
                )

                flip_result = test_model_flip(target_model_path, original_item, edited_item)
                
                transfer_result = {
                    "original_id": original_item["id"],
                    "nli_accepted": nli_result["passes_gate"],
                    "flipped_on_target": flip_result["flipped"],
                    "nli_scores": {
                        "forward": nli_result["forward_score"],
                        "backward": nli_result["backward_score"]
                    }
                }
                
                transfer_results.append(transfer_result)

            total_edits = len(transfer_results)
            nli_accepted = sum(1 for r in transfer_results if r["nli_accepted"])
            flipped_on_target = sum(1 for r in transfer_results if r["flipped_on_target"])

            nli_acceptance_rate = nli_accepted / total_edits if total_edits > 0 else 0
            flip_on_accepted_rate = flipped_on_target / nli_accepted if nli_accepted > 0 else 0
            
            task_results[target_key] = {
                "source_model": source_key,
                "total_edits_tested": total_edits,
                "nli_accepted": nli_accepted,
                "nli_acceptance_rate": nli_acceptance_rate,
                "flipped_on_target": flipped_on_target,
                "flip_on_accepted_rate": flip_on_accepted_rate,
                "transfer_config": {
                    "signal": default_signal,
                    "strategy": default_strategy,
                    "threshold": default_threshold
                }
            }
        
        results[task] = task_results

    with open(run_dir / "results" / "transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results