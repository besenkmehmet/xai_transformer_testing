#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Optional

from xmutant_transformer.mining import mine_errors_for_task, build_candidate_pools
from xmutant_transformer.probing import run_single_probe
from xmutant_transformer.models import train_aft as train_aft_model, evaluate_model
from xmutant_transformer.data_utils import load_split_data
from xmutant_transformer.utils import setup_logging, create_run_directory, update_status_file

class ExperimentRunner:
    def __init__(self, task: str, model: str, explainer: str, strategy: str, threshold: float,
                 foundation_dir: Path = Path("foundation"),
                 experiments_dir: Path = Path("experiments"),
                 experiment_name: Optional[str] = None,
                 max_probe_items: Optional[int] = None,
                 candidates_per_item: int = 3,
                 batch_size: int = 32):

        self.task = task
        self.model = model
        self.model_key = model.replace("/", "_").replace("-", "_")
        self.explainer = explainer
        self.max_probe_items = max_probe_items
        self.candidates_per_item = candidates_per_item
        self.batch_size = batch_size
        self.strategy = strategy
        self.threshold = threshold
        
        self.foundation_dir = foundation_dir
        self.experiments_dir = experiments_dir
        
        # Create experiment directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{task}_{self.model_key}_{explainer}_{strategy}_{threshold}_{timestamp}"
        
        self.experiment_dir = experiments_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.experiment_dir / "logs")
        
        # Validate foundation exists
        self._validate_foundation()
        
        # Save experiment config
        self._save_config()
    
    def _validate_foundation(self):
        """Validate that foundation data and models exist"""
        
        # Check foundation config
        foundation_config_path = self.foundation_dir / "config.json"
        if not foundation_config_path.exists():
            raise FileNotFoundError(f"Foundation not found. Run setup.py first.")
        
        with open(foundation_config_path, 'r') as f:
            foundation_config = json.load(f)
        
        if self.task not in foundation_config["tasks"]:
            raise ValueError(f"Task {self.task} not in foundation. Available: {foundation_config['tasks']}")
        
        if self.model not in foundation_config["models"]:
            raise ValueError(f"Model {self.model} not in foundation. Available: {foundation_config['models']}")
        
        # Check data files
        data_dir = self.foundation_dir / "data" / self.task
        for split in ["fit", "mining", "dev", "test"]:
            split_file = data_dir / f"{split}.jsonl"
            if not split_file.exists():
                raise FileNotFoundError(f"Data file missing: {split_file}")
        
        # Check model
        model_dir = self.foundation_dir / "models" / self.task / self.model_key
        if not model_dir.exists() or not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Model missing: {model_dir}")
        
        self.logger.info("Foundation validation passed")
    
    def _mine_errors_foundation(self):
        """Mine errors using foundation structure directly"""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from xmutant_transformer.data_utils import prepare_model_inputs
        from xmutant_transformer.mining import get_integrated_gradients, get_attention_rollout
        from xmutant_transformer.utils import load_jsonl
        
        # Load model and tokenizer
        model_path = self.foundation_dir / "models" / self.task / self.model_key
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
        model.eval()
        
        # Load mining data
        mining_file = self.foundation_dir / "data" / self.task / "mining.jsonl"
        mining_items = load_jsonl(mining_file)
        
        misclassified_items = []
        
        for item in mining_items:
            # Prepare input
            encodings, labels = prepare_model_inputs([item], tokenizer)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**encodings)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_label = torch.argmax(predictions, dim=-1).item()
            
            true_label = labels[0]
            
            # Check if misclassified
            if predicted_label != true_label:
                # Compute explanations
                explanations = {}
                
                if self.explainer == "ig":
                    try:
                        ig_attrs = get_integrated_gradients(model, tokenizer, encodings, true_label)
                        explanations["ig"] = ig_attrs
                    except Exception as e:
                        print(f"IG failed: {e}")
                        explanations["ig"] = [0.0] * len(encodings['input_ids'][0])
                
                elif self.explainer == "attn":
                    try:
                        attn_scores = get_attention_rollout(model, encodings)
                        explanations["attn"] = attn_scores[0].tolist()
                    except Exception as e:
                        print(f"Attention failed: {e}")
                        explanations["attn"] = [0.0] * len(encodings['input_ids'][0])
                
                # Get tokens
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
        
        return {
            "num_misclassified": len(misclassified_items),
            "total_mining_items": len(mining_items),
            "error_rate": len(misclassified_items) / len(mining_items),
            "misclassified_items": misclassified_items
        }
    
    def _build_pools_foundation(self, mining_result):
        """Build candidate pools from misclassified items"""
        from xmutant_transformer.mining import tokens_to_words
        from collections import Counter
        
        misclassified_items = mining_result["misclassified_items"]
        candidates = []
        
        for item in misclassified_items:
            if self.explainer not in item["explanations"]:
                continue
                
            explanations = item["explanations"][self.explainer]
            tokens = item["tokens"]
            
            # Convert tokens to words with aggregated scores
            words = tokens_to_words(tokens, explanations)
            
            if words:
                # Take top-1 word per error
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
        
        # Compute statistics
        all_words = [c["word"] for c in candidates]
        word_counts = Counter(all_words)
        unique_words = len(set(all_words))
        
        return {
            "candidates": candidates,
            "pool_size": len(candidates),
            "unique_words": unique_words,
            "top_10_words": word_counts.most_common(10),
            "word_counts": dict(word_counts)
        }
    
    def _save_config(self):
        """Save experiment configuration"""
        
        config = {
            "task": self.task,
            "model": self.model,
            "model_key": self.model_key,
            "explainer": self.explainer,
            "strategy": self.strategy,
            "threshold": self.threshold,
            "foundation_dir": str(self.foundation_dir),
            "experiment_dir": str(self.experiment_dir),
            "created_at": datetime.now().isoformat()
        }
        
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def mine_errors(self) -> dict:
        """Load mining results from foundation (already mined during setup)"""

        self.logger.info(f"Loading mining results for {self.explainer} explainer from foundation")
        update_status_file(self.experiment_dir, "mine", "⚡", "Loading mining results...")

        try:
            from xmutant_transformer.utils import load_jsonl, save_jsonl

            # Create mining results directory
            pools_dir = self.experiment_dir / "pools"
            pools_dir.mkdir(exist_ok=True)

            # Load pre-computed mining results from foundation
            foundation_mined_file = self.foundation_dir / "pools" / self.task / f"mined_errors_{self.model_key}.jsonl"
            foundation_pool_file = self.foundation_dir / "pools" / self.task / f"pool_{self.explainer}_{self.model_key}.jsonl"

            if not foundation_mined_file.exists():
                raise FileNotFoundError(f"Mining results not found in foundation: {foundation_mined_file}. Run setup.py first.")

            if not foundation_pool_file.exists():
                raise FileNotFoundError(f"Pool not found in foundation: {foundation_pool_file}. Run setup.py first.")

            # Load pre-computed results
            misclassified_items = load_jsonl(foundation_mined_file)
            candidates = load_jsonl(foundation_pool_file)

            # Copy pool to experiment directory
            target_pool_file = pools_dir / f"pool_{self.explainer}.jsonl"
            save_jsonl(target_pool_file, candidates)

            # Compute statistics
            from collections import Counter
            all_words = [c.get("word", c.get("token", "")) for c in candidates]
            word_counts = Counter(all_words)

            result = {
                "mining": {
                    "num_misclassified": len(misclassified_items),
                    "source": "foundation (pre-computed)"
                },
                "pool": {
                    "pool_size": len(candidates),
                    "unique_words": len(set(all_words)),
                    "top_10_words": word_counts.most_common(10),
                },
                "pool_file": str(target_pool_file)
            }

            # Save mining results
            with open(self.experiment_dir / "mining_results.json", "w") as f:
                json.dump(result, f, indent=2)

            update_status_file(self.experiment_dir, "mine", "✓", f"Loaded {len(misclassified_items)} errors from foundation")

            print(f"✓ Loaded {len(misclassified_items)} misclassified items from foundation")
            print(f"  Pool size: {len(candidates)} candidates")
            print(f"  Unique words: {len(set(all_words))}")
            print(f"  Top words: {', '.join([f'{w}({c})' for w, c in word_counts.most_common(5)])}")

            return result

        except Exception as e:
            update_status_file(self.experiment_dir, "mine", "✗", f"Failed: {str(e)}")
            raise
    
    def probe_edits(self) -> dict:
        """Run probing with specific strategy and threshold"""
        
        self.logger.info(f"Probing with {self.strategy} strategy, threshold {self.threshold}")
        update_status_file(self.experiment_dir, "probe", "⚡", f"Probing {self.strategy}@{self.threshold}...")
        
        try:
            # Create probes directory
            probes_dir = self.experiment_dir / "probes"
            probes_dir.mkdir(exist_ok=True)
            
            # Run custom probing
            result = self._probe_edits_foundation()
            
            # Save probe results
            with open(self.experiment_dir / "probe_results.json", "w") as f:
                json.dump(result, f, indent=2)
            
            success_rate = result["results"]["overall_success_rate"]
            update_status_file(self.experiment_dir, "probe", "✓", f"Success rate: {success_rate:.3f}")
            
            return result
            
        except Exception as e:
            update_status_file(self.experiment_dir, "probe", "✗", f"Failed: {str(e)}")
            raise
    
    def _probe_edits_foundation(self) -> dict:
        """Run probing using foundation structure directly"""
        import torch
        import random
        import numpy as np
        from tqdm import tqdm
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        from xmutant_transformer.data_utils import prepare_model_inputs
        from xmutant_transformer.utils import load_jsonl, save_jsonl
        from xmutant_transformer.probing import apply_edit, check_nli_gate, test_model_flip
        
        # Load candidate pool from experiment directory
        pool_file = self.experiment_dir / "pools" / f"pool_{self.explainer}.jsonl"
        candidates = load_jsonl(pool_file)

        # Load pre-classified items from foundation
        correctly_classified_file = self.foundation_dir / "pools" / self.task / f"correctly_classified_{self.model_key}.jsonl"

        if not correctly_classified_file.exists():
            raise FileNotFoundError(
                f"Correctly classified items not found: {correctly_classified_file}\n"
                "Please re-run setup.py to generate classification results."
            )

        print(f"Loading pre-classified items from foundation...")
        correct_items = load_jsonl(correctly_classified_file)

        # Load model for testing flips
        print(f"Loading SFT model for flip testing...")
        model_path = self.foundation_dir / "models" / self.task / self.model_key

        # Load NLI model once for all probing
        print(f"Loading NLI model for semantic equivalence checking...")
        from xmutant_transformer.probing import NLIChecker, FlipTester
        nli_checker = NLIChecker(model_name="cross-encoder/nli-deberta-v3-small")
        flip_tester = FlipTester(model_path)

        print(f"\n✓ Loaded {len(correct_items)} correctly classified items from foundation")
        print(f"✓ Starting probing with {len(candidates)} candidate words...")

        # Set random seed
        random.seed(42)

        # Run probing experiments
        attempts = []
        accepted_edits = []

        # Use max_probe_items if specified, otherwise use all correctly classified items
        if self.max_probe_items is not None:
            max_items = min(self.max_probe_items, len(correct_items))
        else:
            max_items = len(correct_items)

        candidates_per_item = self.candidates_per_item

        total_attempts = max_items * candidates_per_item
        print(f"✓ Processing {max_items} items × {candidates_per_item} candidates = {total_attempts} total attempts")
        print(f"  Strategy: {self.strategy}, Threshold: {self.threshold}")
        print(f"  Batch size: {self.batch_size} (for NLI and flip checks)")
        print(f"  Device: NLI={nli_checker.device}, SFT={flip_tester.device}\n")

        # Create progress bar
        pbar = tqdm(total=total_attempts, desc="Probing", unit="attempt")

        # Collect edits in batches for faster NLI processing
        batch = []
        batch_metadata = []

        def process_batch(batch, batch_metadata):
            """Process a batch of edits with batched NLI checks and flip tests"""
            if not batch:
                return

            import time
            t0 = time.time()

            # Skip NLI if threshold is negative (accept all)
            if self.threshold < 0:
                nli_results = [
                    {
                        "is_equivalent": True,
                        "forward_score": 1.0,
                        "backward_score": 1.0,
                        "passes_gate": True,
                        "threshold_used": self.threshold,
                    }
                    for _ in batch
                ]
            else:
                # Batch NLI check
                nli_results = nli_checker.check_batch_bidirectional_entailment(
                    batch, threshold=self.threshold
                )

            nli_time = time.time() - t0

            # Collect items that passed NLI for batch flip testing
            flip_test_items = []
            flip_test_indices = []

            for i, (meta, nli_result) in enumerate(zip(batch_metadata, nli_results)):
                if nli_result["passes_gate"]:
                    edited_item = {
                        "id": meta["item"]["id"] + "_edited",
                        "text_a": meta["edited_text_a"],
                        "text_b": meta["edited_text_b"],
                        "label": meta["item"]["label"]
                    }
                    flip_test_items.append((meta["item"], edited_item))
                    flip_test_indices.append(i)

            # Batch flip testing
            flip_results = {}
            flip_time = 0
            if flip_test_items:
                t1 = time.time()
                original_items = [pair[0] for pair in flip_test_items]
                edited_items = [pair[1] for pair in flip_test_items]
                batch_flip_results = flip_tester.test_batch_flips(original_items, edited_items)

                for idx, flip_result in zip(flip_test_indices, batch_flip_results):
                    flip_results[idx] = flip_result
                flip_time = time.time() - t1

            pbar.set_postfix(nli_ms=f"{nli_time*1000:.0f}", flip_ms=f"{flip_time*1000:.0f}", passed=len(flip_test_items))

            # Process each result
            for i, (meta, nli_result) in enumerate(zip(batch_metadata, nli_results)):
                accepted = nli_result["passes_gate"]
                flipped = False

                if not accepted:
                    pbar.set_description(f"'{meta['candidate_word']}' → NLI rejected")
                else:
                    flip_result = flip_results[i]
                    flipped = flip_result["flipped"]

                    if flipped:
                        edited_item = {
                            "id": meta["item"]["id"] + "_edited",
                            "text_a": meta["edited_text_a"],
                            "text_b": meta["edited_text_b"],
                            "label": meta["item"]["label"]
                        }
                        edit_pair = {
                            "original": meta["item"],
                            "edited": edited_item,
                            "candidate_word": meta["candidate_word"],
                            "signal": self.explainer,
                            "strategy": self.strategy,
                            "threshold": self.threshold
                        }
                        accepted_edits.append(edit_pair)
                        pbar.set_description(f"'{meta['candidate_word']}' → ✓✓ FLIPPED!")
                        tqdm.write(f"  ✓✓ FLIP #{len(accepted_edits)}: '{meta['candidate_word']}' at item {meta['item_idx']+1}")
                    else:
                        pbar.set_description(f"'{meta['candidate_word']}' → no flip")

                # Save attempt
                attempt = {
                    "item_id": meta["item"]["id"],
                    "candidate_word": meta["candidate_word"],
                    "signal": self.explainer,
                    "strategy": self.strategy,
                    "threshold": self.threshold,
                    "generator_rejected": False,
                    "nli_scores": {
                        "forward": nli_result["forward_score"],
                        "backward": nli_result["backward_score"]
                    },
                    "accepted": accepted,
                    "flipped": flipped
                }
                attempts.append(attempt)
                pbar.update(1)

        # Main loop - collect edits into batches
        for i, item in enumerate(correct_items[:max_items]):
            if not candidates:
                continue

            for j in range(candidates_per_item):
                candidate = random.choice(candidates)
                candidate_word = candidate.get("word", candidate.get("token", ""))

                try:
                    if self.strategy == "lm" and random.random() < 0.1:
                        # Generator rejection
                        attempt = {
                            "item_id": item["id"],
                            "candidate_word": candidate_word,
                            "signal": self.explainer,
                            "strategy": self.strategy,
                            "threshold": self.threshold,
                            "generator_rejected": True,
                            "nli_scores": None,
                            "accepted": False,
                            "flipped": False
                        }
                        attempts.append(attempt)
                        pbar.set_description(f"'{candidate_word}' → gen rejected")
                        pbar.update(1)
                        continue

                    # Apply edit
                    edited_text_a, edited_text_b = apply_edit(
                        item["text_a"], item.get("text_b"), candidate_word,
                        self.strategy,
                        position_hint=candidate.get("word_indices", [0])[0] if candidate.get("word_indices") else None
                    )

                    # Add to batch
                    batch.append((item["text_a"], edited_text_a))
                    batch_metadata.append({
                        "item": item,
                        "candidate": candidate,
                        "candidate_word": candidate_word,
                        "edited_text_a": edited_text_a,
                        "edited_text_b": edited_text_b,
                        "item_idx": i,
                        "candidate_idx": j
                    })

                    # Process batch when full
                    if len(batch) >= self.batch_size:
                        process_batch(batch, batch_metadata)
                        batch = []
                        batch_metadata = []

                except Exception as e:
                    tqdm.write(f"  Error processing item {item['id']}: {e}")
                    pbar.update(1)
                    continue

        # Process remaining items in batch
        if batch:
            process_batch(batch, batch_metadata)

        pbar.close()
        
        # Save results
        probes_dir = self.experiment_dir / "probes"
        attempts_file = probes_dir / "attempts.jsonl"
        accepted_file = probes_dir / "accepted.jsonl"

        save_jsonl(attempts_file, attempts)
        save_jsonl(accepted_file, accepted_edits)

        # Compute summary statistics
        total_proposals = len(attempts)
        generator_rejections = sum(1 for a in attempts if a["generator_rejected"])
        accepted_count = sum(1 for a in attempts if a["accepted"])
        flipped_count = sum(1 for a in attempts if a["flipped"])

        print(f"\n{'='*60}")
        print(f"PROBING RESULTS")
        print(f"{'='*60}")
        print(f"Total proposals:        {total_proposals}")
        print(f"Generator rejections:   {generator_rejections} ({generator_rejections/total_proposals*100:.1f}%)" if total_proposals > 0 else "")
        print(f"NLI accepted:           {accepted_count} ({accepted_count/total_proposals*100:.1f}%)" if total_proposals > 0 else "")
        print(f"Model flipped:          {flipped_count} ({flipped_count/total_proposals*100:.1f}%)" if total_proposals > 0 else "")
        print(f"Overall success rate:   {flipped_count/total_proposals:.3f}" if total_proposals > 0 else "")
        print(f"{'='*60}\n")
        
        generator_rejection_rate = generator_rejections / total_proposals if total_proposals > 0 else 0
        acceptance_rate = accepted_count / total_proposals if total_proposals > 0 else 0
        flip_on_accepted_rate = flipped_count / accepted_count if accepted_count > 0 else 0
        overall_success_rate = flipped_count / total_proposals if total_proposals > 0 else 0
        
        # NLI score distributions
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
                "task": self.task,
                "model": self.model_key,
                "explainer": self.explainer,
                "strategy": self.strategy,
                "threshold": self.threshold
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
    
    def train_aft(self) -> dict:
        """Train after-training model with accepted edits"""
        
        self.logger.info("Training AFT model")
        update_status_file(self.experiment_dir, "train_aft", "⚡", "Training AFT...")
        
        try:
            # Check if we have accepted edits
            accepted_file = self.experiment_dir / "probes" / "accepted.jsonl"
            if not accepted_file.exists():
                raise FileNotFoundError("No accepted edits found. Run probe first.")
            
            # Load accepted edits
            with open(accepted_file, 'r') as f:
                accepted_edits = [json.loads(line) for line in f]
            
            if not accepted_edits:
                self.logger.warning("No accepted edits to train AFT model")
                print("\n⚠ No accepted edits to train AFT model. Skipping AFT training.")
                return {"message": "No accepted edits"}

            print(f"\n{'='*60}")
            print(f"AFT TRAINING")
            print(f"{'='*60}")

            # Load original training data (fit set)
            fit_data_file = self.foundation_dir / "data" / self.task / "fit.jsonl"
            with open(fit_data_file, 'r') as f:
                fit_items = [json.loads(line) for line in f]

            # Load mining set
            mining_data_file = self.foundation_dir / "data" / self.task / "mining.jsonl"
            with open(mining_data_file, 'r') as f:
                mining_items = [json.loads(line) for line in f]

            # Combine: fit set + mining set + generated adversarial examples
            augmented_items = fit_items + mining_items + [edit_pair["edited"] for edit_pair in accepted_edits]

            print(f"Original fit set:        {len(fit_items)} items")
            print(f"Mining set:              {len(mining_items)} items")
            print(f"Generated adversarial:   {len(accepted_edits)} items")
            print(f"Total training data:     {len(augmented_items)} items")
            print(f"{'='*60}\n")
            
            # Train AFT model
            aft_dir = self.experiment_dir / "models" / "aft"
            aft_dir.mkdir(parents=True, exist_ok=True)

            print(f"Training AFT model on augmented dataset...")

            # Prepare datasets
            from transformers import AutoTokenizer
            from xmutant_transformer.models import train_model, TextDataset, prepare_model_inputs

            tokenizer = AutoTokenizer.from_pretrained(self.model)

            train_encodings, train_labels = prepare_model_inputs(augmented_items, tokenizer)

            # Load dev data for evaluation
            dev_data_file = self.foundation_dir / "data" / self.task / "dev.jsonl"
            with open(dev_data_file, 'r') as f:
                dev_items = [json.loads(line) for line in f]

            dev_encodings, dev_labels = prepare_model_inputs(dev_items, tokenizer)

            train_dataset = TextDataset(train_encodings, train_labels)
            dev_dataset = TextDataset(dev_encodings, dev_labels)

            # Determine number of labels
            unique_labels = set(train_labels + dev_labels)
            num_labels = len(unique_labels)

            # Train AFT model
            training_result = train_model(
                model_name=self.model,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                output_dir=aft_dir,
                num_labels=num_labels,
                seed=42,
                num_epochs=1,
                early_stopping_patience=0
            )

            result = {
                "fit_data_size": len(fit_items),
                "mining_data_size": len(mining_items),
                "generated_adversarial_size": len(accepted_edits),
                "total_training_size": len(augmented_items),
                "model_path": str(aft_dir),
                "training_metrics": training_result
            }
            
            # Save AFT results
            with open(self.experiment_dir / "aft_results.json", "w") as f:
                json.dump(result, f, indent=2)

            update_status_file(self.experiment_dir, "train_aft", "✓", f"Trained on {len(augmented_items)} examples")

            print(f"✓ AFT model saved to: {aft_dir}")

            return result
            
        except Exception as e:
            update_status_file(self.experiment_dir, "train_aft", "✗", f"Failed: {str(e)}")
            raise
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline"""

        print(f"\n{'='*60}")
        print(f"STARTING XMUTANT EXPERIMENT")
        print(f"{'='*60}")
        print(f"Task:        {self.task}")
        print(f"Model:       {self.model}")
        print(f"Explainer:   {self.explainer}")
        print(f"Strategy:    {self.strategy}")
        print(f"Threshold:   {self.threshold}")
        print(f"Experiment:  {self.experiment_dir.name}")
        print(f"{'='*60}\n")

        self.logger.info("Starting full experiment")
        self.logger.info(f"Task: {self.task}, Model: {self.model}")
        self.logger.info(f"Explainer: {self.explainer}, Strategy: {self.strategy}, Threshold: {self.threshold}")
        
        try:
            # Run pipeline
            mining_result = self.mine_errors()
            probe_result = self.probe_edits()
            aft_result = self.train_aft()
            
            # Save summary
            summary = {
                "experiment_config": {
                    "task": self.task,
                    "model": self.model,
                    "explainer": self.explainer,
                    "strategy": self.strategy,
                    "threshold": self.threshold
                },
                "results": {
                    "mining": mining_result,
                    "probe": probe_result,
                    "aft": aft_result
                },
                "completed_at": datetime.now().isoformat()
            }
            
            with open(self.experiment_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n{'='*60}")
            print(f"✓ EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Results saved to: {self.experiment_dir}")
            print(f"{'='*60}\n")

            self.logger.info("Experiment completed successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Run focused XMutant experiment")
    
    parser.add_argument("--task", type=str, required=True,
                       help="Task name (e.g., sst2)")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., distilbert-base-uncased)")
    parser.add_argument("--explainer", type=str, required=True,
                       choices=["ig", "attn"],
                       help="Explainer type")
    parser.add_argument("--strategy", type=str, required=True,
                       choices=["lm", "random", "prefix"],
                       help="Placement strategy")
    parser.add_argument("--threshold", type=float, required=True,
                       help="NLI threshold")
    
    parser.add_argument("--steps", type=str,
                       help="Comma-separated steps to run (mine,probe,aft)")
    parser.add_argument("--foundation-dir", type=Path, default=Path("foundation"),
                       help="Foundation directory path")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"),
                       help="Experiments directory path")
    parser.add_argument("--experiment-name", type=str,
                       help="Custom experiment name")
    parser.add_argument("--max-probe-items", type=int,
                       help="Maximum number of items to probe (default: all)")
    parser.add_argument("--candidates-per-item", type=int, default=3,
                       help="Number of candidates to test per item (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for NLI and flip checks (default: 64, higher = faster)")

    args = parser.parse_args()

    try:
        runner = ExperimentRunner(
            task=args.task,
            model=args.model,
            explainer=args.explainer,
            strategy=args.strategy,
            threshold=args.threshold,
            foundation_dir=args.foundation_dir,
            experiments_dir=args.experiments_dir,
            experiment_name=args.experiment_name,
            max_probe_items=args.max_probe_items,
            candidates_per_item=args.candidates_per_item,
            batch_size=args.batch_size
        )
        
        if args.steps:
            steps = [s.strip() for s in args.steps.split(",")]
            for step in steps:
                if step == "mine":
                    runner.mine_errors()
                elif step == "probe":
                    runner.probe_edits()
                elif step == "aft":
                    runner.train_aft()
                else:
                    print(f"Unknown step: {step}")
        else:
            # Run full experiment
            runner.run_full_experiment()
        
        print(f"✓ Experiment completed: {runner.experiment_dir}")
        
    except Exception as e:
        print(f"✗ Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()