#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import json
from typing import List

from xmutant_transformer.config import ExperimentConfig
from xmutant_transformer.data_utils import split_dataset, load_task_data, convert_to_standard_format, load_adversarial_data
from xmutant_transformer.models import train_model, TextDataset, prepare_model_inputs
from xmutant_transformer.utils import setup_logging, save_jsonl
from xmutant_transformer.mining import mine_errors

def setup_foundation(tasks: List[str], models: List[str], 
                    mining_split_ratio: float = 0.1, seed: int = 42,
                    max_items: int = None, force: bool = False):
    """Set up global foundation: data splits and SFT models"""
    
    base_dir = Path("foundation")
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    
    # Setup logging
    logger = setup_logging(base_dir / "logs")
    
    logger.info("Setting up XMutant foundation...")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    
    # Create foundation directories
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data for each task
    for task in tasks:
        logger.info(f"Setting up data for {task}")
        
        task_data_dir = data_dir / task
        task_data_dir.mkdir(exist_ok=True)
        
        # Check if data already exists
        if not force and all((task_data_dir / f"{split}.jsonl").exists() 
                           for split in ["fit", "mining", "dev", "test"]):
            logger.info(f"Data for {task} already exists, skipping...")
            continue
        
        # Load and split data
        train_data, dev_data, test_data = load_task_data(task, max_items)
        
        # Split training data
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        train_indices = list(range(len(train_data)))
        np.random.seed(seed)
        
        fit_indices, mining_indices = train_test_split(
            train_indices,
            test_size=mining_split_ratio,
            random_state=seed,
            stratify=[train_data[i]["label"] for i in train_indices]
        )
        
        # Create splits
        fit_data = train_data.select(fit_indices)
        mining_data = train_data.select(mining_indices)
        
        # Convert to standard format
        fit_items = convert_to_standard_format(fit_data, task, fit_indices)
        mining_items = convert_to_standard_format(mining_data, task, mining_indices)
        dev_items = convert_to_standard_format(dev_data, task)
        test_items = convert_to_standard_format(test_data, task)
        
        # Save splits
        save_jsonl(task_data_dir / "fit.jsonl", fit_items)
        save_jsonl(task_data_dir / "mining.jsonl", mining_items)
        save_jsonl(task_data_dir / "dev.jsonl", dev_items)
        save_jsonl(task_data_dir / "test.jsonl", test_items)

        # Load and save adversarial dev set
        logger.info(f"Loading adversarial dev data for {task}")
        adv_dev_data = load_adversarial_data(task, max_items)
        if adv_dev_data is not None:
            adv_dev_items = convert_to_standard_format(adv_dev_data, task)
            save_jsonl(task_data_dir / "adv_dev.jsonl", adv_dev_items)
            logger.info(f"Saved adversarial dev set: {len(adv_dev_items)} items")
        else:
            logger.warning(f"No adversarial data available for {task}")
        
        # Save split statistics
        stats = {
            "task": task,
            "splits": {
                "fit": len(fit_items),
                "mining": len(mining_items),
                "dev": len(dev_items),
                "test": len(test_items)
            },
            "mining_split_ratio": mining_split_ratio,
            "seed": seed
        }
        
        with open(task_data_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Created {task} splits: fit={len(fit_items)}, mining={len(mining_items)}, dev={len(dev_items)}, test={len(test_items)}")
    
    # Train SFT models for each task-model combination
    for task in tasks:
        for model_name in models:
            model_key = model_name.replace("/", "_").replace("-", "_")
            
            logger.info(f"Training SFT model: {task} + {model_name}")
            
            task_model_dir = models_dir / task / model_key
            
            # Check if model already exists
            if not force and task_model_dir.exists() and (task_model_dir / "config.json").exists():
                logger.info(f"Model {task}/{model_key} already exists, skipping...")
                continue
            
            task_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Load training data
            task_data_dir = data_dir / task
            with open(task_data_dir / "fit.jsonl", "r") as f:
                fit_items = [json.loads(line) for line in f]
            with open(task_data_dir / "dev.jsonl", "r") as f:
                dev_items = [json.loads(line) for line in f]
            
            # Prepare datasets
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            train_encodings, train_labels = prepare_model_inputs(fit_items, tokenizer)
            dev_encodings, dev_labels = prepare_model_inputs(dev_items, tokenizer)
            
            train_dataset = TextDataset(train_encodings, train_labels)
            dev_dataset = TextDataset(dev_encodings, dev_labels)
            
            # Determine number of labels
            unique_labels = set(train_labels + dev_labels)
            num_labels = len(unique_labels)
            
            # Train model
            training_result = train_model(
                model_name=model_name,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                output_dir=task_model_dir,
                num_labels=num_labels,
                seed=seed
            )
            
            logger.info(f"Completed training {task}/{model_key}")
    
    # Mine errors with both signals after all models are trained
    logger.info("Starting error mining with both signals...")
    pools_dir = base_dir / "pools"
    pools_dir.mkdir(exist_ok=True)
    
    # Create config for mining with both signals
    mining_config = ExperimentConfig(
        tasks=tasks,
        models=models,
        signals=["ig", "attn"],
        strategies=[],  # Not used for mining
        thresholds=[],  # Not used for mining
        mining_split_ratio=mining_split_ratio,
        seed=seed,
        max_items=max_items
    )
    
    # Run mining for each task/model combination
    mining_results = mine_errors(mining_config, base_dir)
    
    # Save mining results summary
    with open(base_dir / "mining_results.json", "w") as f:
        json.dump(mining_results, f, indent=2)
    
    logger.info("Error mining completed!")
    
    # Save foundation config
    foundation_config = {
        "tasks": tasks,
        "models": models,
        "mining_split_ratio": mining_split_ratio,
        "seed": seed,
        "max_items": max_items
    }
    
    with open(base_dir / "config.json", "w") as f:
        json.dump(foundation_config, f, indent=2)
    
    logger.info("Foundation setup completed!")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Models: {models_dir}")
    
    return base_dir

def main():
    parser = argparse.ArgumentParser(description="Setup XMutant foundation data and models")
    
    parser.add_argument("--tasks", type=str, required=True, 
                       help="Comma-separated list of tasks (e.g., sst2,qnli)")
    parser.add_argument("--models", type=str, required=True,
                       help="Comma-separated list of models (e.g., distilbert-base-uncased,roberta-base)")
    parser.add_argument("--mining-split-ratio", type=float, default=0.1,
                       help="Ratio for mining split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--max-items", type=int,
                       help="Max items per split for debugging")
    parser.add_argument("--force", action="store_true",
                       help="Force re-creation even if files exist")
    
    args = parser.parse_args()
    
    tasks = [t.strip() for t in args.tasks.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    
    try:
        setup_foundation(
            tasks=tasks,
            models=models,
            mining_split_ratio=args.mining_split_ratio,
            seed=args.seed,
            max_items=args.max_items,
            force=args.force
        )
        print("✓ Foundation setup completed successfully!")
        
    except Exception as e:
        print(f"✗ Foundation setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()