from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import json

@dataclass
class ExperimentConfig:
    tasks: List[str]
    models: List[str]
    signals: List[str]
    strategies: List[str]
    thresholds: List[float]
    mining_split_ratio: float
    seed: int
    max_items: Optional[int] = None
    debug: bool = False

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'ExperimentConfig':
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, output_path: Path):
        with open(output_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

def create_default_config() -> ExperimentConfig:
    return ExperimentConfig(
        tasks=["sst2", "qnli"],
        models=["distilbert-base-uncased", "roberta-base"],
        signals=["ig", "attn"],
        strategies=["lm", "random", "prefix"],
        thresholds=[0.99, 0.95, 0.90, 0.80, 0.70],
        mining_split_ratio=0.1,
        seed=42
    )
