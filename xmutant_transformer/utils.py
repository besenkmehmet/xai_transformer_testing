import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

def setup_logging(log_dir: Path, level: int = logging.INFO):
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_dir / 'experiment.log')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('xmutant')
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def save_checkpoint(filepath: Path, data: Dict[str, Any]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_checkpoint(filepath: Path) -> Optional[Dict[str, Any]]:
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def save_partial_results(filepath: Path, data: Any):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_partial_results(filepath: Path) -> Any:
    if not filepath.exists():
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_jsonl(filepath: Path, data: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl(filepath: Path) -> list:
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_run_directory(base_dir: Path, run_name: Optional[str] = None) -> Path:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ["data", "models", "pools", "probes", "results", "checkpoints"]:
        (run_dir / subdir).mkdir(exist_ok=True)

    return run_dir

def update_status_file(run_dir: Path, step: str, status: str, message: str = ""):
    status_file = run_dir / "STATUS.txt"

    timestamp = datetime.now().strftime("%H:%M:%S")
    status_line = f"[{status}] {step:<15} - {timestamp} - {message}"

    if status_file.exists():
        with open(status_file, 'r') as f:
            lines = f.readlines()

        updated = False
        for i, line in enumerate(lines):
            if step in line:
                lines[i] = status_line + '\n'
                updated = True
                break

        if not updated:
            lines.append(status_line + '\n')
    else:
        header = f"EXPERIMENT STATUS - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "=" * 60 + "\n"
        lines = [header, status_line + '\n']

    with open(status_file, 'w') as f:
        f.writelines(lines)
