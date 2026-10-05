"""Register only two additional seeds of the already frozen FedAvg control."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REFERENCE = 'research/capacity_compute_control/configs/final/FedAvg-seed-42.json'
REFERENCE_SHA256 = 'aa640fc7fcb72b5cd88a70db7de1942ff25823d59b61ecfde93f1817d294a8bf'
ORIGINAL_RUNNER_SHA256 = 'cd2319c112d44e53f9ddc1ad9c187c68f409ec632d08c9dfb74f912ad6fbefc3'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_final_config(config):
    reference = json.loads((REPO / REFERENCE).read_text())
    if canonical_hash(reference) != REFERENCE_SHA256:
        raise ValueError('Original frozen FedAvg reference changed')
    if type(config.get('seed')) is not int or config['seed'] not in (45, 46):
        raise ValueError('Only the two additional final seeds are registered')
    expected = {**reference, 'seed': config['seed']}
    if config != expected:
        raise ValueError('Configuration must match the original control except for seed')
    return reference
