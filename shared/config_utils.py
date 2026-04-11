"""
shared/config_utils.py
──────────────────────
YAML config loading with dependency check, used by both updaters.
"""

import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_yaml_config(config_path: Path) -> dict:
    """
    Load and return a YAML config file as a dict.
    Exits with a clear message if pyyaml is missing or the file doesn't exist.
    """
    if not HAS_YAML:
        print("✗ pyyaml not installed.  Run:  pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)
