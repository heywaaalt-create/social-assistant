from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _BASE_DIR / "config"


def load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings() -> dict[str, Any]:
    return load_yaml("settings.yaml")


def load_persona() -> dict[str, Any]:
    return load_yaml("persona.yaml")


def init_env() -> None:
    load_dotenv(_BASE_DIR / ".env")
