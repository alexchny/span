import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VerificationConfig:
    syntax: bool = True
    ruff: bool = True
    mypy: bool = False
    mypy_full: bool = True
    pytest: bool = True
    pytest_args: list[str] = field(default_factory=lambda: ["-x", "--tb=short"])


@dataclass
class Config:
    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    ignore: list[str] = field(default_factory=lambda: [".git", "__pycache__", ".venv", "node_modules", ".span"])
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    test_patterns: list[str] = field(default_factory=lambda: ["tests/"])
    fallback_tests: list[str] = field(default_factory=list)
    max_steps: int = 15
    max_retries_per_step: int = 3

    @property
    def api_key(self) -> str | None:
        """Get API key from environment variable."""
        return os.getenv(self.api_key_env)


def load_config(config_path: Path | None = None) -> Config:
    if config_path is None:
        config_path = Path.cwd() / "span.yaml"

    if not config_path.exists():
        if config_path != Path.cwd() / "span.yaml":
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return _dict_to_config(data)


def _dict_to_config(data: dict[str, Any]) -> Config:
    verification_data = data.pop("verification", {})
    verification = VerificationConfig(
        syntax=verification_data.get("syntax", True),
        ruff=verification_data.get("ruff", True),
        mypy=verification_data.get("mypy", False),
        mypy_full=verification_data.get("mypy_full", True),
        pytest=verification_data.get("pytest", True),
        pytest_args=verification_data.get("pytest_args", ["-x", "--tb=short"]),
    )

    return Config(
        model=data.get("model", "claude-sonnet-4-20250514"),
        api_key_env=data.get("api_key_env", "ANTHROPIC_API_KEY"),
        ignore=data.get("ignore", [".git", "__pycache__", ".venv", "node_modules", ".span"]),
        verification=verification,
        test_patterns=data.get("test_patterns", ["tests/"]),
        fallback_tests=data.get("fallback_tests", []),
        max_steps=data.get("max_steps", 15),
        max_retries_per_step=data.get("max_retries_per_step", 3),
    )
