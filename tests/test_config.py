import os
from pathlib import Path

import pytest

from span.config import Config, VerificationConfig, load_config


def test_default_config() -> None:
    config = Config()

    assert config.model == "claude-sonnet-4-20250514"
    assert config.api_key_env == "ANTHROPIC_API_KEY"
    assert ".git" in config.ignore
    assert config.max_steps == 15
    assert config.max_retries_per_step == 3


def test_verification_config_defaults() -> None:
    verification = VerificationConfig()

    assert verification.syntax is True
    assert verification.ruff is True
    assert verification.mypy is False
    assert verification.mypy_full is True
    assert verification.pytest is True
    assert verification.pytest_args == ["-x", "--tb=short"]


def test_load_config_nonexistent_returns_defaults(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    config = load_config()

    assert config.model == "claude-sonnet-4-20250514"


def test_load_config_explicit_path_nonexistent_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "nonexistent.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(config_path)


def test_load_config_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "span.yaml"
    config_path.write_text("""
model: claude-opus-4-20250514
api_key_env: MY_API_KEY
max_steps: 20

verification:
  ruff: false
  mypy: true
  pytest_args: ["-v"]

ignore:
  - ".git"
  - "build"
""")

    config = load_config(config_path)

    assert config.model == "claude-opus-4-20250514"
    assert config.api_key_env == "MY_API_KEY"
    assert config.max_steps == 20
    assert config.verification.ruff is False
    assert config.verification.mypy is True
    assert config.verification.pytest_args == ["-v"]
    assert "build" in config.ignore


def test_load_config_partial_values(tmp_path: Path) -> None:
    config_path = tmp_path / "span.yaml"
    config_path.write_text("""
model: claude-opus-4-20250514
""")

    config = load_config(config_path)

    assert config.model == "claude-opus-4-20250514"
    assert config.max_steps == 15
    assert config.verification.ruff is True


def test_load_config_empty_file(tmp_path: Path) -> None:
    config_path = tmp_path / "span.yaml"
    config_path.write_text("")

    config = load_config(config_path)

    assert config.model == "claude-sonnet-4-20250514"


def test_api_key_property() -> None:
    config = Config(api_key_env="TEST_API_KEY")

    os.environ["TEST_API_KEY"] = "test-key-123"
    assert config.api_key == "test-key-123"

    del os.environ["TEST_API_KEY"]
    assert config.api_key is None


def test_config_with_test_patterns(tmp_path: Path) -> None:
    config_path = tmp_path / "span.yaml"
    config_path.write_text("""
test_patterns:
  - "tests/"
  - "test_*.py"

fallback_tests:
  - "tests/test_smoke.py"
""")

    config = load_config(config_path)

    assert config.test_patterns == ["tests/", "test_*.py"]
    assert config.fallback_tests == ["tests/test_smoke.py"]
