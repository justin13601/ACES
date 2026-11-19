"""Tests for the optional ``aces_version`` key, which declares the ACES versions a task config is written
for."""

import re

import pytest

from aces.config import TaskExtractorConfig

TASK_YAML = """
predicates:
  admission:
    code: ADMISSION
trigger: admission
windows:
  input:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
"""


def load(tmp_path, text: str) -> TaskExtractorConfig:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(text)
    return TaskExtractorConfig.load(config_path)


@pytest.mark.parametrize("installed", ["0.8.2", "0.9.0", "0.8.3.dev1+g0489867"])
def test_config_loads_when_the_installed_version_is_in_range(tmp_path, monkeypatch, installed):
    monkeypatch.setattr("aces.config.__version__", installed)
    cfg = load(tmp_path, 'aces_version: ">=0.8.2"\n' + TASK_YAML)
    assert cfg.trigger.predicate == "admission"


@pytest.mark.parametrize(("required", "installed"), [(">=0.8.2", "0.8.1"), (">=0.8.2, <0.9", "0.9.0")])
def test_config_fails_when_the_installed_version_is_out_of_range(tmp_path, monkeypatch, required, installed):
    monkeypatch.setattr("aces.config.__version__", installed)
    with pytest.raises(
        ValueError, match=re.escape(f"requires ACES {required}, but {installed} is installed")
    ):
        load(tmp_path, f'aces_version: "{required}"\n' + TASK_YAML)


def test_version_is_checked_before_the_rest_of_the_config(tmp_path, monkeypatch):
    """Syntax from a newer ACES must report the version, not an unrelated parsing error."""
    monkeypatch.setattr("aces.config.__version__", "0.8.1")
    with pytest.raises(ValueError, match=re.escape("requires ACES >=0.9,")):
        load(tmp_path, 'aces_version: ">=0.9"\nsome_future_key: 1\n' + TASK_YAML)


def test_invalid_version_range(tmp_path):
    with pytest.raises(ValueError, match=re.escape("Invalid 'aces_version': '0.8.2'")):
        load(tmp_path, 'aces_version: "0.8.2"\n' + TASK_YAML)


def test_unknown_installed_version_skips_the_check(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("aces.config.__version__", "unknown")
    cfg = load(tmp_path, 'aces_version: ">=99"\n' + TASK_YAML)
    assert cfg.trigger.predicate == "admission"
    assert "Could not determine the installed ACES version" in caplog.text


def test_aces_version_is_optional(tmp_path):
    assert load(tmp_path, TASK_YAML).trigger.predicate == "admission"
