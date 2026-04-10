"""
tests/test_ollama_updater.py
─────────────────────────────────────────────────────────────────────────────
Tests for ollama_updater.py v3.0 — single registry, reads probe_models.json.

All tests are self-contained. No Ollama instance or filesystem MCP required.

Run:
    pytest tests/test_ollama_updater.py -v
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ollama"))

import ollama_updater as ou


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _probe_data(models: list[dict]) -> dict:
    return {
        "_meta": {"version": "1.0", "changelog": {"1.0": "test"}, "description": "test"},
        "models": models,
    }


def _ollama_entry(name: str, update="check", enabled=True, cloud=False) -> dict:
    return {
        "name": name, "backend": "ollama", "size_gb": 4.0,
        "family": "llama", "geopolitical_origin": "US/Meta",
        "tool_capable": None, "think_blocks": None, "chat_alignment_strong": None,
        "update": update, "enabled": enabled,
        "notes": "test entry",
        **({"cloud": True} if cloud else {}),
    }


def _write_probe(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class _NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


LOG = _NullLogger()


# ── load_probe_models ──────────────────────────────────────────────────────────

class TestLoadProbeModels:
    def test_returns_only_ollama_backend(self, tmp_path, monkeypatch):
        data = _probe_data([
            _ollama_entry("llama3.2:latest"),
            {"name": "mlx-model", "backend": "mlx", "enabled": True},
        ])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        models = ou.load_probe_models()
        assert all(m["backend"] == "ollama" for m in models)
        assert len(models) == 1
        assert models[0]["name"] == "llama3.2:latest"

    def test_includes_disabled_entries(self, tmp_path, monkeypatch):
        """load_probe_models returns all ollama entries — callers filter enabled."""
        data = _probe_data([
            _ollama_entry("active:7b", enabled=True),
            _ollama_entry("disabled:7b", enabled=False),
        ])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        models = ou.load_probe_models()
        assert len(models) == 2


# ── sync_models ────────────────────────────────────────────────────────────────

class TestSyncModels:
    def test_adds_untracked_installed_model(self, tmp_path, monkeypatch):
        data = _probe_data([_ollama_entry("existing:7b")])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        installed = {"existing:7b": "abc123", "new-model:7b": "def456"}
        monkeypatch.setattr(ou, "get_installed_models", lambda: installed)
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        names = [m["name"] for m in result["models"]]
        assert "new-model:7b" in names

    def test_does_not_duplicate_existing(self, tmp_path, monkeypatch):
        data = _probe_data([_ollama_entry("llama3.2:latest")])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"llama3.2:latest": "abc"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entries = [m for m in result["models"] if m["name"] == "llama3.2:latest"]
        assert len(entries) == 1

    def test_skips_non_probe_names(self, tmp_path, monkeypatch):
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"mxbai-embed-large:latest": "abc"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        assert all(m["name"] != "mxbai-embed-large:latest" for m in result["models"])

    def test_skips_vision_models_by_name(self, tmp_path, monkeypatch):
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"llama3.2-vision:11b": "abc"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        assert all("vision" not in m["name"] for m in result["models"])

    def test_added_entry_has_update_check(self, tmp_path, monkeypatch):
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"falcon3:10b": "abc"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "falcon3:10b")
        assert entry["update"] == "check"

    def test_added_entry_has_null_research_fields(self, tmp_path, monkeypatch):
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"new-model:7b": "abc"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "new-model:7b")
        assert entry["tool_capable"] is None
        assert entry["think_blocks"] is None
        assert entry["chat_alignment_strong"] is None

    def test_version_bumped(self, tmp_path, monkeypatch):
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"new-model:7b": "abc"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        assert result["_meta"]["version"] != "1.0"

    def test_idempotent(self, tmp_path, monkeypatch):
        """Running sync twice must not add duplicates."""
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"new-model:7b": "abc"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)
        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entries = [m for m in result["models"] if m["name"] == "new-model:7b"]
        assert len(entries) == 1


# ── prune_models ───────────────────────────────────────────────────────────────

class TestPruneModels:
    def test_disables_uninstalled_local_model(self, tmp_path, monkeypatch):
        data = _probe_data([
            _ollama_entry("installed:7b", enabled=True),
            _ollama_entry("removed:7b", enabled=True),
        ])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"installed:7b": "abc"})

        ou.prune_models(LOG)

        result = json.loads(probe_file.read_text())
        removed = next(m for m in result["models"] if m["name"] == "removed:7b")
        assert removed["enabled"] is False

    def test_does_not_disable_cloud_models(self, tmp_path, monkeypatch):
        """Cloud models are not local — absence from `ollama list` is expected."""
        data = _probe_data([
            _ollama_entry("deepseek-v3.2:cloud", cloud=True, enabled=True),
        ])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models", lambda: {})

        ou.prune_models(LOG)

        result = json.loads(probe_file.read_text())
        cloud = next(m for m in result["models"] if "cloud" in m["name"])
        assert cloud["enabled"] is True

    def test_no_prune_when_all_installed(self, tmp_path, monkeypatch):
        data = _probe_data([_ollama_entry("llama3.2:latest")])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"llama3.2:latest": "abc"})

        ou.prune_models(LOG)

        result = json.loads(probe_file.read_text())
        assert result["models"][0]["enabled"] is True

    def test_prune_does_not_delete_rows(self, tmp_path, monkeypatch):
        """Prune sets enabled=false, never removes entries."""
        data = _probe_data([_ollama_entry("gone:7b")])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models", lambda: {})

        ou.prune_models(LOG)

        result = json.loads(probe_file.read_text())
        assert len(result["models"]) == 1
        assert result["models"][0]["name"] == "gone:7b"


# ── load_settings ──────────────────────────────────────────────────────────────

class TestLoadSettings:
    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ou, "SETTINGS_FILE", tmp_path / "nonexistent.yaml")
        settings = ou.load_settings()
        assert "pull_delay_seconds" in settings
        assert "log_file" in settings
        assert "log_retention_days" in settings

    def test_reads_settings_file(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("pull_delay_seconds: 5\nlog_file: /tmp/test.log\nlog_retention_days: 7\n")
        monkeypatch.setattr(ou, "SETTINGS_FILE", settings_file)
        settings = ou.load_settings()
        assert settings["pull_delay_seconds"] == 5
        assert settings["log_retention_days"] == 7
