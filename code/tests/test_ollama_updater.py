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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ollama"))

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

    def test_local_model_ollama_pulled_is_today(self, tmp_path, monkeypatch):
        """Local synced models get ollama_pulled set to today's date."""
        from datetime import datetime
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"llama3.2:latest": "abc123"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "llama3.2:latest")
        today = datetime.now().strftime("%Y-%m-%d")
        assert entry["ollama_pulled"] == today

    def test_cloud_model_detected_by_name(self, tmp_path, monkeypatch):
        """Models with 'cloud' in the name get backend=ollama_cloud."""
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"deepseek-r1:cloud": "def456"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "deepseek-r1:cloud")
        assert entry["backend"] == "ollama_cloud"

    def test_cloud_model_ollama_id_from_list(self, tmp_path, monkeypatch):
        """Cloud models get ollama_id populated from `ollama list` digest."""
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"deepseek-r1:cloud": "def456abcdef"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "deepseek-r1:cloud")
        assert entry["ollama_id"] == "def456abcdef"

    def test_cloud_model_ollama_pulled_is_today(self, tmp_path, monkeypatch):
        """Cloud synced models also get ollama_pulled set to today's date."""
        from datetime import datetime
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"deepseek-r1:cloud": "def456"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "deepseek-r1:cloud")
        today = datetime.now().strftime("%Y-%m-%d")
        assert entry["ollama_pulled"] == today

    def test_cloud_model_has_compare_to_field(self, tmp_path, monkeypatch):
        """Cloud entries get a compare_to field; local entries do not."""
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"mymodel:cloud": "abc", "mymodel:latest": "def"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        cloud = next(m for m in result["models"] if m["name"] == "mymodel:cloud")
        local = next(m for m in result["models"] if m["name"] == "mymodel:latest")
        assert "compare_to" in cloud
        assert "compare_to" not in local

    def test_local_model_backend_is_ollama(self, tmp_path, monkeypatch):
        """Non-cloud synced models get backend=ollama."""
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"gemma3:4b": "abc123"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "gemma3:4b")
        assert entry["backend"] == "ollama"

    def test_added_entry_has_raw_capable_null(self, tmp_path, monkeypatch):
        """New synced models default to raw_capable=null (not yet evaluated)."""
        data = _probe_data([])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)
        monkeypatch.setattr(ou, "get_installed_models",
                            lambda: {"gemma3:4b": "abc123"})
        monkeypatch.setattr(ou, "get_ollama_tags", lambda: [])

        ou.sync_models(LOG)

        result = json.loads(probe_file.read_text())
        entry = next(m for m in result["models"] if m["name"] == "gemma3:4b")
        assert "raw_capable" in entry
        assert entry["raw_capable"] is None


# ── migrate_models ────────────────────────────────────────────────────────────

class TestMigrateModels:
    def test_backfills_missing_raw_capable(self, tmp_path, monkeypatch):
        """Entries without raw_capable get it added as null."""
        data = _probe_data([_ollama_entry("llama3.2:latest")])
        # Ensure the field is absent
        data["models"][0].pop("raw_capable", None)
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        ou.migrate_models(LOG)

        result = json.loads(probe_file.read_text())
        assert "raw_capable" in result["models"][0]
        assert result["models"][0]["raw_capable"] is None

    def test_does_not_overwrite_existing_values(self, tmp_path, monkeypatch):
        """Entries that already have raw_capable=false keep their value."""
        entry = _ollama_entry("llama3.2:latest")
        entry["raw_capable"] = False
        data = _probe_data([entry])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        ou.migrate_models(LOG)

        result = json.loads(probe_file.read_text())
        assert result["models"][0]["raw_capable"] is False

    def test_no_changes_when_schema_complete(self, tmp_path, monkeypatch):
        """If all fields present, version is not bumped."""
        entry = _ollama_entry("llama3.2:latest")
        for field in ("tool_capable", "think_blocks", "chat_alignment_strong", "raw_capable"):
            entry[field] = None
        data = _probe_data([entry])
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        ou.migrate_models(LOG)

        result = json.loads(probe_file.read_text())
        assert result["_meta"]["version"] == "1.0"

    def test_version_bumped_when_fields_added(self, tmp_path, monkeypatch):
        """Version increments when at least one entry is updated."""
        data = _probe_data([_ollama_entry("llama3.2:latest")])
        for f in ("raw_capable",):
            data["models"][0].pop(f, None)
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        ou.migrate_models(LOG)

        result = json.loads(probe_file.read_text())
        assert result["_meta"]["version"] != "1.0"

    def test_migrates_multiple_entries(self, tmp_path, monkeypatch):
        """All entries missing raw_capable are updated in one pass."""
        entries = [_ollama_entry(f"model-{i}:7b") for i in range(3)]
        for e in entries:
            e.pop("raw_capable", None)
        data = _probe_data(entries)
        probe_file = tmp_path / "probe_models.json"
        _write_probe(probe_file, data)
        monkeypatch.setattr(ou, "PROBE_FILE", probe_file)

        ou.migrate_models(LOG)

        result = json.loads(probe_file.read_text())
        for m in result["models"]:
            assert "raw_capable" in m


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
