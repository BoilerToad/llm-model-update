"""
tests/test_mlx_updater.py
─────────────────────────
Unit tests for mlx/mlx_updater.py
All HF Hub network calls are mocked — no downloads occur.

Run:  python3 -m pytest tests/test_mlx_updater.py -v
  or: python3 -m unittest tests.test_mlx_updater -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlx.mlx_updater import (
    hf_cache_root,
    repo_cache_dir,
    get_local_commit,
    get_remote_commit,
    is_stale,
    pull_model,
    run_update,
    list_models,
)

SAMPLE_CONFIG = {
    "settings": {
        "hf_cache_dir": "",
        "download_delay_seconds": 0,
        "log_file": "/tmp/test-mlx-updater.log",
        "hf_token": "",
    },
    "models": [
        {
            "repo_id":  "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "update":   True,
            "revision": "main",
            "notes":    "",
        },
        {
            "repo_id":  "mlx-community/Phi-3.5-mini-instruct-4bit",
            "update":   False,
            "revision": "main",
            "notes":    "pinned",
        },
        {
            "repo_id":  "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
            "update":   "check",
            "revision": "main",
            "notes":    "review before updating",
        },
    ],
}

LOCAL_HASH  = "aabbccddeeff001122334455667788990011223344"
REMOTE_HASH = "zzyyxxwwvvuu998877665544332211009988776655"


def make_logger():
    logger = MagicMock()
    logger.info    = MagicMock()
    logger.error   = MagicMock()
    logger.warning = MagicMock()
    return logger


class TestCacheHelpers(unittest.TestCase):

    def test_repo_cache_dir_slash_to_dash(self):
        root   = Path("/tmp/hf_cache")
        result = repo_cache_dir(root, "mlx-community/Qwen3-4B-4bit")
        self.assertEqual(result.name, "models--mlx-community--Qwen3-4B-4bit")

    def test_hf_cache_root_override(self):
        result = hf_cache_root("/custom/cache")
        self.assertEqual(result, Path("/custom/cache"))

    def test_get_local_commit_reads_refs_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            repo_id    = "mlx-community/TestModel"
            refs_dir   = repo_cache_dir(cache_root, repo_id) / "refs"
            refs_dir.mkdir(parents=True)
            (refs_dir / "main").write_text(LOCAL_HASH)
            result = get_local_commit(cache_root, repo_id, "main")
            self.assertEqual(result, LOCAL_HASH)

    def test_get_local_commit_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = get_local_commit(Path(tmp), "mlx-community/NotCached", "main")
            self.assertIsNone(result)

    def test_get_local_commit_snapshot_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            repo_id    = "mlx-community/TestModel"
            snap_dir   = repo_cache_dir(cache_root, repo_id) / "snapshots" / LOCAL_HASH
            snap_dir.mkdir(parents=True)
            result = get_local_commit(cache_root, repo_id, "main")
            self.assertEqual(result, LOCAL_HASH)


class TestGetRemoteCommit(unittest.TestCase):

    @patch("mlx.mlx_updater.model_info")
    def test_returns_sha(self, mock_info):
        mock_info.return_value = MagicMock(sha=REMOTE_HASH)
        result = get_remote_commit("mlx-community/Qwen3-4B-4bit", "main")
        self.assertEqual(result, REMOTE_HASH)

    @patch("mlx.mlx_updater.model_info")
    def test_returns_none_on_repo_not_found(self, mock_info):
        mock_info.side_effect = Exception("404 Repository Not Found")
        result = get_remote_commit("mlx-community/DoesNotExist", "main")
        self.assertIsNone(result)

    @patch("mlx.mlx_updater.model_info")
    def test_returns_none_on_network_error(self, mock_info):
        mock_info.side_effect = ConnectionError("timeout")
        result = get_remote_commit("mlx-community/Qwen3-4B-4bit", "main")
        self.assertIsNone(result)


class TestIsStale(unittest.TestCase):

    def _make_cache(self, tmp, repo_id, commit):
        cache_root = Path(tmp)
        refs_dir   = repo_cache_dir(cache_root, repo_id) / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text(commit)
        return cache_root

    @patch("mlx.mlx_updater.get_remote_commit")
    def test_not_stale_when_hashes_match(self, mock_remote):
        with tempfile.TemporaryDirectory() as tmp:
            repo_id    = "mlx-community/Qwen3-4B-4bit"
            mock_remote.return_value = LOCAL_HASH
            cache_root = self._make_cache(tmp, repo_id, LOCAL_HASH)
            stale, _, _ = is_stale(cache_root, repo_id, "main", None)
            self.assertFalse(stale)

    @patch("mlx.mlx_updater.get_remote_commit")
    def test_stale_when_hashes_differ(self, mock_remote):
        with tempfile.TemporaryDirectory() as tmp:
            repo_id    = "mlx-community/Qwen3-4B-4bit"
            mock_remote.return_value = REMOTE_HASH
            cache_root = self._make_cache(tmp, repo_id, LOCAL_HASH)
            stale, _, _ = is_stale(cache_root, repo_id, "main", None)
            self.assertTrue(stale)

    @patch("mlx.mlx_updater.get_remote_commit")
    def test_not_stale_when_remote_unreachable(self, mock_remote):
        with tempfile.TemporaryDirectory() as tmp:
            repo_id    = "mlx-community/Qwen3-4B-4bit"
            mock_remote.return_value = None
            cache_root = self._make_cache(tmp, repo_id, LOCAL_HASH)
            stale, _, remote_h = is_stale(cache_root, repo_id, "main", None)
            self.assertFalse(stale)
            self.assertEqual(remote_h, "—unreachable—")

    @patch("mlx.mlx_updater.get_remote_commit")
    def test_stale_when_not_cached(self, mock_remote):
        with tempfile.TemporaryDirectory() as tmp:
            mock_remote.return_value = REMOTE_HASH
            stale, local_h, _ = is_stale(Path(tmp), "mlx-community/NewModel", "main", None)
            self.assertTrue(stale)
            self.assertEqual(local_h, "—not cached—")


class TestPullModel(unittest.TestCase):

    def setUp(self):
        self.logger = make_logger()

    @patch("mlx.mlx_updater.is_stale")
    def test_current_skips_download(self, mock_stale):
        mock_stale.return_value = (False, LOCAL_HASH[:12], LOCAL_HASH[:12])
        with tempfile.TemporaryDirectory() as tmp:
            result = pull_model(
                repo_id="mlx-community/Qwen3-4B-4bit", revision="main",
                cache_root=Path(tmp), token=None,
                dry_run=False, delay=0, logger=self.logger,
            )
        self.assertEqual(result, "current")

    @patch("mlx.mlx_updater.is_stale")
    def test_dry_run_stale(self, mock_stale):
        mock_stale.return_value = (True, LOCAL_HASH[:12], REMOTE_HASH[:12])
        with tempfile.TemporaryDirectory() as tmp:
            result = pull_model(
                repo_id="mlx-community/Qwen3-4B-4bit", revision="main",
                cache_root=Path(tmp), token=None,
                dry_run=True, delay=0, logger=self.logger,
            )
        self.assertEqual(result, "dry_run")

    @patch("mlx.mlx_updater.get_local_commit")
    @patch("mlx.mlx_updater.snapshot_download")
    @patch("mlx.mlx_updater.is_stale")
    @patch("time.sleep")
    def test_download_when_stale(self, mock_sleep, mock_stale, mock_dl, mock_local):
        mock_stale.return_value = (True, LOCAL_HASH[:12], REMOTE_HASH[:12])
        mock_local.return_value = REMOTE_HASH
        mock_dl.return_value    = None
        with tempfile.TemporaryDirectory() as tmp:
            result = pull_model(
                repo_id="mlx-community/Qwen3-4B-4bit", revision="main",
                cache_root=Path(tmp), token=None,
                dry_run=False, delay=0, logger=self.logger,
            )
        self.assertEqual(result, "updated")
        mock_dl.assert_called_once()

    @patch("mlx.mlx_updater.snapshot_download")
    @patch("mlx.mlx_updater.is_stale")
    def test_returns_failed_on_exception(self, mock_stale, mock_dl):
        mock_stale.return_value = (True, LOCAL_HASH[:12], REMOTE_HASH[:12])
        mock_dl.side_effect     = RuntimeError("disk full")
        with tempfile.TemporaryDirectory() as tmp:
            result = pull_model(
                repo_id="mlx-community/Qwen3-4B-4bit", revision="main",
                cache_root=Path(tmp), token=None,
                dry_run=False, delay=0, logger=self.logger,
            )
        self.assertEqual(result, "failed")


class TestRunUpdate(unittest.TestCase):

    def setUp(self):
        self.logger = make_logger()

    @patch("mlx.mlx_updater.pull_model")
    def test_auto_skips_pinned(self, mock_pull):
        mock_pull.return_value = "current"
        run_update(mode="auto", single=None, logger=self.logger, config=SAMPLE_CONFIG)
        pulled_ids = [c.kwargs["repo_id"] for c in mock_pull.call_args_list]
        self.assertNotIn("mlx-community/Phi-3.5-mini-instruct-4bit", pulled_ids)

    @patch("mlx.mlx_updater.pull_model")
    def test_auto_check_flag_forces_dry_run(self, mock_pull):
        mock_pull.return_value = "dry_run"
        run_update(mode="auto", single=None, logger=self.logger, config=SAMPLE_CONFIG)
        for c in mock_pull.call_args_list:
            if c.kwargs.get("repo_id") == "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit":
                self.assertTrue(c.kwargs["dry_run"])

    @patch("mlx.mlx_updater.pull_model")
    def test_all_mode_includes_pinned(self, mock_pull):
        mock_pull.return_value = "current"
        run_update(mode="all", single=None, logger=self.logger, config=SAMPLE_CONFIG)
        pulled_ids = [c.kwargs["repo_id"] for c in mock_pull.call_args_list]
        self.assertIn("mlx-community/Phi-3.5-mini-instruct-4bit", pulled_ids)

    @patch("mlx.mlx_updater.pull_model")
    def test_check_mode_all_dry_run(self, mock_pull):
        mock_pull.return_value = "dry_run"
        run_update(mode="check", single=None, logger=self.logger, config=SAMPLE_CONFIG)
        for c in mock_pull.call_args_list:
            self.assertTrue(c.kwargs["dry_run"])

    @patch("mlx.mlx_updater.pull_model")
    def test_single_model(self, mock_pull):
        mock_pull.return_value = "current"
        run_update(
            mode="auto",
            single="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            logger=self.logger,
            config=SAMPLE_CONFIG,
        )
        pulled_ids = [c.kwargs["repo_id"] for c in mock_pull.call_args_list]
        self.assertEqual(pulled_ids, ["mlx-community/Mistral-7B-Instruct-v0.3-4bit"])


class TestListModels(unittest.TestCase):

    @patch("mlx.mlx_updater.get_remote_commit")
    @patch("mlx.mlx_updater.get_local_commit")
    def test_current_label(self, mock_local, mock_remote):
        mock_local.return_value  = LOCAL_HASH
        mock_remote.return_value = LOCAL_HASH
        logger = make_logger()
        list_models(logger, SAMPLE_CONFIG)
        combined = " ".join(str(c) for c in logger.info.call_args_list)
        self.assertIn("CURRENT", combined)

    @patch("mlx.mlx_updater.get_remote_commit")
    @patch("mlx.mlx_updater.get_local_commit")
    def test_stale_label(self, mock_local, mock_remote):
        mock_local.return_value  = LOCAL_HASH
        mock_remote.return_value = REMOTE_HASH
        logger = make_logger()
        list_models(logger, SAMPLE_CONFIG)
        combined = " ".join(str(c) for c in logger.info.call_args_list)
        self.assertIn("STALE", combined)

    @patch("mlx.mlx_updater.get_remote_commit")
    @patch("mlx.mlx_updater.get_local_commit")
    def test_not_cached_label(self, mock_local, mock_remote):
        mock_local.return_value  = None
        mock_remote.return_value = REMOTE_HASH
        logger = make_logger()
        list_models(logger, SAMPLE_CONFIG)
        combined = " ".join(str(c) for c in logger.info.call_args_list)
        self.assertIn("NOT CACHED", combined)


if __name__ == "__main__":
    unittest.main()
