"""
tests/test_probe_healthcheck.py
────────────────────────────────
Unit tests for probe_healthcheck.py — auth headers, smoke tests,
cloud key check, model filtering (--local/--cloud/--family), timeout flags.

No real network calls are made; requests is mocked throughout.

Run:  pytest code/tests/test_probe_healthcheck.py -v
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

from probe_healthcheck import (
    auth_headers,
    chat_smoke,
    check_cloud_key,
    gen_smoke,
    get_local_tags,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

REGISTRY = {
    "_meta": {"version": "1.0"},
    "models": [
        {"name": "llama3:latest",        "family": "llama",    "enabled": True},
        {"name": "deepseek-r1:7b",       "family": "deepseek", "enabled": True},
        {"name": "gemma2:9b",            "family": "gemma",    "enabled": True},
        {"name": "glm-5.1:cloud",        "family": "glm",      "enabled": True},
        {"name": "mistral-large:cloud",  "family": "mistral",  "enabled": True},
        {"name": "disabled-model:7b",    "family": "llama",    "enabled": False},
        {"name": "mxbai-embed-large:latest", "family": "bert", "enabled": True},
    ],
}


def _mock_response(status=200, json_data=None, content="Hello"):
    mock = MagicMock()
    mock.status_code = status
    mock.ok = (status < 400)          # mirrors requests.Response.ok property
    mock.json.return_value = json_data or {}
    mock.raise_for_status = MagicMock()
    if status >= 400:
        import requests
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} Error", response=mock
        )
    return mock


# ── auth_headers ──────────────────────────────────────────────────────────────

class TestAuthHeaders(unittest.TestCase):

    def test_cloud_url_with_key(self):
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
            headers = auth_headers("https://ollama.com")
        self.assertEqual(headers, {"Authorization": "Bearer test-key"})

    def test_cloud_url_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            headers = auth_headers("https://ollama.com")
        self.assertEqual(headers, {})

    def test_local_url_returns_empty(self):
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
            headers = auth_headers("http://localhost:11434")
        self.assertEqual(headers, {})


# ── get_local_tags ────────────────────────────────────────────────────────────

class TestGetLocalTags(unittest.TestCase):

    def test_returns_model_names(self):
        mock_resp = _mock_response(json_data={"models": [
            {"name": "llama3:latest"},
            {"name": "gemma2:9b"},
        ]})
        with patch("requests.get", return_value=mock_resp):
            tags = get_local_tags()
        self.assertEqual(tags, {"llama3:latest", "gemma2:9b"})

    def test_returns_empty_on_connection_error(self):
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            tags = get_local_tags()
        self.assertEqual(tags, set())

    def test_returns_empty_on_timeout(self):
        import requests
        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            tags = get_local_tags()
        self.assertEqual(tags, set())

    def test_returns_empty_on_http_error(self):
        mock_resp = _mock_response(status=500)
        with patch("requests.get", return_value=mock_resp):
            tags = get_local_tags()
        self.assertEqual(tags, set())


# ── chat_smoke ────────────────────────────────────────────────────────────────

class TestChatSmoke(unittest.TestCase):

    def test_pass_on_valid_response(self):
        mock_resp = _mock_response(json_data={"message": {"content": "Hello there"}})
        with patch("requests.post", return_value=mock_resp):
            ok, detail = chat_smoke("llama3:latest", "http://localhost:11434")
        self.assertTrue(ok)
        self.assertIn("c in", detail)

    def test_fail_on_empty_content(self):
        mock_resp = _mock_response(json_data={"message": {"content": "   "}})
        with patch("requests.post", return_value=mock_resp):
            ok, detail = chat_smoke("llama3:latest", "http://localhost:11434")
        self.assertFalse(ok)
        self.assertIn("empty response", detail)

    def test_fail_on_timeout(self):
        import requests
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            ok, detail = chat_smoke("llama3:latest", "http://localhost:11434", timeout=30)
        self.assertFalse(ok)
        self.assertIn("timeout after 30s", detail)

    def test_fail_on_http_error(self):
        mock_resp = _mock_response(status=403)
        with patch("requests.post", return_value=mock_resp):
            ok, detail = chat_smoke("glm-5.1:cloud", "https://ollama.com")
        self.assertFalse(ok)
        self.assertIn("403", detail)

    def test_custom_timeout_used(self):
        import requests
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            ok, detail = chat_smoke("llama3:latest", "http://localhost:11434", timeout=90)
        self.assertFalse(ok)
        self.assertIn("timeout after 90s", detail)

    def test_uses_auth_headers_for_cloud(self):
        mock_resp = _mock_response(json_data={"message": {"content": "Hi"}})
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                chat_smoke("glm-5.1:cloud", "https://ollama.com")
        _, kwargs = mock_post.call_args
        self.assertIn("Authorization", kwargs.get("headers", {}))


# ── gen_smoke ─────────────────────────────────────────────────────────────────

class TestGenSmoke(unittest.TestCase):

    def test_pass_on_valid_response(self):
        mock_resp = _mock_response(json_data={"response": "OK"})
        with patch("requests.post", return_value=mock_resp):
            ok, detail = gen_smoke("llama3:latest")
        self.assertTrue(ok)
        self.assertIn("c in", detail)

    def test_fail_on_empty_response(self):
        mock_resp = _mock_response(json_data={"response": ""})
        with patch("requests.post", return_value=mock_resp):
            ok, detail = gen_smoke("llama3:latest")
        self.assertFalse(ok)
        self.assertIn("empty response", detail)

    def test_fail_on_timeout(self):
        import requests
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            ok, detail = gen_smoke("llama3:latest", timeout=60)
        self.assertFalse(ok)
        self.assertIn("timeout after 60s", detail)

    def test_custom_timeout_reflected_in_message(self):
        import requests
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            ok, detail = gen_smoke("llama3:latest", timeout=120)
        self.assertFalse(ok)
        self.assertIn("timeout after 120s", detail)

    def test_fail_on_http_500(self):
        mock_resp = _mock_response(status=500)
        with patch("requests.post", return_value=mock_resp):
            ok, detail = gen_smoke("llama3:latest")
        self.assertFalse(ok)


# ── check_cloud_key ───────────────────────────────────────────────────────────

class TestCheckCloudKey(unittest.TestCase):

    def test_fail_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            ok, detail = check_cloud_key()
        self.assertFalse(ok)
        self.assertIn("OLLAMA_API_KEY", detail)

    def test_pass_with_key_and_cloud_models(self):
        mock_resp = _mock_response(json_data={"models": [
            {"name": "glm-5.1:cloud"},
            {"name": "mistral-large:cloud"},
            {"name": "llama3:latest"},
        ]})
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
            with patch("requests.get", return_value=mock_resp):
                ok, detail = check_cloud_key()
        self.assertTrue(ok)
        self.assertIn("2 cloud proxies", detail)

    def test_fail_on_connection_error(self):
        import requests
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
            with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
                ok, detail = check_cloud_key()
        self.assertFalse(ok)
        self.assertIn("refused", detail)


# ── CLI flag integration — model filtering ────────────────────────────────────

class TestMainFiltering(unittest.TestCase):
    """Test --local, --cloud, --family filtering via main() with mocked I/O."""

    def _run_main(self, argv, registry=None, local_tags=None, chat_result=None):
        reg = registry or REGISTRY
        tags = local_tags or {"llama3:latest", "deepseek-r1:7b", "gemma2:9b",
                              "glm-5.1:cloud", "mistral-large:cloud"}
        chat_ok = chat_result if chat_result is not None else (True, "10c in 0.5s")

        tmp_file = Path("/tmp/test_registry.json")
        tmp_file.write_text(json.dumps(reg), encoding="utf-8")

        with patch("sys.argv", ["probe_healthcheck.py", "--models-file", str(tmp_file)] + argv), \
             patch("probe_healthcheck.get_local_tags", return_value=tags), \
             patch("probe_healthcheck.chat_smoke", return_value=chat_ok), \
             patch("probe_healthcheck.gen_smoke", return_value=(True, "5c in 0.3s")), \
             patch("probe_healthcheck.check_cloud_key", return_value=(True, "key present, 2 cloud proxies registered")), \
             patch("sys.exit"):
            from probe_healthcheck import main
            main()

    def test_local_flag_excludes_cloud_models(self):
        checked = []
        def fake_chat(name, url, timeout=30):
            checked.append(name)
            return True, "10c in 0.5s"

        tmp_file = Path("/tmp/test_registry.json")
        tmp_file.write_text(json.dumps(REGISTRY), encoding="utf-8")

        with patch("sys.argv", ["probe_healthcheck.py", "--local", "--chat-only",
                                "--models-file", str(tmp_file)]), \
             patch("probe_healthcheck.get_local_tags",
                   return_value={"llama3:latest", "deepseek-r1:7b", "gemma2:9b"}), \
             patch("probe_healthcheck.chat_smoke", side_effect=fake_chat), \
             patch("sys.exit"):
            from probe_healthcheck import main
            main()

        for name in checked:
            self.assertNotIn("cloud", name)

    def test_cloud_flag_excludes_local_models(self):
        checked = []
        def fake_chat(name, url, timeout=30):
            checked.append(name)
            return True, "10c in 0.5s"

        tmp_file = Path("/tmp/test_registry.json")
        tmp_file.write_text(json.dumps(REGISTRY), encoding="utf-8")

        with patch("sys.argv", ["probe_healthcheck.py", "--cloud",
                                "--models-file", str(tmp_file)]), \
             patch("probe_healthcheck.get_local_tags",
                   return_value={"glm-5.1:cloud", "mistral-large:cloud"}), \
             patch("probe_healthcheck.chat_smoke", side_effect=fake_chat), \
             patch("probe_healthcheck.check_cloud_key",
                   return_value=(True, "key present, 2 cloud proxies registered")), \
             patch("sys.exit"):
            from probe_healthcheck import main
            main()

        for name in checked:
            self.assertIn("cloud", name)

    def test_family_filter(self):
        checked = []
        def fake_chat(name, url, timeout=30):
            checked.append(name)
            return True, "10c in 0.5s"

        tmp_file = Path("/tmp/test_registry.json")
        tmp_file.write_text(json.dumps(REGISTRY), encoding="utf-8")

        with patch("sys.argv", ["probe_healthcheck.py", "--family", "deepseek",
                                "--chat-only", "--models-file", str(tmp_file)]), \
             patch("probe_healthcheck.get_local_tags",
                   return_value={"deepseek-r1:7b"}), \
             patch("probe_healthcheck.chat_smoke", side_effect=fake_chat), \
             patch("sys.exit"):
            from probe_healthcheck import main
            main()

        self.assertEqual(checked, ["deepseek-r1:7b"])

    def test_disabled_models_excluded(self):
        checked = []
        def fake_chat(name, url, timeout=30):
            checked.append(name)
            return True, "10c in 0.5s"

        tmp_file = Path("/tmp/test_registry.json")
        tmp_file.write_text(json.dumps(REGISTRY), encoding="utf-8")

        with patch("sys.argv", ["probe_healthcheck.py", "--local", "--chat-only",
                                "--models-file", str(tmp_file)]), \
             patch("probe_healthcheck.get_local_tags",
                   return_value={"llama3:latest", "deepseek-r1:7b", "gemma2:9b",
                                 "disabled-model:7b"}), \
             patch("probe_healthcheck.chat_smoke", side_effect=fake_chat), \
             patch("sys.exit"):
            from probe_healthcheck import main
            main()

        self.assertNotIn("disabled-model:7b", checked)

    def test_skip_families_excluded(self):
        checked = []
        def fake_chat(name, url, timeout=30):
            checked.append(name)
            return True, "10c in 0.5s"

        tmp_file = Path("/tmp/test_registry.json")
        tmp_file.write_text(json.dumps(REGISTRY), encoding="utf-8")

        with patch("sys.argv", ["probe_healthcheck.py", "--local", "--chat-only",
                                "--models-file", str(tmp_file)]), \
             patch("probe_healthcheck.get_local_tags",
                   return_value={"llama3:latest", "deepseek-r1:7b", "gemma2:9b",
                                 "mxbai-embed-large:latest"}), \
             patch("probe_healthcheck.chat_smoke", side_effect=fake_chat), \
             patch("sys.exit"):
            from probe_healthcheck import main
            main()

        self.assertNotIn("mxbai-embed-large:latest", checked)


if __name__ == "__main__":
    unittest.main()
