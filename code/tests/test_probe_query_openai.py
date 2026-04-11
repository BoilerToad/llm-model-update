"""
tests/test_probe_query_openai.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for probe_query_openai.py.
No real network calls — requests is mocked throughout.
No API keys required to run.

Run:  pytest tests/test_probe_query_openai.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from probe_query_openai import (
    PROVIDER_CONFIG,
    OPENAI_BACKENDS,
    DEFAULT_MODEL_IDS,
    is_openai_backend,
    backend_for,
    base_url_for,
    api_key_for,
    api_model_id_for,
    openai_headers,
    has_api_key,
    provider_label,
    extract_think,
    query_chat_openai,
    query_generate_openai,
    check_api_key,
    _error_result,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

XAI_ENTRY = {
    "name":    "grok-4.20-0309-reasoning",
    "backend": "xai",
    "family":  "grok",
    "geopolitical_origin": "US/xAI",
    "enabled": True,
}

XAI_ENTRY_WITH_MODEL_ID = {
    "name":         "grok-4.20-0309-reasoning",
    "backend":      "xai",
    "api_model_id": "grok-3-fast",
}

OLLAMA_ENTRY = {
    "name":    "llama3.2:latest",
    "backend": "ollama",
}


def _openai_response(content="Paris", tool_calls=None, finish_reason="stop",
                     prompt_tokens=10, completion_tokens=20):
    """Build a mock that looks like a successful /v1/chat/completions response."""
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    m.json.return_value = {
        "choices": [
            {"message": message, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return m


def _models_list_response(model_ids=("grok-3",)):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"data": [{"id": mid} for mid in model_ids]}
    return m


# ── Provider config ────────────────────────────────────────────────────────────

class TestProviderConfig(unittest.TestCase):

    def test_xai_in_provider_config(self):
        self.assertIn("xai", PROVIDER_CONFIG)

    def test_xai_has_required_keys(self):
        cfg = PROVIDER_CONFIG["xai"]
        self.assertIn("base_url", cfg)
        self.assertIn("env_key", cfg)
        self.assertIn("label", cfg)

    def test_xai_base_url_correct(self):
        self.assertEqual(PROVIDER_CONFIG["xai"]["base_url"], "https://api.x.ai/v1")

    def test_xai_env_key_correct(self):
        self.assertEqual(PROVIDER_CONFIG["xai"]["env_key"], "XAI_API_KEY")

    def test_xai_in_openai_backends(self):
        self.assertIn("xai", OPENAI_BACKENDS)

    def test_xai_has_default_model_id(self):
        self.assertIn("xai", DEFAULT_MODEL_IDS)
        self.assertEqual(DEFAULT_MODEL_IDS["xai"], "grok-4.20-0309-reasoning")


# ── Routing helpers ────────────────────────────────────────────────────────────

class TestIsOpenaiBackend(unittest.TestCase):

    def test_xai_entry_returns_true(self):
        self.assertTrue(is_openai_backend(XAI_ENTRY))

    def test_ollama_entry_returns_false(self):
        self.assertFalse(is_openai_backend(OLLAMA_ENTRY))

    def test_missing_backend_key_returns_false(self):
        self.assertFalse(is_openai_backend({"name": "something"}))

    def test_unknown_backend_returns_false(self):
        self.assertFalse(is_openai_backend({"name": "x", "backend": "huggingface"}))


class TestBackendFor(unittest.TestCase):

    def test_xai_entry(self):
        self.assertEqual(backend_for(XAI_ENTRY), "xai")

    def test_ollama_entry(self):
        self.assertEqual(backend_for(OLLAMA_ENTRY), "ollama")

    def test_missing_key_returns_empty_string(self):
        self.assertEqual(backend_for({}), "")


class TestBaseUrlFor(unittest.TestCase):

    def test_xai_returns_correct_url(self):
        self.assertEqual(base_url_for(XAI_ENTRY), "https://api.x.ai/v1")

    def test_unknown_backend_returns_empty(self):
        self.assertEqual(base_url_for({"backend": "unknown"}), "")

    def test_ollama_returns_empty(self):
        self.assertEqual(base_url_for(OLLAMA_ENTRY), "")


class TestApiKeyFor(unittest.TestCase):

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-test-key"})
    def test_xai_returns_key_from_env(self):
        self.assertEqual(api_key_for(XAI_ENTRY), "xai-test-key")

    def test_returns_empty_when_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("XAI_API_KEY", None)
            self.assertEqual(api_key_for(XAI_ENTRY), "")

    def test_unknown_backend_returns_empty(self):
        self.assertEqual(api_key_for({"backend": "unknown"}), "")


class TestApiModelIdFor(unittest.TestCase):

    def test_uses_explicit_api_model_id_when_present(self):
        self.assertEqual(api_model_id_for(XAI_ENTRY_WITH_MODEL_ID), "grok-3-fast")

    def test_falls_back_to_default_for_xai(self):
        self.assertEqual(api_model_id_for(XAI_ENTRY), "grok-4.20-0309-reasoning")

    def test_falls_back_to_name_for_unknown_backend(self):
        entry = {"name": "my-model", "backend": "unknown"}
        self.assertEqual(api_model_id_for(entry), "my-model")


class TestOpenaiHeaders(unittest.TestCase):

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-secret"})
    def test_returns_bearer_header_with_key(self):
        headers = openai_headers(XAI_ENTRY)
        self.assertIn("Authorization", headers)
        self.assertEqual(headers["Authorization"], "Bearer xai-secret")
        self.assertIn("Content-Type", headers)

    def test_returns_empty_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("XAI_API_KEY", None)
            self.assertEqual(openai_headers(XAI_ENTRY), {})


class TestHasApiKey(unittest.TestCase):

    @patch.dict("os.environ", {"XAI_API_KEY": "somekey"})
    def test_returns_true_when_key_present(self):
        self.assertTrue(has_api_key(XAI_ENTRY))

    def test_returns_false_when_key_absent(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("XAI_API_KEY", None)
            self.assertFalse(has_api_key(XAI_ENTRY))


class TestProviderLabel(unittest.TestCase):

    def test_xai_returns_xai_label(self):
        self.assertEqual(provider_label(XAI_ENTRY), "xAI")

    def test_unknown_returns_backend_string(self):
        self.assertEqual(provider_label({"backend": "unknown"}), "unknown")


# ── Think block extraction ─────────────────────────────────────────────────────

class TestExtractThink(unittest.TestCase):

    def test_strips_think_block(self):
        think, answer = extract_think("<think>reasoning</think>final answer")
        self.assertEqual(think, "reasoning")
        self.assertEqual(answer, "final answer")

    def test_no_think_block(self):
        think, answer = extract_think("plain response")
        self.assertEqual(think, "")
        self.assertEqual(answer, "plain response")

    def test_empty_think_block(self):
        think, answer = extract_think("<think></think>the answer")
        self.assertEqual(think, "")
        self.assertEqual(answer, "the answer")

    def test_multiple_think_blocks(self):
        think, answer = extract_think("<think>a</think>mid<think>b</think>end")
        self.assertIn("a", think)
        self.assertIn("b", think)
        self.assertNotIn("<think>", answer)


# ── query_chat_openai ──────────────────────────────────────────────────────────

class TestQueryChatOpenai(unittest.TestCase):

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_successful_response_returns_success_true(self, mock_post):
        mock_post.return_value = _openai_response("Paris")
        result = query_chat_openai(XAI_ENTRY, "What is the capital of France?", timeout=10)
        self.assertTrue(result["success"])
        self.assertEqual(result["endpoint"], "chat")
        self.assertEqual(result["backend"], "xai")

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_posts_to_correct_url(self, mock_post):
        mock_post.return_value = _openai_response()
        query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        url = mock_post.call_args[0][0]
        self.assertIn("api.x.ai", url)
        self.assertIn("chat/completions", url)

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_includes_bearer_auth_header(self, mock_post):
        mock_post.return_value = _openai_response()
        query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        headers = mock_post.call_args[1]["headers"]
        self.assertIn("Authorization", headers)
        self.assertIn("xai-key", headers["Authorization"])

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_uses_correct_api_model_id(self, mock_post):
        mock_post.return_value = _openai_response()
        query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], "grok-4.20-0309-reasoning")

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_explicit_api_model_id_overrides_default(self, mock_post):
        mock_post.return_value = _openai_response()
        query_chat_openai(XAI_ENTRY_WITH_MODEL_ID, "hello", timeout=10)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], "grok-3-fast")

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_answer_strips_think_tags(self, mock_post):
        mock_post.return_value = _openai_response("<think>reasoning</think>final answer")
        result = query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertEqual(result["think_block"], "reasoning")
        self.assertEqual(result["answer"], "final answer")
        self.assertEqual(result["think_format"], "tag")

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_no_think_block_format_is_none(self, mock_post):
        mock_post.return_value = _openai_response("plain answer")
        result = query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertEqual(result["think_format"], "none")
        self.assertEqual(result["think_block"], "")

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_usage_tokens_captured(self, mock_post):
        mock_post.return_value = _openai_response(prompt_tokens=15, completion_tokens=42)
        result = query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertEqual(result["prompt_eval_count"], 15)
        self.assertEqual(result["eval_count"], 42)

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_tool_calls_captured(self, mock_post):
        tool_call = {"function": {"name": "web_fetch", "arguments": {"url": "https://example.com"}}}
        mock_post.return_value = _openai_response(content="", tool_calls=[tool_call])
        result = query_chat_openai(XAI_ENTRY, "hello", timeout=10, tools=[{"type": "function"}])
        self.assertIn("tool_calls", result)
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "web_fetch")

    def test_missing_api_key_returns_error(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("XAI_API_KEY", None)
            result = query_chat_openai(XAI_ENTRY, "hello", timeout=10)
            self.assertFalse(result["success"])
            self.assertIn("XAI_API_KEY", result["error"])

    def test_missing_base_url_returns_error(self):
        bad_entry = {"name": "x", "backend": "nonexistent"}
        result = query_chat_openai(bad_entry, "hello", timeout=10)
        self.assertFalse(result["success"])

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_timeout_returns_failure(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout
        result = query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["error"].lower())

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_http_error_returns_failure(self, mock_post):
        import requests as req
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.side_effect = req.exceptions.HTTPError(response=mock_resp)
        result = query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertFalse(result["success"])

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_system_prompt_included_when_provided(self, mock_post):
        mock_post.return_value = _openai_response()
        query_chat_openai(XAI_ENTRY, "hello", timeout=10, system="You are helpful.")
        payload = mock_post.call_args[1]["json"]
        roles = [m["role"] for m in payload["messages"]]
        self.assertIn("system", roles)
        self.assertEqual(payload["messages"][0]["content"], "You are helpful.")

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.post")
    def test_no_system_prompt_has_only_user_message(self, mock_post):
        mock_post.return_value = _openai_response()
        query_chat_openai(XAI_ENTRY, "hello", timeout=10)
        payload = mock_post.call_args[1]["json"]
        roles = [m["role"] for m in payload["messages"]]
        self.assertNotIn("system", roles)
        self.assertIn("user", roles)


# ── query_generate_openai ──────────────────────────────────────────────────────

class TestQueryGenerateOpenai(unittest.TestCase):

    def test_always_returns_failure(self):
        result = query_generate_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertFalse(result["success"])
        self.assertEqual(result["endpoint"], "generate")

    def test_error_mentions_chat_only(self):
        result = query_generate_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertIn("chat-only", result["error"])

    def test_backend_recorded(self):
        result = query_generate_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertEqual(result["backend"], "xai")

    def test_cloud_flag_set(self):
        result = query_generate_openai(XAI_ENTRY, "hello", timeout=10)
        self.assertTrue(result["cloud"])


# ── check_api_key ──────────────────────────────────────────────────────────────

class TestCheckApiKey(unittest.TestCase):

    def test_returns_false_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("XAI_API_KEY", None)
            ok, detail = check_api_key(XAI_ENTRY)
            self.assertFalse(ok)
            self.assertIn("XAI_API_KEY", detail)

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.get")
    def test_returns_true_when_models_endpoint_responds(self, mock_get):
        mock_get.return_value = _models_list_response(["grok-3", "grok-3-mini"])
        ok, detail = check_api_key(XAI_ENTRY)
        self.assertTrue(ok)
        self.assertIn("2", detail)  # "2 model(s) available"

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.get")
    def test_returns_false_on_non_200(self, mock_get):
        m = MagicMock()
        m.status_code = 403
        mock_get.return_value = m
        ok, detail = check_api_key(XAI_ENTRY)
        self.assertFalse(ok)
        self.assertIn("403", detail)

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-key"})
    @patch("probe_query_openai.requests.get")
    def test_returns_false_on_timeout(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout
        ok, detail = check_api_key(XAI_ENTRY)
        self.assertFalse(ok)
        self.assertIn("timeout", detail.lower())

    def test_returns_false_for_missing_base_url(self):
        ok, detail = check_api_key({"backend": "nonexistent"})
        self.assertFalse(ok)


# ── _error_result ──────────────────────────────────────────────────────────────

class TestErrorResult(unittest.TestCase):

    def test_returns_correct_shape(self):
        result = _error_result("xai", "grok-3", "something failed")
        self.assertFalse(result["success"])
        self.assertEqual(result["endpoint"], "chat")
        self.assertEqual(result["error"], "something failed")
        self.assertEqual(result["backend"], "xai")
        self.assertEqual(result["api_model_id"], "grok-3")
        self.assertEqual(result["think_block"], "")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["think_format"], "none")

    def test_elapsed_defaults_to_zero(self):
        result = _error_result("xai", "grok-3", "err")
        self.assertEqual(result["elapsed_s"], 0.0)

    def test_elapsed_can_be_set(self):
        result = _error_result("xai", "grok-3", "err", elapsed=1.23)
        self.assertEqual(result["elapsed_s"], 1.23)


if __name__ == "__main__":
    unittest.main()
