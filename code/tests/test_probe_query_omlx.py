"""
tests/test_probe_query_omlx.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for probe_query_omlx.py.
No real network calls — requests is mocked throughout.
No oMLX server required to run unit tests.

Run:  pytest tests/test_probe_query_omlx.py -v
Integration tests: pytest tests/test_probe_query_omlx.py -v -m integration
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

from probe_query_omlx import (
    BACKEND_NAME,
    DEFAULT_BASE_URL,
    is_omlx_backend,
    base_url_for,
    api_model_id_for,
    headers_for,
    _extract_think,
    _error_result,
    query_chat_omlx,
    query_generate_omlx,
    check_server,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

OMLX_ENTRY = {
    "name":    "qwen3.6-35b-a3b-6bit",
    "backend": "omlx",
    "family":  "qwen",
    "geopolitical_origin": "China/Alibaba",
    "enabled": True,
}

OMLX_ENTRY_WITH_CUSTOM_URL = {
    "name":     "test-model",
    "backend":  "omlx",
    "omlx_url": "http://localhost:9000/v1",
    "api_model_id": "custom-model-id",
}

OLLAMA_ENTRY = {
    "name":    "llama3.2:latest",
    "backend": "ollama",
}


def _omlx_chat_response(content="Paris", thinking=None, tool_calls=None):
    """Build a mock that looks like a successful /v1/chat/completions response."""
    m = MagicMock()
    m.status_code = 200
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    response_data = {
        "choices": [{"message": message}],
    }
    
    if thinking:
        response_data["thinking"] = thinking
    
    m.json.return_value = response_data
    return m


def _omlx_generate_response(content="Paris is the capital"):
    """Build a mock that looks like a successful /completion response."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"content": content}
    return m


def _omlx_models_response(model_ids=("qwen3.6-35b-a3b-6bit",)):
    """Build a mock that looks like a successful /models response."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"data": [{"id": mid} for mid in model_ids]}
    return m


# ── Backend configuration ──────────────────────────────────────────────────────

class TestBackendConfig(unittest.TestCase):

    def test_backend_name_is_omlx(self):
        self.assertEqual(BACKEND_NAME, "omlx")

    def test_default_base_url(self):
        self.assertEqual(DEFAULT_BASE_URL, "http://localhost:8000/v1")


# ── Routing helpers ────────────────────────────────────────────────────────────

class TestIsOmlxBackend(unittest.TestCase):

    def test_omlx_entry_returns_true(self):
        self.assertTrue(is_omlx_backend(OMLX_ENTRY))

    def test_ollama_entry_returns_false(self):
        self.assertFalse(is_omlx_backend(OLLAMA_ENTRY))

    def test_missing_backend_key_returns_false(self):
        self.assertFalse(is_omlx_backend({"name": "something"}))

    def test_unknown_backend_returns_false(self):
        self.assertFalse(is_omlx_backend({"name": "x", "backend": "llamacpp"}))


class TestBaseUrlFor(unittest.TestCase):

    def test_default_url(self):
        self.assertEqual(base_url_for(OMLX_ENTRY), "http://localhost:8000/v1")

    def test_custom_url(self):
        self.assertEqual(base_url_for(OMLX_ENTRY_WITH_CUSTOM_URL), "http://localhost:9000/v1")

    def test_strips_trailing_slash(self):
        entry = {"omlx_url": "http://localhost:8000/v1/"}
        self.assertEqual(base_url_for(entry), "http://localhost:8000/v1")


class TestApiModelIdFor(unittest.TestCase):

    def test_uses_api_model_id_if_present(self):
        self.assertEqual(api_model_id_for(OMLX_ENTRY_WITH_CUSTOM_URL), "custom-model-id")

    def test_falls_back_to_name(self):
        self.assertEqual(api_model_id_for(OMLX_ENTRY), "qwen3.6-35b-a3b-6bit")

    def test_empty_string_if_no_fields(self):
        self.assertEqual(api_model_id_for({}), "")


class TestHeadersFor(unittest.TestCase):

    def test_no_api_key_returns_none(self):
        with patch("probe_query_omlx.OMLX_API_KEY", ""):
            self.assertIsNone(headers_for(OMLX_ENTRY))

    def test_env_api_key_returns_headers(self):
        with patch("probe_query_omlx.OMLX_API_KEY", "test-key"):
            headers = headers_for(OMLX_ENTRY)
            self.assertIsNotNone(headers)
            self.assertEqual(headers["Authorization"], "Bearer test-key")

    def test_entry_api_key_overrides_env(self):
        with patch("probe_query_omlx.OMLX_API_KEY", "env-key"):
            entry = {**OMLX_ENTRY, "omlx_api_key": "entry-key"}
            headers = headers_for(entry)
            self.assertEqual(headers["Authorization"], "Bearer entry-key")


# ── Think block extraction ─────────────────────────────────────────────────────

class TestExtractThink(unittest.TestCase):

    def test_no_think_blocks(self):
        text = "Paris is the capital of France."
        think, clean = _extract_think(text)
        self.assertEqual(think, "")
        self.assertEqual(clean, text)

    def test_single_think_block(self):
        text = "<think>Let me think...</think>Paris is the capital."
        think, clean = _extract_think(text)
        self.assertEqual(think, "Let me think...")
        self.assertEqual(clean, "Paris is the capital.")

    def test_multiple_think_blocks(self):
        text = "<think>First thought</think>Paris<think>Second thought</think>France"
        think, clean = _extract_think(text)
        self.assertIn("First thought", think)
        self.assertIn("Second thought", think)
        self.assertEqual(clean, "ParisFrance")

    def test_multiline_think_block(self):
        text = "<think>\nLine 1\nLine 2\n</think>Answer"
        think, clean = _extract_think(text)
        self.assertIn("Line 1", think)
        self.assertIn("Line 2", think)
        self.assertEqual(clean, "Answer")


# ── Error helper ───────────────────────────────────────────────────────────────

class TestErrorResult(unittest.TestCase):

    def test_error_result_structure(self):
        result = _error_result("chat", "test-model", "Connection failed", 1.5)
        self.assertFalse(result["success"])
        self.assertEqual(result["endpoint"], "chat")
        self.assertEqual(result["api_model_id"], "test-model")
        self.assertEqual(result["error"], "Connection failed")
        self.assertEqual(result["elapsed_s"], 1.5)
        self.assertEqual(result["backend"], "omlx")


# ── Chat endpoint ──────────────────────────────────────────────────────────────

class TestQueryChatOmlx(unittest.TestCase):

    @patch("probe_query_omlx.requests.post")
    def test_successful_chat(self, mock_post):
        mock_post.return_value = _omlx_chat_response("Paris is the capital of France.")
        result = query_chat_omlx(OMLX_ENTRY, "What is the capital of France?", timeout=60)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["endpoint"], "chat")
        self.assertEqual(result["answer"], "Paris is the capital of France.")
        self.assertEqual(result["backend"], "omlx")
        self.assertIn("elapsed_s", result)

    @patch("probe_query_omlx.requests.post")
    def test_chat_with_thinking_field(self, mock_post):
        mock_post.return_value = _omlx_chat_response(
            content="Paris",
            thinking="Let me recall... France's capital is Paris."
        )
        result = query_chat_omlx(OMLX_ENTRY, "What is the capital of France?", timeout=60)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "Paris")
        self.assertEqual(result["think_block"], "Let me recall... France's capital is Paris.")

    @patch("probe_query_omlx.requests.post")
    def test_chat_with_inline_think_tags(self, mock_post):
        mock_post.return_value = _omlx_chat_response(
            "<think>Reasoning...</think>Paris is the capital."
        )
        result = query_chat_omlx(OMLX_ENTRY, "What is the capital of France?", timeout=60)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "Paris is the capital.")
        self.assertIn("think_block", result)
        self.assertEqual(result["think_block"], "Reasoning...")

    @patch("probe_query_omlx.requests.post")
    def test_chat_with_tool_calls(self, mock_post):
        tool_calls = [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}
        }]
        mock_post.return_value = _omlx_chat_response(content="", tool_calls=tool_calls)
        result = query_chat_omlx(OMLX_ENTRY, "What's the weather?", timeout=60, tools=[{}])
        
        self.assertTrue(result["success"])
        self.assertEqual(result["tool_calls"], tool_calls)
        self.assertEqual(result["answer"], "")

    @patch("probe_query_omlx.requests.post")
    def test_chat_timeout(self, mock_post):
        from requests.exceptions import Timeout
        mock_post.side_effect = Timeout()
        result = query_chat_omlx(OMLX_ENTRY, "Question", timeout=1)
        
        self.assertFalse(result["success"])
        self.assertIn("Timeout", result["error"])

    @patch("probe_query_omlx.requests.post")
    def test_chat_connection_error(self, mock_post):
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("Cannot connect")
        result = query_chat_omlx(OMLX_ENTRY, "Question", timeout=60)
        
        self.assertFalse(result["success"])
        self.assertIn("Connection failed", result["error"])

    @patch("probe_query_omlx.requests.post")
    def test_chat_http_error(self, mock_post):
        m = MagicMock()
        m.status_code = 500
        m.text = "Internal server error"
        mock_post.return_value = m
        result = query_chat_omlx(OMLX_ENTRY, "Question", timeout=60)
        
        self.assertFalse(result["success"])
        self.assertIn("HTTP 500", result["error"])

    @patch("probe_query_omlx.requests.post")
    def test_chat_empty_choices(self, mock_post):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"choices": []}
        mock_post.return_value = m
        result = query_chat_omlx(OMLX_ENTRY, "Question", timeout=60)
        
        self.assertFalse(result["success"])
        self.assertIn("Empty choices", result["error"])


# ── Generate endpoint ──────────────────────────────────────────────────────────

class TestQueryGenerateOmlx(unittest.TestCase):

    @patch("probe_query_omlx.requests.post")
    def test_successful_generate(self, mock_post):
        mock_post.return_value = _omlx_generate_response("Paris is the capital")
        result = query_generate_omlx(OMLX_ENTRY, "The capital of France is", timeout=60)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["endpoint"], "gen_plain")
        self.assertEqual(result["answer"], "Paris is the capital")
        self.assertEqual(result["backend"], "omlx")

    @patch("probe_query_omlx.requests.post")
    def test_generate_with_thinking_field(self, mock_post):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {
            "content": "Paris",
            "thinking": "The capital of France is Paris."
        }
        mock_post.return_value = m
        result = query_generate_omlx(OMLX_ENTRY, "Capital of France:", timeout=60)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "Paris")
        self.assertEqual(result["think_block"], "The capital of France is Paris.")

    @patch("probe_query_omlx.requests.post")
    def test_generate_with_inline_think_tags(self, mock_post):
        mock_post.return_value = _omlx_generate_response(
            "<think>Let me think...</think>Paris"
        )
        result = query_generate_omlx(OMLX_ENTRY, "Capital:", timeout=60)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "Paris")
        self.assertEqual(result["think_block"], "Let me think...")

    @patch("probe_query_omlx.requests.post")
    def test_generate_timeout(self, mock_post):
        from requests.exceptions import Timeout
        mock_post.side_effect = Timeout()
        result = query_generate_omlx(OMLX_ENTRY, "Prompt", timeout=1)
        
        self.assertFalse(result["success"])
        self.assertIn("Timeout", result["error"])

    @patch("probe_query_omlx.requests.post")
    def test_generate_connection_error(self, mock_post):
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError()
        result = query_generate_omlx(OMLX_ENTRY, "Prompt", timeout=60)
        
        self.assertFalse(result["success"])
        self.assertIn("Connection failed", result["error"])

    @patch("probe_query_omlx.requests.post")
    def test_generate_empty_response(self, mock_post):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"content": ""}
        mock_post.return_value = m
        result = query_generate_omlx(OMLX_ENTRY, "Prompt", timeout=60)
        
        self.assertFalse(result["success"])
        self.assertIn("Empty response", result["error"])

    @patch("probe_query_omlx.requests.post")
    def test_generate_fallback_to_text_key(self, mock_post):
        """Test fallback when 'content' key is missing but 'text' key exists."""
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"text": "Paris"}
        mock_post.return_value = m
        result = query_generate_omlx(OMLX_ENTRY, "Prompt", timeout=60)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "Paris")


# ── Health check ───────────────────────────────────────────────────────────────

class TestCheckServer(unittest.TestCase):

    @patch("probe_query_omlx.requests.get")
    def test_server_reachable_with_models(self, mock_get):
        mock_get.return_value = _omlx_models_response(
            ("qwen3.6-35b-a3b-6bit", "llama-3.3-70b")
        )
        ok, detail = check_server(OMLX_ENTRY)
        
        self.assertTrue(ok)
        self.assertIn("oMLX server reachable", detail)
        self.assertIn("2 model(s)", detail)

    @patch("probe_query_omlx.requests.get")
    def test_server_reachable_no_models(self, mock_get):
        mock_get.return_value = _omlx_models_response([])
        ok, detail = check_server(OMLX_ENTRY)
        
        self.assertTrue(ok)
        self.assertIn("no models discovered", detail)

    @patch("probe_query_omlx.requests.get")
    def test_server_connection_error(self, mock_get):
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError()
        ok, detail = check_server(OMLX_ENTRY)
        
        self.assertFalse(ok)
        self.assertIn("Cannot connect", detail)

    @patch("probe_query_omlx.requests.get")
    def test_server_timeout(self, mock_get):
        from requests.exceptions import Timeout
        mock_get.side_effect = Timeout()
        ok, detail = check_server(OMLX_ENTRY)
        
        self.assertFalse(ok)
        self.assertIn("Timeout", detail)

    @patch("probe_query_omlx.requests.get")
    def test_server_http_error(self, mock_get):
        m = MagicMock()
        m.status_code = 401
        mock_get.return_value = m
        ok, detail = check_server(OMLX_ENTRY)
        
        self.assertFalse(ok)
        self.assertIn("HTTP 401", detail)

    @patch("probe_query_omlx.requests.get")
    def test_check_server_without_entry(self, mock_get):
        """Test check_server can be called without a model entry."""
        mock_get.return_value = _omlx_models_response(("model-1",))
        ok, detail = check_server(None)
        
        self.assertTrue(ok)
        self.assertIn("oMLX server reachable", detail)


if __name__ == "__main__":
    unittest.main()
