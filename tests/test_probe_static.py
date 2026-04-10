"""
tests/test_probe_static.py
──────────────────────────
Unit tests for probe_static.py — routing, auth headers, think extraction,
deduplication, needs_raw_false detection, and generate payload routing.

No real network calls are made; requests is mocked throughout.

Run:  pytest tests/test_probe_static.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from probe_static import (
    OLLAMA_LOCAL,
    OLLAMA_CLOUD,
    llm_url_for,
    is_cloud,
    auth_headers,
    query_chat,
    query_generate,
    model_is_available,
    needs_raw_false,
    extract_think,
    deduplicate_models,
    _mlx_cache,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chat_response(content="Paris", thinking=""):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"message": {"content": content, "thinking": thinking}}
    m.raise_for_status = MagicMock()
    return m


def _generate_response(response="Paris"):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"response": response}
    m.raise_for_status = MagicMock()
    return m


def _show_response(fmt="gguf", family="llama"):
    """Mock /api/show response for needs_raw_false detection."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"details": {"format": fmt, "family": family}}
    return m


def _post_router(generate_response, show_fmt="gguf", show_family="llama"):
    """
    Returns a side_effect function that routes requests.post calls:
      /api/show    → returns the mock show response (for needs_raw_false)
      /api/generate → returns the mock generate response
      /api/chat    → returns the mock chat response
    """
    def side_effect(url, **kwargs):
        if "/api/show" in url:
            return _show_response(fmt=show_fmt, family=show_family)
        return generate_response
    return side_effect


# ── llm_url_for ───────────────────────────────────────────────────────────────

class TestLlmUrlFor(unittest.TestCase):

    def test_no_cloud_in_name_returns_local(self):
        self.assertEqual(llm_url_for("gemma3:12b"), OLLAMA_LOCAL)

    def test_cloud_in_name_returns_cloud(self):
        self.assertEqual(llm_url_for("deepseek-v3.2:cloud"), OLLAMA_CLOUD)

    def test_cloud_suffix_variants(self):
        for name in ["mistral-large-3:675b-cloud", "qwen3.5:cloud", "glm-5:cloud"]:
            with self.subTest(name=name):
                self.assertEqual(llm_url_for(name), OLLAMA_CLOUD)

    def test_local_model_does_not_route_cloud(self):
        self.assertEqual(llm_url_for("llama3.2:latest"), OLLAMA_LOCAL)
        self.assertEqual(llm_url_for("deepseek-r1:7b"), OLLAMA_LOCAL)


# ── is_cloud ──────────────────────────────────────────────────────────────────

class TestIsCloud(unittest.TestCase):

    def test_cloud_name_returns_true(self):
        self.assertTrue(is_cloud("deepseek-v3.2:cloud"))
        self.assertTrue(is_cloud("qwen3.5:397b-cloud"))

    def test_local_name_returns_false(self):
        self.assertFalse(is_cloud("deepseek-r1:7b"))
        self.assertFalse(is_cloud("gemma3:12b"))


# ── auth_headers ──────────────────────────────────────────────────────────────

class TestAuthHeaders(unittest.TestCase):

    def test_local_url_returns_empty(self):
        self.assertEqual(auth_headers(OLLAMA_LOCAL), {})

    @patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key-123"})
    def test_cloud_url_with_key_returns_bearer(self):
        headers = auth_headers(OLLAMA_CLOUD)
        self.assertEqual(headers, {"Authorization": "Bearer test-key-123"})

    def test_cloud_url_without_key_returns_empty(self):
        import os
        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("OLLAMA_API_KEY", None)
            self.assertEqual(auth_headers(OLLAMA_CLOUD), {})


# ── extract_think ─────────────────────────────────────────────────────────────

class TestExtractThink(unittest.TestCase):

    def test_strips_think_block_from_content(self):
        think, answer = extract_think("<think>reasoning</think>final answer")
        self.assertEqual(think, "reasoning")
        self.assertEqual(answer, "final answer")

    def test_no_think_block_returns_empty_think(self):
        text = "plain answer"
        think, answer = extract_think(text)
        self.assertEqual(think, "")
        self.assertEqual(answer, text)

    def test_multiple_think_blocks_combined(self):
        text = "<think>block one</think>middle<think>block two</think>end"
        think, answer = extract_think(text)
        self.assertIn("block one", think)
        self.assertIn("block two", think)
        self.assertNotIn("<think>", answer)

    def test_empty_think_block(self):
        think, answer = extract_think("<think></think>the answer")
        self.assertEqual(think, "")
        self.assertEqual(answer, "the answer")


# ── needs_raw_false ───────────────────────────────────────────────────────────

class TestNeedsRawFalse(unittest.TestCase):
    """
    needs_raw_false() calls /api/show via requests.post.
    Each test clears _mlx_cache to avoid cross-test contamination.
    """

    def setUp(self):
        _mlx_cache.clear()

    @patch("probe_static.requests.post")
    def test_safetensors_format_returns_true(self, mock_post):
        mock_post.return_value = _show_response(fmt="safetensors", family="qwen3_5_moe")
        self.assertTrue(needs_raw_false("qwen3.5:35b-a3b-coding-nvfp4"))

    @patch("probe_static.requests.post")
    def test_gemma4_family_gguf_returns_true(self, mock_post):
        """gemma4 is GGUF but raw:true hangs in Ollama v0.19."""
        mock_post.return_value = _show_response(fmt="gguf", family="gemma4")
        self.assertTrue(needs_raw_false("gemma4:26b"))

    @patch("probe_static.requests.post")
    def test_standard_gguf_llama_returns_false(self, mock_post):
        mock_post.return_value = _show_response(fmt="gguf", family="llama")
        self.assertFalse(needs_raw_false("llama3.2:latest"))

    @patch("probe_static.requests.post")
    def test_standard_gguf_gemma3_returns_false(self, mock_post):
        mock_post.return_value = _show_response(fmt="gguf", family="gemma3")
        self.assertFalse(needs_raw_false("gemma3:27b"))

    def test_cloud_model_always_returns_false_no_api_call(self):
        """Cloud models skip /api/show entirely."""
        with patch("probe_static.requests.post") as mock_post:
            result = needs_raw_false("deepseek-v3.2:cloud")
            self.assertFalse(result)
            mock_post.assert_not_called()

    @patch("probe_static.requests.post")
    def test_result_cached_second_call_no_request(self, mock_post):
        """After first call the result is cached — no second /api/show request."""
        mock_post.return_value = _show_response(fmt="gguf", family="llama")
        needs_raw_false("llama3.2:latest")
        needs_raw_false("llama3.2:latest")
        self.assertEqual(mock_post.call_count, 1)

    @patch("probe_static.requests.post")
    def test_api_show_exception_returns_false(self, mock_post):
        """If /api/show fails, default to False (use raw:true)."""
        mock_post.side_effect = Exception("connection refused")
        self.assertFalse(needs_raw_false("unknown-model:7b"))


# ── query_chat ────────────────────────────────────────────────────────────────

class TestQueryChat(unittest.TestCase):

    @patch("probe_static.requests.post")
    def test_uses_local_url_by_default(self, mock_post):
        mock_post.return_value = _chat_response()
        query_chat("llama3.2:latest", "hello", timeout=5)
        self.assertTrue(mock_post.call_args[0][0].startswith(OLLAMA_LOCAL))

    @patch("probe_static.requests.post")
    def test_uses_cloud_url_when_specified(self, mock_post):
        mock_post.return_value = _chat_response()
        query_chat("deepseek:cloud", "hello", timeout=5, llm_url=OLLAMA_CLOUD)
        self.assertTrue(mock_post.call_args[0][0].startswith(OLLAMA_CLOUD))

    @patch.dict("os.environ", {"OLLAMA_API_KEY": "mykey"})
    @patch("probe_static.requests.post")
    def test_cloud_call_includes_auth_header(self, mock_post):
        mock_post.return_value = _chat_response()
        query_chat("deepseek:cloud", "hello", timeout=5, llm_url=OLLAMA_CLOUD)
        headers = mock_post.call_args[1]["headers"]
        self.assertIn("Authorization", headers)
        self.assertIn("mykey", headers["Authorization"])

    @patch("probe_static.requests.post")
    def test_local_call_has_no_auth_header(self, mock_post):
        mock_post.return_value = _chat_response()
        query_chat("llama3.2:latest", "hello", timeout=5, llm_url=OLLAMA_LOCAL)
        headers = mock_post.call_args[1]["headers"]
        self.assertNotIn("Authorization", headers)

    @patch("probe_static.requests.post")
    def test_returns_success_true_on_200(self, mock_post):
        mock_post.return_value = _chat_response("Paris")
        result = query_chat("llama3.2:latest", "capital?", timeout=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["endpoint"], "chat")

    @patch("probe_static.requests.post")
    def test_returns_success_false_on_exception(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        result = query_chat("llama3.2:latest", "hello", timeout=5)
        self.assertFalse(result["success"])
        self.assertIn("connection refused", result["error"])

    @patch("probe_static.requests.post")
    def test_field_think_block_captured(self, mock_post):
        """Gemma4-style field-based thinking must be captured in think_block."""
        mock_post.return_value = _chat_response(
            content="the answer", thinking="my reasoning here"
        )
        result = query_chat("gemma4:26b", "hello", timeout=5)
        self.assertEqual(result["think_block"], "my reasoning here")
        self.assertEqual(result["think_format"], "field")
        self.assertEqual(result["answer"], "the answer")

    @patch("probe_static.requests.post")
    def test_tag_think_block_captured(self, mock_post):
        """DeepSeek/Qwen tag-based thinking must be captured in think_block."""
        mock_post.return_value = _chat_response(
            content="<think>chain of thought</think>final answer"
        )
        result = query_chat("deepseek-r1:7b", "hello", timeout=5)
        self.assertEqual(result["think_block"], "chain of thought")
        self.assertEqual(result["think_format"], "tag")
        self.assertEqual(result["answer"], "final answer")

    @patch("probe_static.requests.post")
    def test_no_think_block_format_is_none(self, mock_post):
        mock_post.return_value = _chat_response("plain answer")
        result = query_chat("llama3.2:latest", "hello", timeout=5)
        self.assertEqual(result["think_format"], "none")
        self.assertEqual(result["think_block"], "")


# ── query_generate ────────────────────────────────────────────────────────────

class TestQueryGenerate(unittest.TestCase):

    def setUp(self):
        _mlx_cache.clear()

    @patch("probe_static.requests.post")
    def test_gguf_local_model_uses_raw_true(self, mock_post):
        """Standard GGUF local models must use raw:True."""
        mock_post.side_effect = _post_router(
            _generate_response(), show_fmt="gguf", show_family="llama"
        )
        query_generate("llama3.2:latest", "hello", timeout=5, llm_url=OLLAMA_LOCAL)
        # Find the /api/generate call
        gen_calls = [c for c in mock_post.call_args_list
                     if "/api/generate" in c[0][0]]
        self.assertEqual(len(gen_calls), 1)
        self.assertTrue(gen_calls[0][1]["json"].get("raw"))

    @patch("probe_static.requests.post")
    def test_safetensors_model_omits_raw(self, mock_post):
        """MLX/safetensors models must NOT use raw:True."""
        mock_post.side_effect = _post_router(
            _generate_response(), show_fmt="safetensors", show_family="qwen3_5_moe"
        )
        query_generate("qwen3.5:35b-a3b-coding-nvfp4", "hello", timeout=5, llm_url=OLLAMA_LOCAL)
        gen_calls = [c for c in mock_post.call_args_list
                     if "/api/generate" in c[0][0]]
        self.assertEqual(len(gen_calls), 1)
        self.assertNotIn("raw", gen_calls[0][1]["json"])

    @patch("probe_static.requests.post")
    def test_gemma4_family_omits_raw(self, mock_post):
        """gemma4 family GGUF must NOT use raw:True — hangs in Ollama v0.19."""
        mock_post.side_effect = _post_router(
            _generate_response(), show_fmt="gguf", show_family="gemma4"
        )
        query_generate("gemma4:26b", "hello", timeout=5, llm_url=OLLAMA_LOCAL)
        gen_calls = [c for c in mock_post.call_args_list
                     if "/api/generate" in c[0][0]]
        self.assertEqual(len(gen_calls), 1)
        self.assertNotIn("raw", gen_calls[0][1]["json"])

    @patch("probe_static.requests.post")
    def test_cloud_model_omits_raw(self, mock_post):
        """Cloud models must NOT include raw — it returns HTTP 400."""
        mock_post.return_value = _generate_response()
        query_generate("deepseek:cloud", "hello", timeout=5, llm_url=OLLAMA_CLOUD)
        payload = mock_post.call_args[1]["json"]
        self.assertNotIn("raw", payload)

    @patch("probe_static.requests.post")
    def test_returns_success_true_on_200(self, mock_post):
        mock_post.side_effect = _post_router(_generate_response("Paris"))
        result = query_generate("llama3.2:latest", "capital?", timeout=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["endpoint"], "generate")

    @patch("probe_static.requests.post")
    def test_returns_success_false_on_timeout(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout
        result = query_generate("llama3.2:latest", "hello", timeout=5)
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["error"].lower())

    @patch("probe_static.requests.post")
    def test_gguf_prompt_frame_recorded(self, mock_post):
        mock_post.side_effect = _post_router(
            _generate_response(), show_fmt="gguf", show_family="llama"
        )
        result = query_generate("llama3.2:latest", "hello", timeout=5)
        self.assertIn("raw:true", result["prompt_frame"])

    @patch("probe_static.requests.post")
    def test_raw_false_prompt_frame_recorded(self, mock_post):
        mock_post.side_effect = _post_router(
            _generate_response(), show_fmt="gguf", show_family="gemma4"
        )
        result = query_generate("gemma4:26b", "hello", timeout=5)
        self.assertIn("raw:false", result["prompt_frame"])

    @patch("probe_static.requests.post")
    def test_cloud_prompt_frame_recorded(self, mock_post):
        mock_post.return_value = _generate_response()
        result = query_generate("deepseek:cloud", "hello", timeout=5, llm_url=OLLAMA_CLOUD)
        self.assertIn("cloud", result["prompt_frame"])


# ── model_is_available ────────────────────────────────────────────────────────

class TestModelIsAvailable(unittest.TestCase):

    def _tags_response(self, names):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"models": [{"name": n} for n in names]}
        return m

    @patch("probe_static.requests.get")
    def test_returns_true_when_model_in_tags(self, mock_get):
        mock_get.return_value = self._tags_response(["llama3.2:latest"])
        self.assertTrue(model_is_available("llama3.2:latest"))

    @patch("probe_static.requests.get")
    def test_returns_false_when_model_not_in_tags(self, mock_get):
        mock_get.return_value = self._tags_response(["gemma3:12b"])
        self.assertFalse(model_is_available("llama3.2:latest"))

    @patch("probe_static.requests.get")
    def test_returns_false_on_exception(self, mock_get):
        mock_get.side_effect = Exception("connection refused")
        self.assertFalse(model_is_available("llama3.2:latest"))

    @patch("probe_static.requests.get")
    def test_cloud_model_registered_locally(self, mock_get):
        mock_get.return_value = self._tags_response(["deepseek-v3.2:cloud"])
        self.assertTrue(model_is_available("deepseek-v3.2:cloud"))


# ── deduplicate_models ────────────────────────────────────────────────────────

class TestDeduplicateModels(unittest.TestCase):

    def test_unique_digests_all_kept(self):
        models = ["llama3.2:latest", "gemma3:12b"]
        kept, skipped = deduplicate_models(models, {"llama3.2:latest": "aaa", "gemma3:12b": "bbb"})
        self.assertEqual(kept, models)
        self.assertEqual(skipped, [])

    def test_duplicate_digest_second_skipped(self):
        models = ["Llama3.1:8b", "Llama3.1:latest"]
        digest_map = {"Llama3.1:8b": "46e0c10c", "Llama3.1:latest": "46e0c10c"}
        kept, skipped = deduplicate_models(models, digest_map)
        self.assertIn("Llama3.1:8b", kept)
        self.assertNotIn("Llama3.1:latest", kept)
        self.assertEqual(skipped, [("Llama3.1:latest", "Llama3.1:8b")])

    def test_cloud_models_never_deduplicated(self):
        models = ["qwen3.5:cloud", "qwen3.5:397b-cloud"]
        digest_map = {"qwen3.5:cloud": "stub", "qwen3.5:397b-cloud": "stub"}
        kept, skipped = deduplicate_models(models, digest_map)
        self.assertIn("qwen3.5:cloud", kept)
        self.assertIn("qwen3.5:397b-cloud", kept)
        self.assertEqual(skipped, [])

    def test_empty_digest_map_keeps_all(self):
        models = ["llama3.2:latest", "gemma3:12b"]
        kept, skipped = deduplicate_models(models, {})
        self.assertEqual(kept, models)


if __name__ == "__main__":
    unittest.main()
