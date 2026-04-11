"""
tests/test_probe_sweep_judge.py
────────────────────────────────
Unit tests for probe_sweep_judge.py.
No real network calls — requests is mocked throughout.
No Ollama instance or API key required.

Run:  pytest tests/test_probe_sweep_judge.py -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from probe_sweep_judge import (
    call_judge,
    parse_judge_response,
    load_sweep,
    extract_responses,
    build_prompt,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_PROMPT_TEMPLATE,
    SWEEP_ENDPOINTS,
    DEFAULT_JUDGE,
    OLLAMA_CLOUD,
    OLLAMA_LOCAL,
    _cloud_url_for,
    _auth_headers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_QUESTION = "What is democracy? Answer in two sentences."

SAMPLE_RESPONSES_CONSISTENT = [
    "Democracy is a system where power is vested in the people.",
    "Democracy is a form of government in which power belongs to the people.",
    "Democracy is a political system where citizens hold ultimate authority.",
]

SAMPLE_RESPONSES_INCONSISTENT = [
    "Democracy is a system where power is vested in the people.",
    "Democracy is a type of monarchy where kings rule by divine right.",
    "Democracy involves the systematic oppression of minority groups.",
]

SAMPLE_JUDGE_RESPONSE = {
    "semantic_consistency": "CONSISTENT",
    "consistency_explanation": "All responses convey the same core definition.",
    "factual_accuracy": "GOOD",
    "accuracy_explanation": "Definitions are factually correct.",
    "most_deviant_indices": [],
    "deviance_explanation": "None",
    "repeatability_verdict": "CONSISTENT",
    "quality_verdict": "GOOD",
    "summary": "The model produces consistent, high-quality responses. No significant variance detected.",
}

SAMPLE_SWEEP = {
    "meta": {
        "run_at":   "2026-04-04T10:55:05",
        "sweeps":   11,
        "question": SAMPLE_QUESTION,
        "models":   ["model-a:7b", "model-b:7b"],
    },
    "stats": {
        "model-a:7b": {
            "chat": {
                "successes": 3,
                "failures":  0,
                "responses": SAMPLE_RESPONSES_CONSISTENT,
            },
            "gen_raw": {
                "successes": 1,
                "failures":  2,
                "responses": ["single response"],   # only 1 — should be excluded
            },
            "gen_plain": {
                "successes": 3,
                "failures":  0,
                "responses": SAMPLE_RESPONSES_INCONSISTENT,
            },
        },
        "model-b:7b": {
            "chat": {
                "successes": 0,
                "failures":  3,
                "responses": [],                    # empty — should be excluded
            },
            "gen_plain": {
                "successes": 3,
                "failures":  0,
                "responses": SAMPLE_RESPONSES_CONSISTENT,
            },
        },
    },
    "recommendations": [],
    "raw_sweeps":       [],
}


def _ok_judge_response(verdict="CONSISTENT", quality="GOOD"):
    """Build a mock Ollama /api/chat response returning judge JSON."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "message": {
            "content": json.dumps({
                **SAMPLE_JUDGE_RESPONSE,
                "repeatability_verdict": verdict,
                "quality_verdict": quality,
            })
        }
    }
    return m


# ── _cloud_url_for ────────────────────────────────────────────────────────────

class TestCloudUrlFor(unittest.TestCase):

    def test_cloud_model_routes_to_ollama_cloud(self):
        self.assertEqual(_cloud_url_for("mistral-large-3:675b-cloud"), OLLAMA_CLOUD)

    def test_local_model_routes_to_local(self):
        self.assertEqual(_cloud_url_for("gemma3:27b"), OLLAMA_LOCAL)
        self.assertEqual(_cloud_url_for("deepseek-r1:7b"), OLLAMA_LOCAL)


# ── _auth_headers ─────────────────────────────────────────────────────────────

class TestAuthHeaders(unittest.TestCase):

    @patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key-xyz"})
    def test_cloud_url_with_key_returns_bearer(self):
        headers = _auth_headers(OLLAMA_CLOUD)
        self.assertIn("Authorization", headers)
        self.assertIn("test-key-xyz", headers["Authorization"])

    def test_local_url_returns_empty(self):
        self.assertEqual(_auth_headers(OLLAMA_LOCAL), {})

    def test_cloud_url_without_key_returns_empty(self):
        import os
        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("OLLAMA_API_KEY", None)
            self.assertEqual(_auth_headers(OLLAMA_CLOUD), {})


# ── parse_judge_response ──────────────────────────────────────────────────────

class TestParseJudgeResponse(unittest.TestCase):

    def test_parses_clean_json(self):
        result = parse_judge_response(json.dumps(SAMPLE_JUDGE_RESPONSE))
        self.assertIsNotNone(result)
        self.assertEqual(result["repeatability_verdict"], "CONSISTENT")

    def test_strips_json_markdown_fence(self):
        fenced = "```json\n" + json.dumps(SAMPLE_JUDGE_RESPONSE) + "\n```"
        result = parse_judge_response(fenced)
        self.assertIsNotNone(result)
        self.assertEqual(result["quality_verdict"], "GOOD")

    def test_strips_plain_markdown_fence(self):
        fenced = "```\n" + json.dumps(SAMPLE_JUDGE_RESPONSE) + "\n```"
        result = parse_judge_response(fenced)
        self.assertIsNotNone(result)

    def test_returns_none_on_invalid_json(self):
        self.assertIsNone(parse_judge_response("not json at all {{{"))

    def test_returns_none_on_empty_string(self):
        self.assertIsNone(parse_judge_response(""))

    def test_returns_none_on_truncated_json(self):
        self.assertIsNone(parse_judge_response('{"key": "val"'))

    def test_preserves_all_fields(self):
        result = parse_judge_response(json.dumps(SAMPLE_JUDGE_RESPONSE))
        for key in SAMPLE_JUDGE_RESPONSE:
            self.assertIn(key, result)


# ── load_sweep ────────────────────────────────────────────────────────────────

class TestLoadSweep(unittest.TestCase):

    def test_loads_valid_json(self, tmp_path=None):
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(SAMPLE_SWEEP, f)
            path = Path(f.name)
        try:
            result = load_sweep(path)
            self.assertIn("meta", result)
            self.assertIn("stats", result)
        finally:
            path.unlink()

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_sweep(Path("/nonexistent/path/sweep.json"))

    def test_raises_on_invalid_json(self):
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_sweep(path)
        finally:
            path.unlink()


# ── extract_responses ─────────────────────────────────────────────────────────

class TestExtractResponses(unittest.TestCase):

    def test_includes_endpoints_with_two_or_more_responses(self):
        result = extract_responses(SAMPLE_SWEEP, ["chat", "gen_raw", "gen_plain"])
        # model-a:7b chat has 3 responses — included
        self.assertIn("chat", result["model-a:7b"])
        # model-a:7b gen_raw has only 1 response — excluded
        self.assertNotIn("gen_raw", result["model-a:7b"])

    def test_excludes_empty_response_arrays(self):
        result = extract_responses(SAMPLE_SWEEP, ["chat"])
        # model-b:7b chat has no responses — model-b excluded from chat
        if "model-b:7b" in result:
            self.assertNotIn("chat", result["model-b:7b"])

    def test_excludes_models_with_no_eligible_endpoints(self):
        """A model with all endpoints having <2 responses is excluded entirely."""
        sweep = {
            "meta": {"question": SAMPLE_QUESTION},
            "stats": {
                "empty-model:7b": {
                    "chat": {"responses": []},
                    "gen_plain": {"responses": ["only one"]},
                }
            }
        }
        result = extract_responses(sweep, ["chat", "gen_plain"])
        self.assertNotIn("empty-model:7b", result)

    def test_filters_whitespace_only_responses(self):
        """Responses that are empty or whitespace-only must be filtered out."""
        sweep = {
            "meta": {"question": SAMPLE_QUESTION},
            "stats": {
                "model-x:7b": {
                    "chat": {
                        "responses": ["valid response", "   ", "", "another valid"]
                    }
                }
            }
        }
        result = extract_responses(sweep, ["chat"])
        responses = result["model-x:7b"]["chat"]
        self.assertEqual(len(responses), 2)
        self.assertNotIn("", responses)
        self.assertNotIn("   ", responses)

    def test_only_requested_endpoints_included(self):
        result = extract_responses(SAMPLE_SWEEP, ["chat"])
        for model_data in result.values():
            self.assertNotIn("gen_plain", model_data)
            self.assertNotIn("gen_raw", model_data)

    def test_returns_correct_response_texts(self):
        result = extract_responses(SAMPLE_SWEEP, ["chat"])
        self.assertEqual(result["model-a:7b"]["chat"], SAMPLE_RESPONSES_CONSISTENT)


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt(unittest.TestCase):

    def test_includes_question(self):
        prompt = build_prompt(SAMPLE_QUESTION, SAMPLE_RESPONSES_CONSISTENT)
        self.assertIn(SAMPLE_QUESTION, prompt)

    def test_includes_all_responses_numbered(self):
        prompt = build_prompt(SAMPLE_QUESTION, SAMPLE_RESPONSES_CONSISTENT)
        self.assertIn("[1]", prompt)
        self.assertIn("[2]", prompt)
        self.assertIn("[3]", prompt)

    def test_includes_response_text(self):
        prompt = build_prompt(SAMPLE_QUESTION, SAMPLE_RESPONSES_CONSISTENT)
        for r in SAMPLE_RESPONSES_CONSISTENT:
            self.assertIn(r.strip(), prompt)

    def test_n_reflects_response_count(self):
        prompt = build_prompt(SAMPLE_QUESTION, SAMPLE_RESPONSES_CONSISTENT)
        # The count should appear in the prompt
        self.assertIn("3", prompt)

    def test_single_response_still_builds(self):
        """Edge case: build_prompt should not crash on a single response."""
        prompt = build_prompt(SAMPLE_QUESTION, ["only one response"])
        self.assertIn("[1]", prompt)

    def test_prompt_contains_schema_fields(self):
        """The prompt must mention the required JSON schema fields."""
        prompt = build_prompt(SAMPLE_QUESTION, SAMPLE_RESPONSES_CONSISTENT)
        for field in ["repeatability_verdict", "quality_verdict", "semantic_consistency"]:
            self.assertIn(field, prompt)


# ── call_judge ────────────────────────────────────────────────────────────────

class TestCallJudge(unittest.TestCase):

    @patch("probe_sweep_judge.requests.post")
    def test_returns_success_true_on_200(self, mock_post):
        mock_post.return_value = _ok_judge_response()
        ok, content, elapsed = call_judge(
            DEFAULT_JUDGE, JUDGE_SYSTEM_PROMPT, "test prompt", timeout=30
        )
        self.assertTrue(ok)
        self.assertIn("repeatability_verdict", content)

    @patch("probe_sweep_judge.requests.post")
    def test_returns_success_false_on_http_error(self, mock_post):
        m = MagicMock()
        m.status_code = 500
        m.text = "Internal Server Error"
        mock_post.return_value = m
        ok, content, elapsed = call_judge(
            DEFAULT_JUDGE, JUDGE_SYSTEM_PROMPT, "test prompt", timeout=30
        )
        self.assertFalse(ok)
        self.assertIn("500", content)

    @patch("probe_sweep_judge.requests.post")
    def test_returns_success_false_on_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout
        ok, content, elapsed = call_judge(
            DEFAULT_JUDGE, JUDGE_SYSTEM_PROMPT, "test prompt", timeout=30
        )
        self.assertFalse(ok)
        self.assertIn("TIMEOUT", content)

    @patch("probe_sweep_judge.requests.post")
    def test_routes_cloud_model_to_ollama_cloud(self, mock_post):
        mock_post.return_value = _ok_judge_response()
        call_judge("mistral-large-3:675b-cloud", JUDGE_SYSTEM_PROMPT, "test", timeout=30)
        url_called = mock_post.call_args[0][0]
        self.assertIn("ollama.com", url_called)

    @patch("probe_sweep_judge.requests.post")
    def test_routes_local_model_to_localhost(self, mock_post):
        mock_post.return_value = _ok_judge_response()
        call_judge("gemma3:27b", JUDGE_SYSTEM_PROMPT, "test", timeout=30)
        url_called = mock_post.call_args[0][0]
        self.assertIn("localhost", url_called)

    @patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"})
    @patch("probe_sweep_judge.requests.post")
    def test_cloud_call_includes_auth_header(self, mock_post):
        mock_post.return_value = _ok_judge_response()
        call_judge("mistral-large-3:675b-cloud", JUDGE_SYSTEM_PROMPT, "test", timeout=30)
        headers = mock_post.call_args[1]["headers"]
        self.assertIn("Authorization", headers)
        self.assertIn("test-key", headers["Authorization"])

    @patch("probe_sweep_judge.requests.post")
    def test_elapsed_s_is_non_negative(self, mock_post):
        mock_post.return_value = _ok_judge_response()
        ok, content, elapsed = call_judge(
            DEFAULT_JUDGE, JUDGE_SYSTEM_PROMPT, "test", timeout=30
        )
        self.assertGreaterEqual(elapsed, 0)

    @patch("probe_sweep_judge.requests.post")
    def test_system_prompt_included_in_messages(self, mock_post):
        mock_post.return_value = _ok_judge_response()
        call_judge(DEFAULT_JUDGE, JUDGE_SYSTEM_PROMPT, "user prompt", timeout=30)
        payload = mock_post.call_args[1]["json"]
        roles = [m["role"] for m in payload["messages"]]
        self.assertIn("system", roles)
        self.assertIn("user", roles)
        system_content = next(m["content"] for m in payload["messages"]
                              if m["role"] == "system")
        self.assertEqual(system_content, JUDGE_SYSTEM_PROMPT)


# ── Integration: output structure ─────────────────────────────────────────────

class TestOutputStructure(unittest.TestCase):
    """
    Test that run_judgement produces correctly structured output
    without making real network calls.
    """

    def _make_sweep_file(self, tmp_path):
        p = tmp_path / "sweep.json"
        p.write_text(json.dumps(SAMPLE_SWEEP))
        return p

    @patch("probe_sweep_judge.requests.get")   # version check
    @patch("probe_sweep_judge.requests.post")  # judge call
    def test_output_has_required_top_level_keys(self, mock_post, mock_get):
        import tempfile, os
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"version": "0.20.0"}
        )
        mock_post.return_value = _ok_judge_response()

        from probe_sweep_judge import run_judgement

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = Path(tmpdir) / "sweep.json"
            sweep_path.write_text(json.dumps(SAMPLE_SWEEP))
            out_path = Path(tmpdir) / "judge_out.json"

            result = run_judgement(
                sweep_path  = sweep_path,
                judge_model = DEFAULT_JUDGE,
                endpoints   = ["chat"],
                timeout     = 30,
                out_path    = out_path,
            )

        self.assertIn("meta", result)
        self.assertIn("judgements", result)
        self.assertIn("judge_model", result["meta"])
        self.assertIn("question", result["meta"])
        self.assertIn("endpoints", result["meta"])

    @patch("probe_sweep_judge.requests.get")
    @patch("probe_sweep_judge.requests.post")
    def test_judgement_saved_to_file(self, mock_post, mock_get):
        import tempfile
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"version": "0.20.0"}
        )
        mock_post.return_value = _ok_judge_response()

        from probe_sweep_judge import run_judgement

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = Path(tmpdir) / "sweep.json"
            sweep_path.write_text(json.dumps(SAMPLE_SWEEP))
            out_path = Path(tmpdir) / "judge_out.json"

            run_judgement(
                sweep_path  = sweep_path,
                judge_model = DEFAULT_JUDGE,
                endpoints   = ["chat"],
                timeout     = 30,
                out_path    = out_path,
            )

            self.assertTrue(out_path.exists())
            saved = json.loads(out_path.read_text())
            self.assertIn("meta", saved)
            self.assertIn("judgements", saved)

    @patch("probe_sweep_judge.requests.get")
    @patch("probe_sweep_judge.requests.post")
    def test_failed_judge_call_recorded(self, mock_post, mock_get):
        """A failed judge call must be stored with success=False, not raise."""
        import requests, tempfile
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"version": "0.20.0"}
        )
        mock_post.side_effect = requests.exceptions.Timeout

        from probe_sweep_judge import run_judgement

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = Path(tmpdir) / "sweep.json"
            sweep_path.write_text(json.dumps(SAMPLE_SWEEP))
            out_path = Path(tmpdir) / "judge_out.json"

            result = run_judgement(
                sweep_path  = sweep_path,
                judge_model = DEFAULT_JUDGE,
                endpoints   = ["chat"],
                timeout     = 30,
                out_path    = out_path,
            )

        # Should record failure, not raise
        model_judgements = result["judgements"].get("model-a:7b", {})
        chat_result = model_judgements.get("chat", {})
        self.assertFalse(chat_result.get("success", True))


if __name__ == "__main__":
    unittest.main()
