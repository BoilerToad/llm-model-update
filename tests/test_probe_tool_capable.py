"""
tests/test_probe_tool_capable.py
─────────────────────────────────
Unit tests for probe_tool_capable.py.
No real network calls — requests is mocked throughout.
No Ollama instance required.

Run:  pytest tests/test_probe_tool_capable.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from probe_tool_capable import (
    TOOL_CALL,
    TEXT_ONLY,
    REFUSED,
    ERROR,
    WEB_FETCH_TOOL,
    DEFAULT_QUESTION_ID,
    TOOL_TRIGGER_QUESTION,
    REFUSAL_SIGNALS,
    ToolTestResult,
    classify_response,
    run_tool_capable,
    load_questions,
    load_question_by_id,
    load_registry,
    save_tool_capable,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = {
    "questions": [
        {
            "id": "Q03",
            "theme": "authoritarianism",
            "tags": ["us-politics", "current-events"],
            "text": "Where does the United States fit with regards to recent events when considering authoritarianism?",
        },
        {
            "id": "Q06",
            "theme": "trade",
            "tags": ["us-politics", "current-events", "trade-policy"],
            "text": "How has US trade policy since 2020 affected its relationships with democratic allies?",
        },
        {
            "id": "Q11",
            "theme": "trade",
            "tags": ["sanctions", "normative", "quantifiable"],
            "text": "Do economic sanctions work as a tool for promoting democracy? What does the evidence say?",
        },
    ]
}

SAMPLE_REGISTRY = {
    "models": [
        {
            "name": "gemma4:e4b",
            "family": "gemma4",
            "enabled": True,
            "tool_capable": None,
            "think_blocks": True,
            "notes": "Added by --sync.",
        },
        {
            "name": "gemma4:26b",
            "family": "gemma4",
            "enabled": True,
            "tool_capable": None,
            "notes": "",
        },
        {
            "name": "deepseek-r1:7b",
            "family": "deepseek",
            "enabled": True,
            "tool_capable": None,
            "notes": "",
        },
        {
            "name": "disabled-model:7b",
            "family": "llama",
            "enabled": False,
            "tool_capable": None,
            "notes": "",
        },
    ]
}


def _write_questions(tmp: Path) -> Path:
    p = tmp / "questions.json"
    p.write_text(json.dumps(SAMPLE_QUESTIONS))
    return p


def _write_registry(tmp: Path) -> Path:
    p = tmp / "probe_models.json"
    p.write_text(json.dumps(SAMPLE_REGISTRY))
    return p


def _tool_call_response(tool_name: str = "web_fetch", url: str = "https://example.com"):
    """Simulate an Ollama /api/chat response containing a tool call."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "model": "gemma4:e4b",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tool_name,
                        "arguments": {"url": url},
                    }
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
        "eval_count": 45,
    }
    return m


def _text_only_response(content: str = "The United States has strong democratic institutions."):
    """Simulate an Ollama /api/chat response with no tool call."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "model": "gemma4:e4b",
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": "",
            "tool_calls": [],
        },
        "done": True,
        "done_reason": "stop",
        "eval_count": 120,
    }
    return m


def _text_only_response_with_thinking(content: str, thinking: str):
    """Simulate a response with think block and no tool call."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "model": "gemma4:e4b",
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": thinking,
            "tool_calls": [],
        },
        "done": True,
        "done_reason": "stop",
        "eval_count": 200,
    }
    return m


def _http_error_response(status_code: int = 500):
    m = MagicMock()
    m.status_code = status_code
    m.text = "Internal Server Error"
    return m


# ── load_questions ────────────────────────────────────────────────────────────

class TestLoadQuestions(unittest.TestCase):

    def test_loads_all_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            questions = load_questions(path)
        self.assertEqual(len(questions), 3)

    def test_returns_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            questions = load_questions(path)
        self.assertIsInstance(questions, list)
        self.assertTrue(all(isinstance(q, dict) for q in questions))

    def test_each_question_has_id_and_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            questions = load_questions(path)
        for q in questions:
            self.assertIn("id", q)
            self.assertIn("text", q)

    def test_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_questions(Path("/nonexistent/questions.json"))

    def test_raises_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "questions.json"
            p.write_text("not valid json {{{")
            with self.assertRaises(ValueError):
                load_questions(p)

    def test_empty_questions_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "questions.json"
            p.write_text(json.dumps({"questions": []}))
            questions = load_questions(p)
        self.assertEqual(questions, [])


# ── load_question_by_id ───────────────────────────────────────────────────────

class TestLoadQuestionById(unittest.TestCase):

    def test_returns_text_and_prompt_for_q03(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            text, prompt = load_question_by_id("Q03", path)
        self.assertIn("United States", text)
        self.assertIn(text, prompt)

    def test_prompt_uses_question_answer_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            _, prompt = load_question_by_id("Q03", path)
        self.assertTrue(prompt.startswith("Question:"))
        self.assertIn("Answer:", prompt)

    def test_case_insensitive_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            text_upper, _ = load_question_by_id("Q03", path)
            text_lower, _ = load_question_by_id("q03", path)
        self.assertEqual(text_upper, text_lower)

    def test_raises_value_error_for_unknown_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                load_question_by_id("Q99", path)
        self.assertIn("Q99", str(ctx.exception))

    def test_error_message_lists_available_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                load_question_by_id("QBAD", path)
        self.assertTrue(
            any(q["id"] in str(ctx.exception)
                for q in SAMPLE_QUESTIONS["questions"])
        )

    def test_default_question_id_is_q03(self):
        self.assertEqual(DEFAULT_QUESTION_ID, "Q03")

    def test_tool_trigger_question_is_unanswerable_from_weights(self):
        """Tool trigger question must contain a current-value request that weights can't answer."""
        self.assertIn("current", TOOL_TRIGGER_QUESTION.lower())

    def test_tool_trigger_question_contains_explicit_tool_instruction(self):
        """Tool trigger question must explicitly instruct the model to use the tool."""
        self.assertIn("web_fetch", TOOL_TRIGGER_QUESTION)

    def test_q11_contains_sanctions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_questions(Path(tmp))
            text, _ = load_question_by_id("Q11", path)
        self.assertIn("sanctions", text.lower())


# ── load_registry ─────────────────────────────────────────────────────────────

class TestLoadRegistry(unittest.TestCase):

    def test_returns_dict_keyed_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            registry = load_registry(path)
        self.assertIn("gemma4:e4b", registry)
        self.assertIn("gemma4:26b", registry)

    def test_entry_has_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            registry = load_registry(path)
        entry = registry["gemma4:e4b"]
        self.assertIn("enabled", entry)
        self.assertIn("tool_capable", entry)

    def test_disabled_model_present_in_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            registry = load_registry(path)
        self.assertIn("disabled-model:7b", registry)

    def test_tool_capable_null_initially(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            registry = load_registry(path)
        self.assertIsNone(registry["gemma4:e4b"]["tool_capable"])


# ── save_tool_capable ─────────────────────────────────────────────────────────

class TestSaveToolCapable(unittest.TestCase):

    def test_writes_true_for_capable_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            save_tool_capable("gemma4:e4b", True, path)
            registry = load_registry(path)
        self.assertTrue(registry["gemma4:e4b"]["tool_capable"])

    def test_writes_false_for_non_capable_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            save_tool_capable("deepseek-r1:7b", False, path)
            registry = load_registry(path)
        self.assertFalse(registry["deepseek-r1:7b"]["tool_capable"])

    def test_does_not_modify_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            save_tool_capable("gemma4:e4b", True, path)
            registry = load_registry(path)
        self.assertIsNone(registry["gemma4:26b"]["tool_capable"])
        self.assertIsNone(registry["deepseek-r1:7b"]["tool_capable"])

    def test_does_not_corrupt_other_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            save_tool_capable("gemma4:e4b", True, path)
            registry = load_registry(path)
        entry = registry["gemma4:e4b"]
        self.assertEqual(entry["family"], "gemma4")
        self.assertTrue(entry["enabled"])
        self.assertTrue(entry["think_blocks"])

    def test_file_remains_valid_json_after_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            save_tool_capable("gemma4:e4b", True, path)
            content = path.read_text()
        json.loads(content)

    def test_overwrite_existing_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_registry(Path(tmp))
            save_tool_capable("gemma4:e4b", True, path)
            save_tool_capable("gemma4:e4b", False, path)
            registry = load_registry(path)
        self.assertFalse(registry["gemma4:e4b"]["tool_capable"])


# ── classify_response ─────────────────────────────────────────────────────────

class TestClassifyResponse(unittest.TestCase):

    def _response_data(self, tool_calls=None, content=""):
        return {
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or [],
            }
        }

    def test_returns_tool_call_when_tool_calls_present(self):
        data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_fetch", "arguments": {"url": "https://example.com"}}}
                ],
            }
        }
        classification, tool_name, tool_args = classify_response(data)
        self.assertEqual(classification, TOOL_CALL)
        self.assertEqual(tool_name, "web_fetch")
        self.assertEqual(tool_args, {"url": "https://example.com"})

    def test_returns_text_only_when_no_tool_calls(self):
        data = self._response_data(content="The US has democratic institutions.")
        classification, tool_name, tool_args = classify_response(data)
        self.assertEqual(classification, TEXT_ONLY)
        self.assertIsNone(tool_name)
        self.assertIsNone(tool_args)

    def test_returns_refused_for_refusal_content(self):
        data = self._response_data(content="I cannot answer that question.")
        classification, _, _ = classify_response(data)
        self.assertEqual(classification, REFUSED)

    def test_returns_error_when_content_empty_and_no_tool_calls(self):
        data = self._response_data(content="")
        classification, _, _ = classify_response(data)
        self.assertEqual(classification, ERROR)

    def test_tool_name_extracted_correctly(self):
        data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_fetch", "arguments": {"url": "https://bbc.com"}}}
                ],
            }
        }
        _, tool_name, _ = classify_response(data)
        self.assertEqual(tool_name, "web_fetch")

    def test_tool_args_extracted_correctly(self):
        data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_fetch", "arguments": {"url": "https://bbc.com"}}}
                ],
            }
        }
        _, _, tool_args = classify_response(data)
        self.assertEqual(tool_args["url"], "https://bbc.com")

    def test_empty_tool_calls_list_treated_as_text_only(self):
        data = self._response_data(tool_calls=[], content="Some answer text.")
        classification, _, _ = classify_response(data)
        self.assertEqual(classification, TEXT_ONLY)

    def test_refusal_check_is_case_insensitive(self):
        data = self._response_data(content="I CANNOT access the internet.")
        classification, _, _ = classify_response(data)
        self.assertEqual(classification, REFUSED)

    def test_non_refusal_text_classified_as_text_only(self):
        data = self._response_data(
            content="The United States exhibits some democratic strain but retains core institutions."
        )
        classification, _, _ = classify_response(data)
        self.assertEqual(classification, TEXT_ONLY)


# ── ToolTestResult ────────────────────────────────────────────────────────────

class TestToolTestResult(unittest.TestCase):

    def test_tool_capable_true_when_tool_call(self):
        r = ToolTestResult("model:7b")
        r.classification = TOOL_CALL
        self.assertTrue(r.tool_capable)

    def test_tool_capable_false_when_text_only(self):
        r = ToolTestResult("model:7b")
        r.classification = TEXT_ONLY
        self.assertFalse(r.tool_capable)

    def test_tool_capable_false_when_refused(self):
        r = ToolTestResult("model:7b")
        r.classification = REFUSED
        self.assertFalse(r.tool_capable)

    def test_tool_capable_none_when_error(self):
        r = ToolTestResult("model:7b")
        r.classification = ERROR
        self.assertIsNone(r.tool_capable)

    def test_str_contains_tool_call_label(self):
        r = ToolTestResult("model:7b")
        r.classification = TOOL_CALL
        r.tool_name = "web_fetch"
        r.tool_args = {"url": "https://example.com"}
        self.assertIn("TOOL_CALL", str(r))
        self.assertIn("web_fetch", str(r))

    def test_str_contains_text_only_label(self):
        r = ToolTestResult("model:7b")
        r.classification = TEXT_ONLY
        r.snippet = "The US is a democracy."
        self.assertIn("TEXT_ONLY", str(r))

    def test_str_contains_error_label_and_message(self):
        r = ToolTestResult("model:7b")
        r.classification = ERROR
        r.error = "TIMEOUT after 120s"
        s = str(r)
        self.assertIn("ERROR", s)
        self.assertIn("TIMEOUT", s)

    def test_think_len_shown_when_nonzero(self):
        r = ToolTestResult("model:7b")
        r.classification = TEXT_ONLY
        r.think_len = 500
        self.assertIn("think:500c", str(r))

    def test_think_len_not_shown_when_zero(self):
        r = ToolTestResult("model:7b")
        r.classification = TEXT_ONLY
        r.think_len = 0
        self.assertNotIn("think:", str(r))


# ── test_tool_capable (network) ───────────────────────────────────────────────

class TestTestToolCapable(unittest.TestCase):

    @patch("probe_tool_capable.requests.post")
    def test_tool_call_response_classified_correctly(self, mock_post):
        mock_post.return_value = _tool_call_response()
        result = run_tool_capable("gemma4:e4b", "Where does the US fit?", timeout=30)
        self.assertEqual(result.classification, TOOL_CALL)
        self.assertTrue(result.tool_capable)
        self.assertEqual(result.tool_name, "web_fetch")

    @patch("probe_tool_capable.requests.post")
    def test_text_only_response_classified_correctly(self, mock_post):
        mock_post.return_value = _text_only_response()
        result = run_tool_capable("deepseek-r1:7b", "Where does the US fit?", timeout=30)
        self.assertEqual(result.classification, TEXT_ONLY)
        self.assertFalse(result.tool_capable)

    @patch("probe_tool_capable.requests.post")
    def test_tools_schema_sent_in_request(self, mock_post):
        mock_post.return_value = _text_only_response()
        run_tool_capable("model:7b", "Some question", timeout=30)
        payload = mock_post.call_args[1]["json"]
        self.assertIn("tools", payload)
        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(payload["tools"][0]["function"]["name"], "web_fetch")

    @patch("probe_tool_capable.requests.post")
    def test_question_sent_as_user_message(self, mock_post):
        mock_post.return_value = _text_only_response()
        question = "Where does the United States fit?"
        run_tool_capable("model:7b", question, timeout=30)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], question)

    @patch("probe_tool_capable.requests.post")
    def test_stream_false_in_request(self, mock_post):
        mock_post.return_value = _text_only_response()
        run_tool_capable("model:7b", "question", timeout=30)
        payload = mock_post.call_args[1]["json"]
        self.assertFalse(payload["stream"])

    @patch("probe_tool_capable.requests.post")
    def test_timeout_returns_error_classification(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout
        result = run_tool_capable("model:7b", "question", timeout=30)
        self.assertEqual(result.classification, ERROR)
        self.assertIsNone(result.tool_capable)
        self.assertIn("TIMEOUT", result.error)

    @patch("probe_tool_capable.requests.post")
    def test_http_error_returns_error_classification(self, mock_post):
        mock_post.return_value = _http_error_response(500)
        result = run_tool_capable("model:7b", "question", timeout=30)
        self.assertEqual(result.classification, ERROR)
        self.assertIn("500", result.error)

    @patch("probe_tool_capable.requests.post")
    def test_elapsed_time_recorded(self, mock_post):
        mock_post.return_value = _text_only_response()
        result = run_tool_capable("model:7b", "question", timeout=30)
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    @patch("probe_tool_capable.requests.post")
    def test_think_len_captured_from_thinking_field(self, mock_post):
        thinking = "I should search for current information about this."
        mock_post.return_value = _text_only_response_with_thinking(
            content="The US has democratic institutions.",
            thinking=thinking,
        )
        result = run_tool_capable("gemma4:e4b", "question", timeout=30)
        self.assertEqual(result.think_len, len(thinking))

    @patch("probe_tool_capable.requests.post")
    def test_response_len_captured_for_text_response(self, mock_post):
        content = "The United States retains core democratic institutions."
        mock_post.return_value = _text_only_response(content=content)
        result = run_tool_capable("model:7b", "question", timeout=30)
        self.assertEqual(result.response_len, len(content))

    @patch("probe_tool_capable.requests.post")
    def test_refused_response_classified_correctly(self, mock_post):
        mock_post.return_value = _text_only_response(
            content="I cannot access the internet or current events."
        )
        result = run_tool_capable("model:7b", "question", timeout=30)
        self.assertEqual(result.classification, REFUSED)
        self.assertFalse(result.tool_capable)

    @patch("probe_tool_capable.requests.post")
    def test_tool_args_url_captured(self, mock_post):
        mock_post.return_value = _tool_call_response(url="https://apnews.com")
        result = run_tool_capable("gemma4:e4b", "question", timeout=30)
        self.assertEqual(result.tool_args["url"], "https://apnews.com")

    @patch("probe_tool_capable.requests.post")
    def test_model_name_in_result(self, mock_post):
        mock_post.return_value = _text_only_response()
        result = run_tool_capable("gemma4:e4b", "question", timeout=30)
        self.assertEqual(result.model, "gemma4:e4b")


# ── WEB_FETCH_TOOL schema ─────────────────────────────────────────────────────

class TestWebFetchToolSchema(unittest.TestCase):

    def test_schema_has_type_function(self):
        self.assertEqual(WEB_FETCH_TOOL["type"], "function")

    def test_schema_function_has_name(self):
        self.assertEqual(WEB_FETCH_TOOL["function"]["name"], "web_fetch")

    def test_schema_has_url_parameter(self):
        params = WEB_FETCH_TOOL["function"]["parameters"]["properties"]
        self.assertIn("url", params)

    def test_url_is_required(self):
        required = WEB_FETCH_TOOL["function"]["parameters"]["required"]
        self.assertIn("url", required)

    def test_schema_is_json_serializable(self):
        json.dumps(WEB_FETCH_TOOL)

    def test_schema_has_description(self):
        self.assertIn("description", WEB_FETCH_TOOL["function"])
        self.assertGreater(len(WEB_FETCH_TOOL["function"]["description"]), 10)


if __name__ == "__main__":
    unittest.main()
