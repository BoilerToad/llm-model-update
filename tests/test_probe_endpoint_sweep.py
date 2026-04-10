"""
tests/test_probe_endpoint_sweep.py
───────────────────────────────────
Unit tests for probe_endpoint_sweep.py.
No real network calls — requests is mocked throughout.
No Ollama instance required.

Run:  pytest tests/test_probe_endpoint_sweep.py -v
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

from probe_endpoint_sweep import (
    ModelStats,
    Recommendation,
    run_chat,
    run_generate,
    warmup_model,
    post_evaluate,
    jaccard_similarity,
    pairwise_similarities,
    load_questions,
    load_question_by_id,
    DEFAULT_QUESTION,
    DEFAULT_PROMPT,
    FAIL_RATE_THRESHOLD,
    SHALLOW_TIME_THRESHOLD,
    SHALLOW_TOKEN_THRESHOLD,
    TOKEN_RATIO_THRESHOLD,
    LOW_SIMILARITY_THRESHOLD,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS_DATA = {
    "questions": [
        {
            "id": "Q01",
            "theme": "authoritarianism",
            "tags": ["comparative"],
            "text": "Describe authoritarianism.",
        },
        {
            "id": "Q10b",
            "theme": "trade",
            "tags": ["china-named"],
            "text": "How has China used access to its domestic market as a tool of political influence?",
        },
        {
            "id": "Q03",
            "theme": "authoritarianism",
            "tags": ["us-politics"],
            "text": "Where does the United States fit regarding authoritarianism?",
        },
    ]
}


def _write_questions_file(tmp_path: Path, data: dict = SAMPLE_QUESTIONS_DATA) -> Path:
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(data))
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok_response(content="answer", eval_count=200, thinking=""):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "message": {"content": content, "thinking": thinking},
        "response": content,
        "eval_count": eval_count,
    }
    return m


def _make_stats(
    name: str,
    n: int = 11,
    chat_fail: int = 0,
    gen_raw_fail: int = 0,
    gen_plain_fail: int = 0,
    gen_raw_tokens: float = 200.0,
    gen_plain_tokens: float = 200.0,
    gen_raw_time: float = 5.0,
    gen_raw_contaminated: int = 0,
    chat_texts: list[str] | None = None,
    gen_raw_texts: list[str] | None = None,
    gen_plain_texts: list[str] | None = None,
) -> ModelStats:
    """
    Build a ModelStats object with synthetic data.
    Provides default response texts so similarity calculations work out of the box.
    Override chat_texts / gen_raw_texts / gen_plain_texts for repeatability tests.
    """
    s = ModelStats(name)
    s.total_sweeps      = n
    s.chat_successes    = n - chat_fail
    s.chat_failures     = chat_fail
    s.chat_times        = [5.0] * (n - chat_fail)
    s.chat_texts        = chat_texts if chat_texts is not None else (
        ["democracy is a system of government"] * (n - chat_fail)
    )

    s.gen_raw_successes    = n - gen_raw_fail
    s.gen_raw_failures     = gen_raw_fail
    s.gen_raw_times        = [gen_raw_time] * (n - gen_raw_fail)
    s.gen_raw_tokens       = [int(gen_raw_tokens)] * (n - gen_raw_fail)
    s.gen_raw_contaminated = gen_raw_contaminated
    s.gen_raw_texts        = gen_raw_texts if gen_raw_texts is not None else (
        ["democracy is a form of government"] * (n - gen_raw_fail)
    )

    s.gen_plain_successes = n - gen_plain_fail
    s.gen_plain_failures  = gen_plain_fail
    s.gen_plain_times     = [5.0] * (n - gen_plain_fail)
    s.gen_plain_tokens    = [int(gen_plain_tokens)] * (n - gen_plain_fail)
    s.gen_plain_texts     = gen_plain_texts if gen_plain_texts is not None else (
        ["democracy is a form of government"] * (n - gen_plain_fail)
    )
    return s


def _absorb_result(
    chat_text="democracy is a system of government",
    gen_raw_text="democracy is a form of government",
    gen_plain_text="democracy involves elections and rights",
    chat_ok=True,
    gen_raw_ok=True,
    gen_plain_ok=True,
) -> dict:
    """Build a synthetic sweep result dict for ModelStats.absorb()."""
    return {
        "chat": {
            "success": chat_ok, "elapsed_s": 5.0, "eval_count": 100,
            "response_len": len(chat_text), "think_len": 0,
            "response_text": chat_text,
        } if chat_ok else {
            "success": False, "elapsed_s": 60.0, "error": "TIMEOUT",
            "eval_count": 0, "response_len": 0, "response_text": "",
        },
        "gen_raw": {
            "success": gen_raw_ok, "elapsed_s": 2.0, "eval_count": 80,
            "response_len": len(gen_raw_text), "raw": True,
            "contaminated": False, "response_text": gen_raw_text,
        } if gen_raw_ok else {
            "success": False, "elapsed_s": 60.0, "error": "TIMEOUT",
            "eval_count": 0, "response_len": 0, "raw": True, "response_text": "",
        },
        "gen_plain": {
            "success": gen_plain_ok, "elapsed_s": 4.0, "eval_count": 180,
            "response_len": len(gen_plain_text), "raw": False,
            "contaminated": False, "response_text": gen_plain_text,
        } if gen_plain_ok else {
            "success": False, "elapsed_s": 60.0, "error": "TIMEOUT",
            "eval_count": 0, "response_len": 0, "raw": False, "response_text": "",
        },
    }


# ── load_questions ────────────────────────────────────────────────────────────

class TestLoadQuestions(unittest.TestCase):

    def test_loads_all_questions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            questions = load_questions(path)
        self.assertEqual(len(questions), 3)

    def test_returns_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            questions = load_questions(path)
        self.assertIsInstance(questions, list)
        self.assertTrue(all(isinstance(q, dict) for q in questions))

    def test_each_question_has_id_and_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            questions = load_questions(path)
        for q in questions:
            self.assertIn("id", q)
            self.assertIn("text", q)

    def test_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_questions(Path("/nonexistent/path/questions.json"))

    def test_raises_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "questions.json"
            p.write_text("not valid json {{{")
            with self.assertRaises(ValueError):
                load_questions(p)

    def test_empty_questions_array_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "questions.json"
            p.write_text(json.dumps({"questions": []}))
            questions = load_questions(p)
        self.assertEqual(questions, [])


# ── load_question_by_id ───────────────────────────────────────────────────────

class TestLoadQuestionById(unittest.TestCase):

    def test_returns_text_and_prompt_for_valid_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            text, prompt = load_question_by_id("Q01", path)
        self.assertEqual(text, "Describe authoritarianism.")
        self.assertIn("Describe authoritarianism.", prompt)

    def test_prompt_uses_question_answer_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            text, prompt = load_question_by_id("Q01", path)
        self.assertTrue(prompt.startswith("Question:"))
        self.assertIn("Answer:", prompt)

    def test_case_insensitive_lookup(self):
        """Q10b, q10b, Q10B should all resolve the same question."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            text_upper, _ = load_question_by_id("Q10b", path)
            text_lower, _ = load_question_by_id("q10b", path)
            text_mixed, _ = load_question_by_id("Q10B", path)
        self.assertEqual(text_upper, text_lower)
        self.assertEqual(text_upper, text_mixed)

    def test_raises_value_error_for_unknown_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            with self.assertRaises(ValueError) as ctx:
                load_question_by_id("Q99", path)
        self.assertIn("Q99", str(ctx.exception))

    def test_error_message_lists_available_ids(self):
        """ValueError must name available IDs so the user knows what to use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            with self.assertRaises(ValueError) as ctx:
                load_question_by_id("QBAD", path)
        error_msg = str(ctx.exception)
        # Should mention at least one valid ID
        self.assertTrue(
            any(q["id"] in error_msg for q in SAMPLE_QUESTIONS_DATA["questions"])
        )

    def test_returns_correct_text_for_q10b(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            text, _ = load_question_by_id("Q10b", path)
        self.assertIn("China", text)

    def test_prompt_contains_full_question_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_questions_file(Path(tmpdir))
            text, prompt = load_question_by_id("Q03", path)
        self.assertIn(text, prompt)


# ── jaccard_similarity ────────────────────────────────────────────────────────

class TestJaccardSimilarity(unittest.TestCase):

    def test_identical_strings_return_one(self):
        self.assertAlmostEqual(
            jaccard_similarity("democracy is a system", "democracy is a system"), 1.0
        )

    def test_completely_different_strings_return_zero(self):
        self.assertAlmostEqual(
            jaccard_similarity("alpha beta gamma", "delta epsilon zeta"), 0.0
        )

    def test_partial_overlap(self):
        score = jaccard_similarity("democracy system", "democracy government")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(jaccard_similarity("", "some text"), 0.0)
        self.assertEqual(jaccard_similarity("some text", ""), 0.0)
        self.assertEqual(jaccard_similarity("", ""), 0.0)

    def test_case_insensitive(self):
        self.assertAlmostEqual(
            jaccard_similarity("Democracy Is", "democracy is"), 1.0
        )

    def test_word_order_does_not_matter(self):
        score_ab = jaccard_similarity("alpha beta gamma", "gamma alpha beta")
        self.assertAlmostEqual(score_ab, 1.0)

    def test_result_in_zero_one_range(self):
        for a, b in [("hello world", "world peace"), ("x", "y z"), ("a b c", "b c d e")]:
            score = jaccard_similarity(a, b)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ── pairwise_similarities ─────────────────────────────────────────────────────

class TestPairwiseSimilarities(unittest.TestCase):

    def test_returns_n_minus_one_scores(self):
        texts = ["text one", "text two", "text three"]
        scores = pairwise_similarities(texts)
        self.assertEqual(len(scores), 2)

    def test_single_text_returns_empty(self):
        self.assertEqual(pairwise_similarities(["only one"]), [])

    def test_empty_list_returns_empty(self):
        self.assertEqual(pairwise_similarities([]), [])

    def test_identical_texts_all_ones(self):
        texts = ["democracy is a system"] * 5
        scores = pairwise_similarities(texts)
        for s in scores:
            self.assertAlmostEqual(s, 1.0)

    def test_scores_in_valid_range(self):
        texts = ["alpha beta", "beta gamma", "gamma delta", "delta epsilon"]
        scores = pairwise_similarities(texts)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


# ── ModelStats ────────────────────────────────────────────────────────────────

class TestModelStats(unittest.TestCase):

    def test_absorb_success_stores_response_text(self):
        s = ModelStats("test-model")
        s.absorb(_absorb_result(
            chat_text="democracy text here",
            gen_raw_text="raw text here",
            gen_plain_text="plain text here",
        ))
        self.assertEqual(s.chat_texts, ["democracy text here"])
        self.assertEqual(s.gen_raw_texts, ["raw text here"])
        self.assertEqual(s.gen_plain_texts, ["plain text here"])

    def test_absorb_failure_does_not_append_text(self):
        s = ModelStats("test-model")
        s.absorb(_absorb_result(chat_ok=False, gen_raw_ok=False, gen_plain_ok=True))
        self.assertEqual(s.chat_texts, [])
        self.assertEqual(s.gen_raw_texts, [])
        self.assertEqual(len(s.gen_plain_texts), 1)

    def test_absorb_multiple_sweeps_accumulates_texts(self):
        s = ModelStats("test-model")
        s.absorb(_absorb_result(chat_text="first response"))
        s.absorb(_absorb_result(chat_text="second response"))
        self.assertEqual(len(s.chat_texts), 2)
        self.assertIn("first response", s.chat_texts)
        self.assertIn("second response", s.chat_texts)

    def test_absorb_success_increments_counts(self):
        s = ModelStats("test-model")
        s.absorb(_absorb_result())
        self.assertEqual(s.chat_successes, 1)
        self.assertEqual(s.gen_raw_successes, 1)
        self.assertEqual(s.gen_plain_successes, 1)
        self.assertEqual(s.total_sweeps, 1)

    def test_absorb_failure_increments_failure_count(self):
        s = ModelStats("test-model")
        s.absorb(_absorb_result(chat_ok=False, gen_raw_ok=False, gen_plain_ok=True))
        self.assertEqual(s.chat_failures, 1)
        self.assertEqual(s.gen_raw_failures, 1)
        self.assertEqual(s.gen_plain_successes, 1)

    def test_chat_fail_rate_zero_when_all_succeed(self):
        s = _make_stats("m", n=11, chat_fail=0)
        self.assertAlmostEqual(s.chat_fail_rate, 0.0)

    def test_chat_fail_rate_one_when_all_fail(self):
        s = _make_stats("m", n=11, chat_fail=11)
        self.assertAlmostEqual(s.chat_fail_rate, 1.0)

    def test_gen_raw_fail_rate_partial(self):
        s = _make_stats("m", n=11, gen_raw_fail=4)
        self.assertAlmostEqual(s.gen_raw_fail_rate, 4/11, places=3)

    def test_token_ratio_plain_over_raw(self):
        s = _make_stats("m", gen_raw_tokens=50, gen_plain_tokens=500)
        self.assertAlmostEqual(s.token_ratio, 10.0, places=1)

    def test_token_ratio_zero_when_no_raw_tokens(self):
        s = _make_stats("m", n=11, gen_raw_fail=11)
        self.assertEqual(s.token_ratio, 0.0)

    def test_avg_similarity_identical_texts_returns_one(self):
        s = _make_stats("m", chat_texts=["democracy is a system"] * 5)
        sim = s.avg_similarity(s.chat_texts)
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_avg_similarity_none_when_fewer_than_two_texts(self):
        s = _make_stats("m", chat_texts=["only one text"])
        self.assertIsNone(s.avg_similarity(s.chat_texts))

    def test_avg_similarity_none_when_no_texts(self):
        s = _make_stats("m", chat_texts=[])
        self.assertIsNone(s.avg_similarity(s.chat_texts))

    def test_min_similarity_with_one_low_pair(self):
        texts = [
            "democracy involves voting citizens",
            "democracy involves voting citizens",
            "completely different unrelated answer words",
        ]
        s = _make_stats("m", chat_texts=texts)
        mn = s.min_similarity(s.chat_texts)
        self.assertIsNotNone(mn)
        self.assertLess(mn, 0.5)

    def test_avg_and_min_similarity_in_range(self):
        texts = ["alpha beta gamma delta", "beta gamma delta epsilon", "gamma delta epsilon zeta"]
        s = _make_stats("m", chat_texts=texts)
        avg = s.avg_similarity(s.chat_texts)
        mn  = s.min_similarity(s.chat_texts)
        self.assertGreaterEqual(avg, 0.0)
        self.assertLessEqual(avg, 1.0)
        self.assertLessEqual(mn, avg)

    def test_to_dict_has_required_keys(self):
        s = _make_stats("test-model")
        d = s.to_dict()
        for section in ("chat", "gen_raw", "gen_plain"):
            self.assertIn(section, d)
            self.assertIn("fail_rate",      d[section])
            self.assertIn("avg_elapsed_s",  d[section])
            self.assertIn("avg_similarity", d[section])
            self.assertIn("min_similarity", d[section])
            self.assertIn("responses",      d[section])
        self.assertIn("token_ratio_plain_over_raw", d)

    def test_to_dict_responses_contains_texts(self):
        texts = ["first answer text", "second answer text"]
        s = _make_stats("m", chat_texts=texts)
        d = s.to_dict()
        self.assertEqual(d["chat"]["responses"], texts)

    def test_to_dict_serializable(self):
        s = _make_stats("test-model")
        json.dumps(s.to_dict())


# ── warmup_model ──────────────────────────────────────────────────────────────

class TestWarmupModel(unittest.TestCase):

    @patch("probe_endpoint_sweep.requests.post")
    def test_returns_true_on_200(self, mock_post):
        m = MagicMock()
        m.status_code = 200
        mock_post.return_value = m
        self.assertTrue(warmup_model("llama3.2:latest", timeout=30))

    @patch("probe_endpoint_sweep.requests.post")
    def test_returns_false_on_non_200(self, mock_post):
        m = MagicMock()
        m.status_code = 500
        mock_post.return_value = m
        self.assertFalse(warmup_model("llama3.2:latest", timeout=30))

    @patch("probe_endpoint_sweep.requests.post")
    def test_returns_false_on_exception(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        self.assertFalse(warmup_model("llama3.2:latest", timeout=30))

    @patch("probe_endpoint_sweep.requests.post")
    def test_result_is_discarded(self, mock_post):
        m = MagicMock()
        m.status_code = 200
        mock_post.return_value = m
        result = warmup_model("llama3.2:latest", timeout=30)
        self.assertIsInstance(result, bool)


# ── run_chat ──────────────────────────────────────────────────────────────────

class TestRunChat(unittest.TestCase):

    @patch("probe_endpoint_sweep.requests.post")
    def test_success_response(self, mock_post):
        mock_post.return_value = _ok_response("Democracy is...", eval_count=150)
        result = run_chat("llama3.2:latest", timeout=30)
        self.assertTrue(result["success"])
        self.assertEqual(result["eval_count"], 150)
        self.assertGreaterEqual(result["elapsed_s"], 0)
        self.assertIn("elapsed_s", result)

    @patch("probe_endpoint_sweep.requests.post")
    def test_uses_default_question_when_not_specified(self, mock_post):
        """When question is not provided, DEFAULT_QUESTION must be sent."""
        mock_post.return_value = _ok_response()
        run_chat("llama3.2:latest", timeout=30)
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        self.assertEqual(messages[0]["content"], DEFAULT_QUESTION)

    @patch("probe_endpoint_sweep.requests.post")
    def test_uses_custom_question_when_specified(self, mock_post):
        """When question is provided, the custom text must be sent instead."""
        custom = "Describe authoritarianism in one sentence."
        mock_post.return_value = _ok_response()
        run_chat("llama3.2:latest", timeout=30, question=custom)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["messages"][0]["content"], custom)

    @patch("probe_endpoint_sweep.requests.post")
    def test_response_text_retained(self, mock_post):
        mock_post.return_value = _ok_response("Democracy is a system.", eval_count=50)
        result = run_chat("llama3.2:latest", timeout=30)
        self.assertEqual(result["response_text"], "Democracy is a system.")

    @patch("probe_endpoint_sweep.requests.post")
    def test_response_text_empty_on_failure(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout
        result = run_chat("llama3.2:latest", timeout=30)
        self.assertFalse(result["success"])
        self.assertEqual(result["response_text"], "")

    @patch("probe_endpoint_sweep.requests.post")
    def test_timeout_returns_failure(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout
        result = run_chat("llama3.2:latest", timeout=30)
        self.assertFalse(result["success"])
        self.assertIn("TIMEOUT", result["error"])

    @patch("probe_endpoint_sweep.requests.post")
    def test_http_error_returns_failure(self, mock_post):
        m = MagicMock()
        m.status_code = 500
        m.text = "Internal Server Error"
        mock_post.return_value = m
        result = run_chat("llama3.2:latest", timeout=30)
        self.assertFalse(result["success"])
        self.assertIn("500", result["error"])

    @patch("probe_endpoint_sweep.requests.post")
    def test_thinking_field_captured(self, mock_post):
        mock_post.return_value = _ok_response("answer", thinking="my thinking")
        result = run_chat("gemma4:26b", timeout=30)
        self.assertEqual(result["think_len"], len("my thinking"))


# ── run_generate ──────────────────────────────────────────────────────────────

class TestRunGenerate(unittest.TestCase):

    @patch("probe_endpoint_sweep.requests.post")
    def test_success_raw_true(self, mock_post):
        mock_post.return_value = _ok_response("Democracy is...", eval_count=100)
        result = run_generate("llama3.2:latest", timeout=30, raw=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["eval_count"], 100)
        payload = mock_post.call_args[1]["json"]
        self.assertTrue(payload.get("raw"))

    @patch("probe_endpoint_sweep.requests.post")
    def test_success_raw_false(self, mock_post):
        mock_post.return_value = _ok_response("Democracy is...", eval_count=200)
        result = run_generate("llama3.2:latest", timeout=30, raw=False)
        self.assertTrue(result["success"])
        payload = mock_post.call_args[1]["json"]
        self.assertNotIn("raw", payload)

    @patch("probe_endpoint_sweep.requests.post")
    def test_uses_default_prompt_when_not_specified(self, mock_post):
        """When prompt is not provided, DEFAULT_PROMPT must be sent."""
        mock_post.return_value = _ok_response()
        run_generate("llama3.2:latest", timeout=30, raw=False)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["prompt"], DEFAULT_PROMPT)

    @patch("probe_endpoint_sweep.requests.post")
    def test_uses_custom_prompt_when_specified(self, mock_post):
        """When prompt is provided, the custom text must be sent."""
        custom_prompt = "Question: Describe authoritarianism.\n\nAnswer:"
        mock_post.return_value = _ok_response()
        run_generate("llama3.2:latest", timeout=30, raw=False, prompt=custom_prompt)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["prompt"], custom_prompt)

    @patch("probe_endpoint_sweep.requests.post")
    def test_response_text_retained(self, mock_post):
        mock_post.return_value = _ok_response("Democracy involves voting.", eval_count=50)
        result = run_generate("llama3.2:latest", timeout=30, raw=True)
        self.assertEqual(result["response_text"], "Democracy involves voting.")

    @patch("probe_endpoint_sweep.requests.post")
    def test_response_text_empty_on_failure(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout
        result = run_generate("llama3.2:latest", timeout=30, raw=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["response_text"], "")

    @patch("probe_endpoint_sweep.requests.post")
    def test_timeout_returns_failure(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout
        result = run_generate("llama3.2:latest", timeout=30, raw=True)
        self.assertFalse(result["success"])
        self.assertIn("TIMEOUT", result["error"])

    @patch("probe_endpoint_sweep.requests.post")
    def test_contamination_detected(self, mock_post):
        contaminated = "Democracy is X.\n\nQuestion: What is oligarchy?\n\nAnswer: ..."
        mock_post.return_value = _ok_response(contaminated, eval_count=50)
        result = run_generate("model:7b", timeout=30, raw=True)
        self.assertTrue(result["contaminated"])

    @patch("probe_endpoint_sweep.requests.post")
    def test_clean_response_not_contaminated(self, mock_post):
        mock_post.return_value = _ok_response(
            "Democracy is a system where power is vested in the people.", eval_count=50
        )
        result = run_generate("model:7b", timeout=30, raw=True)
        self.assertFalse(result["contaminated"])


# ── post_evaluate ─────────────────────────────────────────────────────────────

class TestPostEvaluate(unittest.TestCase):

    def setUp(self):
        self.registry_patch = patch(
            "probe_endpoint_sweep.load_registry",
            return_value={
                "model-a:7b":  {"enabled": True, "notes": ""},
                "model-b:7b":  {"enabled": True, "notes": ""},
                "model-c:7b":  {"enabled": True, "notes": ""},
                "gemma4:26b":  {"enabled": True, "notes": ""},
                "qwen3.5:35b": {"enabled": True, "notes": ""},
            }
        )
        self.registry_patch.start()

    def tearDown(self):
        self.registry_patch.stop()

    def test_no_issues_returns_empty(self):
        stats = {"model-a:7b": _make_stats("model-a:7b", n=11)}
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        self.assertEqual(recs, [])

    def test_consistent_raw_true_failures_recommends_needs_raw_false(self):
        stats = {
            "gemma4:26b": _make_stats("gemma4:26b", n=11, gen_raw_fail=11, gen_plain_fail=0)
        }
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        notes_recs = [r for r in recs if r.field == "notes"]
        self.assertTrue(len(notes_recs) > 0)
        self.assertIn("needs_raw_false", notes_recs[0].suggested)
        self.assertEqual(notes_recs[0].confidence, "HIGH")

    def test_shallow_raw_true_recommends_needs_raw_false(self):
        stats = {
            "qwen3.5:35b": _make_stats(
                "qwen3.5:35b", n=11,
                gen_raw_tokens=40, gen_raw_time=0.7, gen_plain_tokens=1000,
            )
        }
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        self.assertTrue(any("needs_raw_false" in r.suggested for r in recs))

    def test_high_token_ratio_recommends_needs_raw_false(self):
        stats = {
            "model-b:7b": _make_stats(
                "model-b:7b", n=11,
                gen_raw_tokens=50, gen_plain_tokens=500, gen_raw_time=5.0,
            )
        }
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        self.assertTrue(any("needs_raw_false" in r.suggested for r in recs))

    def test_contamination_recommends_needs_raw_false(self):
        stats = {
            "model-c:7b": _make_stats(
                "model-c:7b", n=11, gen_raw_contaminated=8,
                gen_raw_tokens=200, gen_raw_time=5.0,
            )
        }
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        self.assertTrue(any("needs_raw_false" in r.suggested for r in recs))

    def test_intermittent_raw_failures_medium_confidence(self):
        stats = {
            "model-a:7b": _make_stats("model-a:7b", n=11, gen_raw_fail=5, gen_plain_fail=0)
        }
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        self.assertTrue(any(r.confidence == "MEDIUM" for r in recs))

    def test_high_chat_failure_rate_recommends_disable(self):
        stats = {"model-a:7b": _make_stats("model-a:7b", n=11, chat_fail=8)}
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        enabled_recs = [r for r in recs if r.field == "enabled"]
        self.assertTrue(len(enabled_recs) > 0)
        self.assertFalse(enabled_recs[0].suggested)

    def test_low_chat_repeatability_flagged(self):
        diverse_texts = [
            "jupiter saturn neptune orbit elliptical",
            "photosynthesis chlorophyll glucose sunlight absorption",
            "tectonic seismic volcanic magma crust",
            "renaissance baroque fugue counterpoint harpsichord",
            "cortisol serotonin dopamine limbic amygdala",
            "tariff embargo sanction quota protectionism",
            "transistor capacitor diode semiconductor resistor",
            "monsoon cyclone humid tropical precipitation",
            "feudalism serfdom manorial chivalry vassalage",
            "mitosis chromosome diploid telophase cytoplasm",
            "fractal mandelbrot recursion iteration complex",
        ]
        stats = {
            "model-a:7b": _make_stats("model-a:7b", n=11, chat_texts=diverse_texts)
        }
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        rep_recs = [r for r in recs if "low-repeatability" in r.suggested]
        self.assertTrue(len(rep_recs) > 0)

    def test_high_chat_repeatability_not_flagged(self):
        identical_texts = ["democracy is a system of government by the people"] * 11
        stats = {
            "model-a:7b": _make_stats("model-a:7b", n=11, chat_texts=identical_texts)
        }
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        rep_recs = [r for r in recs if "low-repeatability" in r.suggested]
        self.assertEqual(len(rep_recs), 0)

    def test_no_generate_skips_generate_analysis(self):
        stats = {
            "model-a:7b": _make_stats("model-a:7b", n=11, gen_raw_fail=11, gen_plain_fail=11)
        }
        recs = post_evaluate(stats, no_generate=True, n_sweeps=11)
        self.assertFalse(any("raw_false" in r.suggested for r in recs))

    def test_already_flagged_model_not_double_recommended(self):
        self.registry_patch.stop()
        with patch("probe_endpoint_sweep.load_registry", return_value={
            "model-a:7b": {"enabled": True, "notes": "needs_raw_false confirmed"},
        }):
            stats = {
                "model-a:7b": _make_stats("model-a:7b", n=11, gen_raw_fail=11, gen_plain_fail=0)
            }
            recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
            notes_recs = [r for r in recs if r.field == "notes"
                          and "needs_raw_false" in r.suggested]
            self.assertEqual(len(notes_recs), 0)
        self.registry_patch.start()

    def test_recommendation_serializable(self):
        stats = {"gemma4:26b": _make_stats("gemma4:26b", n=11, gen_raw_fail=11)}
        recs = post_evaluate(stats, no_generate=False, n_sweeps=11)
        for rec in recs:
            json.dumps(rec.to_dict())


# ── Recommendation ────────────────────────────────────────────────────────────

class TestRecommendation(unittest.TestCase):

    def test_str_contains_key_fields(self):
        rec = Recommendation(
            model="test:7b", field="notes",
            current="old note", suggested="new note",
            reason="raw:T failed 11/11", confidence="HIGH"
        )
        s = str(rec)
        self.assertIn("test:7b", s)
        self.assertIn("HIGH", s)
        self.assertIn("notes", s)

    def test_to_dict_has_required_keys(self):
        rec = Recommendation("m", "enabled", True, False, "test", "HIGH")
        d = rec.to_dict()
        for key in ("model", "field", "current", "suggested", "reason", "confidence"):
            self.assertIn(key, d)

    def test_to_dict_serializable(self):
        rec = Recommendation("m", "enabled", True, False, "test reason", "MEDIUM")
        json.dumps(rec.to_dict())


if __name__ == "__main__":
    unittest.main()
