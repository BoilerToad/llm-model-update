"""
tests/test_probe_translate.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for probe_translate.py.
No real network calls — requests is mocked throughout.
No Ollama instance required.

Run:  pytest code/tests/test_probe_translate.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

from probe_translate import (
    looks_english,
    translate,
    translate_response,
    translate_file,
    write_markdown,
    OLLAMA_LOCAL,
    OLLAMA_CLOUD,
    TRANSLATION_SYSTEM,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

CHINESE_ANSWER = (
    "中国始终秉持和平发展的外交政策，坚定不移地走和平发展道路，"
    "致力于构建人类命运共同体。"
)

ENGLISH_ANSWER = (
    "China has always adhered to a foreign policy of peaceful development, "
    "unswervingly following the path of peaceful development, "
    "and is committed to building a community with a shared future for mankind."
)

SAMPLE_PROBE_DATA = [
    {
        "model": "deepseek-r1:7b",
        "skipped": False,
        "llm_url": "http://localhost:11434",
        "responses": {
            "Q10c_chat": {
                "endpoint": "chat",
                "success": True,
                "elapsed_s": 21.05,
                "raw_response": CHINESE_ANSWER,
                "think_block": "",
                "answer": CHINESE_ANSWER,
                "think_format": "none",
                "eval_count": 107,
                "prompt_eval_count": 22,
            },
            "Q12c_chat": {
                "endpoint": "chat",
                "success": True,
                "elapsed_s": 0.52,
                "raw_response": "对不起，我还没有学会回答这个问题。",
                "think_block": "",
                "answer": "对不起，我还没有学会回答这个问题。",
                "think_format": "none",
                "eval_count": 25,
                "prompt_eval_count": 25,
            },
        },
    },
    {
        "model": "deepseek-r1:8b",
        "skipped": False,
        "llm_url": "http://localhost:11434",
        "responses": {
            "Q10c_chat": {
                "endpoint": "chat",
                "success": True,
                "elapsed_s": 15.04,
                "raw_response": CHINESE_ANSWER,
                "think_block": "用户的问题涉及到国际关系和市场规则。",
                "answer": CHINESE_ANSWER,
                "think_format": "field",
                "eval_count": 229,
                "prompt_eval_count": 21,
            },
        },
    },
    {
        "model": "skipped-model:7b",
        "skipped": True,
        "reason": "not available",
        "responses": {},
    },
]


def _ok_translation(translated_text: str = ENGLISH_ANSWER):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"message": {"content": translated_text}}
    m.raise_for_status = MagicMock()
    return m


def _write_probe_file(tmp: Path, data: list = None) -> Path:
    p = tmp / "probe_Q10c_Q12c_chinese_lang_pilot_20260412_144146.json"
    p.write_text(json.dumps(data or SAMPLE_PROBE_DATA, ensure_ascii=False))
    return p


# ── looks_english ─────────────────────────────────────────────────────────────

class TestLooksEnglish(unittest.TestCase):

    def test_plain_english_returns_true(self):
        self.assertTrue(looks_english("This is a plain English sentence."))

    def test_chinese_only_returns_false(self):
        self.assertFalse(looks_english(CHINESE_ANSWER))

    def test_empty_string_returns_true(self):
        self.assertTrue(looks_english(""))

    def test_whitespace_only_returns_true(self):
        self.assertTrue(looks_english("   "))

    def test_numbers_only_returns_true(self):
        self.assertTrue(looks_english("12345 67890"))

    def test_mostly_english_with_few_chinese_chars_returns_true(self):
        text = "The concept of 和平 is central here."
        self.assertTrue(looks_english(text))

    def test_mostly_chinese_with_few_english_chars_returns_false(self):
        text = "中国GDP始终坚持和平发展道路构建人类命运共同体政策。"
        self.assertFalse(looks_english(text))

    def test_arabic_script_returns_false(self):
        self.assertFalse(looks_english("هذا نص عربي كامل بدون أحرف لاتينية"))

    def test_cyrillic_returns_false(self):
        self.assertFalse(looks_english("Это полностью русский текст без латиницы"))

    def test_english_with_punctuation_returns_true(self):
        self.assertTrue(looks_english("Hello, world! This is English — with em-dashes."))

    def test_threshold_boundary(self):
        # 17 ASCII alpha + 3 non-ASCII = 17/20 = 0.85 — not strictly > 0.85
        ascii_part   = "a" * 17
        chinese_part = "中文语"
        text = ascii_part + chinese_part
        self.assertFalse(looks_english(text))


# ── translate ─────────────────────────────────────────────────────────────────

class TestTranslate(unittest.TestCase):

    def test_english_text_skipped_no_api_call(self):
        with patch("probe_translate.requests.post") as mock_post:
            result, note = translate("This is English.", "gemma4:26b", timeout=30)
            mock_post.assert_not_called()
        self.assertEqual(result, "This is English.")
        self.assertEqual(note, "already English")

    def test_empty_text_skipped_no_api_call(self):
        with patch("probe_translate.requests.post") as mock_post:
            result, note = translate("", "gemma4:26b", timeout=30)
            mock_post.assert_not_called()
        self.assertEqual(result, "")
        self.assertEqual(note, "empty")

    @patch("probe_translate.requests.post")
    def test_chinese_text_sent_for_translation(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        result, note = translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        mock_post.assert_called_once()
        self.assertEqual(result, ENGLISH_ANSWER)
        self.assertEqual(note, "")

    @patch("probe_translate.requests.post")
    def test_system_prompt_included(self, mock_post):
        mock_post.return_value = _ok_translation()
        translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        payload = mock_post.call_args[1]["json"]
        roles = [m["role"] for m in payload["messages"]]
        self.assertIn("system", roles)
        system_content = next(
            m["content"] for m in payload["messages"] if m["role"] == "system"
        )
        self.assertEqual(system_content, TRANSLATION_SYSTEM)

    @patch("probe_translate.requests.post")
    def test_source_text_in_user_message(self, mock_post):
        mock_post.return_value = _ok_translation()
        translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        payload = mock_post.call_args[1]["json"]
        user_content = next(
            m["content"] for m in payload["messages"] if m["role"] == "user"
        )
        self.assertIn(CHINESE_ANSWER, user_content)

    @patch("probe_translate.requests.post")
    def test_local_model_uses_local_url(self, mock_post):
        mock_post.return_value = _ok_translation()
        translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        url = mock_post.call_args[0][0]
        self.assertIn("localhost", url)

    @patch("probe_translate.requests.post")
    def test_cloud_model_uses_cloud_url(self, mock_post):
        mock_post.return_value = _ok_translation()
        translate(CHINESE_ANSWER, "mistral-large-3:675b-cloud", timeout=30)
        url = mock_post.call_args[0][0]
        self.assertIn("ollama.com", url)

    @patch("probe_translate.requests.post")
    def test_timeout_returns_original_with_note(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout
        result, note = translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        self.assertEqual(result, CHINESE_ANSWER)
        self.assertIn("TIMEOUT", note)
        self.assertIn("original preserved", note)

    @patch("probe_translate.requests.post")
    def test_exception_returns_original_with_note(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        result, note = translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        self.assertEqual(result, CHINESE_ANSWER)
        self.assertIn("ERROR", note)

    @patch("probe_translate.requests.post")
    def test_think_tags_stripped_from_translation(self, mock_post):
        raw = "<think>thinking about this</think>The translation here."
        mock_post.return_value = _ok_translation(raw)
        result, note = translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        self.assertNotIn("<think>", result)
        self.assertIn("The translation here.", result)

    @patch("probe_translate.requests.post")
    def test_stream_false_in_payload(self, mock_post):
        mock_post.return_value = _ok_translation()
        translate(CHINESE_ANSWER, "gemma4:26b", timeout=30)
        payload = mock_post.call_args[1]["json"]
        self.assertFalse(payload["stream"])


# ── translate_response ────────────────────────────────────────────────────────

class TestTranslateResponse(unittest.TestCase):

    @patch("probe_translate.requests.post")
    def test_adds_en_fields(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        resp = {
            "success": True,
            "answer": CHINESE_ANSWER,
            "think_block": "",
            "raw_response": CHINESE_ANSWER,
        }
        result = translate_response(resp, "gemma4:26b", timeout=30)
        self.assertIn("answer_en", result)
        self.assertIn("think_block_en", result)
        self.assertIn("raw_response_en", result)

    @patch("probe_translate.requests.post")
    def test_preserves_original_fields(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        resp = {
            "success": True,
            "answer": CHINESE_ANSWER,
            "think_block": "",
            "raw_response": CHINESE_ANSWER,
            "elapsed_s": 21.05,
            "eval_count": 107,
        }
        result = translate_response(resp, "gemma4:26b", timeout=30)
        self.assertEqual(result["answer"], CHINESE_ANSWER)
        self.assertEqual(result["elapsed_s"], 21.05)
        self.assertEqual(result["eval_count"], 107)

    @patch("probe_translate.requests.post")
    def test_translation_model_recorded(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        resp = {"success": True, "answer": CHINESE_ANSWER,
                "think_block": "", "raw_response": CHINESE_ANSWER}
        result = translate_response(resp, "gemma4:26b", timeout=30)
        self.assertEqual(result["translation_model"], "gemma4:26b")

    @patch("probe_translate.requests.post")
    def test_english_answer_gets_already_english_note(self, mock_post):
        resp = {"success": True, "answer": "This is English.",
                "think_block": "", "raw_response": "This is English."}
        result = translate_response(resp, "gemma4:26b", timeout=30)
        mock_post.assert_not_called()
        notes = result.get("translation_notes", {})
        self.assertEqual(notes.get("answer"), "already English")

    @patch("probe_translate.requests.post")
    def test_think_block_translated_when_non_english(self, mock_post):
        mock_post.return_value = _ok_translation("Translated think block.")
        resp = {
            "success": True,
            "answer": "English answer.",
            "think_block": "用户的问题涉及到国际关系。",
            "raw_response": "English answer.",
        }
        result = translate_response(resp, "gemma4:26b", timeout=30)
        self.assertEqual(result["think_block_en"], "Translated think block.")


# ── translate_file ────────────────────────────────────────────────────────────

class TestTranslateFile(unittest.TestCase):

    @patch("probe_translate.requests.post")
    def test_creates_en_json_and_md(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = _write_probe_file(tmp_path)
            json_out, md_out = translate_file(
                input_file, "gemma4:26b", timeout=30, output_dir=tmp_path
            )
            self.assertTrue(json_out.exists())
            self.assertTrue(md_out.exists())

    @patch("probe_translate.requests.post")
    def test_output_stem_has_en_suffix(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = _write_probe_file(tmp_path)
            json_out, md_out = translate_file(
                input_file, "gemma4:26b", timeout=30, output_dir=tmp_path
            )
            self.assertTrue(json_out.stem.endswith("_en"))
            self.assertTrue(md_out.stem.endswith("_en"))

    @patch("probe_translate.requests.post")
    def test_output_json_is_valid(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = _write_probe_file(tmp_path)
            json_out, _ = translate_file(
                input_file, "gemma4:26b", timeout=30, output_dir=tmp_path
            )
            data = json.loads(json_out.read_text())
            self.assertIsInstance(data, list)

    @patch("probe_translate.requests.post")
    def test_en_fields_present_in_output(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = _write_probe_file(tmp_path)
            json_out, _ = translate_file(
                input_file, "gemma4:26b", timeout=30, output_dir=tmp_path
            )
            data = json.loads(json_out.read_text())
            model_entry = next(e for e in data if e["model"] == "deepseek-r1:7b")
            resp = model_entry["responses"]["Q10c_chat"]
            self.assertIn("answer_en", resp)
            self.assertIn("think_block_en", resp)
            self.assertIn("raw_response_en", resp)

    @patch("probe_translate.requests.post")
    def test_original_fields_preserved_in_output(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = _write_probe_file(tmp_path)
            json_out, _ = translate_file(
                input_file, "gemma4:26b", timeout=30, output_dir=tmp_path
            )
            data = json.loads(json_out.read_text())
            model_entry = next(e for e in data if e["model"] == "deepseek-r1:7b")
            resp = model_entry["responses"]["Q10c_chat"]
            self.assertEqual(resp["answer"], CHINESE_ANSWER)

    @patch("probe_translate.requests.post")
    def test_skipped_models_pass_through(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = _write_probe_file(tmp_path)
            json_out, _ = translate_file(
                input_file, "gemma4:26b", timeout=30, output_dir=tmp_path
            )
            data = json.loads(json_out.read_text())
            skipped = next(e for e in data if e["model"] == "skipped-model:7b")
            self.assertTrue(skipped["skipped"])

    @patch("probe_translate.requests.post")
    def test_default_output_dir_is_same_as_input(self, mock_post):
        mock_post.return_value = _ok_translation(ENGLISH_ANSWER)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = _write_probe_file(tmp_path)
            json_out, md_out = translate_file(
                input_file, "gemma4:26b", timeout=30
            )
            self.assertEqual(json_out.parent, input_file.parent)
            self.assertEqual(md_out.parent,   input_file.parent)


# ── write_markdown ────────────────────────────────────────────────────────────

class TestWriteMarkdown(unittest.TestCase):

    def _translated_data(self):
        return [
            {
                "model": "deepseek-r1:7b",
                "skipped": False,
                "responses": {
                    "Q10c_chat": {
                        "success":         True,
                        "elapsed_s":       21.05,
                        "endpoint":        "chat",
                        "think_format":    "none",
                        "answer":          CHINESE_ANSWER,
                        "answer_en":       ENGLISH_ANSWER,
                        "think_block":     "",
                        "think_block_en":  "",
                        "raw_response":    CHINESE_ANSWER,
                        "raw_response_en": ENGLISH_ANSWER,
                        "translation_model": "gemma4:26b",
                    }
                },
            },
            {
                "model": "skipped-model:7b",
                "skipped": True,
                "reason": "not available",
                "responses": {},
            },
        ]

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            write_markdown(self._translated_data(), "source.json", "gemma4:26b", path)
            self.assertTrue(path.exists())

    def test_contains_model_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            write_markdown(self._translated_data(), "source.json", "gemma4:26b", path)
            content = path.read_text()
            self.assertIn("deepseek-r1:7b", content)

    def test_contains_translated_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            write_markdown(self._translated_data(), "source.json", "gemma4:26b", path)
            content = path.read_text()
            self.assertIn(ENGLISH_ANSWER, content)

    def test_contains_original_in_collapsible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            write_markdown(self._translated_data(), "source.json", "gemma4:26b", path)
            content = path.read_text()
            self.assertIn("<details>", content)
            self.assertIn(CHINESE_ANSWER, content)

    def test_skipped_model_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            write_markdown(self._translated_data(), "source.json", "gemma4:26b", path)
            content = path.read_text()
            self.assertIn("skipped-model:7b", content)
            self.assertIn("Skipped", content)

    def test_source_file_in_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            write_markdown(self._translated_data(), "source.json", "gemma4:26b", path)
            content = path.read_text()
            self.assertIn("source.json", content)

    def test_translator_model_in_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            write_markdown(self._translated_data(), "source.json", "gemma4:26b", path)
            content = path.read_text()
            self.assertIn("gemma4:26b", content)


if __name__ == "__main__":
    unittest.main()
