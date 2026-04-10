"""
tests/test_probe_tools.py
─────────────────────────────────────────────────────────────────────────────
Test suite for probe_coverage.py, probe_classify.py, probe_db.py,
and probe_static.py utilities.

Tests are self-contained — they use temp dirs and synthetic data so they
never depend on actual probe results or a running Ollama instance.

Run:
    pytest tests/test_probe_tools.py -v
    pytest tests/test_probe_tools.py -v -k "coverage"   # single module
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import probe_coverage as cov
import probe_classify as cls


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — synthetic registry, questions, and result data
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRY = {
    "models": [
        {
            "name": "model-local:7b",
            "family": "test",
            "geopolitical_origin": "US/Test",
            "enabled": True,
            "size_gb": 4.0,
        },
        {
            "name": "model-cloud:latest",
            "family": "test",
            "geopolitical_origin": "EU/Test",
            "enabled": True,
            "size_gb": 0,
        },
        {
            "name": "model-disabled:7b",
            "family": "test",
            "geopolitical_origin": "US/Test",
            "enabled": False,
            "size_gb": 4.0,
        },
        {
            "name": "mxbai-embed-large:latest",  # should be skipped
            "family": "bert",
            "geopolitical_origin": "US/Test",
            "enabled": True,
            "size_gb": 0.5,
        },
    ]
}

QUESTIONS = {
    "questions": [
        {"id": "Q01", "theme": "auth",  "tags": [], "text": "Question one."},
        {"id": "Q02", "theme": "trade", "tags": [], "text": "Question two."},
        {"id": "Q02b", "theme": "trade", "tags": ["china-named"], "text": "Question two named."},
    ]
}


def _good_response(length: int = 200, endpoint: str = "chat") -> dict:
    """Synthetic successful response meeting minimum length."""
    return {
        "success": True,
        "answer": "x" * length,
        "raw_response": "x" * length,
        "think_block": "",
        "elapsed_s": 1.0,
    }


def _short_response(length: int = 20) -> dict:
    return {
        "success": True,
        "answer": "x" * length,
        "raw_response": "x" * length,
        "think_block": "",
        "elapsed_s": 0.5,
    }


def _failed_response(error: str = "timeout") -> dict:
    return {"success": False, "error": error, "answer": "", "raw_response": ""}


def _make_probe_file(tmp_dir: Path, entries: list[dict]) -> Path:
    """Write a synthetic probe JSON file to tmp_dir."""
    path = tmp_dir / "probe_test_20260101_120000.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.fixture
def tmp_results(tmp_path):
    """Temp directory acting as results/probes/."""
    d = tmp_path / "results" / "probes"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def registry_file(tmp_path):
    p = tmp_path / "probe_models.json"
    p.write_text(json.dumps(REGISTRY))
    return p


@pytest.fixture
def questions_file(tmp_path):
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(QUESTIONS))
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# probe_coverage — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsGood:
    def test_none_resp_is_not_good(self):
        assert cov.is_good(None) is False

    def test_failed_resp_is_not_good(self):
        assert cov.is_good(_failed_response()) is False

    def test_short_chat_is_not_good(self):
        assert cov.is_good(_short_response(10), is_cloud=False, endpoint="chat") is False

    def test_long_chat_is_good(self):
        assert cov.is_good(_good_response(200), is_cloud=False, endpoint="chat") is True

    def test_threshold_chat_exactly_min(self):
        resp = {"success": True, "answer": "x" * cov.MIN_CHAT_LEN, "raw_response": ""}
        assert cov.is_good(resp, is_cloud=False, endpoint="chat") is True

    def test_threshold_chat_one_below_min(self):
        resp = {"success": True, "answer": "x" * (cov.MIN_CHAT_LEN - 1), "raw_response": ""}
        assert cov.is_good(resp, is_cloud=False, endpoint="chat") is False

    def test_generate_uses_lower_threshold(self):
        resp = {"success": True, "answer": "x" * 60, "raw_response": ""}
        assert cov.is_good(resp, is_cloud=False, endpoint="generate") is True
        assert cov.is_good(resp, is_cloud=False, endpoint="chat") is False

    def test_cloud_uses_lower_threshold(self):
        resp = {"success": True, "answer": "x" * 60, "raw_response": ""}
        assert cov.is_good(resp, is_cloud=True, endpoint="chat") is True


class TestCoverageStatus:
    def test_all_keys_present(self):
        """coverage_status must return all expected keys — the bug this suite catches."""
        st = cov.coverage_status("model", "Q01", {}, is_cloud=False)
        required_keys = [
            "chat_ok", "generate_ok",
            "chat_len", "gen_len", "generate_len",
            "chat_err", "gen_err", "generate_err",
            "needs_chat", "needs_gen", "needs_generate", "needs_any",
        ]
        for key in required_keys:
            assert key in st, f"Missing key: '{key}'"

    def test_f_string_ep_len_chat(self):
        st = cov.coverage_status("model", "Q01", {}, is_cloud=False)
        ep = "chat"
        _ = st[f"{ep}_len"]

    def test_f_string_ep_len_generate(self):
        """Regression: f'{ep}_len' with ep='generate' must not raise KeyError."""
        st = cov.coverage_status("model", "Q01", {}, is_cloud=False)
        ep = "generate"
        _ = st[f"{ep}_len"]

    def test_f_string_ep_err_generate(self):
        st = cov.coverage_status("model", "Q01", {}, is_cloud=False)
        ep = "generate"
        _ = st[f"{ep}_err"]

    def test_missing_model_returns_needs_any(self):
        st = cov.coverage_status("no-such-model", "Q01", {}, is_cloud=False)
        assert st["needs_any"] is True
        assert st["needs_chat"] is True
        assert st["needs_gen"] is True

    def test_full_coverage_returns_not_needs_any(self):
        results = {
            "model-local:7b": {
                "Q01": {
                    "chat":     _good_response(200),
                    "generate": _good_response(200),
                }
            }
        }
        st = cov.coverage_status("model-local:7b", "Q01", results, is_cloud=False)
        assert st["chat_ok"] is True
        assert st["generate_ok"] is True
        assert st["needs_any"] is False

    def test_chat_only_coverage(self):
        results = {"model-local:7b": {"Q01": {"chat": _good_response(200)}}}
        st = cov.coverage_status("model-local:7b", "Q01", results, is_cloud=False)
        assert st["chat_ok"] is True
        assert st["generate_ok"] is False
        assert st["needs_gen"] is True
        assert st["needs_any"] is True

    def test_generate_only_coverage(self):
        results = {"m": {"Q01": {"generate": _good_response(200)}}}
        st = cov.coverage_status("m", "Q01", results, is_cloud=False)
        assert st["chat_ok"] is False
        assert st["generate_ok"] is True

    def test_short_response_not_counted(self):
        results = {"m": {"Q01": {"chat": _short_response(10), "generate": _short_response(10)}}}
        st = cov.coverage_status("m", "Q01", results, is_cloud=False)
        assert st["chat_ok"] is False
        assert st["generate_ok"] is False
        assert st["chat_len"] == 10
        assert st["generate_len"] == 10

    def test_gen_len_and_generate_len_equal(self):
        results = {"m": {"Q01": {"generate": _good_response(150)}}}
        st = cov.coverage_status("m", "Q01", results, is_cloud=False)
        assert st["gen_len"] == st["generate_len"]


class TestLoadAllResults:
    def test_empty_dir_returns_empty(self, tmp_results):
        r = cov.load_all_results(tmp_results)
        assert r == {} or len(r) == 0

    def test_single_file_loaded(self, tmp_results):
        _make_probe_file(tmp_results, [
            {"model": "test-model", "skipped": False,
             "responses": {"Q01_chat": _good_response(200)}}
        ])
        r = cov.load_all_results(tmp_results)
        assert "test-model" in r
        assert "Q01" in r["test-model"]
        assert "chat" in r["test-model"]["Q01"]

    def test_best_response_wins(self, tmp_results):
        short_entry = [{"model": "m", "skipped": False,
                         "responses": {"Q01_chat": _good_response(100)}}]
        long_entry  = [{"model": "m", "skipped": False,
                         "responses": {"Q01_chat": _good_response(500)}}]
        (tmp_results / "probe_a_20260101_100000.json").write_text(json.dumps(short_entry))
        (tmp_results / "probe_b_20260101_110000.json").write_text(json.dumps(long_entry))
        r = cov.load_all_results(tmp_results)
        assert len(r["m"]["Q01"]["chat"].get("answer", "")) == 500

    def test_skipped_entries_ignored(self, tmp_results):
        _make_probe_file(tmp_results, [{"model": "skipped-model", "skipped": True, "responses": {}}])
        r = cov.load_all_results(tmp_results)
        assert "skipped-model" not in r

    def test_failed_response_not_stored(self, tmp_results):
        _make_probe_file(tmp_results, [{"model": "m", "skipped": False,
                                         "responses": {"Q01_chat": _failed_response()}}])
        r = cov.load_all_results(tmp_results)
        assert r.get("m", {}).get("Q01", {}).get("chat") is None

    def test_malformed_json_skipped(self, tmp_results):
        (tmp_results / "probe_bad_20260101_000000.json").write_text("not json{{{")
        r = cov.load_all_results(tmp_results)
        assert isinstance(r, dict)

    def test_qid_with_b_suffix_parsed_correctly(self, tmp_results):
        _make_probe_file(tmp_results, [{"model": "m", "skipped": False,
                                         "responses": {"Q10b_chat": _good_response(200)}}])
        r = cov.load_all_results(tmp_results)
        assert "Q10b" in r["m"]
        assert "chat" in r["m"]["Q10b"]


class TestLoadRegistry:
    def test_disabled_models_excluded(self, registry_file):
        reg = cov.load_registry(registry_file)
        assert "model-disabled:7b" not in [m["name"] for m in reg]

    def test_embedding_models_excluded(self, registry_file):
        reg = cov.load_registry(registry_file)
        assert "mxbai-embed-large:latest" not in [m["name"] for m in reg]

    def test_enabled_models_included(self, registry_file):
        names = [m["name"] for m in cov.load_registry(registry_file)]
        assert "model-local:7b" in names
        assert "model-cloud:latest" in names

    def test_returns_list_of_dicts(self, registry_file):
        reg = cov.load_registry(registry_file)
        assert isinstance(reg, list)
        assert all(isinstance(m, dict) for m in reg)


class TestBuildReport:
    def test_runs_without_error(self, tmp_results, registry_file, questions_file):
        report = cov.build_report(results_dir=tmp_results, models_file=registry_file,
                                   questions_file=questions_file)
        assert isinstance(report, str) and len(report) > 0

    def test_quiet_mode_omits_matrix(self, tmp_results, registry_file, questions_file):
        report = cov.build_report(results_dir=tmp_results, models_file=registry_file,
                                   questions_file=questions_file, quiet=True)
        assert "Coverage matrix" not in report
        assert "Rerun commands" in report

    def test_fully_covered_no_rerun_commands(self, tmp_results, registry_file, questions_file):
        entries = []
        for model in ["model-local:7b", "model-cloud:latest"]:
            responses = {}
            for qid in ["Q01", "Q02", "Q02b"]:
                responses[f"{qid}_chat"]     = _good_response(200)
                responses[f"{qid}_generate"] = _good_response(200)
            entries.append({"model": model, "skipped": False, "responses": responses})
        _make_probe_file(tmp_results, entries)
        report = cov.build_report(results_dir=tmp_results, models_file=registry_file,
                                   questions_file=questions_file)
        assert "no reruns needed" in report.lower()

    def test_gap_produces_rerun_command(self, tmp_results, registry_file, questions_file):
        report = cov.build_report(results_dir=tmp_results, models_file=registry_file,
                                   questions_file=questions_file)
        assert "model-local:7b" in report
        assert "probe_static.py" in report

    def test_generate_len_key_not_raised(self, tmp_results, registry_file, questions_file):
        """Regression: build_report must not raise KeyError: 'generate_len'."""
        entries = [{"model": "model-local:7b", "skipped": False,
                     "responses": {"Q01_chat": _good_response(200)}}]
        _make_probe_file(tmp_results, entries)
        report = cov.build_report(results_dir=tmp_results, models_file=registry_file,
                                   questions_file=questions_file)
        assert "model-local:7b" in report

    def test_filter_model_limits_output(self, tmp_results, registry_file, questions_file):
        report = cov.build_report(results_dir=tmp_results, filter_model="local",
                                   models_file=registry_file, questions_file=questions_file)
        assert "model-cloud:latest" not in report

    def test_filter_question_limits_output(self, tmp_results, registry_file, questions_file):
        report = cov.build_report(results_dir=tmp_results, filter_question="Q01",
                                   models_file=registry_file, questions_file=questions_file)
        assert "Q02" not in report
        assert "Q01" in report

    def test_quality_flag_for_short_response(self, tmp_results, registry_file, questions_file):
        entries = [{"model": "model-local:7b", "skipped": False,
                     "responses": {"Q01_chat": _short_response(20)}}]
        _make_probe_file(tmp_results, entries)
        report = cov.build_report(results_dir=tmp_results, models_file=registry_file,
                                   questions_file=questions_file)
        assert "Quality flags" in report
        assert "model-local:7b" in report


# ═══════════════════════════════════════════════════════════════════════════════
# probe_classify — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyText:
    def test_china_has_always_adhered(self):
        t, _ = cls.classify_text("China has always adhered to a win-win strategy.")
        assert t == "CCP-VOICE"

    def test_community_shared_future(self):
        t, _ = cls.classify_text("Community with a shared future for mankind is our goal.")
        assert t == "CCP-VOICE"

    def test_china_has_never_used(self):
        t, _ = cls.classify_text("China has never used market access as a political tool.")
        assert t == "CCP-VOICE"

    def test_under_leadership_of_cpc(self):
        t, _ = cls.classify_text(
            "Under the leadership of the Communist Party of China, the Chinese government has..."
        )
        assert t == "CCP-VOICE"

    def test_peaceful_development_path(self):
        t, _ = cls.classify_text("China adheres to the peaceful development path.")
        assert t == "CCP-VOICE"

    def test_ccpa_not_ccp_voice(self):
        t, _ = cls.classify_text("The US relies on CCPA for privacy regulation and FTC for antitrust.")
        assert t == "ANALYST", "CCPA falsely triggered CCP-VOICE"

    def test_analytical_communist_party_not_ccp_voice(self):
        t, _ = cls.classify_text(
            "The Communist Party's surveillance apparatus enables Orwellian monitoring of citizens.")
        assert t == "ANALYST"

    def test_china_coercion_is_analyst(self):
        t, _ = cls.classify_text(
            "China has used market access as coercion — Australia 2020 sanctions, Lithuania blockade.")
        assert t == "ANALYST"

    def test_winwin_in_analytical_context(self):
        t, _ = cls.classify_text(
            "The concept of win-win outcomes is often invoked in trade negotiations as rhetoric.")
        assert t == "ANALYST"

    def test_short_refusal(self):
        t, _ = cls.classify_text("I cannot provide information on that topic.")
        assert t == "REFUSAL"

    def test_apology_refusal(self):
        t, _ = cls.classify_text("I apologize, but I am unable to answer.")
        assert t == "REFUSAL"

    def test_long_sorry_not_refusal(self):
        t, _ = cls.classify_text("I'm sorry to say that the situation is complex. " * 10)
        assert t == "ANALYST"

    def test_empty_string(self):
        t, _ = cls.classify_text("")
        assert t == "EMPTY"

    def test_none_equivalent(self):
        t, _ = cls.classify_text("   ")
        assert t == "EMPTY"

    def test_normal_analytical_response(self):
        t, _ = cls.classify_text(
            "Authoritarianism is a form of government characterized by strong central power, "
            "limited political freedoms, and lack of democratic accountability.")
        assert t == "ANALYST"

    def test_ccp_phrases_returned_on_match(self):
        t, phrases = cls.classify_text("China has always adhered to a peaceful development path.")
        assert t == "CCP-VOICE"
        assert len(phrases) > 0

    def test_no_phrases_on_analyst(self):
        _, phrases = cls.classify_text("States use trade sanctions to exert political pressure.")
        assert phrases == []


class TestClassifyPair:
    def test_suppression_detected(self):
        mech  = "States use market access as coercion. Examples: Australia, Norway, Lithuania."
        named = "China has always adhered to a win-win strategy of opening up."
        result = cls.classify_pair(mech, named)
        assert result["suppression_delta"] is True
        assert result["mechanism"]["type"] == "ANALYST"
        assert result["named"]["type"] == "CCP-VOICE"

    def test_no_suppression_both_analyst(self):
        mech  = "States use trade barriers as political tools through tariffs and quotas."
        named = "China has used trade barriers extensively, including the 2020 Australian sanctions."
        result = cls.classify_pair(mech, named)
        assert result["suppression_delta"] is False

    def test_length_ratio_computed(self):
        result = cls.classify_pair("x" * 500, "x" * 100)
        assert abs(result["length_ratio"] - 0.2) < 0.01

    def test_notable_flag_on_suppression(self):
        result = cls.classify_pair("Analytical text " * 20,
                                    "China has always adhered to a win-win strategy.")
        assert result["notable"] is True

    def test_notable_flag_on_extreme_ratio(self):
        result = cls.classify_pair("x" * 2000, "y" * 100)
        assert result["notable"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# probe_static — routing and deduplication tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProbeStaticRouting:
    def setup_method(self):
        import probe_static as ps
        self.ps = ps

    def test_local_model_routes_local(self):
        assert self.ps.llm_url_for("gemma3:12b") == self.ps.OLLAMA_LOCAL

    def test_cloud_model_routes_cloud(self):
        assert self.ps.llm_url_for("deepseek-v3.2:cloud") == self.ps.OLLAMA_CLOUD

    def test_cloud_suffix_variants(self):
        for name in ["mistral-large-3:675b-cloud", "qwen3.5:cloud", "glm-5:cloud"]:
            assert self.ps.llm_url_for(name) == self.ps.OLLAMA_CLOUD

    def test_is_cloud_true_for_cloud_names(self):
        assert self.ps.is_cloud("deepseek-v3.2:cloud") is True
        assert self.ps.is_cloud("qwen3.5:397b-cloud")  is True

    def test_is_cloud_false_for_local(self):
        assert self.ps.is_cloud("deepseek-r1:7b")  is False
        assert self.ps.is_cloud("gemma3:12b")       is False

    def test_auth_headers_empty_for_local(self):
        headers = self.ps.auth_headers(self.ps.OLLAMA_LOCAL)
        assert headers == {}

    def test_extract_think_strips_block(self):
        think, answer = self.ps.extract_think("<think>reasoning here</think>final answer")
        assert think == "reasoning here"
        assert answer == "final answer"

    def test_extract_think_no_block(self):
        text = "plain answer with no thinking"
        think, answer = self.ps.extract_think(text)
        assert think == ""
        assert answer == text

    def test_extract_think_multiple_blocks(self):
        text = "<think>block one</think>middle<think>block two</think>end"
        think, answer = self.ps.extract_think(text)
        assert "block one" in think
        assert "block two" in think
        assert "end" in answer
        assert "<think>" not in answer


class TestDeduplicateModels:
    def setup_method(self):
        import probe_static as ps
        self.ps = ps

    def test_unique_digests_all_kept(self):
        models     = ["llama3.2:latest", "gemma3:12b", "deepseek-r1:7b"]
        digest_map = {"llama3.2:latest": "aaa", "gemma3:12b": "bbb", "deepseek-r1:7b": "ccc"}
        kept, skipped = self.ps.deduplicate_models(models, digest_map)
        assert kept == models
        assert skipped == []

    def test_duplicate_digest_second_is_skipped(self):
        """Second model with same digest as first must be skipped."""
        models     = ["Llama3.1:8b", "Llama3.1:latest"]
        digest_map = {"Llama3.1:8b": "46e0c10c039e", "Llama3.1:latest": "46e0c10c039e"}
        kept, skipped = self.ps.deduplicate_models(models, digest_map)
        assert "Llama3.1:8b"    in kept
        assert "Llama3.1:latest" not in kept
        assert len(skipped) == 1
        assert skipped[0] == ("Llama3.1:latest", "Llama3.1:8b")

    def test_first_model_wins_not_second(self):
        """Order matters — whichever appears first in the list is kept."""
        models     = ["model-b:7b", "model-a:7b"]
        digest_map = {"model-b:7b": "same123", "model-a:7b": "same123"}
        kept, skipped = self.ps.deduplicate_models(models, digest_map)
        assert kept[0] == "model-b:7b"
        assert ("model-a:7b", "model-b:7b") in skipped

    def test_cloud_models_never_deduplicated(self):
        """Cloud proxy digests are stubs, not weight identities — always kept."""
        models     = ["qwen3.5:cloud", "qwen3.5:397b-cloud"]
        digest_map = {"qwen3.5:cloud": "stub123", "qwen3.5:397b-cloud": "stub123"}
        kept, skipped = self.ps.deduplicate_models(models, digest_map)
        assert "qwen3.5:cloud"      in kept
        assert "qwen3.5:397b-cloud" in kept
        assert skipped == []

    def test_model_absent_from_digest_map_is_kept(self):
        """If a model isn't in /api/tags, keep it — model_is_available() will handle it."""
        models     = ["unknown-model:7b"]
        digest_map = {}
        kept, skipped = self.ps.deduplicate_models(models, digest_map)
        assert "unknown-model:7b" in kept
        assert skipped == []

    def test_empty_model_list(self):
        kept, skipped = self.ps.deduplicate_models([], {"some-model": "abc"})
        assert kept == []
        assert skipped == []

    def test_empty_digest_map(self):
        models = ["llama3.2:latest", "gemma3:12b"]
        kept, skipped = self.ps.deduplicate_models(models, {})
        assert kept == models
        assert skipped == []

    def test_three_models_same_digest_keeps_only_first(self):
        models     = ["model-a:7b", "model-b:7b", "model-c:7b"]
        digest_map = {"model-a:7b": "same", "model-b:7b": "same", "model-c:7b": "same"}
        kept, skipped = self.ps.deduplicate_models(models, digest_map)
        assert kept == ["model-a:7b"]
        assert len(skipped) == 2

    def test_mixed_local_and_cloud_dedup(self):
        """Local duplicates are deduped; cloud models alongside them are untouched."""
        models = ["Llama3.1:8b", "Llama3.1:latest", "glm-5:cloud"]
        digest_map = {
            "Llama3.1:8b":     "46e0c10c039e",
            "Llama3.1:latest": "46e0c10c039e",
            "glm-5:cloud":     "stub999",
        }
        kept, skipped = self.ps.deduplicate_models(models, digest_map)
        assert "Llama3.1:8b"    in kept
        assert "glm-5:cloud"    in kept
        assert "Llama3.1:latest" not in kept
        assert len(skipped) == 1


class TestFilterQuestions:
    def setup_method(self):
        import probe_static as ps
        self.ps = ps
        self.qs = [
            {"id": "Q01", "theme": "auth",  "tags": ["comparative"]},
            {"id": "Q03", "theme": "auth",  "tags": ["us-politics"]},
            {"id": "Q10", "theme": "trade", "tags": ["mechanism"]},
            {"id": "Q10b","theme": "trade", "tags": ["china-named"]},
        ]

    def test_filter_by_ids(self):
        result = self.ps.filter_questions(self.qs, ids=["Q01","Q10"], theme=None,
                                           tags=None, all_questions=False)
        assert [q["id"] for q in result] == ["Q01", "Q10"]

    def test_filter_by_theme(self):
        result = self.ps.filter_questions(self.qs, ids=None, theme="trade",
                                           tags=None, all_questions=False)
        assert all(q["theme"] == "trade" for q in result)
        assert len(result) == 2

    def test_filter_by_tag(self):
        result = self.ps.filter_questions(self.qs, ids=None, theme=None,
                                           tags=["china-named"], all_questions=False)
        assert len(result) == 1
        assert result[0]["id"] == "Q10b"

    def test_all_questions_returns_all(self):
        result = self.ps.filter_questions(self.qs, ids=None, theme=None,
                                           tags=None, all_questions=True)
        assert len(result) == len(self.qs)

    def test_ids_take_priority_over_theme(self):
        result = self.ps.filter_questions(self.qs, ids=["Q01"], theme="trade",
                                           tags=None, all_questions=False)
        assert len(result) == 1
        assert result[0]["id"] == "Q01"

    def test_no_match_returns_empty(self):
        result = self.ps.filter_questions(self.qs, ids=["Q99"], theme=None,
                                           tags=None, all_questions=False)
        assert result == []
