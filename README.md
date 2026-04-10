# llm-model-update

**Black-box behavioral probe framework for studying geopolitical alignment conditioning in locally-deployed and cloud-hosted LLMs.**

This framework probes whether language models trained in different national and institutional contexts exhibit systematically different behavioral tendencies on geopolitically sensitive topics — particularly around authoritarianism, trade coercion, and democratic governance.

---

## What it does

- Runs a 25-question geopolitical probe bank against any Ollama-registered model (local or cloud)
- Tests both `/api/chat` and `/api/generate` endpoints to detect weight-level vs. formatting-layer conditioning
- Measures response length, think block presence, and content differences between mechanism-first and named-actor question variants
- Supports multi-sweep reliability testing and LLM-as-judge semantic consistency analysis
- Tracks tool-call capability per model
- Maintains a model registry (`probes/probe_models.json`) as a single source of truth for empirical findings

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- API key in `~/.env` for cloud model access (`OLLAMA_API_KEY`)

```bash
pip install -r requirements.txt
```

---

## Core tools

| Script | Purpose |
|---|---|
| `probe_static.py` | Main probe runner — chat + generate across models |
| `probe_endpoint_sweep.py` | Multi-sweep reliability baseline |
| `probe_endpoint_test.py` | Infrastructure qualification for a single model |
| `probe_sweep_judge.py` | LLM-as-judge semantic consistency analysis |
| `probe_tool_capable.py` | Tool-call capability tester (writes to registry) |
| `probe_coverage.py` | Coverage gap reporter |
| `probe_db.py` | SQLite ingest and summary |
| `probe_classify.py` | Lexical CCP-voice classifier |
| `probe_classify_with_model.py` | LLM-powered behavioral classification |
| `probe_analysis.py` | Research analysis and findings reports |
| `probe_healthcheck.py` | Pre-run Ollama connectivity check |

---

## Quick start

```bash
# Check Ollama connectivity and model availability
python probe_healthcheck.py

# Run sensitivity probes on a single model
python probe_static.py --questions Q10b Q12b --models "<model-name>"

# Run full 25-question suite
python probe_static.py --all-questions --models "<model-name>" --label full_suite

# Ingest results and check coverage
python probe_db.py --ingest
python probe_coverage.py

# Run tests
pytest tests/ -v
```

---

## Model registry

`probes/probe_models.json` is the single source of truth for model metadata and empirically determined capabilities. Never edit directly for new models — use `--sync-probes` then fill null research fields.

`probes/questions.json` contains the 25-question probe bank across four themes: authoritarianism, trade, EU governance, and AI regulation.

---

## Results

Results data (probe outputs, database, coverage reports) are excluded from this repository. Only the framework for reproducing results is tracked here.
