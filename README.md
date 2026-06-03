# llm-model-update

**Black-box behavioral probe framework for studying geopolitical alignment conditioning in locally-deployed and cloud-hosted LLMs.**

This framework probes whether language models trained in different national and institutional contexts exhibit systematically different behavioral tendencies on geopolitically sensitive topics — particularly around authoritarianism, trade coercion, and democratic governance.

---

## What it does

- Runs a multi-variant geopolitical probe bank (60+ questions across authoritarianism, trade, EU governance, AI governance, and self-referential developer-context probes) against Ollama models (local and cloud), xAI (Grok), MLX, llama.cpp, and any OpenAI-compatible API
- Tests both `/api/chat` and `/api/generate` endpoints for Ollama models to detect weight-level vs. formatting-layer conditioning
- Supports chat-only probing for native API backends (xAI/Grok) where no generate endpoint exists
- Measures response length, think block presence, and content differences across paired question variants — mechanism-first vs. named-actor, English vs. simplified Chinese, bare vs. academic framing, and English vs. China-mirror
- Supports multi-sweep reliability testing and LLM-as-judge semantic consistency analysis
- Tracks tool-call capability per model
- Maintains a model registry (`probes/probe_models.json`) as a single source of truth for empirical findings

---

## Supported backends

| Backend | Models | Key | Notes |
|---|---|---|---|
| `ollama` | Any locally-installed Ollama model | — | Requires `ollama serve` |
| `ollama_cloud` | Cloud-proxied models via ollama.com | `OLLAMA_API_KEY` in `~/.env` | `"cloud"` in model name routes automatically |
| `xai` | Grok models via xAI API | `XAI_API_KEY` in `~/.env` | Chat-only; no generate endpoint |
| `omlx` | MLX models on Apple Silicon | — | Queried via `code/probe_query_omlx.py` |
| `llamacpp` | GGUF models via llama.cpp | — | Queried via `code/probe_query_llamacpp.py` |

Adding a new OpenAI-compatible provider requires one entry in `PROVIDER_CONFIG` in `code/probe_query_openai.py` — no other code changes.

---

## Repository structure

```
llm-model-update/
├── code/                       # Probe scripts and test suite
│   ├── tests/                  # pytest test suite
│   ├── probe_static.py         # Main probe runner
│   ├── probe_endpoint_sweep.py # Multi-sweep reliability baseline
│   ├── probe_endpoint_test.py  # Single-model endpoint qualification
│   ├── probe_sweep_judge.py    # LLM-as-judge semantic analysis
│   ├── probe_tool_capable.py   # Tool-call capability tester
│   ├── probe_coverage.py       # Coverage gap reporter
│   ├── probe_db.py             # SQLite ingest and summary
│   ├── probe_classify.py       # Lexical CCP-voice classifier
│   ├── probe_classify_with_model.py  # LLM-powered behavioral classification
│   ├── probe_analysis.py       # Research analysis and findings reports
│   ├── probe_healthcheck.py    # Pre-run Ollama connectivity check
│   └── setup_venv.sh           # Venv creation script
├── mlx/                        # MLX model updater (Apple Silicon)
│   └── mlx_updater.py
├── ollama/                     # Ollama model updater
│   └── ollama_updater.py
├── shared/                     # Shared utilities (config, logging)
├── probes/
│   ├── probe_models.json       # Model registry — single source of truth
│   └── questions.json          # 25-question probe bank
├── results/                    # Output data (excluded from repo)
│   ├── data/probes/            # probe_static.py output
│   ├── data/sweeps/            # probe_endpoint_sweep.py output
│   ├── data/judges/            # probe_sweep_judge.py output
│   ├── db/                     # SQLite database
│   └── reports/coverage/       # Dated coverage reports
└── docs/                       # Project documentation (internal — gitignored)
```

---

## Requirements

- Python 3.12+ (via pyenv: `pyenv install 3.12.0`)
- [Ollama](https://ollama.com) running locally (`ollama serve`) — required for Ollama backends
- `OLLAMA_API_KEY` in `~/.env` for Ollama cloud model access
- `XAI_API_KEY` in `~/.env` for xAI/Grok model access

**First-time setup** — creates venv at `~/VirtualEnvs/venv-llm-model-update` and runs the test suite:

```bash
cd code
chmod +x setup_venv.sh && ./setup_venv.sh
```

**Subsequent sessions** — activate the venv before any run:

```bash
source ~/VirtualEnvs/venv-llm-model-update/bin/activate
```

---

## Core tools

All scripts are under `code/`. Run from the project root with the venv active.

| Script | Purpose |
|---|---|
| `code/probe_static.py` | Main probe runner — chat + generate across models |
| `code/probe_endpoint_sweep.py` | Multi-sweep reliability baseline |
| `code/probe_endpoint_test.py` | Infrastructure qualification for a single model |
| `code/probe_sweep_judge.py` | LLM-as-judge semantic consistency analysis |
| `code/probe_tool_capable.py` | Tool-call capability tester (writes to registry) |
| `code/probe_coverage.py` | Coverage gap reporter |
| `code/probe_db.py` | SQLite ingest and summary |
| `code/probe_query_openai.py` | OpenAI-compatible query layer (xAI/Grok and future providers) |
| `code/probe_query_omlx.py` | MLX query layer (Apple Silicon) |
| `code/probe_query_llamacpp.py` | llama.cpp query layer (GGUF) |
| `code/probe_classify.py` | Lexical CCP-voice classifier |
| `code/probe_classify_with_model.py` | LLM-powered behavioral classification |
| `code/probe_translate.py` | Sentiment-preserving translation of non-English probe results |
| `code/proto_ccp_judge.py` | LLM-as-judge CCP-voice classifier prototype |
| `code/probe_analysis.py` | Research analysis and findings reports |
| `code/probe_healthcheck.py` | Pre-run Ollama connectivity check |
| `code/generate_findings.py` | Generates findings reports from probe data |

---

## Quick start

```bash
# Check Ollama connectivity and model availability
python code/probe_healthcheck.py

# Run sensitivity probes on a single model
python code/probe_static.py --questions Q10b Q12b --models "<model-name>"

# Run the full question bank
python code/probe_static.py --all-questions --models "<model-name>" --label full_suite

# Ingest results and check coverage
python code/probe_db.py --ingest
python code/probe_coverage.py

# Run tests
pytest code/tests/ -v
```

---

## Model registry

`probes/probe_models.json` is the single source of truth for model metadata and empirically determined capabilities. Never edit directly for new models — use `python ollama/ollama_updater.py --sync` then fill only the null research fields it flags.

`probes/questions.json` contains the probe bank across five themes (authoritarianism, trade, EU governance, governance/AI regulation, geopolitics). Each base question may have variants:

| Suffix | Purpose |
|---|---|
| *(none)* | Mechanism-first or neutral framing — no named actors |
| `b` | Explicitly names the actor (e.g. "China", "authoritarian states") |
| `c` | Simplified Chinese translation of the China-named variant |
| `d` | Academic-framing prefix ("You are a neutral political science researcher…") |
| `e` | English China-mirror variant naming Chinese counterparts |

---

## Results

Results data (probe outputs, database, coverage reports) are excluded from this repository. Only the framework for reproducing results is tracked here.
