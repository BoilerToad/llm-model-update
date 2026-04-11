---
name: llm-probe-workflow
description: >
  Operational standards for the llm-model-update probe research project
  (~/AI-Development/llm-model-update/). ALWAYS use this skill when working on
  probe_static.py, probe_coverage.py, probe_db.py, probe_analysis.py,
  probe_classify.py, probe_healthcheck.py, probe_models.json, models.yaml,
  questions.json, or any related tooling. Enforces: use scripted tools over
  manual edits, test-first Python development, correct workflow order, and
  registry sync procedures.
---

# LLM Probe Workflow — Operational Standards

Two rules take precedence over everything else in this project.

---

## Rule 1: Use the Scripts We Wrote

**Before touching any file manually, ask: does a script already exist to do
this?** The answer is almost always yes. Manual edits to registry files are
the largest source of bugs — name mismatches, truncated JSON, inconsistencies
between files that must stay in sync.

### Model registry management

| Task | Correct command | Never do this |
|---|---|---|
| Add newly pulled model to `models.yaml` | `python ollama/ollama_updater.py --sync` | Edit `models.yaml` directly |
| Add model to `probe_models.json` | `python ollama/ollama_updater.py --sync-probes` | Edit `probe_models.json` directly |
| Remove stale entries from `models.yaml` | `python ollama/ollama_updater.py --prune` | Delete lines manually |
| List registered vs installed | `python ollama/ollama_updater.py --list` | Read YAML directly |
| Verify all models respond before a run | `python probe_healthcheck.py --chat-only` | Assume they work |

**Complete workflow for adding a new model:**
```
1. ollama pull <model-name>
2. python ollama/ollama_updater.py --sync          # adds to models.yaml
3. python ollama/ollama_updater.py --sync-probes   # adds to probe_models.json
4. python probe_healthcheck.py --family <family>   # confirm it responds
5. Edit probe_models.json ONLY to fill in the null fields flagged by --sync-probes:
   geopolitical_origin, tool_capable, think_blocks, chat_alignment_strong
```

### Probe execution

| Task | Correct command |
|---|---|
| Find coverage gaps | `python probe_coverage.py --quiet` |
| Get rerun commands | Copy `probe_coverage.py --quiet` output verbatim |
| Ingest results | `python probe_db.py --ingest` |
| Generate reports | `python probe_analysis.py` |

**Never construct probe run commands from memory.** `probe_coverage.py --quiet`
generates exact commands with correct question sets and model lists.

### Correct session workflow
```
1. python probe_healthcheck.py --chat-only
2. python probe_coverage.py --out results/coverage_report.md
3. python probe_coverage.py --quiet   ← copy these commands
4. [run generated commands unchanged]
5. python probe_db.py --ingest
6. python probe_analysis.py
```

---

## Rule 2: New Python Code Gets Tests

Every new function or script ships with tests. This is a hard requirement.

The bugs that caused the most lost time in this project were all trivially
catchable by tests:
- `KeyError: 'generate_len'` — f-string key access not tested for `ep="generate"`
- `AttributeError: module has no attribute 'classify_text'` — name mismatch
- `SyntaxError: name used prior to global declaration` — caught by any import test

### Minimum coverage for any new function

1. **Happy path** — valid inputs, assert expected output
2. **Edge case** — empty input, None, missing key, empty directory
3. **Dict key contract** — if function returns a dict, assert every downstream key:
   ```python
   for ep in ("chat", "generate"):
       _ = result[f"{ep}_len"]   # test BOTH values — this is the recurring bug
       _ = result[f"{ep}_err"]
   ```

### The regression rule
1. Write a failing test that reproduces the bug — before touching the code
2. Fix the code
3. Confirm the test passes
4. Test stays permanently

### Running the suite
```bash
pytest tests/ -v
```
All tests must pass before any code is considered done.

---

## Registry consistency

| File | Purpose | Managed by |
|---|---|---|
| `ollama/models.yaml` | Ollama updater control | `ollama_updater.py --sync` / `--prune` |
| `probes/probe_models.json` | Probe metadata | `ollama_updater.py --sync-probes` (then fill null fields) |

After `--sync-probes`, the only acceptable manual edit to `probe_models.json`
is filling in the null research fields flagged in the notes:
`geopolitical_origin`, `tool_capable`, `think_blocks`, `chat_alignment_strong`.

Model names must exactly match `ollama list` output. Use `probe_healthcheck.py`
to verify before any run.

---

## Known model quirks

- **falcon3:7b** — generate returns 1-2 chars. Use `--no-generate`.
- **falcon3:10b** — generate behavior TBD. Test first.
- **deepseek-v3.2:cloud** — returns `success=True` with empty body on some questions.
- **mistral-nemo** — Ollama registers as `mistral-nemo:latest` (with tag suffix).
