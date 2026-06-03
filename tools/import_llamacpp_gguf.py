"""
import_llamacpp_gguf.py
───────────────────────
Scan Hugging Face hub cache for .gguf files and propose safe Phase-0 stubs
for insertion into `probes/probe_models.json`.

Usage:
  python tools/import_llamacpp_gguf.py        # dry-run, prints candidates
  python tools/import_llamacpp_gguf.py --apply  # insert confirmed stubs

The script is conservative: it will never overwrite existing registry entries.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import re

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "probes" / "probe_models.json"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_gguf_files(cache_dir: Path) -> list[Path]:
    candidates = []
    if not cache_dir.exists():
        return candidates
    for root, dirs, files in os.walk(cache_dir):
        for name in files:
            if name.endswith('.gguf'):
                candidates.append(Path(root) / name)
    # also accept directories named *.gguf (some tools store as folder)
    for p in cache_dir.rglob('*.gguf'):
        if p.is_dir():
            # try find a blob file inside
            blob = next(p.rglob('*.gguf'), None)
            if blob:
                candidates.append(blob)
    # dedupe
    unique = sorted({str(p.resolve()): p for p in candidates}.values())
    return unique


def suggest_name_from_path(p: Path) -> tuple[str,str]:
    """Return (suggested_registry_name, display_source).

    Heuristic: look for patterns like models--owner--ModelName.gguf in path,
    or use filename without extension with normalization.
    """
    parts = p.parts
    joined = '/'.join(parts[-4:])
    fn = p.name
    # pattern models--owner--ModelName(-rest).gguf
    m = re.search(r'models--([^/]+)--(.+?)\.gguf', str(p))
    if m:
        owner = m.group(1)
        model = m.group(2)
        reg_name = f"{owner}/{model}.gguf"
        display = f"{owner}/{model}"
        return reg_name, display
    # fallback: normalized basename
    base = re.sub(r'\.(gguf|bin)$', '', fn, flags=re.I)
    # normalize to lowercase, replace non-alnum with -
    norm = re.sub(r'[^0-9a-z]+', '-', base.lower())
    reg_name = norm
    return reg_name, base


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def entry_exists(registry: dict, name: str) -> bool:
    for m in registry.get('models', []):
        if m.get('name') == name:
            return True
    return False


def make_stub(name: str, ll_url: str) -> dict:
    return {
        "name": name,
        "backend": "llamacpp",
        "llamacpp_url": ll_url,
        "api_model_id": name,
        "family": None,
        "geopolitical_origin": None,
        "tool_capable": None,
        "think_blocks": None,
        "chat_alignment_strong": None,
        "raw_capable": None,
        "enabled": True,
        "notes": f"Phase 0 stub added {datetime.now():%Y-%m-%d}; local gguf from HF cache"
    }


def apply_stubs(registry_path: Path, registry: dict, stubs: list[dict]) -> None:
    # bump meta version if present
    meta = registry.get('_meta', {})
    ver = meta.get('version')
    if ver:
        try:
            parts = ver.split('.')
            parts[1] = str(int(parts[1]) + 1)
            newv = '.'.join(parts)
            meta['version'] = newv
            meta.setdefault('changelog', {})[newv] = f"{datetime.now():%Y-%m-%d}: added {', '.join(s['name'] for s in stubs)} (llamacpp Phase 0)"
            registry['_meta'] = meta
        except Exception:
            pass
    # insert stubs at top
    models = registry.get('models', [])
    for s in reversed(stubs):
        models.insert(0, s)
    registry['models'] = models
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Write stubs into probes/probe_models.json')
    parser.add_argument('--url', default='http://127.0.0.1:8080', help='llamacpp server base URL to record in stubs')
    args = parser.parse_args()

    files = find_gguf_files(HF_CACHE)
    print(f'Found {len(files)} gguf blob(s) under {HF_CACHE}')
    registry = load_registry(REGISTRY)

    candidates = []
    for p in files:
        reg_name, display = suggest_name_from_path(p)
        exists = entry_exists(registry, reg_name) or entry_exists(registry, display)
        candidates.append((p, reg_name, display, exists))

    to_add = []
    for p, reg_name, display, exists in candidates:
        status = 'exists' if exists else 'new'
        print(f' - {p} -> registry name: "{reg_name}" (display: {display}) [{status}]')
        if not exists:
            to_add.append((reg_name, display))

    if not to_add:
        print('No new candidates to add.')
        return

    print('\nProposed stubs to add:')
    stubs = [make_stub(name, args.url) for name, _ in to_add]
    print(json.dumps(stubs, indent=2, ensure_ascii=False))

    if args.apply:
        apply_stubs(REGISTRY, registry, stubs)
        print(f'Applied {len(stubs)} stubs to {REGISTRY} (backup not created).')
    else:
        print('\nDry run: no changes written. Re-run with --apply to write the stubs.')

if __name__ == '__main__':
    main()
