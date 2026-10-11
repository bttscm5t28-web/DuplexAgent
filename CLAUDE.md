# DuplexAgent — notes for agents

- Python ≥ 3.11, core is stdlib-only. Run `python -m pytest -q` and `python -m duplexagent eval scenarios/*.toml` before pushing.
- Modules are plugins: `duplexagent/plugins/<kind>/<name>.py` with `@register(kind, name)`; selected via `impl` in TOML config or `--set kind.impl=name`. Import optional deps lazily.
- Do not change interfaces in `duplexagent/core/` casually; follow docs/02-skeleton-and-experiments.md (experiment branches own only their module's directories).
- Docs and reports are written in Chinese.
