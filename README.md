# local-ai

A coding assistant CLI that connects a repo to either a **local LM Studio model** or
**Claude** (Anthropic API), and produces reviews, answers, plans, patches, run
diagnostics, and multi-turn chat — with full context built automatically from the repo.

```
local-ai index  .
local-ai review .
local-ai ask    . "Where is the homepage component?"
local-ai plan   . "Refactor this app into cleaner components"
local-ai patch  . "Fix the mobile layout issue"
local-ai apply  . .local-ai/patches/patch_20260627_143015.diff
local-ai diff   .
local-ai run    . "npm run build"
local-ai chat   .                              # persistent multi-turn chat
local-ai chat   . --resume session_20260627_143000
```

**Two backends:**

| Backend | When to use | Config |
|---|---|---|
| LM Studio (default) | Fully local, no API costs | `backend = "lmstudio"` |
| Claude | Stronger reasoning, cloud API | `backend = "claude"` |

**Key features:**
- **Streaming output** — token-by-token printing (`stream = true` or `LOCAL_AI_STREAM=1`)
- **Agentic loop** — model calls `read_file` / `search_code` / `list_directory` mid-reasoning (`agentic = true`)
- **Persistent chat** — multi-turn sessions saved to `.local-ai/sessions/`, resumable

---

## How it differs from aider / Continue

- **No manual file wrangling.** You don't `/add` files or fight an IDE context window.
  `local-ai` scans the repo, builds a repo map, ranks files by relevance to your
  question/task with cheap local heuristics, and assembles a token-budgeted context
  packet automatically.
- **Repo-aware by default.** Every command starts from an up-to-date scan that respects
  `.gitignore` and skips junk (`node_modules`, `dist`, `.venv`, lockfiles, binaries…).
- **Audit trail.** Reviews, plans, patches, and run logs are written under `.local-ai/`
  with timestamps. Patches are never applied automatically.
- **Honest context.** Each report lists exactly which files were sent to the model, and
  the system prompts forbid the model from claiming to read files it wasn't given.
- **Local-only.** Built for LM Studio + GGUF models like
  `deepseek-coder-v2-lite-instruct`.

---

## 1. Start LM Studio Developer Mode

1. Open LM Studio.
2. Download and load a coding model (e.g. **DeepSeek-Coder-V2-Lite-Instruct GGUF**).
3. Go to the **Developer** tab → **Start Server**.
4. Confirm it's serving at `http://localhost:1234` (OpenAI-compatible at `/v1`).

You can sanity-check it from a terminal:

```bash
curl http://localhost:1234/v1/models
```

---

## 2. Install the CLI

```bash
cd local-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then:

```bash
local-ai --help
```

---

## 3. Configure

The first time you run any command in a repo, a default config is written to
`.local-ai/config.toml`:

```toml
model = "deepseek-coder-v2-lite-instruct"
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
max_context_tokens = 12000
temperature = 0.2
top_p = 0.9
request_timeout = 600
```

Inspect the effective config and probe the server with:

```bash
local-ai config .
```

### Environment variable overrides

| Variable | Overrides |
|---|---|
| `LOCAL_AI_BASE_URL` | `base_url` |
| `LOCAL_AI_MODEL` | `model` |
| `LOCAL_AI_API_KEY` | `api_key` |
| `LOCAL_AI_MAX_CONTEXT_TOKENS` | `max_context_tokens` |

Example:

```bash
LOCAL_AI_MODEL="qwen2.5-coder-7b-instruct" local-ai ask . "What does this project do?"
```

---

## 4. Commands

| Command | What it does |
|---|---|
| `local-ai index PATH` | Scan the repo → `.local-ai/repo_map.md` + `repo_index.json` |
| `local-ai review PATH` | Full repo review → `.local-ai/reports/repo_review_*.md` |
| `local-ai ask PATH "Q"` | Answer a question, auto-selecting relevant files |
| `local-ai plan PATH "TASK"` | Implementation plan → `.local-ai/plans/plan_*.md` (no edits) |
| `local-ai patch PATH "TASK"` | Proposed unified diff → `.local-ai/patches/patch_*.diff` (not applied) |
| `local-ai apply PATH FILE` | Apply a patch via `git apply`, with confirmation |
| `local-ai diff PATH` | Review the current `git diff HEAD` |
| `local-ai run PATH "CMD"` | Run a command; on failure, diagnose it → `.local-ai/runs/run_*.md` |
| `local-ai config PATH` | Show effective config + backend reachability |
| `local-ai chat PATH` | Persistent multi-turn chat session about this repo |

### Safety guarantees

- Patches are **never** applied automatically — only via `local-ai apply`, after a
  `git apply --check` dry run and a `[y/N]` confirmation. If the patch doesn't apply
  cleanly, **no files are touched**.
- `local-ai run` flags destructive commands (`rm -rf`, `sudo`, recursive `chmod`/`chown`,
  `curl … | bash`, fork bombs, raw `dd`, …) and asks before running them.
- The model is told it can only see the included files and must say what's missing
  rather than hallucinating file access.

---

## 5. Context engineering (how it picks files)

`local-ai` does **not** dump the whole repo every time. For each command it:

1. Scans the repo and builds a **repo map** (tree, languages, frameworks, important files).
2. Extracts cheap structural signals per file (imports, exported symbols, function/class
   names).
3. **Ranks** files against your question/task using filename relevance, folder importance
   (`src`, `app`, `components`, `routes`, `api`…), symbol matches, config/entry-point
   importance, and framework conventions (e.g. "homepage" → `index`/`page`/`Home`).
4. Fills a configurable **token budget** (`max_context_tokens`), summarizing oversized
   files (head + imports/symbols outline) instead of including them whole.
5. Reports exactly which files made it into the context.

---

## 6. Troubleshooting

**Connection refused / "Could not connect"**
The LM Studio server isn't running. Open LM Studio → Developer → Start Server.
Verify with `curl http://localhost:1234/v1/models`. If you use a different port/host,
set `LOCAL_AI_BASE_URL` or edit `base_url` in `.local-ai/config.toml`.

**Model name mismatch (HTTP 404)**
The `model` value must match the identifier LM Studio is serving. Check the model id in
LM Studio's server panel and set it via `model` in the config or `LOCAL_AI_MODEL`.

**Context too large (HTTP 400 about tokens/length)**
Lower `max_context_tokens` (or `LOCAL_AI_MAX_CONTEXT_TOKENS`), or load a model with a
larger context window in LM Studio.

**Request timed out**
Local models can be slow, especially first token. Raise `request_timeout` in the config,
or use a smaller/faster model.

**Patch failed to apply**
`local-ai apply` runs `git apply --check` first and aborts without modifying anything if
the patch is stale or malformed. Re-run `local-ai patch` against the current code, or
inspect the `.diff` manually. Patching requires the directory to be a git repo.

**Empty model response**
The model returned nothing. Reload the model in LM Studio, or nudge `temperature`
slightly up in the config.

---

## 7. Development

```bash
pip install -e ".[dev]"
pytest
```

Tests cover the scanner (ignore rules, framework detection, structure extraction) and the
context builder (relevance ranking + token budgeting). They run fully offline — no model
server required.

---

## Project layout

```
local-ai/
  pyproject.toml
  local_ai/
    cli.py             # Typer CLI: index/review/ask/plan/patch/apply/diff/run/config
    config.py          # TOML config + env overrides
    scanner.py         # repo walk, ignore rules, structure extraction, file tree
    gitignore_utils.py # ignore lists + .gitignore PathSpec
    file_ranker.py     # relevance scoring heuristics
    context_builder.py # repo map + token-budgeted context packets
    model_client.py    # OpenAI-compatible LM Studio client + error handling
    prompts.py         # system prompts per mode
    patcher.py         # diff extraction, stats, git apply (--check + apply)
    command_runner.py  # command execution + destructive-command guard
    memory.py          # per-repo audit log (.local-ai/memory.json)
    report_writer.py   # timestamped artifact writers
    types.py           # shared dataclasses
  tests/
```
