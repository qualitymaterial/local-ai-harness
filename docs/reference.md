# local-ai — Full Reference

## Table of contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [LM Studio setup](#lm-studio-setup)
4. [Configuration](#configuration)
5. [Commands](#commands)
   - [index](#index)
   - [review](#review)
   - [ask](#ask)
   - [plan](#plan)
   - [patch](#patch)
   - [apply](#apply)
   - [diff](#diff)
   - [run](#run)
   - [config](#config)
6. [How context is built](#how-context-is-built)
7. [Safety model](#safety-model)
8. [Output artifacts](#output-artifacts)
9. [Troubleshooting](#troubleshooting)
10. [Architecture](#architecture)

---

## Overview

`local-ai` is a command-line coding assistant that:

- **Scans your repo** intelligently — respects `.gitignore`, skips binaries/lockfiles/junk, extracts lightweight structural signals (imports, exported symbols, function/class names).
- **Builds context automatically** — ranks files against your question or task using filename heuristics, folder importance, symbol overlap, and framework conventions. No manual file-adding.
- **Calls your local model** via the OpenAI-compatible API exposed by LM Studio (default: `http://localhost:1234/v1`).
- **Produces safe, auditable output** — reviews, plans, patches, and run reports are written under `.local-ai/`. Patches are never applied without an explicit `apply` command and a `[y/N]` confirmation after a `git apply --check` dry run.

No cloud APIs. No OpenAI key. No data leaves your machine.

---

## Installation

### Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai) (macOS / Windows / Linux)
- A GGUF coding model loaded in LM Studio (e.g. DeepSeek-Coder-V2-Lite-Instruct)

### Steps

```bash
git clone https://github.com/qualitymaterial/local-ai-harness local-ai
cd local-ai
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
local-ai --help
```

To run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

---

## LM Studio setup

1. Open LM Studio and download a coding model. Recommended:
   - **DeepSeek-Coder-V2-Lite-Instruct** — solid code understanding, low RAM
   - **Qwen2.5-Coder-7B-Instruct** — strong alternative
   - Any model ≥7B with a coding instruction tune works well
2. Load the model (click Load in the model card).
3. Go to **Developer** tab → click **Start Server**.
4. Confirm it's listening:

```bash
curl http://localhost:1234/v1/models
```

You should get a JSON list of loaded models.

### Finding the exact model identifier

The `model` value in your config must match what LM Studio reports. The JSON above shows
it, and it appears in the LM Studio server panel. Copy it exactly — it often includes
the full filename without the `.gguf` extension.

---

## Configuration

`local-ai` writes a default config to `.local-ai/config.toml` the first time you run any
command in a repo. You can also create it manually:

```toml
# .local-ai/config.toml
model = "deepseek-coder-v2-lite-instruct"
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
max_context_tokens = 12000
temperature = 0.2
top_p = 0.9
request_timeout = 600
```

| Key | Default | Description |
|---|---|---|
| `model` | `deepseek-coder-v2-lite-instruct` | Model identifier (must match LM Studio) |
| `base_url` | `http://localhost:1234/v1` | LM Studio server URL |
| `api_key` | `lm-studio` | Dummy key (LM Studio ignores it, but the field is required) |
| `max_context_tokens` | `12000` | Token budget for the model context (prompt side). Set this to ≤ your model's context window. |
| `temperature` | `0.2` | Sampling temperature. Lower = more deterministic. |
| `top_p` | `0.9` | Nucleus sampling. |
| `request_timeout` | `600` | Seconds before giving up on a model response. Local models can be slow. |

### Environment variable overrides

All keys can be overridden per-command without editing the file:

```bash
LOCAL_AI_MODEL="qwen2.5-coder-7b-instruct"  local-ai ask . "…"
LOCAL_AI_BASE_URL="http://192.168.1.10:1234/v1"  local-ai review .
LOCAL_AI_MAX_CONTEXT_TOKENS=4096  local-ai ask . "…"
```

| Variable | Config key |
|---|---|
| `LOCAL_AI_BASE_URL` | `base_url` |
| `LOCAL_AI_MODEL` | `model` |
| `LOCAL_AI_API_KEY` | `api_key` |
| `LOCAL_AI_MAX_CONTEXT_TOKENS` | `max_context_tokens` |

### Tuning `max_context_tokens`

This is the most important tuning knob. It controls how much of the repo `local-ai` can
include in each model call. The right value is slightly below your loaded model's actual
context window (leave ~1,500 tokens headroom for the model's response).

Common model context windows:

| Model | Context window | Recommended `max_context_tokens` |
|---|---|---|
| DeepSeek-Coder-V2-Lite | 4k–16k depending on quant | 3000–14000 |
| Qwen2.5-Coder-7B | 32k | 28000 |
| Qwen2.5-Coder-14B | 128k | 120000 |
| CodeLlama-13B | 16k | 14000 |

If you hit a 400 "context too large" error, lower this value.

---

## Commands

All commands take `PATH` as their first argument — the root directory of the repo to work
with. Use `.` when you're already inside the repo.

---

### index

```bash
local-ai index PATH
```

Scan the repo and write:

- `.local-ai/repo_map.md` — readable tree + stack summary
- `.local-ai/repo_index.json` — full machine-readable index (file list, symbols, imports, sizes, languages)

This is the fastest command and runs fully offline. All other commands re-scan before
building context (the scan is fast; it never re-reads files it can skip).

**What it ignores:**

Always skipped regardless of `.gitignore`:
`.git`, `node_modules`, `dist`, `build`, `.next`, `.nuxt`, `.svelte-kit`, `.venv`,
`venv`, `__pycache__`, `coverage`, `target`, `vendor`, `.idea`, `.vscode`, `.local-ai`,
`.turbo`, `.cache`, `out`

Always skipped files:
`.DS_Store`, `package-lock.json`, `pnpm-lock.yaml`, `yarn.lock`, `poetry.lock`,
`Cargo.lock`, `composer.lock`, `Gemfile.lock`

Binary extensions (content not read): images, fonts, archives, compiled objects, model
weights (`.gguf`, `.safetensors`, `.pt`), databases, audio/video, WebAssembly.

`.gitignore` and `.git/info/exclude` are loaded and applied on top of the above.

---

### review

```bash
local-ai review PATH
```

Generates a full repository review. Sends the repo map + the most relevant config/source
files to the model, which produces a structured report covering:

- Project summary and detected stack
- Architecture overview and likely entry points
- Strengths
- Risks, bugs, and suspicious patterns
- Cleanup opportunities
- Dependency, security, and performance concerns
- Recommended next steps

**Output:** `.local-ai/reports/repo_review_YYYYMMDD_HHMMSS.md`

The terminal prints a truncated summary and the path to the full report.

---

### ask

```bash
local-ai ask PATH "your question here"
```

Answer any question about the codebase. `local-ai` ranks files for relevance to the
question, includes the most useful ones within the token budget, and answers with
citations to filenames and (where possible) line ranges or symbol names.

If the question can't be answered from the included context, the model will say so and
name the files that would need to be inspected.

**Examples:**

```bash
local-ai ask . "Where is the homepage component?"
local-ai ask . "How does authentication work?"
local-ai ask . "What API routes does this server expose?"
local-ai ask . "Where is the database schema defined?"
local-ai ask . "How are environment variables loaded?"
```

---

### plan

```bash
local-ai plan PATH "task description"
```

Produces a step-by-step implementation plan for the described task. The model identifies
the files likely to change and flags risks and open questions — but **does not write or
modify any code**.

**Output:** `.local-ai/plans/plan_YYYYMMDD_HHMMSS.md`

**Plan sections:**
- Goal
- Affected files (with justification)
- Step-by-step plan
- Risks and edge cases
- Testing strategy
- Open questions / missing context

---

### patch

```bash
local-ai patch PATH "task description"
```

Asks the model to produce a **unified diff** for the described change. The diff is
written to disk but **not applied**. The model is instructed to only modify files present
in the context and to produce a valid `git apply`-compatible diff.

After generating, `local-ai` runs `git apply --check` to verify the patch would apply
cleanly, and reports the result.

If the model cannot produce a safe diff from the visible context, it produces an
explanation + plan instead.

**Output:**
- `.local-ai/patches/patch_TIMESTAMP.diff` — the raw unified diff (if produced)
- `.local-ai/patches/patch_TIMESTAMP.md` — explanation / plan

**To apply the patch:**

```bash
local-ai apply . .local-ai/patches/patch_TIMESTAMP.diff
```

---

### apply

```bash
local-ai apply PATH PATCH_FILE [--yes]
```

Applies a patch file via `git apply`. Before touching any file:

1. Shows a table of files changed with addition/deletion counts.
2. Runs `git apply --check` — if this fails, aborts immediately with no changes.
3. Asks for confirmation: `Apply this patch? [y/N]`
4. Applies atomically with `git apply` (all hunks validated before any write).

Pass `--yes` / `-y` to skip the confirmation prompt (e.g. in scripts).

**Requires:** a git repository. If the directory is not a git repo, the command refuses
and exits without touching files.

---

### diff

```bash
local-ai diff PATH
```

Reads `git diff HEAD` from the repository and sends it to the model for review. The
model flags:

- Bugs and correctness issues
- Risky changes
- Missing tests
- Cleanup opportunities
- Overall verdict (approve / request changes)

Does not write any files. Useful before committing or opening a PR.

---

### run

```bash
local-ai run PATH "shell command" [--yes]
```

Runs a shell command from `PATH` and:

1. **Captures** stdout, stderr, and exit code.
2. **If successful** — prints output and writes a run report.
3. **If failed** — sends the failure (command, exit code, stdout tail, stderr tail, repo
   map) to the model for diagnosis. Prints the diagnosis and writes it to the run report.

**Dangerous command detection:** Commands matching patterns like `rm -rf`, `sudo`,
`chmod -R`, `chown -R`, `curl … | bash`, fork bombs, raw `dd`, and destructive git
operations are flagged and require confirmation. Pass `--yes` / `-y` to bypass.

**Output:** `.local-ai/runs/run_YYYYMMDD_HHMMSS.md`

**Examples:**

```bash
local-ai run . "npm run build"
local-ai run . "pytest"
local-ai run . "cargo test"
local-ai run . "python manage.py migrate"
```

---

### config

```bash
local-ai config [PATH]
```

Shows the effective configuration for the repo (file + env overrides merged), masks the
API key, and probes `GET /models` to confirm LM Studio is reachable.

Creates a default `config.toml` if one doesn't exist yet.

---

## How context is built

`local-ai` never dumps the whole repo blindly. For each command:

1. **Scan** — walks the repo, applies ignore rules, extracts imports/symbols/line counts.
2. **Repo map** — always included: tree, stack, important files. Cheap.
3. **Rank** — scores every non-binary file against the query/task using:
   - Filename/path token overlap with the query (split on camelCase and delimiters)
   - Concept expansion (e.g. "homepage" → searches for `index`, `home`, `page`, `app`,
     `main`, `landing`; "mobile" → `css`, `style`, `responsive`, `media`, `tailwind`)
   - Exported symbol overlap with query tokens
   - Folder importance weights (`app`=1.8, `pages`=1.8, `routes`=1.6, `components`=1.6,
     `api`=1.5, `src`=1.5, …)
   - Config/entry-point bonus (package.json, pyproject.toml, vite.config.*, etc.)
4. **Fill budget** — includes files in score order. Oversized files are summarized (head
   + imports/symbols outline) rather than truncated silently. Files that don't fit are
   tracked and listed in the report footer.
5. **Report** — every output includes a "Context used" section listing exactly which
   files were sent and which were dropped.

---

## Safety model

| Concern | What local-ai does |
|---|---|
| Applying patches | Never automatic. Requires `local-ai apply`, `git apply --check`, and `[y/N]` confirmation. |
| Destructive shell commands | Flagged by pattern (rm -rf, sudo, curl\|bash, etc.) and require confirmation. |
| Hallucinated file access | System prompts explicitly forbid the model from claiming to read files not in context. |
| Partial patch writes | `git apply` validates all hunks before writing anything; impossible to leave files half-patched. |
| Non-git repos | `local-ai apply` refuses to run if `git rev-parse` fails. |
| Audit trail | Every action is logged to `.local-ai/memory.json` with a timestamp. |

---

## Output artifacts

All output is written under `.local-ai/` in the target repo. This directory is in
`.gitignore` by default and is never committed.

```
.local-ai/
  config.toml                      # per-repo config (commit this if you want shared defaults)
  repo_map.md                      # human-readable repo tree + stack summary
  repo_index.json                  # machine-readable full file index
  memory.json                      # audit log of local-ai actions
  reports/
    repo_review_TIMESTAMP.md
  plans/
    plan_TIMESTAMP.md
  patches/
    patch_TIMESTAMP.diff           # raw unified diff
    patch_TIMESTAMP.md             # explanation / fallback plan
  runs/
    run_TIMESTAMP.md               # command output + diagnosis if failed
```

---

## Troubleshooting

**"Could not connect to http://localhost:1234"**
LM Studio's server isn't started. Open LM Studio → Developer tab → Start Server. Then
`curl http://localhost:1234/v1/models` to confirm.

**HTTP 404 / "Model endpoint returned 404"**
The `model` config value doesn't match what LM Studio has loaded. Check the exact
identifier in LM Studio's server panel and paste it into `model` in
`.local-ai/config.toml` or set `LOCAL_AI_MODEL`.

**"The request exceeded the model's context window" (HTTP 400)**
Lower `max_context_tokens` — it needs to be smaller than the loaded model's context
window. Common fix: set it to 4096 for small models (3B–7B at 4-bit quant with a 4k
window) or 12000–28000 for larger/longer-context models.

**Model takes forever / times out**
Raise `request_timeout` in the config. First token from a cold model can take 30–60s
on CPU. This is normal for large GGUF files.

**Empty model response**
The model returned content but it was blank. Reload the model in LM Studio, or try
nudging `temperature` up slightly (e.g. 0.3). Sometimes happens when a model gets stuck
on stop tokens.

**Patch fails to apply**
`local-ai apply` runs `git apply --check` first — if that fails, nothing is touched.
Possible causes: the model referenced line numbers that have since changed, or it output
a malformed diff. Re-run `local-ai patch` against the current state of the files, or
inspect the `.diff` manually and adjust hunk offsets.

**"Not a git repository"** on `local-ai apply`
`git apply` requires a git repo. Run `git init` in the repo root first.

**Files I care about weren't included in context**
The ranker may have scored them below the budget cutoff. Options:
- Increase `max_context_tokens` if your model supports it.
- Reference the filename explicitly in your question/task — filename token matches get
  the highest weight.
- Check the "Context used" footer of any report to see what was included vs. dropped.

---

## Architecture

```
local_ai/
  cli.py             Typer app — one function per command, wires everything together
  config.py          TOML config loader, env overrides, Config dataclass
  scanner.py         Repo walk (ignore-aware), language detection, structure extraction,
                     framework detection, file tree renderer
  gitignore_utils.py Hard-coded ignore lists + pathspec .gitignore loader
  file_ranker.py     Relevance scoring: path/symbol/folder/concept heuristics
  context_builder.py Repo map builder + token-budgeted context packet assembler
  model_client.py    httpx-based OpenAI-compatible client, typed error hierarchy
  prompts.py         System prompts per mode + message assembler
  patcher.py         Diff extraction from model output, stats, git apply wrapper
  command_runner.py  Shell command runner + destructive pattern guard
  memory.py          Per-repo audit log (.local-ai/memory.json)
  report_writer.py   Timestamped artifact writers for each output type
  types.py           Shared dataclasses (FileEntry, RepoIndex, ContextPacket, …)
```

### Data flow for `local-ai ask . "question"`

```
user query
  └─► scanner.scan_repo()          walk repo, extract signals
        └─► file_ranker.rank_files()   score files against query
              └─► context_builder.build_context()   fill token budget
                    └─► model_client.ModelClient.chat()   POST /chat/completions
                          └─► prompts.build_messages()    system + context + query
                                └─► rich console output + memory.record()
```
