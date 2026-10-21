# AGENTS.md — working notes for this repository

Instructions for anyone (human or agent) changing this repository. Read this
before your first commit.

The product contract lives in [`SPEC.md`](./SPEC.md) and the user-facing
behaviour in [`README.md`](./README.md). This file is about **how to not let
them drift**.

---

## The rule that keeps getting broken

> **A change that alters user-visible behaviour must update `README.md` in the
> same commit.**

`README.md` is 850+ lines of operational detail (flags, error codes, allowed
paths, verdicts, installed weights). It is the document a human actually reads
before spending money on a GPU. It has lagged behind the code every single
time it was left out of a commit — twice now it took a separate request to fix,
after it had already told the reader things that were no longer true.

### What counts as a README-visible change

Update `README.md` when you touch any of these:

| Changed | README section that must change |
| --- | --- |
| A CLI subcommand, flag, or its default | §15 command table |
| An error code, exit code, or `E_*` name | §15 error table |
| Allowed `dest` paths, upload whitelist, size limits | §13 transfer boundaries |
| Config keys, defaults, or precedence rules | §4 (and `config/default.yaml`) |
| Model/plugin list, pinned revisions, install order | §10, §10b |
| Smoke cases, verdict rules, what `READY` means | §11 |
| Which profile is default, or a new profile | §10b, header |
| Download/stop/destroy semantics | §12, §14 |
| Any new module with a user-facing entry point (`lora.py` did) | new section + TOC |
| Verification status, test counts, what was proven on a real machine | §20 |

### What does **not** need a README change

- Internal refactors that keep every flag, code and path identical.
- Test-only changes.
- Fixing a typo that is not itself a factual claim.

If you are unsure: it is a README change. A stale line costs more than an extra
paragraph.

---

## Checklist before committing

Run through this, then commit:

1. `python -m pytest` is green with the project venv (`.venv/bin/python -m pytest`).
   Never install into the system Python; never let a test run mutate the parent
   environment.
2. `config/default.yaml` still matches the built-in defaults (a test enforces
   this — if that test fails, change the code, not the test).
3. Every factual claim you added to `README.md` was verified against the code,
   not against memory. Numbers (counts, sizes, timings) come from a real run and
   say when that run happened.
4. If you changed behaviour, `README.md` is in the same commit — not a follow-up.
5. Commit messages state **why** the behaviour changed, and what proved it.

---

## Things this repo deliberately does

These are decisions, not accidents. Do not "clean them up":

- **No `config/models.yaml`.** Model lists live per profile under
  `config/profiles/`. One copy only.
- **No hardcoded case ids, node names, or model filenames in code.** They come
  from the active profile's workflows and smoke cases. A fixed list reports a
  false failure for whichever profile it was not written for.
- **No directory scanning for the deploy package.** Files are a whitelist
  derived from the profile's declared smoke cases. Stray files must never be
  uploaded.
- **A subset smoke run can never be `READY`.** `PARTIAL` exists exactly so a
  partial run cannot be dressed up as a pass.
- **Fixed seed is refused, not silently replaced.** Random per-run seeds defeat
  ComfyUI's node cache; a fixed seed that quietly becomes random would be worse.
- **`nsfw` stays out of the NSFW case's declared negative prompt** while the
  workflow file keeps it as a deliberate regression guard.
- **LoRA pins are exact.** sd-scripts commit, torch build, transformers and the
  WD14 revision are asserted non-floating by `tests/test_lora.py`.
- **HF token is read at call time and never stored.** No token on the publisher
  object, no token in a repr, no token in a log.
- **A saved LoRA file is not a passing run.** Verification (same seed, LoRA
  off/on, measured difference) is part of the plan and can fail.

---

## Repo conventions

- **Assets that make a result reproducible go in the repo** (training images,
  captions, reference photos, the verification pair). Large or intermediate
  artifacts do not: no LoRA weights (114 MB), no parameter-sweep variants, no
  R18 output.
- **Reference images get a README** recording how they are consumed and what
  went wrong, not just the file. `assets/reference/noobai/README.md` is the
  model to copy.
- **Two model families must both stay working.** A change that only runs for
  `noobai` is half a change; check `anima` too.
- **Secrets never enter the repo.** `.env`, keys and `known_hosts` are
  gitignored; tests cover the packaging refusal.

---

## Environment notes

- The project venv is `.venv` at the repo root; use `.venv/bin/python` on Linux.
- Vast.ai operations need `--env-file .env` explicitly; the CLI never reads
  `.env` on its own.
- On this account the Vast.ai proxy endpoint is often dead while the direct
  port works. If SSH returns `Connection reset`/`kex_exchange_identification`,
  add `--set ssh.strategy=direct_only`. Do not disable host key verification.
- The repo pins `vastai/comfy:v0.40.0-cuda-12.9-py312`; the last real-machine
  acceptance ran `v0.41.0-cuda-13.2-py312`. That gap is known and unresolved —
  do not quietly "fix" one side without deciding which.
