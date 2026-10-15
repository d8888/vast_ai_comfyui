# vast-comfy test suite

Offline unit + mock-integration tests for `SPEC.md` section 17. **No test touches
the real Vast.ai API, real SSH or real GPU instances.** Everything external is
replaced by local mocks under `tests/mocks/`.

## How to run

```powershell
# from the repository root, using the project venv
C:\vast_ai_comfyui\.venv\Scripts\python.exe -m pytest

# verbose, list skipped reasons
C:\vast_ai_comfyui\.venv\Scripts\python.exe -m pytest -v -rs
```

On Linux (or WSL) the same command works; the real-`flock` session tests then
run instead of being skipped.

## Layout

| File | SPEC 17 coverage |
| --- | --- |
| `test_config.py` | 1, 2 — YAML strictness, precedence, resolved paths |
| `test_gpu_aliases.py` | 1, 2 — alias resolution, custom aliases, mapping updates |
| `test_offers_query.py` | 3 — search query/unit contract |
| `test_offers_evaluate.py` | 3-5 — hard conditions, cheapest selection, price-cap errors |
| `test_rental.py` | 6, 7 — paid confirmation, create/verify/wait errors |
| `test_vast_client.py` | 6, 12, 19 — REST wire contract against a local mock server |
| `test_ssh.py` | 8 — proxy-only SSH, known_hosts TOFU, tunnels, scrubbing |
| `test_deploy_package.py` | 14 — upload whitelist + deterministic packaging |
| `test_remote_agent.py` | 2, 11, 16 — remote-agent pure logic, install manifest |
| `test_smoke_runner.py` | 16, 17 — READY/FAILED/PARTIAL/busy and image validation |
| `test_smoke_assets.py` | 16 — committed smoke PNG assets are usable |
| `test_jobs.py` | 12 — job wait/timeout/log parsing, remote `invoke` |
| `test_session_prepare.py` | 9-11, 13, 18 — lock calls, prepare flow, deploy idempotence |
| `test_download.py` | 13 — overwrite refusal, sha256 verification, outputs listing |
| `test_cli.py` | 19 — one-JSON-object stdout, exit codes, stable codes |
| `test_linux_session.py` | 9-11, 15, 18 — real `flock` daemon semantics (Linux only) |

`tests/mocks/` contains the scripted fake SSH runner (`fake_ssh.py`), the local
Vast.ai API mock server (`vast_api_server.py`) and the in-process ComfyUI fake
(`fake_comfy.py`).

## Linux-only tests

`test_linux_session.py` drives the real `vc_session.py` remote script through
subprocesses (`VAST_COMFY_RUNTIME_DIR` / `VAST_COMFY_COMFY_DIR` pointed at tmp
dirs) and proves the real `flock` mutual exclusion, heartbeat, release refusal
and stale-takeover rules. It is marked `@pytest.mark.linux_only` +
`skipif(sys.platform != "linux")` and is therefore **skipped on Windows**; run
it on a Linux machine (or WSL) for the platform-level evidence required by
SPEC.md section 17.

Everything else runs identically on Windows and Linux.
