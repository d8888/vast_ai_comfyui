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
| `test_offers_evaluate.py` | 3-5 — hard conditions, cheapest selection, price-cap errors, geolocation exclusion fails closed |
| `test_rental.py` | 6, 7 — paid confirmation, create/verify/wait errors |
| `test_vast_client.py` | 6, 12, 19 — REST wire contract against a local mock server |
| `test_ssh.py` | 8 — proxy-only SSH, known_hosts TOFU, tunnels, scrubbing, same-binary scp protocol/path safety |
| `test_deploy_package.py` | 14 — upload whitelist + deterministic packaging |
| `test_remote_agent.py` | 2, 11, 16 — remote-agent pure logic, install manifest |
| `test_smoke_runner.py` | 16, 17 — READY/FAILED/PARTIAL/busy and image validation |
| `test_smoke_assets.py` | 16 — committed smoke PNG assets are usable |
| `test_jobs.py` | 12 — job wait/timeout/log parsing, remote `invoke` |
| `test_session_prepare.py` | 9-11, 13, 18 — lock calls, prepare flow, deploy idempotence + file-coverage and manifest-content verification |
| `test_download.py` | 13 — overwrite refusal, sha256 verification, outputs listing |
| `test_cli.py` | 19 — one-JSON-object stdout, exit codes, stable codes |
| `test_state_store.py` | local state persistence: roundtrip, merge and serialized updates under threads |
| `test_tunnel_stop.py` | tunnel stop identity verification: unsafe/reused PIDs refused, already-exited handling, real command-line reads |
| `test_linux_session.py` | 9-11, 15, 18 — real `flock` daemon semantics (Linux only) |

### Audit regression suites

Each audit finding has a dedicated offline regression file (all use mocks;
none touch the network, a real GPU or a real instance):

| File | Scope |
| --- | --- |
| `test_audit_safety_regressions.py` | paid-before-preflight, rejected platform mutations, NaN/finite, bootstrap whitelist, secret redaction, cold start |
| `test_audit_cli_regressions.py` | CLI wiring: jobs argv, report/download ownership, tunnel argv, install verify session forwarding |
| `test_audit_smoke_regressions.py` | READY strictness: case identity/order, graph contract, cached samplers, no mtime grace, secrets entropy |
| `test_audit_linux_operations.py` | real kernel locks: same-session operation exclusion, inherited FD, overdue without cancellation (Linux only) |
| `test_audit2_platform.py` | create-response classification, post-paid SSH errors, paid-request interrupt handoff, agent-only auth, queue schema, service forwarding, release config, release failure/unknown/not-held outcomes, prepare handoff incl. summary read, configured health/install timeouts |
| `test_audit2_manifests.py` | manifest/URL strictness before payment, run-id and case-selection validation, full smoke-case document validation before payment, report path containment, configured smoke case timeout |
| `test_audit2_controller_locks.py` | per-thread operation contexts, true nesting, uncertain stop/destroy never releasing the lock |
| `test_audit2_installer.py` | remote defense-in-depth manifest validation, atomic no-clobber publication, redaction, empty/duplicate-manifest refusal, install stage deadline incl. the last item |
| `test_audit2_download.py` | download path containment, batch prevalidation, atomic no-clobber promotion |
| `test_audit2_inference.py` | HTTP-inference lifetime vs management process, persistent fence, `resolve-inference` rescue incl. schema-invalid/non-UTF8 records, live-child and finished-job liveness |
| `test_audit2_linux_inference.py` | real FD held until terminal HTTP evidence, crashed watcher fence (Linux only) |

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
