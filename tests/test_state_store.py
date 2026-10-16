"""Local state persistence: atomic updates and no lost records under threads."""
from __future__ import annotations

import threading

from vast_comfy.state import StateStore


def test_state_store_roundtrip(tmp_path):
    store = StateStore(tmp_path)
    store.put("jobs", "job-1", {"job_id": "job-1"})
    store.record_session("s1", instance_id="123")
    assert store.get("jobs", "job-1")["job_id"] == "job-1"
    assert store.get_session("s1")["instance_id"] == "123"
    store.remove("jobs", "job-1")
    assert store.get("jobs", "job-1") is None


def test_record_session_merges_existing_fields(tmp_path):
    store = StateStore(tmp_path)
    store.record_session("s1", instance_id="1")
    store.record_session("s1", state="released")
    record = store.get_session("s1")
    assert record["instance_id"] == "1"
    assert record["state"] == "released"


def test_concurrent_updates_do_not_lose_records(tmp_path):
    # Regression: the read-modify-write of one state file must be serialized;
    # parallel writers previously overwrote each other without any error.
    store = StateStore(tmp_path)
    errors: list[BaseException] = []

    def write(index: int) -> None:
        try:
            store.put("jobs", f"job-{index}", {"index": index})
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert len(store.all("jobs")) == 12
