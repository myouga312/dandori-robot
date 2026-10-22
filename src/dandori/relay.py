"""GCS relay so the simulator VM can use an LLM CLI that is logged in only on another machine.

VM side  (LLM_BACKEND=relay): put a request object, then poll for the response object.
Worker   (scripts/relay_worker.py, e.g. on a laptop where `claude` / `codex` are logged in):
         pick up requests, call the local backend, write responses.
No credentials ever live on the VM.

Layout under RELAY_GCS (gs://bucket/prefix):
  req/<id>.json   {"system", "content", "schema", "model", "effort", "max_tokens", "backend"}
  resp/<id>.json  {"ok": true, "out": {...}, "meta": {...}}  |  {"ok": false, "error": "..."}
"""
from __future__ import annotations

import json
import os
import time
import uuid

from google.cloud import storage

POLL_SEC = 0.5
TIMEOUT_SEC = 600


def _bucket_prefix(url: str) -> tuple[storage.Bucket, str]:
    assert url.startswith("gs://"), url
    bucket, _, prefix = url[5:].partition("/")
    return storage.Client().bucket(bucket), prefix.rstrip("/")


def call(system, content, schema, model, effort, max_tokens) -> tuple[dict, dict]:
    bucket, prefix = _bucket_prefix(os.environ["RELAY_GCS"])
    rid = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    req = {"system": system, "content": content, "schema": schema, "model": model, "effort": effort,
           "max_tokens": max_tokens, "backend": os.environ.get("RELAY_BACKEND", "claude-cli")}
    bucket.blob(f"{prefix}/req/{rid}.json").upload_from_string(json.dumps(req), content_type="application/json")
    resp_blob = bucket.blob(f"{prefix}/resp/{rid}.json")
    t0 = time.time()
    while time.time() - t0 < TIMEOUT_SEC:
        if resp_blob.exists():
            resp = json.loads(resp_blob.download_as_text())
            resp_blob.delete()
            if not resp.get("ok"):
                raise RuntimeError(f"relay worker error: {resp.get('error')}")
            meta = resp.get("meta", {})
            meta["relay_wait_sec"] = round(time.time() - t0, 2)
            return resp["out"], meta
        time.sleep(POLL_SEC)
    raise TimeoutError(f"no relay response for {rid} within {TIMEOUT_SEC}s (is relay_worker.py running?)")
