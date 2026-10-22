"""Relay worker: serve LLM requests that the simulator VM puts on GCS.

Run where the CLI is logged in (e.g. your laptop):
  python scripts/relay_worker.py gs://bucket/pai2026/relay
Each request names its backend (claude-cli / codex-cli); see src/dandori/relay.py.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from dandori import llm
from dandori.relay import _bucket_prefix


def handle(bucket, prefix: str, name: str) -> None:
    rid = name.rsplit("/", 1)[-1].removesuffix(".json")
    blob = bucket.blob(name)
    try:
        req = json.loads(blob.download_as_text())
        blob.delete()  # claim it
    except Exception:
        return  # taken by another worker or half-written
    try:
        out, meta = llm.call_backend(req["backend"], req["system"], req["content"], req["schema"],
                                     req.get("model"), req.get("effort", "low"), req.get("max_tokens", 4000))
        resp = {"ok": True, "out": out, "meta": meta}
    except Exception as e:
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1000]}
    bucket.blob(f"{prefix}/resp/{rid}.json").upload_from_string(json.dumps(resp, ensure_ascii=False))
    print(time.strftime("%H:%M:%S"), rid, req["backend"], "ok" if resp["ok"] else resp["error"][:120],
          resp.get("meta", {}).get("sec", ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("relay_gcs")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    bucket, prefix = _bucket_prefix(args.relay_gcs)
    print(f"relay worker on {args.relay_gcs} (Ctrl-C to stop)", flush=True)
    inflight: set[str] = set()
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            try:
                for b in bucket.list_blobs(prefix=f"{prefix}/req/"):
                    if b.name.endswith(".json") and b.name not in inflight:
                        inflight.add(b.name)
                        pool.submit(handle, bucket, prefix, b.name).add_done_callback(
                            lambda _f, n=b.name: inflight.discard(n))
            except Exception as e:  # network blips (DNS, timeouts) must not kill a long eval
                print(time.strftime("%H:%M:%S"), "list failed, retrying:", str(e)[:120], flush=True)
                time.sleep(10)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
