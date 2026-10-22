"""JSON-schema constrained LLM calls with pluggable backends.

LLM_BACKEND selects how the model is reached (all take images + a JSON schema):
  claude-cli     Claude Code headless (`claude -p`), minimal setup: no CLAUDE.md / MCP / tools.
                 Auth: CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) or an existing login.
  codex-cli      OpenAI Codex CLI (`codex exec`), e.g. GPT-6. Auth: `codex login`.
  anthropic-api  Claude API via the Anthropic SDK. Auth: ANTHROPIC_API_KEY.
  vertex         Claude on Vertex AI. Auth: GCP ADC + ANTHROPIC_VERTEX_PROJECT_ID.
  relay          Hand the call to scripts/relay_worker.py through GCS (RELAY_GCS); the worker
                 runs one of the backends above where it is logged in. No credentials on this host.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_CODEX_MODEL = "gpt-6-astra"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CLI_TIMEOUT = 180

_client = None


def backend() -> str:
    return os.environ.get("LLM_BACKEND", "claude-cli")


def model_from_env(var: str) -> str:
    if backend() == "codex-cli" or (backend() == "relay" and os.environ.get("RELAY_BACKEND") == "codex-cli"):
        return os.environ.get(var + "_CODEX") or DEFAULT_CODEX_MODEL
    return os.environ.get(var) or DEFAULT_MODEL


def _png_bytes(img: np.ndarray | Image.Image, max_side: int = 512) -> bytes:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_block(img: np.ndarray | Image.Image, max_side: int = 512) -> dict:
    """RGB array / PIL image -> base64 PNG content block (Messages API shape)."""
    data = base64.standard_b64encode(_png_bytes(img, max_side)).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def call_json(
    system: str,
    content: list[dict] | str,
    schema: dict,
    model: str | None = None,
    effort: str = "low",
    max_tokens: int = 4000,
) -> tuple[dict, dict]:
    """Call the configured backend; returns (parsed_json, meta) where meta is for logging."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    t0 = time.time()
    fn = {"claude-cli": _call_claude_cli, "codex-cli": _call_codex_cli,
          "anthropic-api": _call_sdk, "vertex": _call_sdk, "relay": _call_relay}[backend()]
    out, meta = fn(system, content, schema, model, effort, max_tokens)
    meta.update({"backend": backend(), "sec": round(time.time() - t0, 2)})
    return out, meta


def call_backend(name: str, system, content, schema, model=None, effort="low", max_tokens=4000):
    """Run one call on an explicit backend (used by the relay worker)."""
    prev = os.environ.get("LLM_BACKEND")
    os.environ["LLM_BACKEND"] = name
    try:
        return call_json(system, content, schema, model, effort, max_tokens)
    finally:
        if prev is None:
            os.environ.pop("LLM_BACKEND", None)
        else:
            os.environ["LLM_BACKEND"] = prev


def _call_relay(system, content, schema, model, effort, max_tokens):
    from . import relay

    return relay.call(system, content, schema, model, effort, max_tokens)


# ---- Claude Code headless ---------------------------------------------------------------

def _call_claude_cli(system, content, schema, model, effort, max_tokens):
    model = model or DEFAULT_MODEL
    with tempfile.TemporaryDirectory() as tmp:
        mcp = Path(tmp) / "mcp.json"
        mcp.write_text('{"mcpServers":{}}')
        cmd = [
            "claude", "-p", "--model", model,
            # strip Claude Code's own context (~55k tokens) so a call costs about what the API does
            "--system-prompt", system,
            "--setting-sources", "", "--strict-mcp-config", "--mcp-config", str(mcp),
            "--tools", "", "--no-session-persistence",
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--json-schema", json.dumps(schema),
        ]
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        proc = subprocess.run(cmd, input=json.dumps(msg) + "\n", capture_output=True, text=True,
                              timeout=CLI_TIMEOUT, cwd=tmp)
    result = None
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            result = ev
    if result is None or result.get("is_error") or result.get("structured_output") is None:
        detail = (result or {}).get("result") or proc.stderr[-500:] or proc.stdout[-500:]
        raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {detail}")
    usage = result.get("usage", {})
    meta = {"model": model, "cost_usd": result.get("total_cost_usd"),
            "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0),
            "output_tokens": usage.get("output_tokens")}
    return result["structured_output"], meta


# ---- Codex CLI (GPT) ----------------------------------------------------------------------

def _call_codex_cli(system, content, schema, model, effort, max_tokens):
    model = model or DEFAULT_CODEX_MODEL
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "schema.json").write_text(json.dumps(schema))
        images, parts = [], [system, ""]
        for block in content:  # images go as attachments, referenced by order in the text
            if block["type"] == "image":
                p = tmp / f"img{len(images)}.png"
                p.write_bytes(base64.b64decode(block["source"]["data"]))
                images.append(str(p))
                parts.append(f"[attached image {len(images)}]")
            else:
                parts.append(block["text"])
        cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "-m", model,
               "--output-schema", str(tmp / "schema.json"), "-o", str(tmp / "out.json")]
        for p in images:
            cmd += ["-i", p]
        # `-i` takes several files, so a trailing positional prompt would be swallowed as an image;
        # pass the prompt on stdin instead.
        proc = subprocess.run(cmd, input="\n".join(parts), capture_output=True, text=True,
                              timeout=CLI_TIMEOUT, cwd=tmp)
        out_path = tmp / "out.json"
        if proc.returncode != 0 or not out_path.exists():
            # stderr echoes the whole prompt; keep only the error lines
            errs = [l for l in proc.stderr.splitlines() if l.startswith("ERROR")] or [proc.stderr[-300:]]
            raise RuntimeError(f"codex exec failed (rc={proc.returncode}): {' | '.join(dict.fromkeys(errs))}")
        return json.loads(out_path.read_text()), {"model": model}


# ---- Anthropic SDK (Claude API / Vertex) --------------------------------------------------

def client():
    global _client
    if _client is None:
        import anthropic
        if backend() == "vertex":
            _client = anthropic.AnthropicVertex(
                project_id=os.environ["ANTHROPIC_VERTEX_PROJECT_ID"],
                region=os.environ.get("CLOUD_ML_REGION", "global"),
            )
        else:
            _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY or `ant auth login` profile
    return _client


def _call_sdk(system, content, schema, model, effort, max_tokens):
    model = model or DEFAULT_MODEL
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if backend() == "vertex":
        # server-side `fallbacks` is not available on Vertex; a refusal raises below
        resp = client().messages.create(**kwargs)
    else:
        resp = client().beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
    meta = {
        "model": resp.model,
        "stop_reason": resp.stop_reason,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"LLM refused the request: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("LLM output hit max_tokens before finishing the JSON")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), meta
