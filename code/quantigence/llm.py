"""Thin transport to a local llama-server (OpenAI-compatible).

Deliberately minimal: the orchestrator owns the tool-execution loop; this class
only sends chat requests and returns the assistant message. Two extras that
small local models need are baked in: `enable_thinking:false` (avoids the
thinking+tool-call parser edge cases in Qwen3.5) and a `complete_json` helper
that uses server-side json_schema grammar constraint for reliable structured
output.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


class LlamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1",
                 model: str = "local", temperature: float = 0.0,
                 thinking: bool = False):
        self.client = OpenAI(base_url=base_url, api_key="sk-none", timeout=600)
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.n_calls = 0  # cheap usage counter for the eval harness

    def _extra_body(self) -> dict[str, Any]:
        # Qwen3.5 tool-calling is most reliable with thinking disabled.
        return {"chat_template_kwargs": {"enable_thinking": self.thinking}}

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 2048, response_format: dict | None = None):
        """Return the raw assistant message (has .content and .tool_calls)."""
        self.n_calls += 1
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "extra_body": self._extra_body(),
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message

    def complete_text(self, messages: list[dict], max_tokens: int = 2048) -> str:
        return self.chat(messages, max_tokens=max_tokens).content or ""

    def complete_json(self, messages: list[dict], schema: dict,
                      max_tokens: int = 2048) -> dict:
        """Grammar-constrained structured output. `schema` is a JSON Schema."""
        rf = {"type": "json_schema",
              "json_schema": {"name": "response", "schema": schema, "strict": True}}
        raw = self.chat(messages, max_tokens=max_tokens, response_format=rf).content or "{}"
        return json.loads(raw)

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Requires a running llama-server. Skips cleanly if none is up.
    import sys

    c = LlamaClient()
    if not c.health():
        print("no llama-server at :8080 — start one to run this check; skipping")
        sys.exit(0)

    txt = c.complete_text([{"role": "user", "content": "Reply with exactly: PONG"}],
                          max_tokens=16)
    print("text:", txt.strip()[:40])

    out = c.complete_json(
        [{"role": "user", "content": "Return JSON with the capital of France."}],
        schema={"type": "object", "properties": {"capital": {"type": "string"}},
                "required": ["capital"], "additionalProperties": False})
    assert out.get("capital", "").lower().startswith("paris"), out
    print("json:", out)
    print("llm self-check passed")
