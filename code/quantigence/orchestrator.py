"""The Quantigence orchestrator and the three evaluation conditions.

- run_zeroshot:     single model call, no tools, no roles (baseline 1).
- run_single_agent: one generalist agent with the tool loop (baseline 2).
- run_quantigence:  supervisor decomposes -> role workers execute with review/
                    retry -> supervisor synthesizes (Algorithm 1 of the paper).

All three return a RunResult with the fields the eval harness scores.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from . import personas
from .llm import LlamaClient
from .registry import Registry

MAX_TOOL_CALLS = 6

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "role": {"type": "string", "enum": list(personas.WORKERS)},
                    "task": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["id", "role", "task", "depends_on"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "critique": {"type": "string"},
    },
    "required": ["verdict", "critique"],
    "additionalProperties": False,
}


@dataclass
class RunResult:
    answer: str
    condition: str
    sources: list[str] = field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    plan: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0


def _assistant_dict(msg) -> dict:
    """Convert an openai message object to a re-sendable dict."""
    d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d


def _validate_args(name: str, args: dict, schemas: list[dict]) -> str | None:
    """Return an error string if args are invalid, else None."""
    spec = next((s["function"]["parameters"] for s in schemas
                 if s["function"]["name"] == name), None)
    if spec is None:
        return f"unknown tool {name}"
    try:
        jsonschema.validate(args, spec)
        return None
    except jsonschema.ValidationError as e:
        return f"invalid arguments: {e.message}"


def agent_loop(client: LlamaClient, system: str, user: str, reg: Registry,
               max_tool_calls: int = MAX_TOOL_CALLS) -> tuple[str, list[dict]]:
    """Run one agent with tool access until it produces a final answer."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    trace: list[dict] = []
    budget = 9000  # cap cumulative tool-output chars so context stays bounded
    for _ in range(max_tool_calls + 1):
        if budget <= 0:  # spent the tool budget: force a final answer
            messages.append({"role": "user",
                             "content": "Stop using tools and give your final answer now."})
            return client.complete_text(messages), trace
        try:
            msg = client.chat(messages, tools=reg.schemas())
        except Exception:
            # Server-side tool-call parse failures (truncated JSON) are transient;
            # fall back to a tool-free final answer rather than aborting the run.
            messages.append({"role": "user",
                             "content": "Answer now without calling any tools."})
            try:
                return client.complete_text(messages), trace
            except Exception:
                return "", trace
        if not getattr(msg, "tool_calls", None):
            return msg.content or "", trace
        messages.append(_assistant_dict(msg))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                result = {"error": "arguments were not valid JSON"}
            else:
                err = _validate_args(name, args, reg.schemas())
                if err:
                    result = {"error": err}
                else:
                    try:
                        result = reg.dispatch(name, args)
                    except Exception as e:  # tool failure -> tell the model, don't crash
                        result = {"error": f"tool failed: {e}"}
            content = json.dumps(result)[:2000]
            budget -= len(content)
            trace.append({"tool": name, "args": tc.function.arguments})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    # Ran out of tool budget: force a final answer with no tools.
    messages.append({"role": "user",
                     "content": "Stop using tools and give your final answer now."})
    return client.complete_text(messages), trace


# --- Condition 1: zero-shot -------------------------------------------------

def run_zeroshot(client: LlamaClient, query: str) -> RunResult:
    t0 = time.time()
    client.n_calls = 0
    ans = client.complete_text([
        {"role": "system", "content": personas.GENERALIST},
        {"role": "user", "content": query},
    ])
    return RunResult(answer=ans, condition="zeroshot", llm_calls=client.n_calls,
                     elapsed_s=time.time() - t0)


# --- Condition 2: single agent + tools --------------------------------------

def run_single_agent(client: LlamaClient, query: str, reg: Registry) -> RunResult:
    t0 = time.time()
    client.n_calls = 0
    ans, trace = agent_loop(client, personas.GENERALIST, query, reg)
    return RunResult(answer=ans, condition="single_agent", sources=list(reg.sources),
                     tool_calls=len(reg.calls), llm_calls=client.n_calls,
                     trace=trace, elapsed_s=time.time() - t0)


# --- Condition 3: full Quantigence (Algorithm 1) ----------------------------

def _decompose(client: LlamaClient, query: str) -> list[dict]:
    msgs = [
        {"role": "system", "content": personas.SUPERVISOR},
        {"role": "user", "content":
            f"Decompose this query into 1-4 focused sub-tasks, each assigned to one "
            f"specialist role ({', '.join(personas.WORKERS)}). Use depends_on to order "
            f"tasks that need earlier results (e.g. risk_assessor depends on the "
            f"analysts). Query: {query}"},
    ]
    try:
        plan = client.complete_json(msgs, PLAN_SCHEMA).get("tasks", [])
    except Exception:
        plan = []
    if not plan:  # fallback: a single generalist task keeps the query answerable
        plan = [{"id": 0, "role": "standards_specialist", "task": query, "depends_on": []}]
    return plan[:4]


def _review(client: LlamaClient, task: str, result: str) -> dict:
    msgs = [
        {"role": "system", "content": personas.SUPERVISOR},
        {"role": "user", "content":
            f"Sub-task: {task}\n\nSpecialist's answer:\n{result}\n\nReview it. "
            f"Verdict 'fail' only if it is clearly wrong, unsupported, or ignores "
            f"the task; otherwise 'pass'. Give a one-line critique."},
    ]
    try:
        return client.complete_json(msgs, REVIEW_SCHEMA)
    except Exception:
        return {"verdict": "pass", "critique": "review unavailable"}


def _ordered(plan: list[dict]) -> list[dict]:
    """Return tasks in dependency order (stable; tolerates missing deps)."""
    done: set[int] = set()
    out: list[dict] = []
    remaining = list(plan)
    while remaining:
        progressed = False
        for t in list(remaining):
            if all(d in done for d in t.get("depends_on", [])):
                out.append(t)
                done.add(t["id"])
                remaining.remove(t)
                progressed = True
        if not progressed:  # dependency cycle / dangling dep: append the rest as-is
            out.extend(remaining)
            break
    return out


def run_quantigence(client: LlamaClient, query: str, reg: Registry) -> RunResult:
    t0 = time.time()
    client.n_calls = 0
    plan = _decompose(client, query)
    memory: dict[int, dict] = {}
    trace: list[dict] = []

    for task in _ordered(plan):
        role = task["role"]
        system = personas.WORKERS.get(role, personas.GENERALIST)
        context = "\n".join(
            f"[From {memory[d]['role']}]: {memory[d]['answer'][:600]}"
            for d in task.get("depends_on", []) if d in memory)
        # Workers see the original query (so concrete details like risk parameters
        # are never lost in the supervisor's paraphrase) plus their sub-task.
        parts = [f"Overall user query: {query}", f"Your specific sub-task: {task['task']}"]
        if context:
            parts.append(f"Relevant findings from the team so far:\n{context}")
        user = "\n\n".join(parts)

        answer, ttrace = agent_loop(client, system, user, reg)
        review = _review(client, task["task"], answer)
        if review["verdict"] == "fail":
            retry_user = f"{user}\n\nSupervisor feedback: {review['critique']}\nRevise."
            answer, ttrace2 = agent_loop(client, system, retry_user, reg)
            ttrace += ttrace2
        memory[task["id"]] = {"role": role, "answer": answer}
        trace.append({"task": task["task"], "role": role,
                      "review": review["verdict"], "tools": ttrace})

    findings = "\n\n".join(f"### {m['role']}\n{m['answer'][:1200]}" for m in memory.values())
    try:
        synthesis = client.complete_text([
            {"role": "system", "content": personas.SUPERVISOR},
            {"role": "user", "content":
                f"Query: {query}\n\nTeam findings:\n{findings}\n\nWrite the final, "
                f"integrated answer. Be concise and cite the sources the team used."},
        ], max_tokens=1500)
    except Exception:  # fall back to concatenated findings if synthesis fails
        synthesis = findings

    return RunResult(answer=synthesis, condition="quantigence", sources=list(reg.sources),
                     tool_calls=len(reg.calls), llm_calls=client.n_calls, plan=plan,
                     trace=trace, elapsed_s=time.time() - t0)


CONDITIONS = {
    "zeroshot": run_zeroshot,
    "single_agent": run_single_agent,
    "quantigence": run_quantigence,
}
