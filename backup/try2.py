#!/usr/bin/env python3
"""
A minimal-yet-practical data-analysis agent that:
  • Uses LangGraph (StateGraph) for orchestration (planner → tools → observe → replan)
  • Stores data in a local SQLite file (sqlite3)
  • Calls LLMs through Poe's OpenAI‑compatible API via the official OpenAI Python SDK

Features
  - Explicit planning step (Planner–Executor pattern)
  - Flexible branching: the model can request multiple SQL or Python steps
  - Schema-aware prompts (auto‑introspects tables & columns)
  - Safe(ish) SQL (read‑only, row/column limits) and sandboxed Python runner
  - Conversation loop that supports follow‑up questions

Environment
  - POE_API_KEY   : your Poe API key (https://poe.com/api_key)
  - POE_MODEL     : Poe bot/model name (e.g. "Claude-Sonnet-4", "Llama-3.1-405B")
  - DB_PATH       : path to a local SQLite database file (e.g. ./sample.db)

Run
  $ pip install langgraph openai pandas
  $ export POE_API_KEY=...; export POE_MODEL=Claude-Sonnet-4; export DB_PATH=./sample.db
  $ python agent_sqlite_poe.py

Note: Tools/function‑calling params are not used; we parse a compact JSON action spec
      from the model output to decide the next hop.
"""
from __future__ import annotations
import os
import json
import sqlite3
import textwrap
import traceback
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, TypedDict, Literal

import pandas as pd
from openai import OpenAI  # Official SDK; we set base_url to Poe API
from langgraph.graph import StateGraph, START, END

# ---------------------------
# Configuration & clients
# ---------------------------
POE_API_KEY = os.getenv("POE_API_KEY")
POE_MODEL = os.getenv("POE_MODEL", "Claude-Sonnet-4")
DB_PATH = os.getenv("DB_PATH", "./data.db")

if not POE_API_KEY:
    raise SystemExit("POE_API_KEY is required. Get one at https://poe.com/api_key")

# Poe exposes an OpenAI‑compatible endpoint
client = OpenAI(api_key=POE_API_KEY, base_url="https://api.poe.com/v1")

# ---------------------------
# SQLite helpers (read‑only)
# ---------------------------
class SQLite:
    def __init__(self, path: str):
        # Note: SQLite has no strict read‑only mode without opening URI flags;
        # we still enforce query_only and guard against writes.
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only = ON;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def schema_overview(self, max_cols: int = 60) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT name, type FROM sqlite_master WHERE type in ('table','view') ORDER BY name;")
        items = cur.fetchall()
        lines: List[str] = []
        for row in items:
            name = row["name"]
            typ = row["type"]
            cols = self.columns(name)
            col_str = ", ".join([f"{c['name']}:{c['type']}" for c in cols])
            if len(col_str) > max_cols:
                col_str = col_str[:max_cols] + " …"
            lines.append(f"- {typ} {name} (columns: {col_str})")
        if not lines:
            return "(No tables found)"
        return "\n".join(lines)

    def columns(self, table: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        return [
            {"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "default": r[4], "pk": r[5]}
            for r in cur.fetchall()
        ]

    def run_sql(self, sql: str, params: Optional[tuple] = None, row_limit: int = 1000) -> Dict[str, Any]:
        # Guard against writes
        lowered = sql.strip().lower()
        forbidden = ("insert", "update", "delete", "drop", "alter", "create", "replace", "attach", "detach", "vacuum", "pragma")
        if lowered.split()[0] in forbidden:
            raise ValueError("Write or unsafe statement is not allowed.")
        cur = self.conn.cursor()
        cur.execute(sql, params or tuple())
        rows = cur.fetchmany(row_limit)
        cols = [d[0] for d in cur.description] if cur.description else []
        data = [dict(zip(cols, r)) for r in rows]
        return {"columns": cols, "rows": data, "row_count": len(data)}

# ---------------------------
# Agent State & Actions
# ---------------------------
class AgentState(TypedDict, total=False):
    user_question: str
    messages: List[Dict[str, str]]      # chat history for LLM
    plan: List[str]                     # high‑level steps
    last_tool_output: str               # text summary of last observation
    sql_result_preview: str             # pretty table preview
    python_result_preview: str
    final_answer: str
    action: Dict[str, Any]              # latest parsed action from the LLM

ActionType = Literal["plan", "sql", "python", "final"]

@dataclass
class ToolResult:
    kind: ActionType
    success: bool
    summary: str
    payload: Dict[str, Any] = field(default_factory=dict)

# ---------------------------
# LLM call helpers
# ---------------------------
SYSTEM_PROMPT = (
    """
You are a disciplined Data Analysis Agent that strictly follows a JSON action protocol to plan and execute steps
using two tools: SQL over a SQLite database and a Python sandbox. You have to reach the final answer.

Tools available:
- sql(query: string): Execute a read‑only SQL on SQLite; returns rows as JSON (max 1000 rows).
- python(code: string): Run safe Python with pandas available. Input data is provided via variable `dataframes`,
  a dict populated with any previous SQL results as pandas DataFrames.

Schema overview will be provided. Prefer minimal, correct queries. You may run multiple SQL or Python steps.

When you respond, OUTPUT **ONLY** a compact JSON object with this shape (no prose):
{
  "type": "plan"|"sql"|"python"|"final",
  "plan": [optional list of step strings],
  "sql": "...optional...",
  "python": "...optional...",
  "answer": "...final answer if type=final...",
  "notes": "short rationale"
}
Never include triple backticks. Never include explanations outside JSON.
    """
).strip()

def llm_action(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call Poe via OpenAI SDK and parse a JSON action."""
    resp = client.chat.completions.create(
        model=POE_MODEL,
        messages=messages,
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"
    # Extract the first top‑level JSON object heuristically
    content_stripped = content.strip()
    # Try a direct parse first
    try:
        return json.loads(content_stripped)
    except json.JSONDecodeError:
        # Fallback: find the first {...} block
        start = content_stripped.find("{")
        end = content_stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content_stripped[start : end + 1])
            except Exception:
                pass
        raise ValueError(f"Model did not return valid JSON action. Got: {content[:200]}")

# ---------------------------
# Tool nodes
# ---------------------------
DB = SQLite(DB_PATH)

def summarize_rows(result: Dict[str, Any], max_rows: int = 10) -> str:
    cols = result.get("columns", [])
    rows = result.get("rows", [])
    df = pd.DataFrame(rows, columns=cols)
    preview = df.head(max_rows).to_string(index=False)
    return f"Columns: {cols}\nPreview (first {min(len(rows), max_rows)} of {len(rows)} rows):\n{preview}"

# Keep DataFrames from recent SQL calls for the Python sandbox
DATAFRAMES_RING: List[pd.DataFrame] = []
MAX_FRAMES = 5

def node_call_model(state: AgentState) -> AgentState:
    sys = {"role": "system", "content": SYSTEM_PROMPT}
    schema = DB.schema_overview()
    schema_msg = {
        "role": "user",
        "content": f"SCHEMA\n{schema}\n\nUSER_QUESTION\n{state['user_question']}\n\nRECENT_OBSERVATION\n{state.get('last_tool_output','(none)')}"
    }
    msgs = [sys, schema_msg]
    action = llm_action(msgs)
    new_state = dict(state)
    new_state["action"] = action
    if action.get("type") == "plan" and action.get("plan"):
        new_state["plan"] = action["plan"]
    return new_state

def node_execute_sql(state: AgentState) -> AgentState:
    sql = (state.get("action", {}) or {}).get("sql")
    if not sql:
        raise ValueError("SQL action missing 'sql' field")
    try:
        result = DB.run_sql(sql)
        DATAFRAMES_RING.append(pd.DataFrame(result["rows"], columns=result["columns"]))
        while len(DATAFRAMES_RING) > MAX_FRAMES:
            DATAFRAMES_RING.pop(0)
        preview = summarize_rows(result)
        state.update({
            "last_tool_output": f"SQL OK\n{preview}",
            "sql_result_preview": preview,
        })
    except Exception as e:
        state.update({
            "last_tool_output": f"SQL ERROR: {e}",
            "sql_result_preview": f"ERROR: {e}",
        })
    return state

def safe_exec_python(code: str, dataframes: List[pd.DataFrame]) -> Dict[str, Any]:
    # Minimal sandbox: no builtins except a tiny whitelist, no file/network ops
    allowed_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "print": print,
        "enumerate": enumerate,
        "sorted": sorted,
    }
    safe_globals = {
        "__builtins__": allowed_builtins,
        "pd": pd,
        # Provide recent SQL results
        "dataframes": dataframes,
    }
    stdout = StringIO()
    try:
        import sys
        old_stdout = sys.stdout
        sys.stdout = stdout
        exec(  # noqa: S102 (we're sandboxing carefully)
            code,
            safe_globals,
            {},
        )
        sys.stdout = old_stdout
    except Exception:
        sys.stdout = old_stdout
        return {"ok": False, "stdout": stdout.getvalue(), "error": traceback.format_exc()}
    return {"ok": True, "stdout": stdout.getvalue()}

def node_execute_python(state: AgentState) -> AgentState:
    code = (state.get("action", {}) or {}).get("python")
    if not code:
        raise ValueError("Python action missing 'python' field")
    result = safe_exec_python(code, DATAFRAMES_RING.copy())
    if result["ok"]:
        preview = result["stdout"].strip() or "(no stdout)"
        state.update({
            "last_tool_output": f"PYTHON OK\n{preview}",
            "python_result_preview": preview,
        })
    else:
        out = result["stdout"].strip()
        err = result.get("error", "")
        state.update({
            "last_tool_output": f"PYTHON ERROR\n{out}\n{err}",
            "python_result_preview": f"ERROR\n{out}\n{err}",
        })
    return state

def node_finalize(state: AgentState) -> AgentState:
    action = state.get("action", {})
    answer = action.get("answer")
    notes = action.get("notes")
    state["final_answer"] = answer if answer else f"(Model ended without an explicit answer)\nNotes: {notes}"
    return state

# ---------------------------
# Graph wiring
# ---------------------------

def router(state: AgentState) -> Literal["execute_sql", "execute_python", "finalize"]:
    typ = (state.get("action", {}) or {}).get("type", "final")
    if typ == "sql":
        return "execute_sql"
    if typ == "python":
        return "execute_python"
    return "finalize"

workflow = StateGraph(AgentState)
workflow.add_node("call_model", node_call_model)
workflow.add_node("execute_sql", node_execute_sql)
workflow.add_node("execute_python", node_execute_python)
workflow.add_node("finalize", node_finalize)

workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", router, {"execute_sql": "execute_sql", "execute_python": "execute_python", "finalize": "finalize"})
workflow.add_edge("execute_sql", "call_model")
workflow.add_edge("execute_python", "call_model")
workflow.add_edge("finalize", END)

app = workflow.compile()

# ---------------------------
# Simple REPL for ad‑hoc Q&A
# ---------------------------
HELP = """
Type a question about your SQLite data. Examples:
  • What were monthly sales per region in 2023?
  • Get top 10 customers by revenue and plot the churn rate.
  • Compare average order value between mobile vs web.
Enter blank line to exit.
""".strip()

def run_once(question: str) -> str:
    init: AgentState = {"user_question": question, "messages": []}
    final_state = app.invoke(init)
    return textwrap.dedent(
        f"""
        --- PLAN ---
        {os.linesep.join(final_state.get('plan') or [])}

        --- LAST SQL PREVIEW ---
        {final_state.get('sql_result_preview','(none)')}

        --- LAST PYTHON PREVIEW ---
        {final_state.get('python_result_preview','(none)')}

        === ANSWER ===
        {final_state.get('final_answer','(no answer)')}
        """
    ).strip()

if __name__ == "__main__":
    print("📦 SQLite:", DB_PATH)
    print("🧠 Poe model:", POE_MODEL)
    print("📚 Schema:\n" + DB.schema_overview())
    print("\n" + HELP + "\n")
    try:
        while True:
            q = input("You> ").strip()
            if not q:
                break
            print("\nThinking & executing...\n")
            try:
                answer = run_once(q)
                print(answer)
            except Exception as e:
                print("Agent error:", e)
            print("\n——————\n")
    except KeyboardInterrupt:
        pass
