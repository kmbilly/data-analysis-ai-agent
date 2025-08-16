#!/usr/bin/env python3
import os
from typing import Any, Dict, List, Optional, TypedDict, Literal
from langchain_ollama import ChatOllama
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
from database import Database

load_dotenv()

# ---------------------------
# Configuration & clients
# ---------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-coder-v2")
DB_PATH = os.getenv("DB_PATH", "./data.db")

DB = Database(DB_PATH)

llm = ChatOllama(
    model=LLM_MODEL, 
    temperature=0,
    streaming=True,
)

@tool
def execute_sql(queries: list[str]) -> list[Dict[str, Any]]:
    """Execute a read‑only SQL on SQLite.

    Args:
        queries: a list of sql query strings to execute.
    Return:
        A list of sql execution results, each is a dictionary with the following keys:
        - sql: the executed SQL query.
        - columns: list of column names.
        - rows: list of rows as dictionaries.
        - row_count: number of rows returned.
    """
    if not queries:
        raise ValueError("SQL action missing 'queries' field")

    results = []
    for sql in queries:
        print(f"Executing SQL: {sql}\n")

        result = {}
        try:
            result = DB.run_sql(sql)
            print(f"SQL result:\n{result}\n-----\n")

            result["sql"] = sql
            # DATAFRAMES_RING.append(pd.DataFrame(result["rows"], columns=result["columns"]))
            # preview = summarize_rows(result)
        except Exception as e:
            result = {
                "sql": sql,
                "error": f"SQL ERROR: {e}",
            }

        results.append(result)

    return results


tools = [
    execute_sql
]
llm_with_tools = llm.bind_tools(tools)

# prompt = hub.pull("wfh/react-agent-executor")
# prompt.pretty_print()

agent_executor = create_react_agent(llm, tools)

SYSTEM_PROMPT = (
    """
You are a disciplined Data Analysis Agent that answers data analysis questions.
You have a tool to run multiple SQLs over a SQLite database. You can call the tool multiple times.
You have to reach the final answer.

Schema overview will be provided. Prefer minimal, correct queries. You may run multiple SQL.
    """
).strip()

def run_once(question: str) -> str:
    system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
    schema = DB.schema_overview()
    user_prompt = {
        "role": "user",
        "content": f"SCHEMA\n{schema}\n\nUSER_QUESTION\n{question}"
    }

    messages = [
        system_prompt,
        user_prompt,
    ]
    for chunk, _ in agent_executor.stream(
        {"messages": messages},
        stream_mode="messages"
    ):
        if hasattr(chunk, "tool_call_id"):
            continue  # This chunk is a tool output, skip it
        if hasattr(chunk, "content"):
            print(chunk.content, end="", flush=True)
            # print(str(chunk) + "\n", end="", flush=True)
        else:
            print("Token: " + str(chunk) + "\n", end="", flush=True)

# ---------------------------
# Simple REPL for ad‑hoc Q&A
# ---------------------------
HELP = """
Type a question about your SQLite data. Examples:
  • What were monthly sales per region in 2023?
  • Get top 10 customers by revenue and plot the churn rate.
  • Compare average order value between mobile vs web.
  * How many customers are there in the country of Brazil?
  * Is artist with names having "Berliner" has relatively high number of albums?
Enter blank line to exit.
""".strip()

if __name__ == "__main__":
    print("📦 SQLite:", DB_PATH)
    print("🧠 LLM model:", LLM_MODEL)
    print("📚 Schema:\n" + DB.schema_overview())
    print("\n" + HELP + "\n")
    try:
        while True:
            q = input("You> ").strip()
            if not q:
                break
            print("\nThinking & executing...\n")
            try:
                run_once(q)
            except Exception as e:
                print("Agent error:", e)
            print("\n——————\n")
    except KeyboardInterrupt:
        pass
