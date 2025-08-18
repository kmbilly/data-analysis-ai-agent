#!/usr/bin/env python3
import os
from typing import Any, Dict, List, Optional, TypedDict, Literal
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import pandas as pd
from database import Database
from python_sandbox import safe_exec_python
from langchain.callbacks.base import BaseCallbackHandler
import json

load_dotenv()

# ---------------------------
# Configuration & clients
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.poe.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
DB_PATH = os.getenv("DB_PATH", "./data.db")

DB = Database(DB_PATH)

class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        print("=== LLM RAW RESPONSE ===")
        print(response)

# llm = ChatOllama(
#     model=LLM_MODEL, 
#     temperature=0,
#     streaming=True,
# )
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY, 
    base_url=OPENAI_API_BASE,
    temperature=0,
    streaming=True,
    # callbacks=[LoggingCallbackHandler()],
    # model_kwargs={"function_call": "auto"}
    # model_kwargs={
    #     "reasoning": {"effort": "high"}
    # }
)

# Keep DataFrames from recent SQL calls for the Python sandbox
DATAFRAMES_RING: List[pd.DataFrame] = []
GENERATED_IMAGES: List[str] = []

# def showImage(base64Img:str):
#     GENERATED_IMAGES.append(str)

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
        print(f"Tool - Executing SQL: {sql}\n")

        result = {}
        try:
            result = DB.run_sql(sql)
            print(f"Tool - SQL result:\n{result}\n-----\n")

            result["sql"] = sql
            DATAFRAMES_RING.append(pd.DataFrame(result["rows"], columns=result["columns"]))
            # preview = summarize_rows(result)
        except Exception as e:
            result = {
                "sql": sql,
                "error": f"SQL ERROR: {e}",
            }

        results.append(result)

    return results

@tool
def execute_python(code: str) -> str:
    """Execute a safe Python script, print out results or save any image to image_bytes.

    A variable `dataframes` is populated with a list of any previous SQL results as pandas DataFrames.
    A variable `image_bytes` is provided to save any generated image in PNG format, e.g.
        plt.savefig(image_bytes, format='png')
    
    DO NOT print any image in any format.  DO NOT call plt.show() as it WOULD NOT work.
        
    The following packages are available.

    - pd
    - numpy
    - matplotlib
    - seaborn
    - scipy
    - statsmodels
    - sklearn
    - PIL
    - pyarrow

    Args:
        code: Python script to execute
    Return:
        The print out result
    """
    if not code:
        raise ValueError("Python action missing 'python' field")
    
    print(f"[Executing Python]\n{code}\n", flush=True)

    result = ""

    import io
    import base64
    image_bytes = io.BytesIO()

    execResult = safe_exec_python(code, DATAFRAMES_RING.copy(), image_bytes)
    if execResult["ok"]:
        result = execResult["stdout"].strip()
        # if len(GENERATED_IMAGES) > 0:
        if image_bytes.getbuffer().nbytes > 0:
            image_bytes.seek(0)
            base64_img = base64.b64encode(image_bytes.read()).decode('utf-8')
            GENERATED_IMAGES.append("data:image/png;base64," + base64_img)
            result = "Image has been shown to the user.\n" + result
    else:
        out = execResult["stdout"].strip()
        err = execResult.get("error", "")
        result = f"PYTHON ERROR\n{out}\n{err}"

    print(f"[Python result]\n{result}\n-----\n", flush=True)

    return result

# @tool
# def show_plotly_json(plotly_json) -> str:
#     """Show a chart in Plotly JSON format.

#     Args:
#         plotly_json: Plotly JSON string for the chart
#     """
#     print(json.dumps(plotly_json))
#     return "Chart has been shown to the user."

tools = [
    execute_sql,
    # show_plotly_json,
    execute_python
]
llm_with_tools = llm.bind_tools(tools)

agent_executor = create_react_agent(llm, tools)


SYSTEM_PROMPT = (
    """
You are a disciplined Data Analysis Agent that answers data analysis questions from a business user.
You have a tool execute_sql to run multiple SQLs over a SQLite database. You can call the tool multiple times.
For complicated calculations, you may use another tool execute_python to execute a Python script. 
If the user asks to generate a chart, you may use execute_python to generate the chart image.
You have to reach the final answer.
Do not mention the existance of database and any SQLs.

Schema overview will be provided. Prefer minimal, correct queries. You may run multiple SQL.
    """
).strip()

def get_agent_executor():
    schema = DB.schema_overview()
    return {
        "executor": agent_executor,
        "system_prompt": {"role": "system", "content": SYSTEM_PROMPT + "\n\nSCHEMA\n" + schema + "\n"}
    }

def get_generated_images():
    return GENERATED_IMAGES

def run_once(question: str) -> str:
    schema = DB.schema_overview()
    system_prompt = {"role": "system", "content": SYSTEM_PROMPT + "\n\nSCHEMA\n" + schema + "\n"}
    user_prompt = {
        "role": "user",
        "content": question
    }

    messages = [
        system_prompt,
        user_prompt,
    ]
    for chunk, _ in agent_executor.stream(
        {
            "messages": messages,
            "reasoning": {
                "enabled": True
            }
        },
        stream_mode="messages"
    ):
        if hasattr(chunk, "tool_call_id"):
            continue  # This chunk is a tool output, skip it
        if hasattr(chunk, "content"):
            print(chunk.content, end="", flush=True)
            # print(str(chunk) + "\n", end="", flush=True)
        else:
            print("Token: " + str(chunk) + "\n", end="", flush=True)

    print("\n")
    if len(GENERATED_IMAGES) > 0:
        print("[Result Image]\n")
        for img in GENERATED_IMAGES:
            print(img)

def run_interactive(question: str, print_handler, img_handler) -> str:
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
        {
            "messages": messages,
            "reasoning": {
                "enabled": True
            }
        },
        stream_mode="messages"
    ):
        if hasattr(chunk, "tool_call_id"):
            continue  # This chunk is a tool output, skip it
        if hasattr(chunk, "content"):
            print_handler(chunk.content, end="", flush=True)
            # print(str(chunk) + "\n", end="", flush=True)
        else:
            print_handler("Token: " + str(chunk) + "\n", end="", flush=True)

    print_handler("\n")
    if len(GENERATED_IMAGES) > 0:
        for img in GENERATED_IMAGES:
            img_handler(img)

def get_help() -> str:
    return "📦 SQLite:" + DB_PATH + "\n" + "🧠 LLM model:" + LLM_MODEL + "\n" + "📚 Schema:\n" + DB.schema_overview() + "\n" + HELP + "\n"

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
  * Can you plot a vertical bar chart for the number of customers per country?
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
