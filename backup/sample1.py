from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI  # You can swap to open-source LLM wrapper
import pandas as pd
import json
from datetime import datetime

# -------------------------
# Simulated dataset (replace with DB query)
# -------------------------
data = pd.DataFrame({
    "date": pd.date_range("2025-08-01", periods=10, freq="D"),
    "offence_type": ["illegal dumping", "littering", "illegal dumping", "noise"] * 2 + ["littering", "noise"],
    "location": ["Loc A", "Loc B", "Loc C", "Loc A", "Loc C", "Loc A", "Loc B", "Loc C", "Loc A", "Loc B"]
})

# -------------------------
# Tool: Data Retrieval
# -------------------------
def get_data_between_dates(start_date: str, end_date: str) -> str:
    """
    Retrieve tabular offence records for a given date range.
    Use this tool when the user asks for data filtered between a start and end date.
    
    Inputs:
    - start_date: string in YYYY-MM-DD format
    - end_date: string in YYYY-MM-DD format

    Output:
    - JSON string with:
        columns: list of column names
        rows: list of row values
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD."})
    
    filtered = data[(data["date"] >= start) & (data["date"] <= end)]
    return json.dumps({
        "columns": list(filtered.columns),
        "rows": filtered.astype(str).values.tolist()
    })

# -------------------------
# Tool: Python Analysis
# -------------------------
def analyze_data_with_python(json_data: str, python_code: str) -> str:
    """
    Perform custom data analysis on the provided dataset using Python & Pandas.
    Use this after retrieving data when calculations, aggregations, or trend analysis are required.
    
    Inputs:
    - json_data: JSON string with 'columns' and 'rows' from get_data_between_dates
    - python_code: Python code that processes a Pandas DataFrame 'df' and prints or returns a result
    
    Output:
    - String representation of the analysis result
    """
    try:
        dataset = json.loads(json_data)
        df = pd.DataFrame(dataset["rows"], columns=dataset["columns"])
        local_vars = {"df": df, "pd": pd}
        exec(python_code, {}, local_vars)
        return str(local_vars.get("result", "No result variable found"))
    except Exception as e:
        return f"Error during analysis: {e}"

# -------------------------
# Wrap tools for LangChain
# -------------------------
tools = [
    Tool.from_function(
        func=get_data_between_dates,
        name="GetDataBetweenDates",
        description=get_data_between_dates._doc_
    ),
    Tool.from_function(
        func=analyze_data_with_python,
        name="AnalyzeDataWithPython",
        description=analyze_data_with_python._doc_
    )
]

# -------------------------
# LLM (swap with Open Source for PoC)
# -------------------------
llm = OpenAI(temperature=0)  # Replace with local model wrapper

# -------------------------
# Initialize agent
# -------------------------
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",  # Chooses tools based on descriptions
    verbose=True
)

# -------------------------
# Example queries
# -------------------------
agent.run("How many offence records happened between 2025-08-02 and 2025-08-06?")
agent.run("List the unique offence types recorded between 2025-08-01 and 2025-08-05")