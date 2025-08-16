from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], list]

load_dotenv()

# Read testing database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    # model_name="GPT-OSS-20B",
    model_name="GPT-4o-mini",
    openai_api_base=os.getenv("OPENAI_API_BASE"), 
)

# Initialize SQL tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
# print(tools)

system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect="SQLite",
    top_k=5,
)

checkpointer = InMemorySaver()
agent = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=system_message,
    checkpointer=checkpointer
)

question = "Which country's customers spent the most?"

# for step in agent.stream(
#     {"messages": [{"role": "user", "content": question}]}, 
#     stream_mode="messages"
# ):
#     print(step)

config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config  
)

print(response)