from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from sse_starlette import EventSourceResponse
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from main import get_agent_executor, get_generated_images, clear_generated_images 
from dotenv import load_dotenv

load_dotenv()

# app = FastAPI()

# origins = [
#     "*"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins, 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],      
# )

agent = get_agent_executor()

async def event_generator(messages):
    # Stream the agent output
    agent_executor = agent["executor"]
    system_prompt = agent["system_prompt"]

    final_messages = [system_prompt] + messages

    async for chunk, _ in agent_executor.astream(
        {
            "messages": final_messages,
            "reasoning": {
                "enabled": True
            }
        },
        stream_mode="messages"
    ):
        text = ""
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            if len(chunk.tool_calls) > 0:
                tool_name = chunk.tool_calls[0]["name"]
                text = "\n\n[Executing Tool...]\n\n"
                if tool_name == "execute_sql":
                    text = "\n\n[Retrieving Data...]\n\n"
                elif tool_name == "execute_python":
                    text = "\n\n[Executing Python...]\n\n"
                else:
                    text = "\n\n[Executing Tool...]\n\n"
        elif hasattr(chunk, "tool_call_id") and chunk.tool_call_id:
            tool_name = chunk.name
            text = "\n\n[Execution completed]\n\n"
            if tool_name == "execute_sql":
                text = "\n\n[Data retrieved]\n\n"
            elif tool_name == "execute_python":
                text = "\n\n[Python executed]\n\n"
            else:
                text = "\n\n[Execution completed]\n\n"
        else:
            text = chunk.content

        if len(text) > 0:
            print(text)
            yield f"{json.dumps({'choices':[{'delta':{'content': text}}]})}"

    # Signal end of stream
    images = get_generated_images()
    if len(images) > 0:
        for image in images:
            md_image = f"\n![Chart]({image})\n"
            yield f"{json.dumps({'choices':[{'delta':{'content': md_image}}]})}"

    clear_generated_images()

    yield "[DONE]"

# @app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    body = await request.json()
    messages = body["messages"]
    stream = request.get("stream", False)

    return EventSourceResponse(event_generator(messages))
    # else:
    #     # Non-streaming mode
    #     output = await run_agent(messages, stream=False)
    #     return {
    #         "id": "cmpl-123",
    #         "object": "chat.completion",
    #         "choices": [{"message": {"role": "assistant", "content": output}}],
    #     }

middleware = [
    Middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=["*"], allow_credentials=True)
]

app = Starlette(routes=[
    Route("/v1/chat/completions", chat_completions, methods=["POST"])
], middleware=middleware)
