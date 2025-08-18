from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from sse_starlette.sse import EventSourceResponse
from main import get_agent_executor, get_generated_images

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],      
)

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request["messages"]
    stream = request.get("stream", False)

    async def event_generator(messages):
        # Stream the agent output
        agent = get_agent_executor()
        agent_executor = agent["executor"]
        system_prompt = agent["system_prompt"]

        final_messages = [system_prompt] + messages

        for chunk, _ in agent_executor.stream(
            {
                "messages": final_messages,
                "reasoning": {
                    "enabled": True
                }
            },
            stream_mode="messages"
        ):
            if hasattr(chunk, "tool_call_id"):
                text = "[Executing Tool]\n\n"
            else:
                text = chunk.content

            if len(text) > 0:
                yield f"{json.dumps({'choices':[{'delta':{'content': text}}]})}"

        # Signal end of stream
        images = get_generated_images()
        if len(images) > 0:
            for image in images:
                md_image = f"\n![Chart]({image})\n"
                yield f"{json.dumps({'choices':[{'delta':{'content': md_image}}]})}"

        yield "[DONE]"

    if stream:
        return EventSourceResponse(event_generator(messages))
    # else:
    #     # Non-streaming mode
    #     output = await run_agent(messages, stream=False)
    #     return {
    #         "id": "cmpl-123",
    #         "object": "chat.completion",
    #         "choices": [{"message": {"role": "assistant", "content": output}}],
    #     }