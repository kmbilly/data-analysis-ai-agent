import asyncio
from langchain_sandbox import PyodideSandbox

async def main():
    sandbox = PyodideSandbox(
        # Allow Pyodide to install python packages that
        # might be required.
        allow_read=True,
        allow_net=True,
    )
    code = """\
with open('/tmp/sandbox/test.txt', 'r') as file:
    content = file.read()  # Read entire content as a string

print(content)
    """

    # Execute Python code
    print(await sandbox.execute(code))

if __name__ == "__main__":
    asyncio.run(main())
