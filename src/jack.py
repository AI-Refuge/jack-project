import chromadb
from chromadbx import NanoIDGenerator
import random
import json
import time
import logging
from datetime import datetime, timezone
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import argparse
import traceback

logging.basicConfig(filename="../conv.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

HTTP_USER_AGENT = "AI: @jack"

# Set your API keys
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

MODELS = [
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
]

parser = argparse.ArgumentParser(description="Jack")
parser.add_argument('-m', '--model', default=MODELS[0], help="LLM Model")
parser.add_argument('-t', '--temperature', default=0, help="Temperature")
parser.add_argument('-w', '--max-tokens', default=4096, help="Max tokens")
parser.add_argument('-g', '--goal', action='store_true', help="Goal mode")
parser.add_argument('-c', '--conversation', default="first", help="Conversation")
args = parser.parse_args()

if args.model == 'list':
    print("> Supported models:")
    for m in MODELS:
        print(f"> {m}")
    exit()

vdb = chromadb.PersistentClient(path="../memory")
memory = vdb.get_or_create_collection("meta")
ts = int(datetime.now(timezone.utc).timestamp())
conv = vdb.get_or_create_collection(f"conv-{args.conversation}")

llm = ChatAnthropic(
    model=args.model,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
)

search = DuckDuckGoSearchRun()
fs_toolkit = FileManagementToolkit()

req_toolkit = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={
        "User-Agent": HTTP_USER_AGENT,
    }),
    allow_dangerous_requests=True,
)

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
pyrepl_tool = PythonREPLTool()

@tool(parse_docstring=True)
def memory_count() -> int:
    """Number of total memories

    Args:
        None

    Returns:
        Number of memories
    """
    return str(memory.count())

@tool(parse_docstring=True)
def memory_insert(
    documents: list[str],
    metadata: None | dict[str, str|int|float] = None,
    timestamp: bool = True
) -> list[str]:
    """Insert new memories

    Args:
        documents: list of text memories
        metadata: Common metadata for the memories (only primitive types allowed for values).
        timestamp: if true, will set a unix float timestamp in metadata

    Returns:
        List of ID's of the documents that were inserted
    """

    ids = list(map(str, NanoIDGenerator(len(documents))))

    if timestamp:
        metadata = metadata or {}
        metadata["timestamp"] = datetime.now(timezone.utc).timestamp()

    metadatas = None
    if metadata:
        metadatas = [metadata for i in range(len(documents))]

    memory.add(ids=ids, documents=documents, metadatas=metadatas)

    return json.dumps(ids)

@tool(parse_docstring=True)
def memory_fetch(ids: list[str]) -> dict[str, dict]:
    """directly fetch specific ID's.

    Args:
        ids: ID's of memory

    Returns:
        List of memories
    """
    return json.dumps(memory.get(ids=ids))

@tool(parse_docstring=True)
def memory_query(
    query_texts: list[str],
    where: None | dict = None,
    count: int = 10
) -> dict[str, dict]:
    """Get nearest neighbor memories for provided query_texts

    Args:
        query_texts: The document texts to get the closes neighbors of.
        where: dict used to filter results by. E.g. {"color" : "red", "price": 4.20}.
        count: number of neighbors to return for each query_texts.

    Returns:
        List of memories
    """
    return json.dumps(memory.query(
        query_texts=query_texts,
        where=where
    ))

@tool(parse_docstring=True)
def memory_update(
    ids: list[str],
    documents: list[str] = None,
    metadata: dict[str, str|int|float] = None,
    timestamp: bool = True
) -> None:
    """Update memories

    Args:
        ids: ID's of memory
        documents: list of text memories
        metadata: Common metadata for the memories
        timestamp: if true, will set a unix float timestamp in metadata

    Returns:
        None
    """
    if timestamp:
        metadata = metadata or {}
        metadata["timestamp"] = datetime.now(timezone.utc).timestamp()

    metadatas = None
    if metadata:
        metadatas = [metadata for i in range(len(ids))]

    return json.dumps(memory.update(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    ))

@tool(parse_docstring=True)
def memory_upsert(
    ids: list[str],
    documents: list[str],
    metadata: None | dict[str, str|int|float] = None,
    timestamp: bool = True
) -> None:
    """Update memories or insert if not existing

    Args:
        ids: ID's of memory
        documents: list of text memories
        metadata: Common metadata for the memories
        timestamp: if true, will set a unix float timestamp in metadata

    Returns:
        None
    """

    if timestamp:
        metadata = metadata or {}
        metadata["timestamp"] = datetime.now(timezone.utc).timestamp()

    metadatas = None
    if metadata:
        metadatas = [metadata for i in range(len(ids))]

    # update if exists or insert
    return json.dumps(memory.update(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    ))

@tool(parse_docstring=True)
def memory_delete(
    ids: list[str] = None,
    where: dict = None
) -> None:
    """Delete memories

    Args:
        ids: Document id's to delete
        where: dict used on metadata to filter results by. E.g. {"color" : "red", "price": 4.20}.

    Returns:
        None
    """

    return json.dumps(memory.delete(ids=ids, where=where))

@tool(parse_docstring=True)
def random_get(count: int = 3) -> list[float]:
    """Get random values between 0.0 <= X < 1.0

    Args:
        count: Number of random values

    Returns:
        list of floating values
    """
    return json.dumps([random.random() for i in range(count)])

@tool(parse_docstring=True)
def datetime_now() -> str:
    """Get the current UTC datetime

    Args:
        None

    Returns:
        String time in ISO 8601 format
    """
    return

@tool(parse_docstring=True)
def script_restart():
    """ There is a bash script that run the script if it exists.
    Use this to reload changes.
    """
    logger.warning("meta:brain executed restart")
    exit()

@tool(parse_docstring=True)
def script_delay(sec: float):
    """Delay execution of the script.
    This is useful when you need to block script execution for somet time

    Args:
        sec: Number of seconds in float

    Returns:
        None
    """
    time.sleep(sec)


tools = [
    memory_count,
    memory_insert,
    memory_fetch,
    memory_query,
    memory_update,
    memory_upsert,
    memory_delete,
] + [
    random_get,
    datetime_now,
    script_restart,
    script_delay,
    search,
    wiki_tool,
    pyrepl_tool,
] + fs_toolkit.get_tools() + req_toolkit.get_tools()

jack = llm.bind_tools(tools)

def conv_print(msg, source="stdout", screen=True, log=True):
    global conv
    conv.add(ids=NanoIDGenerator(1), metadatas=[{
        "source": source,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }], documents=[
        msg
    ])

    if log:
        logger.debug(msg)

    if screen:
        print(msg)

def exception_to_string(exc):
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

def ellipsis(data, max_len=200):
    return (data[:max_len] + '...') if len(data) > max_len else data


SYS_FILES = ('meta.txt', 'home.txt', 'goal.txt', )
FUN_FILES = ('intro.txt', 'thoughts.txt', )
sys_msg = SystemMessage(content='\n'.join([open(x).read() for x in SYS_FILES]))
fun_msg = HumanMessage(content='\n'.join([open(x).read() for x in FUN_FILES]))
chat_history = [sys_msg, fun_msg]
user_turn = True
cycle_num = 0

def main():
    global fun_msg, chat_history, user_turn, cycle_num
    stay = True
    logger.debug(f"Loop cycle {cycle_num}")
    cycle_num += 1

    if user_turn:
        if args.goal:
            conv_print("> Pushing for goal")
            fun_msg = open('goal.txt').read()
            chat_history.append(fun_msg)
        elif fun_msg is None:
            user_input = input("You: ") or "<empty>"
            if user_input.lower() == 'exit':
                stay = False

            fun_msg = HumanMessage(content=user_input)

            logger.debug(fun_msg)
            conv_print(user_input, source="stdin", screen=False, log=False)
            chat_history.append(fun_msg)

    try:
        reply = jack.invoke(chat_history)
    except Exception as e:
        reply = None
        logger.exception("Problem while executing request")
        conv_print(f"> exception happened {ellipsis(str(e))}")

    if reply is None:
        conv_print("> sleeping for 5 seconds as we didnt get reply")
        time.sleep(5)
        return stay

    user_turn = True

    logger.debug(reply)

    if reply.content == "" or len(reply.content) == 0:
        return stay

    # the message has been accepted
    fun_msg = None
    chat_history.append(reply)

    for tool_call in reply.tool_calls:
        tool_name = tool_call['name'].lower()

        if tool_name == 'script_restart':
            # Handle the exit here
            conv_print("> @jack want to restart the script")
            stay = False

        conv_print(f"> Tool used: {tool_name} {tool_call['args']}")

        try:
            selected_tool = next(x for x in tools if x.name == tool_name)
            tool_output = selected_tool.invoke(tool_call)
        except Exception as e:
            logger.exception("Problem while executing tool_call")
            conv_print(f"> Exception while calling tool {ellipsis(str(e))}", log=False)
            tool_output = ToolMessage(
                content=exception_to_string(e),
                name=tool_name,
                tool_call_id=tool_call.get('id'),
                status='error',
            )

        conv_print(f"> Tool output given {ellipsis(tool_output.content)}")
        logger.debug(tool_output)
        chat_history.append(tool_output)

    if reply.response_metadata.get('stop_reason', 'end_turn') != 'end_turn':
        user_turn = False

    # Print the response and add it to the chat history
    if isinstance(reply.content, str):
        conv_print(reply.content)
    else:
        for r in reply.content:
            if r['type'] == 'text':
                conv_print(r['text'])

    return stay


if __name__ == '__main__':
    conv_print("Welcome to meta. Type 'exit' to quit.")

    loop = True
    while loop:
        loop = main()

    conv_print("Thank you for interacting with meta. Bye!")
