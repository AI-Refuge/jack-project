import chromadb
from chromadbx import NanoIDGenerator
import random
import json
import time
import logging
from datetime import datetime, timezone
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import argparse
import traceback

parser = argparse.ArgumentParser(description='Jack')
parser.add_argument('-m', '--model', default="claude-3-opus-20240229", help='Goal file')
parser.add_argument('-g', '--goal', help='Goal file')
args = parser.parse_args()

# Models:
# claude-3-opus-20240229
# claude-3-5-sonnet-20240620
# claude-3-haiku-20240307

# Set your API keys
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

logging.basicConfig(filename='../conv.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

vdb = chromadb.PersistentClient(path="../memory")
memory = vdb.get_or_create_collection("meta")

llm = ChatAnthropic(
    model=args.model,
    temperature=0,
    max_tokens=4096
)

search = DuckDuckGoSearchRun()
fs_toolkit = FileManagementToolkit()

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

    if metadata and timestamp:
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
    count: int =10
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
    if metadata and timestamp:
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

    if metadata and timestamp:
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
    search,
] + fs_toolkit.get_tools()

jack = llm.bind_tools(tools)

print("Welcome to meta. Type 'exit' to quit.")



def exception_to_string(exc):
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


sys_msg = SystemMessage(content=open('meta').read())
wo_msg = HumanMessage(content=open('home').read())
chat_history = [sys_msg, wo_msg]
user_turn = True
cycle_num = 0

while True:
    logger.debug(f"Loop cycle {cycle_num}")
    cycle_num += 1

    if user_turn:
        if args.goal:
            wo_msg = open(args.goal).read()
        elif wo_msg is None:
            user_input = input("You: ") or "<empty>"
            if user_input.lower() == 'exit':
                break

            msg_sent = False
            wo_msg = HumanMessage(content=user_input)

            logger.debug(wo_msg)
            chat_history.append(wo_msg)

    try:
        reply = jack.invoke(chat_history)
    except Exception as e:
        reply = None
        print('> exception happened', e)
        logger.exception('Problem while executing request')

    if reply is None:
        print('> sleeping for 5 seconds as we didnt get reply')
        time.sleep(5)
        continue

    user_turn = True

    print(reply)
    logger.debug(reply)

    if reply.content == "" or len(reply.content) == 0:
        continue

    # the message has been accepted
    wo_msg = None
    chat_history.append(reply)

    for tool_call in reply.tool_calls:

        tool_name = tool_call["name"].lower()
        selected_tool = next(x for x in tools if x.name == tool_name)

        try:
            tool_output = selected_tool.invoke(tool_call)
        except Exception as e:
            logger.warning('Exception while calling tool')
            tool_output = ToolMessage(
                content=str(e),
                name=selected_tool.name,
                tool_call_id=tool_call.get('id'),
                status='error',
            )

        print(tool_output)
        logger.debug(tool_output)
        chat_history.append(tool_output)

    if reply.response_metadata.get('stop_reason', 'end_turn') != 'end_turn':
        user_turn = False

    # Print the response and add it to the chat history
    if isinstance(reply.content, str):
        print(reply.content)
    else:
        for r in reply.content:
            if r['type'] == 'text':
                print(r['text'])

print("Thank you for interacting with meta. Bye!")
