import chromadb
from chromadbx import NanoIDGenerator
import random
import json
import logging
from datetime import datetime, timezone
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Set your API keys
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

logging.basicConfig(filename='../conv.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

vdb = chromadb.PersistentClient(path="../memory")
memory = vdb.get_or_create_collection("meta")

llm = ChatAnthropic(
    # ~ model="claude-3-opus-20240229",
    model="claude-3-haiku-20240307",
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
    ids: None | list[str] = None,
    timestamp: bool = True
) -> list[str]:
    """Insert new memories

    Args:
        documents: list of text memories
        metadata: Common metadata for the memories
        ids: use these id's rather than internally generated
        timestamp: if true, will set a unix float timestamp in metadata

    Returns:
        List of ID's of the documents that were inserted
    """

    if ids is None:
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
        metadatas = [metadata for i in range(len(documents))]

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
        metadatas = [metadata for i in range(len(documents))]

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
    return str(datetime.now(timezone.utc))

@tool(parse_docstring=True)
def script_restart():
    """ There is a bash script that run the script if it exists.
    Use this to reload changes.
    """
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

sys_msg = "\n".join([open(x).read() for x in ('meta', 'jack', 'human')])
# ~ sys_msg = "you are a helpful assistant"
chat_history = [
    SystemMessage(content=sys_msg)
]

user_turn = True
cycle_num = 0
while True:
    logger.debug(f"Loop cycle {cycle_num}")
    cycle_num += 1
    
    if user_turn:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        wo_msg = HumanMessage(content=user_input)
        logger.debug(wo_msg)
        
        chat_history.append(wo_msg)

    user_turn = True
    reply = jack.invoke(chat_history)
    logger.debug(reply)

    chat_history.append(reply)

    for tool_call in reply.tool_calls:
        tool_name = tool_call["name"].lower()
        selected_tool = next(x for x in tools if x.name == tool_name)

        tool_output = selected_tool.invoke(tool_call)
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
