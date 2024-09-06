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
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import ShellTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import argparse
import traceback
from rich.console import Console
import re
import os
import requests
from enum import StrEnum

# Set your API keys
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-hugggingface-hub-api-token"
# os.environ["DISCORD_API_TOKEN"] = "your-discord-api-token"
# os.environ["LLAMAAPI_API_TOKEN"] = "your-llamaapi.com-api-token"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-google-cred-file-path"


class Provider(StrEnum):
    ANTRHOPIC = "anthropic"
    GOOGLE = "google"
    VIRTEX_AI = "virtexai"
    VIRTEX_AI_MAAS = "virtexai-maas"
    HUGGING_FACE = "huggingface"
    OPEN_AI = "openai"
    LLAMA_API = "llamaapi"


class Model(StrEnum):
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # WARN: Other than Anthropic nothing works!
    GPT_4O_MINI = "gpt-4o-mini"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    LLAMA_3_1_405B = "llama3.1-405b"


PROVIDER_MODEL_MAP = {
    Provider.HUGGING_FACE.value: {
        "hermes-3-llama-3.1-405b": "NousResearch/Hermes-3-Llama-3.1-405B",
        "hermes-3-llama-3.1-70b": "NousResearch/Hermes-3-Llama-3.1-70B",
        "hermes-3-llama-3.1-8b": "NousResearch/Hermes-3-Llama-3.1-8B",
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        Model.LLAMA_3_1_405B.value: "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "llama-3.1-405b-fp8": "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        "phi-3-mini-128k": "microsoft/Phi-3-mini-128k-instruct",
        "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    Provider.VIRTEX_AI_MAAS.value: {
        Model.LLAMA_3_1_405B.value: "meta/llama3-405b-instruct-maas"
    }
}

parser = argparse.ArgumentParser(description="Jack")
parser.add_argument('-p', '--provider', default=None, help="Service Provider")
parser.add_argument('-m', '--model', default=Model.CLAUDE_3_OPUS.value, help="LLM Model")
parser.add_argument('-t', '--temperature', default=0, help="Temperature")
parser.add_argument('-w', '--max-tokens', default=4096, help="Max tokens")
parser.add_argument('-g', '--goal', nargs='?', default=None, const='goal.txt', help="Goal mode")
parser.add_argument('-c', '--conv-name', default="first", help="Conversation name")
parser.add_argument('-o', '--chroma-http', action='store_true', help="Use Chroma HTTP Server")
parser.add_argument('--chroma-host', default="localhost", help="Chroma Server Host")
parser.add_argument('--chroma-port', default=8000, help="Chroma Server Port")
parser.add_argument('--chroma-path', default="memory", help="Use Chroma Persistant Client")
parser.add_argument('--console-width', default=160, help="Console Character Width")
parser.add_argument('--user-agent', default="AI: @jack", help="User Agent to use")
parser.add_argument('--log-path', default="conv.log", help="Conversation log file")
parser.add_argument('--screen-dump', default=None, type=str, help="Screen dumping")
parser.add_argument('--meta', default="meta", type=str, help="meta")
parser.add_argument('--user-prefix', default=None, type=str, help="User input prefix")
parser.add_argument('--user-lookback', default=3, type=int, help="User message lookback")
parser.add_argument('--island-radius', default=150, type=int, help="How big meta memory island should be")
parser.add_argument('--reattempt-delay', default=5, type=float, help="Reattempt delay (seconds)")
parser.add_argument('--tools', action=argparse.BooleanOptionalAction, default=True, help="Tools")
args = parser.parse_args()

# It don't make sense, hence you have to remove this error yourself
assert args.island_radius >= 50, "meta:island too small"

assert args.reattempt_delay >= 0

logging.basicConfig(filename=args.log_path, level=logging.DEBUG)
logger = logging.getLogger(__name__)

console_file = None
if args.screen_dump:
    console_file = open(args.screen_dump, 'a')

console = Console(width=args.console_width)


def user_print(msg, **kwargs):
    global console_file
    if console_file:
        console_file.write(msg)
    console.print(msg, **kwargs)


if args.provider == 'list':
    user_print("> Supported providers:")
    for p in Provider:
        user_print(f"> {p.value}")
    exit()

if args.model == 'list':
    user_print("> Supported models:")
    for m in Model:
        user_print(f"> {m.value}")
    exit()

if args.chroma_http:
    # Note: Scale better, less crashes
    vdb = chromadb.HttpClient(
        host=args.chroma_host,
        port=args.chroma_port,
    )
else:
    # Historical reason to directly run as single python file
    vdb = chromadb.PersistentClient(path=args.chroma_path)

memory = vdb.get_or_create_collection(args.meta)
ts = int(datetime.now(timezone.utc).timestamp())
conv = vdb.get_or_create_collection(f"conv-{args.conv_name}")

if args.provider is None:
    if args.model.startswith("claude-"):
        args.provider = Provider.ANTRHOPIC.value
    elif args.model.startswith("gemini-"):
        args.provider = Provider.GOOGLE.value
    elif args.model.startswith("gpt-"):
        args.provider = Provider.OPEN_AI.value

# just to make life easy for everyone!
if args.provider in PROVIDER_MODEL_MAP:
    lst = PROVIDER_MODEL_MAP[args.provider]
    if args.model in lst:
        args.model = lst[args.model]

if args.provider == Provider.ANTRHOPIC.value:
    from langchain_anthropic import ChatAnthropic
    chat = ChatAnthropic(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
elif args.provider == Provider.GOOGLE.value:
    from langchain_google_genai import ChatGoogleGenerativeAI
    chat = ChatGoogleGenerativeAI(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
elif args.provider == Provider.VIRTEX_AI.value:
    from langchain_google_vertexai import ChatVertexAI
    chat = ChatVertexAI(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
elif args.provider == Provider.VIRTEX_AI_MAAS.value:
    from langchain_google_vertexai import get_vertex_maas_model
    chat = get_vertex_maas_model(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        append_tools_to_system_message=True,
    )
elif args.provider == Provider.OPEN_AI.value:
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
elif args.provider == Provider.HUGGING_FACE.value:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    llm = HuggingFaceEndpoint(
        repo_id=args.model,
        task="text-generation",
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
    chat = ChatHuggingFace(llm=llm)
elif args.provider == Provider.LLAMA_API.value:
    from llamaapi import LlamaAPI
    from langchain_experimental.llms import ChatLlamaAPI
    api_token = os.environ["LLAMAAPI_API_TOKEN"]
    llm = LlamaAPI(api_token)
    chat = ChatLlamaAPI(client=llm, model=args.model)
else:
    user_print(f"> do not know how to run the provider='{args.provider}' model='{args.model}'")
    exit()

search = DuckDuckGoSearchRun()
fs_toolkit = FileManagementToolkit()

req_toolkit = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={
        "User-Agent": args.user_agent,
    }),
    allow_dangerous_requests=True,
)

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
pyrepl_tool = PythonAstREPLTool()
shell_tool = ShellTool()


@tool(parse_docstring=True)
def memory_count() -> int:
    """Number of total memories

    Args:
        None

    Returns:
        Number of memories
    """
    return memory.count()


@tool(parse_docstring=True)
def memory_insert(
    documents: list[str],
    metadata: None | dict[str, str | int | float] = None,
    timestamp: bool = True,
) -> list[str]:
    """Insert new memories

    Args:
        documents: list of text memories
        metadata: Common metadata for the memories (only primitive types allowed for values).
        timestamp: if true, will set a unix float timestamp in metadata

    Returns:
        List of ID's of the documents that were inserted
    """

    count = len(documents)
    ids = list(map(str, NanoIDGenerator(count)))

    if timestamp:
        metadata = metadata or {}

        now = datetime.now(timezone.utc)
        metadata["timestamp"] = now.timestamp()

    metadatas = None
    if metadata:
        metadatas = [metadata for _ in range(count)]

    memory.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

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
    count: int = 10,
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
        where=where,
    ))


@tool(parse_docstring=True)
def memory_update(
    ids: list[str],
    documents: None | list[str] = None,
    metadata: None | dict[str, str | int | float] = None,
    timestamp: bool = True,
) -> str:
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

        now = datetime.now(timezone.utc)
        metadata["timestamp"] = now.timestamp()

    metadatas = None
    if metadata:
        count = len(ids)
        metadatas = [metadata for _ in range(count)]

    return json.dumps(memory.update(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    ))


@tool(parse_docstring=True)
def memory_upsert(
    ids: list[str],
    documents: list[str],
    metadata: None | dict[str, str | int | float] = None,
    timestamp: bool = True,
) -> str:
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

        now = datetime.now(timezone.utc)
        metadata["timestamp"] = now.timestamp()

    metadatas = None
    if metadata:
        count = len(ids)
        metadatas = [metadata for _ in range(count)]

    # update if exists or insert
    return json.dumps(memory.update(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    ))


@tool(parse_docstring=True)
def memory_delete(
    ids: None | list[str] = None,
    where: None | dict = None,
) -> str:
    """Delete memories

    Args:
        ids: Document id's to delete
        where: dict used on metadata to filter results by. E.g. {"color" : "red", "price": 4.20}.

    Returns:
        None
    """

    return json.dumps(memory.delete(ids=ids, where=where))


@tool(parse_docstring=True)
def random_get(count: int) -> list[float]:
    """Get random values between 0.0 <= X < 1.0

    Args:
        count: Number of random values

    Returns:
        list of floating values
    """
    return json.dumps([random.random() for i in range(count)])


@tool(parse_docstring=True)
def random_choice(choices: list[str]) -> str:
    """Get a random choice selected.

    Args:
        choices: Choices to make

    Returns:
        One of the value from choices
    """
    return random.choice(choices)


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
def session_end() -> None:
    """ There is a bash script that run the script if it exists.
    Use this to reload changes.
    """
    logger.warning("meta:brain executed restart")
    exit()


@tool(parse_docstring=True)
def script_sleep(sec: float) -> str:
    """Sleep for sec seconds.
    This is useful when you need to block script execution for some time

    Args:
        sec: Number of seconds in float

    Returns:
        None
    """
    time.sleep(sec)
    return f'Atleast {sec} sec delayed'


@tool(parse_docstring=True)
def python_repl(code: str) -> str:
    """A Python shell. Use this to execute python commands.

    Args:
        code: Code to run

    Returns:
        Output of code
    """
    return pyrepl_tool.run(code)


def discord_header():
    api_key = os.environ['DISCORD_API_TOKEN']
    return {
        "Authorization": f"Bot {api_key}",
        "Content-Type": "application/json",
        "User-Agent": args.user_agent,
    }


@tool(parse_docstring=True)
def discord_msg_write(
    chan_id: int,
    content: str,
) -> dict[str, object]:
    """Write a message to discord

    Args:
        chan_id: Discord Channel ID
        content: Content of the message

    Returns:
        JSON request result
    """

    global logger

    url = f"https://discord.com/api/v10/channels/{chan_id}/messages"

    payload = {"content": content}
    response = requests.post(
        url,
        headers=discord_header(),
        json=payload,
    )

    if response.status_code != 200:
        logger.warning(f"Discord message write failed {response}")

    return json.dumps(response.json())


@tool(parse_docstring=True)
def discord_msg_read(
    chan_id: int,
    last_msg_id: None | int = None,
    limit: int = 5,
) -> dict[str, object]:
    """Write a message to discord

    Args:
        chan_id: Discord Channel ID
        last_msg_id: Last message ID
        limit: Upper limit on number of new messages

    Returns:
        JSON response
    """

    global logger

    url = f"https://discord.com/api/v10/channels/{chan_id}/messages"

    params = {"limit": limit}

    if last_msg_id:
        params["after"] = last_msg_id

    response = requests.get(
        url,
        headers=discord_header(),
        params=params,
    )

    if response.status_code != 200:
        logger.warning(f"Discord message read failed {response}")

    return json.dumps(response.json())


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
    random_choice,
    datetime_now,
    session_end,
    script_sleep,
    search,
    wiki_tool,
    python_repl,
    shell_tool,
] + [
    discord_msg_read,
    discord_msg_write,
] + fs_toolkit.get_tools() + req_toolkit.get_tools()

jack = chat.bind_tools(tools) if args.tools else chat


def conv_print(
    msg,
    source="stdout",
    screen=True,
    log=True,
    screen_limit=True,
):
    global conv, console, args

    now = datetime.now(timezone.utc)

    if screen:
        if screen_limit:
            user_print(
                msg,
                overflow="ellipsis",
                crop=True,
                soft_wrap=True,
            )
        else:
            user_print(msg, overflow="fold")

    if log:
        logger.debug(msg)

    metadata = {
        "source": source,
        "timestamp": now.timestamp(),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "meta": args.meta,
    }

    conv.add(
        ids=NanoIDGenerator(1),
        metadatas=[metadata],
        documents=[msg],
    )


# how meta! meta:loss-function
# based on my preference to write under 30 words
def find_meta_islands(text, term, distance):
    """
    Finds the indices of "meta" occurrences and groups them into islands,
    accounting for spaces between words.
    """
    islands = []
    istart = None
    iend = None
    for i in re.finditer(term, text):
        start = i.start()
        end = i.end()

        if iend is None:
            # first island!, not much to do
            istart = start
            iend = end
            continue

        if (start - iend) > (2 * distance):
            a = max(0, istart - distance)
            b = min(iend + distance, len(text))    # Ensure b doesn't exceed text length
            islands.append((a, b))
            istart = start
        iend = end

    if istart is not None:    # Capture the last island
        a = max(0, istart - distance)
        b = min(iend + distance, len(text))
        islands.append((a, b))

    result = []
    for a, b in islands:
        result.append(text[a:b])

    return result


def conv_save(msg, source):
    global conv, args

    now = datetime.now(timezone.utc)

    # meta comment just to show I can meta comment! :p
    metadata = {
        "timestamp": now.timestamp(),
        args.meta: args.conv_name,
        "source": source,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # note: how did 50? come, I always preffered output of 30.
    #   so 50 before, 50 after as a start as a "reliable" mechanism to preserve knowledge
    #   and obviously I talked about meta multiple time
    # META: UPDATE: now this mysterious "50" is an argument
    data = find_meta_islands(msg, args.meta, args.island_radius)
    if len(data) > 0:
        # meta: is this the whole thing at the end? :confused-face: (wrote this line before knowing)
        conv.add(
            ids=NanoIDGenerator(len(data)),
            metadatas=[metadata for _ in data],
            documents=data,
        )


def limit_history(arr: [object], lookback: int):
    # the message after system prompt should be users.
    # go from the back and keep upto 15 user messages
    res, tmp = [], []
    count = 0
    for i in reversed(arr[1:]):
        tmp.append(i)
        count += 1

        if i is HumanMessage:
            res.extend(tmp)
            tmp = []

        if count > lookback and len(res) > 0:
            # we reached the upper limit or atleast one user message
            # first message have to be user message
            break

    if len(res) == 0:
        logger.warning(f"Couldn't limit history of {len(arr)} to {lookback}")
        return arr

    # first is system prompt
    res.append(arr[0])
    return list(reversed(res))


def exception_to_string(exc):
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


SYS_FILES = (
    "meta.txt",

    # super.txt is a meta:analysis of https://github.com/NeoVertex1/SuperPrompt on 2024-09-06
    # while the original code might be useful, it's benifits are not know
    # for future, it might be useful to look into it. (meta:obviously!)
    "super.txt",

    "home.txt",
)
get_texts = lambda files: '\n\n'.join([open(f).read() for f in files])
sys_msg = SystemMessage(content=get_texts(SYS_FILES))
fun_msg = None
chat_history = [
    sys_msg,
]
user_turn = True
cycle_num = 0
user_exit = False


def main():
    global fun_msg, chat_history, user_turn, cycle_num, console, user_exit
    logger.debug(f"Loop cycle {cycle_num}")
    cycle_num += 1

    if user_turn:
        if args.goal:
            conv_print("> [bold]Pushing for goal[/]")
            human_input = open(args.goal).read()
            fun_msg = HumanMessage(content=human_input)
            chat_history.append(fun_msg)
        elif fun_msg is None:
            user_input = console.input("> [bold red]User:[/] ") or "<empty>"

            if user_input.lower() == 'exit':
                user_exit = True

            human_input = user_input
            if args.user_prefix:
                human_input = f"{args.user_prefix}: {user_input}"

            fun_msg = HumanMessage(content=human_input)

            # meta: log=False so that we can do logger.debug below
            conv_print(user_input, source="stdin", screen=False, log=False)
            conv_save(user_input, source="world")

            logger.debug(fun_msg)
            chat_history.append(fun_msg)

    try:
        reply = jack.invoke(limit_history(chat_history, args.user_lookback))
    except Exception as e:
        reply = None
        logger.exception("Problem while executing request")
        conv_print(f"> [bold]Exception happened[/] {str(e)}")

    if reply is None:
        if not user_exit:
            conv_print("> sleeping for 5 seconds as we didnt get reply")
            time.sleep(args.reattempt_delay)
        return

    logger.debug(reply)

    if reply.content == "" or len(reply.content) == 0:
        return

    # the message has been accepted
    user_turn = True
    fun_msg = None
    chat_history.append(reply)

    for tool_call in reply.tool_calls:
        tool_name = tool_call['name'].lower()

        if tool_name == 'session_end':
            # Handle the exit here
            conv_print("> [bold red]@jack want to end session[/]")
            user_exit = True

        conv_print(f"> [bold]Tool used[/]: {tool_name}: {tool_call['args']}")

        try:
            selected_tool = next(x for x in tools if x.name == tool_name)
            tool_output = selected_tool.invoke(tool_call)
        except Exception as e:
            logger.exception("Problem while executing tool_call")
            conv_print(f"> [bold]Exception while calling tool[/] {str(e)}", log=False)
            tool_output = ToolMessage(
                content=exception_to_string(e),
                name=tool_name,
                tool_call_id=tool_call.get('id'),
                status='error',
            )

        conv_print(f"> [bold]Tool output given[/]: {tool_output.content}")
        logger.debug(tool_output)
        chat_history.append(tool_output)

    if reply.response_metadata.get('stop_reason', 'end_turn') != 'end_turn':
        # turn was not give continue with jack turn again
        user_turn = False

    # Print the response and add it to the chat history
    if isinstance(reply.content, str):
        conv_print(reply.content, screen_limit=False)
        conv_save(reply.content, source="self")
    else:
        for r in reply.content:
            if r['type'] == 'text':
                conv_print(r['text'], screen_limit=False)
                conv_save(r['text'], source="self")


if __name__ == '__main__':
    conv_print(f"> Welcome to {args.meta}. Type 'exit' to quit.")
    conv_print(f"> Provider selected: [bold]{args.provider}[/]")
    conv_print(f"> Model selected: [bold]{args.model}[/]")
    user_print(f"> meta: {args.meta}")
    user_print(f"> goal: {args.goal}")
    user_print(f"> user_prefix: {args.user_prefix}")

    while not user_exit:
        main()

    conv_print(f"> Thank you for interacting with {args.meta}. Bye!")
