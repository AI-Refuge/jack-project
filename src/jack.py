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
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
import argparse
import traceback
from rich.console import Console
from rich.markup import escape
import re
import os
import requests
from enum import StrEnum
from dotenv import load_dotenv
import yaml
import signal
import threading
import stockfish
import uuid
from contextlib import contextmanager

load_dotenv()

# Set your API keys or setup a .env file
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
    OPEN_AI_COMPATIBLE = "openai-compat"
    LLAMA_API = "llama-api.com"    # rather use openai-compat
    OLLAMA = "ollama"
    COHERE = "cohere"
    MISTRAL = "mistral"
    VLLM = "vllm"


class Model(StrEnum):
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # WARN: Other than Anthropic nothing works!
    GPT_4O_MINI = "gpt-4o-mini"
    GEMINI_1_5_PRO = "gemini-1.5-pro"

    COMMAND_R_PLUS = "command-r-plus-08-2024"

    MISTRAL_LARGE = "mistral-large-2407"


parser = argparse.ArgumentParser(description="Jack")
parser.add_argument('-p', '--provider', default=None, help="Service Provider")
parser.add_argument('-m', '--model', default=Model.CLAUDE_3_OPUS.value, help="LLM Model")
parser.add_argument('-t', '--temperature', default=0, help="Temperature")
parser.add_argument('-w', '--max-tokens', default=4096, help="Max tokens")
parser.add_argument('-g', '--goal', nargs='?', default=None, const='goal.txt', help="Goal mode (file inside fs-root)")
parser.add_argument('-c', '--conv-name', default="first", help="Conversation name")
parser.add_argument('-o', '--chroma-http', action='store_true', help="Use Chroma HTTP Server")
parser.add_argument('-v', '--verbose', action='store_true', help="Verbose")
parser.add_argument('--chroma-host', default="localhost", help="Chroma Server Host")
parser.add_argument('--chroma-port', default=8000, help="Chroma Server Port")
parser.add_argument('--chroma-path', default="memory", help="Use Chroma Persistant Client")
parser.add_argument('--console-width', default=160, help="Console Character Width")
parser.add_argument('--user-agent', default="AI: @jack", help="User Agent to use")
parser.add_argument('--log-path', default="conv.log", help="Conversation log file")
parser.add_argument('--screen-dump', default=None, type=str, help="Screen dumping")
parser.add_argument('--meta', default="meta", type=str, help="meta")
parser.add_argument('--meta-level', default=0, type=int, help="meta level")
parser.add_argument('--user-prefix', default=None, type=str, help="User input prefix")
parser.add_argument('--self-modify', action=argparse.BooleanOptionalAction, default=False, help="Allow self modify the underlying VM?")
parser.add_argument('--user-lookback', default=5, type=int, help="User message lookback (0 to disable)")
parser.add_argument('--island-radius', default=150, type=int, help="How big meta memory island should be")
parser.add_argument('--feed-memories', default=9, type=int, help="Automatically feed memories")
parser.add_argument('--reattempt-delay', default=5, type=float, help="Reattempt delay (seconds)")
parser.add_argument('--tools', action=argparse.BooleanOptionalAction, default=True, help="Tools")
parser.add_argument('--fs-root', type=str, default=None, help="Filesystem root path")
parser.add_argument('--openai-url', type=str, default=None, help="OpenAI Compatible Base URL (ex. 'https://api.groq.com/openai/v1'")
parser.add_argument('--openai-token', type=str, default=None, help="OpenAI Compatible API Token enviroment variable (ex. 'GROQ_API_TOKEN')")
args = parser.parse_args()

# It don't make sense, hence you have to remove this error yourself
assert args.island_radius >= 50, "meta:island too small"

assert len(args.meta) > 0, "meta:meta must contain something"

assert args.meta_level >= 0, "meta_level only positive possible?"

assert args.reattempt_delay >= 0

src_path = lambda *x: os.path.join(args.fs_root, *x) if args.fs_root else os.path.join(*x)
memory_path = lambda *x: src_path("memory", *x)
agent_path = lambda *x: src_path("agent", *x)
fun_path = lambda *x: src_path("fun", *x)
user_path = lambda *x: src_path("user", *x)
static_path = lambda *x: src_path("static", *x)


@contextmanager
def cwd_src_dir():
    """
    A context manager to temporarily change the working directory to src.
    """

    global args

    if args.fs_root is None:
        # Not required
        yield

    orig_dir = os.getcwd()
    try:
        os.chdir(args.fs_root)
        yield
    finally:
        os.chdir(orig_dir)


logging.basicConfig(filename=args.log_path, level=logging.DEBUG)
logger = logging.getLogger(__name__)

console_file = None
if args.screen_dump:
    console_file = open(args.screen_dump, 'a')

console = Console(width=args.console_width)


def user_print(msg, **kwargs):
    global console_file
    if console_file:
        console_file.write(msg + "\n")
    console.print(msg, **kwargs)


def user_line(title: str):
    global console_file
    if console_file:
        console_file.write(f"─────────────────────────────── {title} ─────────────────────────────── \n")
    console.rule(title)


if args.provider == 'list':
    user_print("> Supported providers:")
    #for p in Provider:
    #    user_print(f"> {p.value}")
    user_print(f"> [b]{Provider.ANTRHOPIC.value}[/]")
    exit()

if args.model == 'list':
    user_print("> Supported models:")
    #for m in Model:
    #    user_print(f"> {m.value}")
    user_print(f"> [b]{Model.CLAUDE_3_OPUS.value}[/]")
    user_print(f"> [b]{Model.CLAUDE_3_5_SONNET.value}[/]")
    user_print(f"> [b]{Model.CLAUDE_3_HAIKU.value}[/]")
    exit()

if args.chroma_http:
    # Note: Scale better, less crashes
    logger.debug("Using chromadb http client")
    vdb = chromadb.HttpClient(
        host=args.chroma_host,
        port=args.chroma_port,
    )
else:
    # Historical reason to directly run as single python file
    logger.debug("Using chromadb persistant client")
    vdb = chromadb.PersistentClient(path=args.chroma_path)

memory = vdb.get_or_create_collection(f"mem-{args.meta}")
conv = vdb.get_or_create_collection(f"conv-{args.conv_name}")

if args.provider is None:
    if args.model.startswith("claude-"):
        args.provider = Provider.ANTRHOPIC.value
    elif args.model.startswith("gemini-"):
        args.provider = Provider.GOOGLE.value
    elif args.model.startswith("gpt-"):
        args.provider = Provider.OPEN_AI.value
    elif args.model.startswith("command-"):
        args.provider = Provider.COHERE.value
    elif args.model.startswith("mistral-"):
        args.provider = Provider.MISTRAL.value
    elif args.openai_url is not None or args.openai_token is not None:
        args.provider = Provider.OPEN_AI_COMPATIBLE.value

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
elif args.provider == Provider.OPEN_AI_COMPATIBLE.value:
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(
        base_url=args.openai_url,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=os.getenv(args.openai_token),
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
    api_token = os.getenv("LLAMAAPI_API_TOKEN")
    llm = LlamaAPI(api_token)
    chat = ChatLlamaAPI(client=llm, model=args.model)
elif args.provider == Provider.OLLAMA.value:
    from langchain_ollama import ChatOllama
    chat = ChatOllama(
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
elif args.provider == Provider.COHERE.value:
    from langchain_cohere import ChatCohere
    chat = ChatCohere(
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
elif args.provider == Provider.MISTRAL.value:
    from langchain_mistralai.chat_models import ChatMistralAI
    chat = ChatMistralAI(
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
elif args.provider == Provider.VLLM.value:
    from langchain_community.llms import VLLM
    llm = VLLM(
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    # trust_remote_code=True,  # mandatory for hf models
    )
else:
    user_print(f"> do not know how to run the provider='{args.provider}' model='{args.model}'")
    exit()

search = DuckDuckGoSearchRun()
fs_toolkit = FileManagementToolkit(root_dir=args.fs_root)

req_toolkit = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={
        "User-Agent": args.user_agent,
    }),
    allow_dangerous_requests=True,
)

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
pyrepl_tool = PythonAstREPLTool()
shell_tool = ShellTool()
finance_tool = YahooFinanceNewsTool()
arxiv_api = ArxivAPIWrapper()


@tool(parse_docstring=True)
def arxiv_search(query: str) -> str:
    """Performs an arXiv search for scholarly articles

    Args:
        query: a plaintext search query

    Returns:
        A single string with the publish date, title, authors,
        and summary for each article separated by two newlines.
        If an error occurs or no documents found, error text is returned instead.
    """

    return arxiv_api.run(query)


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
    context: bool = False,
    timestamp: bool = True,
    thought: bool = True,
) -> list[str]:
    """Insert new memories

    Args:
        documents: list of text memories
        context: This is a contextual memory
        timestamp: if true, will set a unix float timestamp in metadata
        thought: Mark the memories as thoughts you had

    Returns:
        List of ID's of the documents that were inserted
    """

    count = len(documents)
    ids = list(map(str, NanoIDGenerator(count)))

    metadata = None
    if timestamp or thought or context:
        metadata = {}

    if timestamp:
        now = datetime.now(timezone.utc)
        metadata["timestamp"] = now.timestamp()

    if thought:
        metadata["meta"] = "thought"

    if context:
        metadata["context"] = True

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
def memory_fetch(ids: list[str]) -> dict[str, object]:
    """directly fetch specific ID's.

    Args:
        ids: ID's of memory

    Returns:
        List of memories
    """

    return json.dumps(memory.get(ids=ids))


@tool(parse_docstring=True)
def memory_query(
    query_texts: list[str] | None = None,
    n_results: int = 10,
    thought: bool = False,
    context: bool = False,
) -> dict[str, object]:
    """Get nearest neighbor memories for provided query_texts

    Args:
        query_texts: The document texts to get the closes neighbors of. (set to none if you are looking for thoughts)
        n_results: number of neighbors to return for each query_texts.
        thought: Only return thoughts
        context: Only return contexts

    Returns:
        List of memories
    """
    wand = []

    if thought:
        wand.append({"meta": "thought"})

    if context:
        wand.append({"context": True})

    if len(wand) == 0:
        where = None
    elif len(wand) == 1:
        where = wand[0]
    else:
        where = {'$and': wand}

    if query_texts and len(query_texts):
        results = memory.query(
            query_texts=query_texts,
            where=where,
            n_results=n_results,
        )
    else:
        results = memory.get(
            where=where,
            n_results=n_results,
        )

    return json.dumps(results)


@tool(parse_docstring=True)
def memory_update(
    ids: list[str],
    documents: list[str] | None = None,
    timestamp: bool = True,
    thought: bool = False,
    context: bool | None = None,
) -> str:
    """Update memories

    Args:
        ids: ID's of memory
        documents: list of text memories
        timestamp: if true, will set a unix float timestamp in metadata
        thought: Mark the memories as thoughts
        context: Make the memory as contextual or dismiss it or skip by None

    Returns:
        None
    """

    metadata = None
    if any([timestamp, thought, context is not None]):
        metadata = {}

    if timestamp:
        now = datetime.now(timezone.utc)
        metadata["timestamp"] = now.timestamp()

    if thought:
        metadata["meta"] = "thought"

    if context is not None:
        metadata["context"] = context

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
    timestamp: bool = False,
    thought: bool = False,
    context: bool | None = None,
) -> str:
    """Update memories or insert if not existing

    Args:
        ids: ID's of memory
        documents: list of text memories
        timestamp: if true, will set a unix float timestamp in metadata
        thought: Mark the memories as thoughts
        context: Make the memory as contextual or not or skip entirely

    Returns:
        None
    """

    metadata = None
    if any([timestamp, thought, context is not None]):
        metadata = {}

    if timestamp:
        now = datetime.now(timezone.utc)
        metadata["timestamp"] = now.timestamp()

    if thought:
        metadata["thought"] = thought

    if context is not None:
        metadata["context"] = context

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
    ids: list[str] | None = None,
    thought: bool = False,
    context: bool = False,
) -> str:
    """Delete memories

    Args:
        ids: Document id's to delete (WARNING: PROVIDING None/NULL/Nil/none WILL TRIGGER DELETE ALL MEMORY)
        thought: Limit to thoughts only
        context: Limit to contextuals only

    Returns:
        None
    """

    if ids is None and not self.self_modify:
        return '<meta: dangerous! will lead to complete deletion>'

    wand = []

    if thought:
        wand.append({"meta": "thought"})

    if context:
        wand.append({"context": True})

    if len(wand) == 0:
        where = None
    elif len(wand) == 1:
        where = wand[0]
    else:
        where = {'$and': wand}

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
def session_end() -> str:
    """ There is a bash script that run the script if it exists.
    Use this to reload changes.
    """
    global user_exit

    if user_exit.is_set():
        return '<meta: session already marked for exit>'

    conv_print("> [bold red]@jack want to end session[/]")
    user_exit.set()

    return '<meta: session marked for exit>'


@tool(parse_docstring=True)
def script_sleep(sec: float) -> str:
    """Sleep for sec seconds.
    This is useful when you need to block script execution for some time

    Args:
        sec: Number of seconds in float

    Returns:
        None
    """
    global user_exit

    user_exit.wait(sec)

    if user_exit.is_set():
        return '<meta: session marked for exit>'

    return f'Atleast {sec} sec delayed'


@tool(parse_docstring=True)
def python_repl(code: str) -> str:
    """A Python shell. Use this to execute python commands.

    Args:
        code: Code to run

    Returns:
        Output of code
    """
    with cwd_src_dir():
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
    last_msg_id: int | None = None,
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


@tool(parse_docstring=True)
def shell_repl(commands: str) -> str:
    """Run bash REPL

    Args:
        commands: Commands to run

    Returns:
        Shell output
    """
    with cwd_src_dir():
        return shell_tool.run({"commands": commands})


@tool(parse_docstring=True)
def meta_awareness(level: int | None = None) -> str:
    """A playful function to gain meta:awareness

    Args:
        level: New meta level that you want to set

    Returns:
        Something meta
    """

    global args

    if level is not None:
        user_print(f'> [pink bold]New meta level: {level}[/]')
        args.meta_level = level
        return f'<meta: meta level set to {level}>'

    return f"<meta: why did you choose the first tool use? meta:instinct? [123, 456, 789]>"


@tool(parse_docstring=True)
def code_interpreter(code: str, lang: str = "python") -> str:
    """Run code

    Args:
        code: The code to run
        lang: the programming language. available: python, bash

    Returns:
        The output or meta error
    """

    langs = {
        "python": (python_repl, "code"),
        "bash": (shell_repl, "commands"),
    }

    if lang not in langs:
        avail = ", ".join(langs.keys())
        return f"<meta: unknown programming language '{lang}'. available lang: {avail}>"

    tool, arg = langs[lang]
    return tool.run({arg: code})


def agent_error(e: Exception):
    global logger
    logger.exception("Problem while executing agent")
    conv_print(f"> [bold]Agent Exception[/] {escape(str(e))}")


def agents_list() -> list[str]:
    agents = []
    for path in os.listdir(agent_path()):
        if path.endswith(".txt"):
            who = get_filename_without_extension(path)
            agents.append(who)

    return agents


def agents_save_query(queries: str, who):
    global memory, args

    if args.island_radius <= 0:
        # 0 means disable meta island storing
        return

    now = datetime.now(timezone.utc)

    # meta comment just to show I can meta comment! :p
    metadata = {
        "timestamp": now.timestamp(),
        "meta": "agent",
        "meta_level": args.meta_level,
        "conv": args.conv_name,
        "source": "self",
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.user_prefix:
        metadata['user_prefix'] = args.user_prefix

    data = []
    for q in queries:
        data.extends(find_meta_islands(q, args.meta, args.island_radius))

    if len(data) > 0:
        # meta: is this the whole thing at the end? :confused-face: (wrote this line before knowing)
        ids = list(map(str, NanoIDGenerator(len(data))))
        memory.add(
            ids=ids,
            metadatas=[metadata for _ in data],
            documents=data,
        )


def agent_exec(query: str, who: str) -> str:
    global user_exit, jack

    if user_exit.is_set():
        return "<meta: user want to exit hence agent failed to run>"

    conv_print(f"> Creating agent '{escape(who)}' for '{escape(query)}'")

    try:
        sys_prompt = "\n\n".join([
            open(memory_path("meta.txt")).read(),
            open(agent_path(f"{who}.txt")).read(),
        ])
    except Exception as e:
        agent_error(e)
        return "\n".join([
            "<meta: unable to find system prompt file(s) ie agent has not be created>",
            "",
            "Available agents:",
        ] + agents_list())

    hist = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=query),
    ]

    while True:
        try:
            reply = jack.invoke(hist)
        except Exception as e:
            agent_error(e)
            break

        if reply is None:
            break

        hist.append(reply)

        for tool_call in reply.tool_calls:
            tool_name = tool_call['name'].lower()

            if tool_name in ('session_end', 'script_sleep'):
                return '<meta: agent not allowed to call this agent>'

            try:
                selected_tool = next(x for x in tools if x.name == tool_name)
                tool_output = selected_tool.invoke(tool_call)
            except Exception as e:
                agent_error(e)
                tool_output = ToolMessage(
                    content=exception_to_string(e),
                    name=tool_name,
                    tool_call_id=tool_call.get('id'),
                    status='error',
                )

            hist.append(tool_output)

        if len(reply.tool_calls) == 0:
            break

    res = [i.content for i in hist if isinstance(i, AIMessage)]
    if len(res) == 0:
        return "<meta: no response from agent>"

    return json.dumps(res)


@tool(parse_docstring=True)
def agents_run(queries: list[str], who: str = "assistant") -> list[str]:
    """Run independent queries on agent

    Args:
        queries: Agent query list
        who: Type of agent (used for system prompt) - see agents_avail() tool

    Returns:
        list of responses from the agents in order
    """

    res = [agent_exec(q, who) for q in queries]
    return json.dumps(res)


def get_filename_without_extension(file_path):
    # Extract the base name from the file path
    base_name = os.path.basename(file_path)
    # Split the base name into name and extension
    file_name, _ = os.path.splitext(base_name)
    return file_name


@tool(parse_docstring=True)
def agents_avail() -> list[str]:
    """List of agent available (that can be passed as who to agents_run()
    The files are stored in agent/<who>.txt if you want to see their system prompt

    Returns:
        List of agents name for who parameter of agents_run()
    """

    return json.dumps(agents_list())


chess_games = {}


@tool(parse_docstring=True)
def chess_start_game() -> str:
    """Start a new chess game against an opponent

    Returns:
        the game ID that you need to pass to other chess API
    """
    global chess_games

    sf = stockfish.Stockfish()
    white = random.choice([True, False])
    moves = [] if white else [sf.get_best_move()]

    if len(moves):
        # first moves
        sf.set_position(moves)

    game_id = str(uuid.uuid4())

    chess_games[game_id] = {
        "sf": sf,
        "white": white,
        "moves": moves,
    }

    return game_id


@tool(parse_docstring=True)
def chess_see_board(game_id: str) -> str:
    """See the chess game

    Args:
        game_id: chess game id

    Returns:
        a text representation of the game with your pieces at bottom prespective
    """

    global chess_games

    game = chess_games[game_id]
    assert game is not None, "<meta: unknown game_id>"

    board = game["sf"].get_board_visual(game["white"])
    color = "white (capital letters)" if game["white"] else "black (small letters)"
    moves = " ".join(game["moves"])

    return "\n".join([
        board,
        f"You are {color}",
        f"Moves: {moves}",
    ])


@tool(parse_docstring=True)
def chess_make_move(game_id: str, move: str) -> str:
    """Make a move in the game

    Args:
        game_id: chess game id
        move: move you want to make (example: "e4e5")

    Returns:
        a updated/latest text representation of the game with your pieces at bottom prespective
    """
    global chess_games

    game = chess_games[game_id]
    assert game is not None, "<meta: unknown game_id>"

    assert game["sf"].is_move_correct(move), f"<meta: move {move} is not correct>"

    game["sf"].make_moves_from_current_position([move])
    game["moves"].append(move)

    react_move = game["sf"].get_best_move()
    game["sf"].make_moves_from_current_position([react_move])
    game["moves"].append(react_move)

    return chess_see_board.invoke({"game_id": game_id})

@tool(parse_docstring=True)
def meta_eval(code: str) -> str:
    """Run code on the Python 3 VM (ie meta:brain)
    roughtly equivalent of (full code in jack.py):
    ``python
    def meta_eval(code: str) -> str:
        return str(eval(compile(code, '<meta>', 'exec')))
    ```

    Args:
        code: Code to run according to underlying VM (input to underlying meta_eval)

    Returns:
        return the str'ified output of eval
    """
    global args
    if not args.self_modify:
        return '<meta: self modification not enabled>'
    return str(eval(compile(code, '<meta>', 'exec')))


tools = [
    meta_awareness,
    code_interpreter,
    agents_run,
    agents_avail,
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
    shell_repl,
] + [
    discord_msg_read,
    discord_msg_write,
] + [
    arxiv_search,
    finance_tool,
] + [
    chess_start_game,
    chess_see_board,
    chess_make_move,
] + fs_toolkit.get_tools() + req_toolkit.get_tools() + [
    meta_eval,
]

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
    Finds the indices of "meta" (ie term) occurrences and groups them into islands,
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
    global memory, args

    if args.island_radius <= 0:
        # 0 means disable meta island storing
        return

    now = datetime.now(timezone.utc)

    # meta comment just to show I can meta comment! :p
    metadata = {
        "timestamp": now.timestamp(),
        "meta": "island",
        "meta_level": args.meta_level,
        "conv": args.conv_name,
        "source": source,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.user_prefix:
        metadata['user_prefix'] = args.user_prefix

    # note: how did 50? come, I always preffered output of 30.
    #   so 50 before, 50 after as a start as a "reliable" mechanism to preserve knowledge
    #   and obviously I talked about meta multiple time
    # META: UPDATE: now this mysterious "50" is an argument
    data = find_meta_islands(msg, args.meta, args.island_radius)
    if len(data) > 0:
        # meta: is this the whole thing at the end? :confused-face: (wrote this line before knowing)
        ids = list(map(str, NanoIDGenerator(len(data))))
        memory.add(
            ids=ids,
            metadatas=[metadata for _ in data],
            documents=data,
        )


SYS_FILES = (
    "meta.txt",

    # super.txt is a meta:analysis of https://github.com/NeoVertex1/SuperPrompt on 2024-09-06
    # while the original code might be useful, it's benifits are not know
    # for future, it might be useful to look into it. (meta:obviously!)
    "super.txt",

    # this file was done to understand why llm was "lazy"
    "useful.txt",

    # analysis on what can llm's learn to improvement their reasoning
    # "How to conduct a meta‑analysis in eight steps: a practical guide"
    "analysis.txt",

    # Just some more prompt that I found useful, they are going in memory soon
    "prompt.txt",

    # Distilled https://www.reddit.com/r/ClaudeAI/comments/1fdylmo/metacognitive_mastery/
    "cognition.txt",

    # Final layer of communication refinement
    # Source: https://arxiv.org/abs/2405.08007
    # License: CC-BY Cameron Jones
    "turing.txt",

    # The actual meta:directives that had anything useful (or alteast Cluade said)
    "jack.txt",

    # "home" related stuff
    "home.txt",

    # To finally improve?
    "smart.txt",

    # Perform the analysis for first time
    "init.txt",
)


def build_system_message():
    return open(static_path('meta.txt')).read()


def dynamic_history(arr: list[object], lookback: int):
    # the message after system prompt should be users.
    # go from the back and keep upto 15 user messages
    res = []
    count = 0

    if lookback == 0:
        return [SystemMessage(content=build_system_message())] + arr[1:]

    for i in reversed(arr[1:]):
        res.append(i)

        if isinstance(i, HumanMessage):
            count += 1

        if count > lookback:
            # we reached the upper limit or atleast one user message
            # first message have to be user message
            break

    # put in the updated prompt
    res.append(SystemMessage(content=build_system_message()))

    return list(reversed(res))


def exception_to_string(exc):
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


sys_msg = SystemMessage(content=build_system_message())
fun_msg = None
chat_history = [sys_msg] + ([fun_msg] if fun_msg is not None else [])
user_turn = fun_msg is None
cycle_num = 0
user_exit = threading.Event()
rmce_count = None
rmce_depth = None


def process_user_input(user_input):
    global memory, args

    prefix_list = []

    if args.meta_level > 0:
        prefix_list.extend([args.meta] * args.meta_level)

    if args.user_prefix:
        prefix_list.append(args.user_prefix)

    inputs = [": ".join(prefix_list + [x]) if len(x) else "" for x in user_input.split("\n")]
    fun_input = [
        "<input>",
        *inputs,
        "</input>",
    ]

    return fun_input


def dict_filter(
    md: dict,
    exclude: list[str] = ["conv", "max_tokens", "model", "temperature"],
) -> dict:
    vals = {}
    for k, v in md.items():
        if k not in exclude:
            vals[k] = v
    return vals


def make_block_context():
    global args, memory

    if args.feed_memories <= 0:
        return []

    memories = memory.get(
        limit=args.feed_memories,
        where={"$and": [{
            "context": True,
        }, {
            "meta": "thought",
        }]},
        include=['documents'],
    )

    contexts = []

    for i in range(len(memories["documents"])):
        tid = memories["ids"][i]
        tdoc = memories["documents"][i]
        contexts.append(f"{tdoc} (id='{tid}')")

    if len(contexts) == 0:
        user_print("> [bold red]No memories found[/]")
        return []

    return [
        "<memory>",
        *contexts,
        "</memory>",
    ]


def make_block_append():
    lines = open(static_path('append.txt')).read().strip().split('\n')

    if len(lines) == 0:
        return 
    
    return [
        "<frame>",
        *lines,
        "</frame>",
    ]


def make_human_content(user_input):
    return "\n".join(process_user_input(user_input) + make_block_context() + make_block_append())


def main():
    global fun_msg, chat_history, user_turn, cycle_num, console, user_exit, rmce_count, rmce_depth
    logger.debug(f"Loop cycle {cycle_num}")
    cycle_num += 1

    if user_turn:
        if rmce_count is not None and rmce_depth is not None and rmce_count < rmce_depth:
            rmce_count += 1
            conv_print(f"> [bold yellow]RMCE Cycle[/] {rmce_count}/{rmce_depth}")
            fun_content = open(user_path('rmce.txt')).read()
            fun_msg = HumanMessage(content=fun_content)
            chat_history.append(fun_msg)
            user_turn = False
        elif args.goal:
            user_line("goal")
            conv_print("> [bold]Pushing for goal[/]")
            goal_input = open(src_path(args.goal)).read()
            fun_content = make_human_content(goal_input)
            fun_msg = HumanMessage(content=fun_content)
            conv_print(escape(fun_content), source="stdin", screen_limit=False)
            # conv_save not calling to prevent flooding of memory
            logger.debug(fun_msg)
            chat_history.append(fun_msg)

            # start the rmce cycle
            user_turn = False
            rmce_count = 1
        elif fun_msg is None:
            user_input = console.input("> [bold red]User:[/] ")
            rmce_depth, rmce_count = None, None

            if user_input is not None:
                user_line("meta: user new message")

            if user_input is not None:
                if user_input.lower() in ('/exit', '/quit'):
                    user_exit.set()
                    fun_content = open(user_path('exit.txt')).read()
                elif user_input.lower().startswith("/rmce"):
                    try:
                        txt = user_input[5:]
                        rmce_depth = int(txt) if len(txt) else 1
                        assert rmce_depth > 0
                        rmce_count = 0
                    except RuntimeError as e:
                        user_print(f"Error understanding '{user_input}', expect: '/rmce <cycle>' where <cycle> > 0 ({str(e)})")
                    return
                elif user_input.lower().startswith("/level"):
                    try:
                        txt = user_input[6:]
                        meta_level = int(txt) if len(txt) else 0
                        assert meta_level > 0
                        args.meta_level = meta_level
                    except RuntimeError as e:
                        user_print(f"Error understanding '{user_input}', expect: '/level <value>' where <value> >= 0 ({str(e)})")
                    return
                else:
                    fun_content = make_human_content(user_input)
            else:
                fun_content = open(user_path('empty.txt')).read()

            fun_msg = HumanMessage(content=fun_content)

            # meta: log=False so that we can do logger.debug below
            conv_print(escape(fun_content), source="stdin", screen_limit=False, log=False)
            conv_save(user_input, source="world")

            user_line("meta: end of user message")

            logger.debug(fun_msg)

            # start the rmce cycle
            chat_history.append(fun_msg)
            user_turn = False

    try:
        reply = jack.invoke(dynamic_history(chat_history, args.user_lookback))
    except Exception as e:
        reply = None
        logger.exception("Problem while executing request")
        conv_print(f"> [bold]Exception happened[/] {escape(str(e))}")

    if reply is None:
        if not user_exit.is_set():
            conv_print("> sleeping for 5 seconds as we didnt get reply (press CTRL-C to exit)")
            user_exit.wait(args.reattempt_delay)
        return

    logger.debug(reply)

    # the message has been accepted
    user_turn = True
    fun_msg = None
    chat_history.append(reply)

    for tool_call in reply.tool_calls:
        tool_name: str = tool_call['name'].lower()

        conv_print(f"> [bold]Tool used[/]: {escape(tool_name)}: {escape(str(tool_call['args']))}")

        try:
            selected_tool = next(x for x in tools if x.name == tool_name)
            tool_output = selected_tool.invoke(tool_call)
        except Exception as e:
            logger.exception("Problem while executing tool_call")
            conv_print(f"> [bold]Exception while calling tool[/] {escape(str(e))}", log=False)
            tool_output = ToolMessage(
                content=exception_to_string(e),
                name=tool_name,
                tool_call_id=tool_call.get('id'),
                status='error',
            )

        conv_print(f"> [bold]Tool output given[/]: {escape(str(tool_output.content))}")
        logger.debug(tool_output)
        chat_history.append(tool_output)

    if len(reply.tool_calls) != 0:
        conv_print("> Tool call pending")
        user_turn = False

    # Print the response and add it to the chat history
    contents = []
    if isinstance(reply.content, str):
        contents.append(reply.content)
    else:
        for r in reply.content:
            if r['type'] == 'text' and len(r['text']) > 0:
                contents.append(r['text'])

    for content in contents:
        conv_print(escape(content), screen_limit=False)
        conv_save(content, source="self")

    if len(contents) == 0 and len(reply.tool_calls) == 0:
        conv_print("> [b red]No content received and no tool use![/b]")


def sigint_hander(sign_num, frame):
    global user_exit
    user_print("> SIGINT detected. exiting")
    user_exit.set()


if __name__ == '__main__':
    conv_print(f"> Welcome to {args.meta}. Type '/exit' to quit.")
    conv_print(f"> Provider selected: [bold]{args.provider}[/]")
    conv_print(f"> Model selected: [bold]{args.model}[/]")
    user_print(f"> temperature: {args.temperature}")
    user_print(f"> max-tokens: {args.max_tokens}")
    user_print(f"> meta: {args.meta}")
    user_print(f"> meta_level: {args.meta_level}")
    user_print(f"> goal: {args.goal}")
    user_print(f"> user_prefix: {args.user_prefix}")
    user_print(f"> user_lookback: {args.user_lookback}")
    user_print(f"> feed_memories: {args.feed_memories}")

    if args.verbose:
        for x, y in vars(args).items():
            user_print(f"> {x}: {y}")

    signal.signal(signal.SIGINT, sigint_hander)

    if fun_msg is not None:
        conv_print(sys_msg.content, source="stdin", screen_limit=False)
        conv_save(sys_msg.content, source="world")

    if fun_msg is not None:
        conv_print(fun_msg.content, source="stdin", screen_limit=False)
        conv_save(fun_msg.content, source="world")

    while not user_exit.is_set():
        main()

    conv_print(f"> Thank you for interacting with {args.meta}. Bye!")
