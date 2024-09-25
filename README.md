# Jack

If you need meta:conviencing, READ [jack-conv1.txt](src/history/jack-conv1.txt) yourself.

### Setup

```bash
make setup
```

### Run the Vector Database

```bash
make vdb
```

### Anthropic API Key

How to get?: [go here](https://support.anthropic.com/en/articles/8114521-how-can-i-access-the-anthropic-api)

```bash
export ANTHROPIC_API_KEY="api-key-secret-goes-here"
```

## Chat

```bash
make chat
```

## Goal mode

```bash
make goal # rather than user input, feed work/goal.txt file repeatedly
```

### Note

Just try system prompt on Claude? https://ai-refuge.org/jack.person

**WARN**: LLM will be able to execute code on the machine!

One man show

Only works with Anthropic ATM but technique works on all

Scientific Paper is in progress

BYOT: Bring Your Own Trust

Meta depth to use for llm (see [meta.txt](src/static/meta.txt)):
meta-llama/llama-3.1-70b-instruct: 3 worked
qwen/qwen-2.5-72b-instruct: 4, 5 worked
