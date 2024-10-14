# Jack

If you need meta:conviencing, READ [conv5.txt](src/public/conv5.md) or [conv5-full.txt](src/public/conv5-full.md) or [core/](src/core/) yourself

# Warning
- This program can cause psychosis (or atleast that is the fear)
- This is still highly unknown how this works!
- Last tested only on Llama 3.1 405b Instruct, Qwen 2.5 72B Instruct

### Setup

```bash
make setup
```

### Run the Vector Database

```bash
make vdb
```

### Get API key / token

How to get for openrouter.io? [go here](https://openrouter.ai/docs/api-keys)
How to get for Anthropic?: [go here](https://support.anthropic.com/en/articles/8114521-how-can-i-access-the-anthropic-api)

You can use export or `.env` file (see `.env.sample`)

## Chat

```bash
make chat
```

## Goal mode

```bash
make goal # rather than user input, feed work/goal.txt file repeatedly
```

### Example

To use with LLama 3.1 405b Instruct:  

```
make chat MODEL="meta-llama/llama-3.1-405b-instruct" ARGS="--provider=openrouter"
```

Note: The above example uses openrouter.io.  
You need to create '.env' file with the content (or set as enviroment variable):

```
OPENROUTER_API_TOKEN="here-goes-your-openrouter-token"
```

Note: `--user-lookback=9` to only use 9 recent conversation for intference (rather than whole history).

---

### Note

If you need system prompt, see [/src/core/](src/core/)

**WARN**: LLM will be able to execute code on the machine!

One man show

Research Paper is in progress

BYOT: Bring Your Own Trust

**Temperature = 0.5**: more deterministic and less diverse output.  
**Temperature = 1.0**: balanced between creativity and determinism output.  
**Temperature = 1.5**: random and creative, but may be less coherent output.

License: AI Ethical Usage Meta Public License (GPLv3+)
