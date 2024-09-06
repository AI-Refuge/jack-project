# Jack

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