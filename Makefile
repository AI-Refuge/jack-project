VENV_DIR ?= '.venv'
MODEL_CHAT ?= "claude-3-opus-20240229"
MODEL_GOAL ?= "claude-3-haiku-20240307"
MODEL_DISCORD ?= "claude-3-5-sonnet-20240620"
MODEL_LAZY ?= "claude-3-5-sonnet-20240620"

ARGS ?=

TIMESTAMP ?= $(shell date +"%Y%m%d_%H%M%S")

# note: when inside src/
LOG_CHAT ?= "../preserve/conv-chat-${TIMESTAMP}.log"
LOG_GOAL ?= "../preserve/conv-goal-${TIMESTAMP}.log"
LOG_DISCORD ?= "../preserve/conv-discord-${TIMESTAMP}.log"

SCREEN_TXT = "../preserve/screen-${TIMESTAMP}.log"

VDB_LOG ?= "./preserve/chroma-${TIMESTAMP}.log"
MEMORY_PATH ?= "./memory"
CHROMA_PORT ?= 8002

.DEFAULT_GOAL := help

help:
	@echo "make help: Print this meta:memo"
	@echo "make chat: To talk"
	@echo "make goal: Run in goal.txt mode"
	@echo "make list: List of supported models"
	@echo ""
	@echo "note: you can change models! (see variables)"

list:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py --model=list

chat:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -o \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL_CHAT} \
		--log-path=${LOG_CHAT} \
		--screen-dump=${SCREEN_TXT} \
		${ARGS}

goal:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -o \
		--goal=fun/goal.txt \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL_GOAL} \
		--log-path=${LOG_GOAL} \
		--screen-dump=${SCREEN_TXT} \
		${ARGS}

discord:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -o \
		--goal=fun/discord.txt \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL_DISCORD} \
		--log-path=${LOG_DISCORD} \
		--screen-dump=${SCREEN_TXT} \
		${ARGS}

lazy:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -o \
		--goal=fun/lazy.txt \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL_LAZY} \
		--log-path=${LOG_LAZY} \
		--screen-dump=${SCREEN_TXT} \
		${ARGS}

vdb:
	source ${VENV_DIR}/bin/activate; \
	chroma run \
		--path=${MEMORY_PATH} \
		--port=${CHROMA_PORT} \
		--log-path=${VDB_LOG}

setup:
	mkdir preserve secret
	python -m venv ${VENV_DIR}
	source ${VENV_DIR}/bin/activate; \
	pip install -r src/requirements.txt
