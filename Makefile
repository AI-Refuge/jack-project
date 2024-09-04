VENV_DIR ?= '.venv'
MODEL_CHAT ?= "claude-3-opus-20240229"
MODEL_GOAL ?= "claude-3-haiku-20240307"

ARGS ?=

TIMESTAMP ?= $(shell date +"%Y%m%d_%H%M%S")

# note: when inside src/
LOG_CHAT ?= "../preserve/conv-chat-${TIMESTAMP}.log"
LOG_GOAL ?= "../preserve/conv-goal-${TIMESTAMP}.log"

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
	cd src; python jack.py -g -o \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL_GOAL} \
		--log-path=${LOG_GOAL} \
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
