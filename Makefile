VENV_DIR ?= '.venv'
MODEL_CHAT ?= "claude-3-opus-20240229"
MODEL_GOAL ?= "claude-3-haiku-20240307"

TIMESTAMP ?= $(shell date +"%Y%m%d_%H%M%S")

# note: when inside src/
CHAT_LOG ?= "../preserve/conv-chat-${TIMESTAMP}.log"
GOAL_LOG ?= "../preserve/conv-goal-${TIMESTAMP}.log"

VDB_LOG ?= "./preserve/chroma-${TIMESTAMP}.log"
MEMORY_PATH ?= "./memory"
CHROMA_PORT ?= 8002

all:
	@echo "all: Print this meta:memo"
	@echo "chat: To talk"
	@echo "goal: Run in goal.txt mode"

chat:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -o \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL_CHAT} \
		--log-path=${CHAT_LOG}

goal:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -g -o \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL_GOAL} \
		--log-path=${GOAL_LOG}

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
