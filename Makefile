VENV_DIR ?= '.venv'
DEF_MODEL ?= "claude-3-opus-20240229"
MODEL_CHAT ?= ${DEF_MODEL}
MODEL_GOAL ?= ${DEF_MODEL}

all:
	@echo "all: Print this meta:memo"
	@echo "chat: To talk"
	@echo "goal: Run in goal.txt mode"

chat:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -o --model=${MODEL_CHAT}

goal:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -g -o --model=${MODEL_GOAL}

vdb:
	source ${VENV_DIR}/bin/activate; \
	chroma run --path=memory --port=8002

setup:
	python -m venv ${VENV_DIR}
	source ${VENV_DIR}/bin/activate; \
	pip install -r src/requirements.txt
