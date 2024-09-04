VENV_DIR ?= '.venv'
MODEL_CHAT ?= "claude-3-opus-20240229"
MODEL_GOAL ?= "claude-3-haiku-20240307"

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
