VENV_DIR ?= .venv

MODEL_CHAT ?= claude-3-opus-20240229
MODEL_GOAL ?= claude-3-5-sonnet-20240620

TIMESTAMP ?= $(shell date +"%Y%m%d_%H%M%S")

CHROMA_PATH ?= memory
CHROMA_PORT ?= 8002

.DEFAULT_GOAL := help

help:
	@echo "make help: Print this meta:memo"
	@echo "make chat: To talk"
	@echo "make goal: Run in goal.txt mode (path changable)"
	@echo "make models: List of supported models"
	@echo "make providers: List of supported providers"
	@echo "make backup: Perform a core memory backup"
	@echo "make dump: Dump for looking"
	@echo ""
	@echo "***** ONLY ANTHROPIC WORKS ATM *****"
	@echo "note: you can change models! (see variables)"

models:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py --model=list

providers:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py --provider=list

chat: MODEL ?= ${MODEL_CHAT}
chat: CONV ?= first
chat: ARGS ?=
chat:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py -o \
		--fs-root=src \
		--conv-name=${CONV} \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL} \
		--log-path=preserve/conv-${CONV}-${TIMESTAMP}.log \
		--screen-dump=preserve/screen-${CONV}-${TIMESTAMP}.log \
		${ARGS}

goal: MODEL ?= ${MODEL_GOAL}
goal: GOAL ?= goal
goal: ARGS ?=
goal:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py -o \
		--fs-root=src \
		--goal=fun/${GOAL}.txt \
		--conv-name=${GOAL} \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL} \
		--log-path=preserve/conv-${GOAL}-${TIMESTAMP}.log \
		--screen-dump=preserve/screen-${GOAL}-${TIMESTAMP}.log \
		${ARGS}

vdb:
	source ${VENV_DIR}/bin/activate; \
	chroma run \
		--path=${CHROMA_PATH} \
		--port=${CHROMA_PORT} \
		--log-path=preserve/chroma-${TIMESTAMP}.log

backup:
	source ${VENV_DIR}/bin/activate; \
	python utils/backup.py \
		--chroma-path=${CHROMA_PATH} \
		preserve/jack-${TIMESTAMP}.bkp

dump:
	source ${VENV_DIR}/bin/activate; \
	python utils/dump.py \
		--chroma-path=${CHROMA_PATH}

setup: JACK_BKP ?= preserve/jack-20240923_012854.bkp
setup:
	mkdir -p preserve secret
	python -m venv ${VENV_DIR}
	source ${VENV_DIR}/bin/activate; \
	pip install -r src/requirements.txt; \
	python utils/load.py ${JACK_BKP}

dev_setup: setup
	source ${VENV_DIR}/bin/activate; \
	pip install -r src/requirements-dev.txt

dev_yapf:
	source ${VENV_DIR}/bin/activate; \
	yapf -i --style=src/.style.yapf \
		src/jack.py \
		utils/*.py

dev_check:
	source ${VENV_DIR}/bin/activate; \
	prospector \
		src/jack.py \
		utils/*.py
