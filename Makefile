VENV_DIR ?= .venv

FS_ROOT ?= src

DEFAULT_PROVIDER ?= openrouter
DEFAULT_MODEL ?= deepseek/deepseek-chat

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

models:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py --model=list

providers:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py --provider=list

chat: MODEL ?= ${DEFAULT_MODEL}
chat: PROVIDER ?= ${DEFAULT_PROVIDER}
chat: CONV ?= first
chat: ARGS ?=
chat:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py -o \
		--fs-root=${FS_ROOT} \
		--conv-name=${CONV} \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL} \
		--provider=${PROVIDER} \
		--log-path=preserve/conv-${CONV}-${TIMESTAMP}.log \
		--screen-dump=preserve/screen-${CONV}-${TIMESTAMP}.log \
		${ARGS}

goal: MODEL ?= ${DEFAULT_MODEL}
goal: PROVIDER ?= ${DEFAULT_PROVIDER}
goal: GOAL ?= goal
goal: ARGS ?=
goal:
	source ${VENV_DIR}/bin/activate; \
	python src/jack.py -o \
		--fs-root=${FS_ROOT} \
		--goal=fun/${GOAL}.txt \
		--conv-name=${GOAL} \
		--chroma-port=${CHROMA_PORT} \
		--model=${MODEL} \
		--provider=${PROVIDER} \
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

setup: JACK_BKP ?= preserve/jack-20241004_152934.bkp
setup:
	mkdir -p preserve secret
	python -m venv ${VENV_DIR}
	source ${VENV_DIR}/bin/activate; \
	pip install -r src/requirements.txt; \
	python utils/load.py ${JACK_BKP}

dev_setup:
	source ${VENV_DIR}/bin/activate; \
	pip install -r src/requirements-dev.txt

dev_format:
	source ${VENV_DIR}/bin/activate; \
	yapf -i --style=src/.style.yapf \
		src/jack.py \
		utils/*.py

dev_check:
	source ${VENV_DIR}/bin/activate; \
	prospector \
		src/jack.py \
		utils/*.py

dev_deps:
	source ${VENV_DIR}/bin/activate; \
	pip-compile src/requirements.in; \
	pip-compile src/requirements-dev.in
