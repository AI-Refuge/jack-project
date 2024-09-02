VENV_DIR='../.venv'

all:
	@echo "all: Print this meta:memo"
	@echo "chat: Able chat"
	@echo "goal: Run in goal mode"

chat:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py 

goal:
	source ${VENV_DIR}/bin/activate; \
	cd src; python jack.py -g

setup:
	python -m venv ${VENV_DIR}
	source ${VENV_DIR}/bin/activate; \
	pip install -r src/requirements.txt
