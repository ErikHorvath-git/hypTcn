.PHONY: all clean deps venv python-service

GO=go
CC=gcc
PYTHON=python3
VENV=.venv
PIP=$(VENV)/bin/pip
UVICORN=$(VENV)/bin/uvicorn
GOCACHE=$(CURDIR)/.cache/go-build
REQUIREMENTS=analyzer_service/requirements.txt

all: bin/hyptcn

bin/hyptcn:
	@mkdir -p bin
	GOCACHE=$(GOCACHE) $(GO) build -o $@ ./cmd/hyptcn

deps: $(VENV)/bin/activate

$(VENV)/bin/activate: $(REQUIREMENTS)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

python-service: deps
	$(UVICORN) analyzer_service.server:APP --uds /tmp/hyptcn.sock --log-level info

clean:
	rm -rf bin
	rm -rf $(VENV)
	rm -rf .cache
