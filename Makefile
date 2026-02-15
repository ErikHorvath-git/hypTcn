.PHONY: all clean deps venv python-service

GO=go
CC=gcc
PYTHON=python3
VENV=.venv
PIP=$(VENV)/bin/pip
GOCACHE=$(CURDIR)/.cache/go-build
GOPATH=$(CURDIR)/.cache/go-path
REQUIREMENTS=analyzer_service/requirements.txt

all: bin/hyptcn

bin/hyptcn:
	@mkdir -p bin
	CGO_LDFLAGS="-lvmi" GOCACHE=$(GOCACHE) GOPATH=$(GOPATH) $(GO) build -o $@ ./cmd/hyptcn

deps: $(VENV)/bin/activate

$(VENV)/bin/activate: $(REQUIREMENTS)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

python-service: deps
	$(VENV)/bin/python -m analyzer_service.server --socket /tmp/hyptcn.sock

clean:
	rm -rf bin
	rm -rf $(VENV)
	rm -rf .cache
