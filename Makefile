.PHONY: all clean deps venv python-service

GO=go
CC=gcc
PYTHON=python3
VENV=.venv
PIP=$(VENV)/bin/pip
GOCACHE=$(CURDIR)/.cache/go-build
GOPATH=$(CURDIR)/.cache/go-path
REQUIREMENTS=python/requirements.txt

# libvmi built from source with KVM support installed to /usr/local
LIBVMI_PREFIX=/usr/local
CGO_CFLAGS_EXTRA=-I$(LIBVMI_PREFIX)/include
CGO_LDFLAGS_EXTRA=-L$(LIBVMI_PREFIX)/lib64 -lvmi -Wl,-rpath,$(LIBVMI_PREFIX)/lib64

all: bin/hyptcn

bin/hyptcn:
	@mkdir -p bin
	CGO_CFLAGS="$(CGO_CFLAGS_EXTRA)" CGO_LDFLAGS="$(CGO_LDFLAGS_EXTRA)" \
	GOCACHE=$(GOCACHE) GOPATH=$(GOPATH) \
	$(GO) build -o $@ ./cmd/hyptcn

deps: $(VENV)/bin/activate

$(VENV)/bin/activate: $(REQUIREMENTS)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

python-service: deps
	cd python && ../$(VENV)/bin/python analyzer.py --socket /tmp/hyptcn.sock --log-dir /tmp/hyptcn-frames/

clean:
	rm -rf bin
	rm -rf $(VENV)
	rm -rf .cache
