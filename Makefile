.PHONY: all build build-debug build-release test test-debug test-release fmt fmt-check clippy check catalog-guard clean install uninstall

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
BINARIES = turbospark-check turbospark-model turbospark-server turbospark-bench

all: check

build: build-debug

build-debug:
	cargo build --workspace

build-release:
	cargo build --workspace --release

install: build-release
	mkdir -p $(DESTDIR)$(BINDIR)
	for bin in $(BINARIES); do \
		install -m 755 target/release/$$bin $(DESTDIR)$(BINDIR)/$$bin; \
	done

uninstall:
	for bin in $(BINARIES); do \
		rm -f $(DESTDIR)$(BINDIR)/$$bin; \
	done

test: test-debug

test-debug:
	cargo test --workspace

test-release:
	cargo test --workspace --release

fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

clippy:
	cargo clippy --workspace --tests

check: fmt-check clippy test-debug

# The catalog's rot guard. Network, no download: a file list and a HEAD per
# row, ~26 s. Not part of `check` -- it needs the network and it asserts
# something about the outside world rather than about this tree. Run it after
# editing crates/catalog/src/models.json.
catalog-guard:
	cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture

clean:
	cargo clean

