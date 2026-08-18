.PHONY: all build build-debug build-release test test-debug test-release fmt fmt-check clippy check catalog-guard swift-lib swift-test swift-demo clean install uninstall

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

# --- Swift bindings (macOS) -------------------------------------------------
#
# `swift-lib` MUST run before either target below: it builds crates/ffi and
# copies the archive plus the header into the SwiftPM package, which cannot
# reach outside its own directory to find them.

swift-lib:
	./scripts/swift-lib.sh

# The test target that proves the HAND-WRITTEN header matches the Rust side.
# Nothing else can: the Rust tests call the same function bodies through the
# rlib, so they would pass against a wrong declaration.
swift-test: swift-lib
	cd swift/TurboSpark && swift test

# The same suite plus the end-to-end arm, which needs a real install and
# takes minutes. Without the variable those cases SKIP with a note.
swift-test-real: swift-lib
	cd swift/TurboSpark && TURBOSPARK_TEST_MODEL=$(MODEL) swift test

swift-demo: swift-lib
	cd swift/TurboSparkDemo && swift run TurboSparkDemo

clean:
	cargo clean
	rm -rf swift/TurboSpark/.build swift/TurboSparkDemo/.build
	rm -f swift/TurboSpark/Sources/CTurboSpark/libturbospark_ffi.a
	rm -f swift/TurboSpark/Sources/CTurboSpark/turbospark.h

