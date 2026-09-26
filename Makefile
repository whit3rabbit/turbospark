.PHONY: all build build-debug build-release test test-debug test-release fmt fmt-check clippy check catalog-guard swift-lib swift-lib-app compile-strings swift-test swift-test-real swift-app-build swift-app-release swift-app swift-demo app-bundle dmg clean clean-cargo clean-swift clean-dist clean-tmp install uninstall

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
# turbospark (the unified front end) ships too: the app's Launch
# Claude/Codex buttons and `turbospark start <agent>` resolve it from
# ~/.local/bin.
BINARIES = turbospark turbospark-check turbospark-model turbospark-server turbospark-bench

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

# The app links Syntext's Rust archive as well as TurboSpark's. Keep symbol
# localization out of `swift-lib`, because the binding package needs the
# runtime definition to remain global for its own linker.
swift-lib-app:
	TURBOSPARK_LOCALIZE_RUST_SYMBOLS=1 ./scripts/swift-lib.sh

# Compiles Localization/Localizable.xcstrings into the .lproj resources the
# app actually reads. `swift build`/`swift run` never do this themselves --
# only Xcode's own build phase compiles a String Catalog, and this package
# has no Xcode project -- so without it every localized string is dead JSON
# sitting unused in the resource bundle (swift/docs/SWIFT_SETTINGS_AUDIT.md item 1).
compile-strings:
	./scripts/compile-strings.sh

# The test target that proves the HAND-WRITTEN header matches the Rust side.
# Nothing else can: the Rust tests call the same function bodies through the
# rlib, so they would pass against a wrong declaration.
swift-test: swift-lib
	cd swift/TurboSpark && swift test

# The same suite plus the end-to-end arm, which needs a real install and
# takes minutes. Without the variable those cases SKIP with a note.
#
# BLOCKED is a SECOND install and covers the cases MODEL structurally cannot:
# every speculation assertion is a property of the install it opens, so one
# variable can only ever gate one shape. Point it at a MoE or sub-4-bit
# install (`make swift-test-real MODEL=~/models/qwen38-27b-mtp.gturbo
# BLOCKED=~/models/ornith35b.gturbo`). Unset, those two cases skip like the
# rest.
# IMAGE is the vision fixture path. IMAGE_MODEL is the separate verified image
# install for the native image-session tests. Point MODEL at an install with a
# vision tower when setting IMAGE, and use an `.image.gturbo` install for the
# image-generation arm.
swift-test-real: swift-lib
	cd swift/TurboSpark && TURBOSPARK_TEST_MODEL=$(MODEL) \
	  TURBOSPARK_TEST_MODEL_NO_SPECULATION=$(BLOCKED) \
	  TURBOSPARK_TEST_IMAGE=$(IMAGE) \
	  TURBOSPARK_TEST_IMAGE_MODEL=$(IMAGE_MODEL) swift test

swift-app-build: swift-lib-app compile-strings
	cd swift/TurboSparkApp && swift build -Xbuild-tools-swiftc -suppress-warnings

swift-app-release: swift-lib-app compile-strings
	cd swift/TurboSparkApp && swift build -c release -Xbuild-tools-swiftc -suppress-warnings

swift-app: swift-lib-app compile-strings
	cd swift/TurboSparkApp && swift run -Xbuild-tools-swiftc -suppress-warnings TurboSparkApp

swift-demo: swift-app

# --- Release artifacts (macOS) ----------------------------------------------
#
# `app-bundle` turns the bare SwiftPM executable into a real TurboSpark.app,
# with the three CLI binaries inside it; `dmg` wraps that into the disk image
# the Homebrew cask installs. Both land in `dist/`. Neither is part of `check`:
# they build the whole workspace in release plus the Swift app, which is
# minutes, and CI runs them on its own (see .github/workflows/release.yml).
#
# What they produce is documented in docs/RELEASE.md, including the Gatekeeper
# situation: the bundle is AD-HOC signed, which is not notarization.
app-bundle:
	./scripts/make-app-bundle.sh

dmg: app-bundle
	./scripts/make-dmg.sh

clean-cargo:
	cargo clean

clean-swift:
	rm -rf swift/TurboSpark/.build swift/TurboSparkApp/.build
	rm -f swift/TurboSpark/Sources/CTurboSpark/libturbospark_ffi.a
	rm -f swift/TurboSpark/Sources/CTurboSpark/turbospark.h

clean-dist:
	rm -rf dist

clean-tmp:
	find "$${TMPDIR:-/tmp}" -maxdepth 1 \( -name 'turbospark*' -o -name 'qwen*' -o -name 'TurboSparkTests*' \) -exec rm -rf {} + 2>/dev/null || true

clean: clean-cargo clean-swift clean-dist clean-tmp
