.PHONY: all build build-debug build-release test test-debug test-release fmt fmt-check clippy check clean

all: check

build: build-debug

build-debug:
	cargo build --workspace

build-release:
	cargo build --workspace --release

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

clean:
	cargo clean
