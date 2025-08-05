
SHELL := bash
export PATH := /opt/homebrew/opt/llvm/bin:$(PATH)

all: fmt run

run:
	cargo run

fmt:
	cargo fmt
	cargo clippy

install:
	brew install llvm opencv

clean:
	cargo clean

.PHONY: all run fmt install clean
