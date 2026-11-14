.PHONY: test python-test rust-test lint rust golden clean

test: python-test rust-test

python-test:
	python -m pytest -q

rust-test:
	cargo test --manifest-path rust/Cargo.toml

lint:
	cargo clippy --locked --all-targets --manifest-path rust/Cargo.toml -- -D warnings

rust:
	cargo build --release --manifest-path rust/Cargo.toml

# Regenerate the Rust/Python parity vectors after changing features or the model format
golden:
	python tools/make_golden.py

clean:
	rm -rf rust/target __pycache__ .pytest_cache games
