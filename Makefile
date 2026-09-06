PYTHON ?= python3
CFLOW_DIR := native/cflow
REVERSE_ETA_DIR := native/reverse_eta

.PHONY: all native test audit benchmark benchmark-smoke benchmark-core benchmark-transcription benchmark-full-ocp benchmark-all benchmark-reference benchmark-tests clean

all: native

native:
	$(MAKE) -C $(CFLOW_DIR) all
	$(MAKE) -C $(REVERSE_ETA_DIR) all
	$(MAKE) -C native/segment all
	$(MAKE) -C native/crossing all
	$(MAKE) -C native/reverse all
	$(MAKE) -C native/dd_yaw all

# Public regression suite shipped with this repository snapshot.
test: native
	PYTHONPATH=. $(PYTHON) -m pytest -q tests

benchmark: benchmark-core

benchmark-smoke: native
	PYTHONPATH=. $(PYTHON) -m benchmarks.run --profile smoke --overwrite

benchmark-core: native
	PYTHONPATH=. $(PYTHON) -m benchmarks.run --profile core

benchmark-transcription: native
	PYTHONPATH=. $(PYTHON) -m benchmarks.run --profile transcription

benchmark-full-ocp: native
	PYTHONPATH=. $(PYTHON) -m benchmarks.run --profile full-ocp

benchmark-all: native
	PYTHONPATH=. $(PYTHON) -m benchmarks.run --profile all

benchmark-reference: native
	PYTHONPATH=. $(PYTHON) -m benchmarks.run --profile all --reference --overwrite

benchmark-tests: native
	PYTHONPATH=. $(PYTHON) -m pytest -q benchmarks/tests -m "not slow"

audit: native
	$(MAKE) -C $(CFLOW_DIR) audit
	$(MAKE) -C $(REVERSE_ETA_DIR) audit
	$(MAKE) -C native/segment audit
	$(MAKE) -C native/reverse audit
	$(MAKE) -C native/dd_yaw audit
	ldd ./native/crossing/libame_crossing.so || true
	nm -D --defined-only ./native/crossing/libame_crossing.so | grep ' ame_crossing_' || true

clean:
	$(MAKE) -C native/dd_yaw clean
	$(MAKE) -C native/reverse clean
	$(MAKE) -C native/crossing clean
	$(MAKE) -C native/segment clean
	$(MAKE) -C $(CFLOW_DIR) clean
	$(MAKE) -C $(REVERSE_ETA_DIR) clean
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache
