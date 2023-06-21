
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# ================================================================
# python
# ================================================================

PACKAGE_NAME := cuda_ext_example

PYTHON_DIRS := $(PACKAGE_NAME) tests
PYTHON := python
PYRIGHT := pyright
PYTEST_ARGS ?= -s --tb=short --disable-pytest-warnings
PYTEST_COMMAND := $(PYTHON) -m pytest $(PYTEST_ARGS)

# remove all build, test, coverage and Python artifacts
.PHONY: clean
clean: clean-build clean-pyc clean-test clean-cpp

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	rm -rf .tox/
	rm -rf .pytest_cache

.PHONY: pyright
pyright:
	$(PYRIGHT) $(PYTHON_DIRS)

.PHONY: lint
lint: ## check style with yapf + isort + docformatter + pyright
	$(PYTHON) -m isort -c --diff $(PYTHON_DIRS)
	$(PYTHON) -m yapf -dr $(PYTHON_DIRS)
	$(PYTHON) -m docformatter -r --wrap-summaries 90 --wrap-descriptions 90 $(PYTHON_DIRS)
	$(PYRIGHT) $(PYTHON_DIRS)

.PHONY: style
style: ## enforce style with yapf + isort + docformatter
	$(PYTHON) -m isort $(PYTHON_DIRS)
	$(PYTHON) -m yapf -rip $(PYTHON_DIRS)
	$(PYTHON) -m docformatter -ri --wrap-summaries 90 --wrap-descriptions 90 $(PYTHON_DIRS)

.PHONY: test
test: ## run tests quickly with the default Python
	$(PYTEST_COMMAND) --doctest-modules $(PACKAGE_NAME)
	$(PYTEST_COMMAND) tests/

.PHONY: test-aws
test-aws:  ## run tests that require working aws credentials (streaming datasets)
	$(PYTEST_COMMAND) tests/ -m aws

.PHONY: dist
dist: clean ## builds source and wheel package
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	ls -l dist

.PHONY: install
install: clean ## install the package to the active Python's site-packages
	pip install -e .


# ================================================================
# cpp / pytorch custom op
# ================================================================

# Uncomment for debugging
# DEBUG := 1
# Pretty build
# Q ?= @

CXX := g++

# PYTHON Header path
PYTHON_HEADER_DIR := $(shell python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')
PYTORCH_INCLUDES := $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LIBRARIES := $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

# CUDA ROOT DIR that contains bin/ lib64/ and include/
# CUDA_DIR := /usr/local/cuda
CUDA_DIR := $(shell python -c 'from torch.utils.cpp_extension import _find_cuda_home; print(_find_cuda_home())')

# Assume pytorch > v1.1
WITH_ABI := $(shell python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')

INCLUDE_DIRS := ./ $(CUDA_DIR)/include

INCLUDE_DIRS += $(PYTHON_HEADER_DIR)
INCLUDE_DIRS += $(PYTORCH_INCLUDES)

# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# SRC_DIR := ./csrc/example_op/   # TODO include other op(s) too
SRC_DIR := ./csrc
OBJ_DIR := ./objs
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
STATIC_LIB := $(OBJ_DIR)/libmy_custom_ops.a

# Do codegen at compile time to save runtime overhead; see
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# for list of configurations and associated gpus; the ones below are probably
# the ones we care about (K40,K80 + all of {pascal,volta,turing,ampere});
# see explanation for all of this here: https://stackoverflow.com/a/35657430
# the last line uses compute_86 to generate PTX rather than SASS (binary),
# so that future gpus can compile it for themselves; can remove this
# to include no ptx in the binary and better obfuscate internals
#
# put CUDA_ARCH here to also target K40,K80
# 		-gencode arch=compute_35,code=sm_35 \
# 		-gencode arch=compute_37,code=sm_37 \
# 		-gencode arch=compute_52,code=sm_52 \
# 		-gencode arch=compute_60,code=sm_60 \
# 		-gencode arch=compute_61,code=sm_61 \
# 		-gencode arch=compute_62,code=sm_62 \
#
# Currently just targets Volta, Turing, and Ampere
CUDA_ARCH := \
		-gencode arch=compute_70,code=sm_70 \
		-gencode arch=compute_72,code=sm_72 \
		-gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_80,code=sm_80 \
		-gencode arch=compute_86,code=sm_86 \
		-gencode arch=compute_86,code=compute_86  # ship ptx for fwd compatibility

# 		remove the above line and uncomment these lines if you have Hopper support
# 		-gencode arch=compute_90,code=compute_90 \
# 		-gencode arch=compute_90,code=compute_90  # uncomment to ship ptx

# We will also explicitly add stdc++ to the link target.
# LIBRARIES += stdc++ cudart c10 caffe2 torch torch_python caffe2_gpu
LIBRARIES += stdc++ cudart torch torch_python

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	# https://gcoe-dresden.de/reaching-the-shore-with-a-fog-warning-my-eurohack-day-4-morning-session/
	NVCCFLAGS += -g -G # -rdc true
else
	COMMON_FLAGS += -DNDEBUG -O3
endif

WARNINGS := -Wall -Wno-sign-compare -Wcomment

INCLUDE_DIRS += $(BLAS_INCLUDE)

# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) \
	-DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=$(WITH_ABI)
CXXFLAGS += -pthread -fPIC -fwrapv -std=c++17 $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -std=c++17 -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
TIDY_FLAGS := '--warnings-as-errors=*'
TIDY_FLAGS += $(foreach flag,$(CXXFLAGS),--extra-arg $(flag))


cpp: $(STATIC_LIB)  ## build C++/CUDA code

$(OBJ_DIR):
	@ mkdir -p $@
	@ mkdir -p $@/cuda

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@ echo CXX $<
	$(Q)$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@ echo NVCC $<
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(STATIC_LIB): $(OBJS) $(CU_OBJS) | $(OBJ_DIR)
	$(RM) -f $(STATIC_LIB)
	$(RM) -rf build dist
	@ echo LD -o $@
	ar rc $(STATIC_LIB) $(OBJS) $(CU_OBJS)

.PHONY: clean-cpp
clean-cpp:  ## remove C++ build outputs
	@- $(RM) -rf $(OBJ_DIR) build dist

# to get clang-tidy on macos, `brew install llvm`
.PHONY: lint-cpp
lint-cpp:  # lint C++ code; needs to run in env with clang + needed libs
	clang-tidy --format-style=file $(TIDY_FLAGS) $(CPP_SRCS) --

.PHONY: style-cpp
style-cpp:  # lint C++ code; needs to run in env with clang + needed libs
	clang-tidy -fix --format-style=file $(TIDY_FLAGS) $(CPP_SRCS) --

.PHONY: reinstall
reinstall:  ## pip uninstall and reinstall
	pip uninstall -y $(PACKAGE_NAME) && \
	pip install -e .

.PHONY: fulltest
fulltest:  ## pip uninstall, clean, build C++, reinstall, run tests
	pip uninstall -y $(PACKAGE_NAME) && \
	make clean && \
	pip install -e . && \
	pytest tests
