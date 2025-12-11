# llmlib

<!-- TOC -->
- [llmlib](#llmlib)
  - [introduction](#introduction)
  - [Package Overview](#package-overview)
  - [Installation (editable / dev mode)](#installation-editable--dev-mode)
  - [Running Tests](#running-tests)
  - [Working on the Code (reload changes)](#working-on-the-code-reload-changes)
  - [Building a Package (optional)](#building-a-package-optional)

<!-- /TOC -->


## introduction

Small utility library for playing with tiny Transformer / GPT-style models.

It provides:
- a simple character-level tokenizer,
- a HuggingFace-style TinyConfig for defining model hyperparameters,
- a small TinyTransformerModel,
- clean I/O helpers for saving/loading models + configs,
- reproducible experiment structure.

---

## Package Overview

```
llmlib/
├── pyproject.toml            # Build + packaging config (PEP 621)
├── ReadMe.md                 # This file
├── src/
│   ├── llmlib/               # Library source code
│   │   ├── __init__.py
│   │   ├── io.py             # I/O helpers: configs, model & data paths, save/load
│   │   ├── tiny_config.py    # TinyConfig (HF-style config for tiny Transformer)
│   │   ├── tiny_model.py     # TinyTransformerModel implementation
│   │   └── tokenizer.py      # Simple char-level tokenizer + VOCAB_SIZE
│   └── llmlib.egg-info/      # Generated metadata (created by packaging tools)
└── test/
    ├── __init__.py
    ├── ReadMe.md             # Notes about tests (optional)
    ├── test_io.py            # Tests / smoke checks for io.py
    └── test_tiny_model.py    # Tests / smoke checks for tiny_model.py

```
* src/llmlib: the actual library code you import.
* test: tests for the library (run via pytest or similar).
* llmlib.egg-info: auto-generated packaging metadata (you don’t edit this manually).


## Installation (editable / dev mode)
From the `llmlib` root directory:

 🔹 Step 1 — Activate your venv
    ```
    source ~/.venv/llm_course/bin/activate
    ```

🔹 Step 2 — install the library in editable mode
   ```
   cd ~/PoojaVault/Professional/Workbench/Tools/Python/PLibraries/llmlib
   
   ```
   `-e` = editable mode: changes in `src/llmlib/*.py` are picked up immediately without reinstalling.

🔹 Step 3 — Verify
   ```
   $ python -c "import llmlib, inspect; print(llmlib.__file__)"
   $ $HOME/PoojaVault/Professional/Workbench/Tools/Python/PLibraries/llmlib/src/llmlib/__init__.py
   ```

🔹 Step 3 — To uninstall later
   ```
   pip uninstall llmlib
   ```

## Running Tests
From the `llmlib` root:
```
# Recommended:
$ python -m pytest
=================================================== test session starts ===================================================
platform linux -- Python 3.11.0, pytest-9.0.1, pluggy-1.6.0
rootdir: /home/pooja-saxena/PoojaVault/Professional/Workbench/Tools/Python/PLibraries/llmlib
configfile: pyproject.toml
plugins: anyio-4.11.0
collected 5 items                                                                                                         

test/test_io.py .                                                                              [ 20%]
test/test_tiny_model.py ...                                                                    [ 80%]
test/test_tokenizer. py .                                                                      [100%]

=========================================== 5 passed in 2.67s ========================================


# Or specifically:
$ python -m pytest test/test_io.py
$ python -m pytest test/test_tiny_model.py
```
> Make sure the same virtualenv that has `llmlib` installed is active.


## Working on the Code (reload changes)

Because you installed with `pip install -e` :
* Any change in `src/llmlib/*.py` is visible immediately in your projects that do import llmlib.
* You do not need to reinstall after each edit.

If Python keeps using stale bytecode, you can safely delete:
```
find src/llmlib -name "*.pyc" -delete
find src/llmlib -name "__pycache__" -type d -exec rm -rf {} +
```


## Building a Package (optional)

If you ever want to build a wheel / sdist:
```
# Install build tools once:
pip install build

# From llmlib root:
python -m build
```

This will create `dist/` with `.whl` and `.tar.gz` files that you can install elsewhere:
```
pip install dist/llmlib-<version>-py3-none-any.whl
```
* Version and filename are taken from pyproject.toml.

