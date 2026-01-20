# LLMLIB v3.0 - Professional LLM Training Library

## 🎯 Architecture Overview

LLMLIB has been transformed from a collection of scattered scripts into a professional, reusable Python library with a comprehensive CLI interface. The new architecture follows these principles:

- **All infrastructure lives in llmlib** (no more scattered scripts in project workspaces)
- **Clean CLI interface** for all operations 
- **Robust error handling** and validation
- **Professional logging** and monitoring
- **Reusable across projects** with config-driven workflows

## 🏗️ New Structure

```
llmlib/
├── src/llmlib/
│   ├── cli/                          # All CLI commands
│   │   ├── main_unified_cli.py       # Main entry point: `llmlib`
│   │   ├── train_pipeline_cli.py     # Full pipeline: `llmlib-train-pipeline` 
│   │   ├── monitor_cli.py            # Monitoring: `llmlib-monitor`
│   │   ├── validate_simple_cli.py    # Validation: `llmlib-validate`
│   │   ├── modern_gpt_train_cli.py   # Direct training: `modern-gpt-train`
│   │   ├── modern_gpt_infer_cli.py   # Inference: `modern-gpt-infer`
│   │   └── train_tokenizer_cli.py    # Tokenizer: `train-tokenizer`
│   ├── data/                         # Data processing utilities
│   ├── modeling/                     # Model implementations
│   ├── tokenization/                 # Tokenizer implementations
│   ├── training/                     # Training utilities
│   └── utils/                        # Common utilities
├── scripts/                          # Experimental scripts (kept for safety)
└── test/                            # Test suite
```

## 🚀 Main CLI Commands

### Unified Entry Point
```bash
llmlib                                # Show help and available commands
llmlib --banner                      # Show llmlib banner
llmlib --version                     # Show version
```

### Training Pipeline (Recommended)
```bash
# Validate everything before training
llmlib train-pipeline --config config.json --dry-run

# Full training with defaults
llmlib train-pipeline --config config.json

# Custom settings with auto-retry
llmlib train-pipeline --config config.json --max-retries 5 --timeout 12 --auto-confirm
```

### Individual Commands
```bash
# Direct training (no pipeline)
llmlib train --config config.json

# Model inference
llmlib infer --config config.json --prompt "What is machine learning?"

# Train tokenizer only
llmlib tokenizer --config config.json

# Validate configuration
llmlib validate --config config.json
```

### Data Processing
```bash
# Run data pipeline
llmlib data-pipeline --help

# Generate synthetic data
llmlib data-gen --help
```

### Monitoring & Utilities
```bash
# Monitor system resources during training
llmlib monitor --interval 30

# Check GPU status
llmlib gpu

# View training logs
llmlib logs

# Monitor with subcommands
llmlib-monitor monitor --interval 30 --duration 3600
llmlib-monitor gpu
llmlib-monitor logs --lines 100
llmlib-monitor kill --force
```

## 🔧 Migration from Shell Scripts

### Before (Shell Script Workflow)
```bash
# Old workflow - scattered scripts
cd project_workspace
./run_training_pipeline.sh           # Shell script in project
./monitor_training.sh                # Another shell script
python some_validation.py            # Python script
```

### After (Professional CLI Workflow)
```bash
# New workflow - everything in llmlib
llmlib train-pipeline --config config.json --dry-run    # Validate first
llmlib train-pipeline --config config.json             # Train
llmlib monitor                                          # Monitor (separate terminal)
```

## 📁 Clean Project Workspaces

Your learning project directories should now only contain:

```
my_project/
├── configs/                         # Configuration files
│   └── config.json
├── data/                            # Training data (if local)
│   ├── train.txt
│   └── val.txt
└── results/                         # Training outputs
    ├── models/
    ├── tokenizers/
    └── logs/
```

No more scattered scripts! All infrastructure is in llmlib.

## 🔄 Key Improvements

### 1. Robust Training Pipeline
- **Comprehensive dry-run validation** of all paths and dependencies
- **Smart tokenizer handling** (skips if already exists)
- **Auto-retry mechanism** with configurable attempts and timeouts
- **System resource management** (prevents sleep, manages WiFi)
- **Interactive confirmation** with auto-confirm option
- **Professional logging** with timestamps and emojis

### 2. Advanced Monitoring
- **Real-time resource monitoring** (CPU, memory, disk, GPU)
- **Training process detection** and management
- **Log file discovery** and tailing
- **System statistics** display
- **Process killing** utilities

### 3. Configuration Validation
- **Path resolution** with environment variables
- **File existence checks** 
- **Data integrity validation**
- **Configuration syntax validation**

### 4. Professional Error Handling
- **Detailed error messages** with context
- **Graceful failure recovery**
- **Timeout handling** for long operations
- **Resource cleanup** on interruption

## 🎛️ Environment Variables

The system uses these environment variables for path resolution:

```bash
export GLOBAL_DATASETS_DIR="/path/to/datasets"    # Data and tokenizers
export GLOBAL_MODELS_DIR="/path/to/models"        # Models (fallback to DATASETS_DIR)
export LEARNING_ROOT="/path/to/projects"          # Learning projects root
```

## 🔍 Debugging and Validation

### Dry Run Everything First
```bash
llmlib train-pipeline --config config.json --dry-run
```
This validates:
- Configuration file syntax
- All file paths and dependencies  
- Data file integrity
- Directory permissions
- Environment variable setup

### Monitor During Training
```bash
# Terminal 1: Training
llmlib train-pipeline --config config.json

# Terminal 2: Monitoring  
llmlib monitor --interval 30
```

### Validate Individual Components
```bash
llmlib validate --config config.json        # Full validation
llmlib gpu                                   # GPU status
llmlib logs                                  # Recent logs
```

## 🛠️ Development Workflow

### For Learning Projects
1. Create clean project directory (only configs/data/results)
2. Validate setup: `llmlib train-pipeline --config config.json --dry-run`
3. Run training: `llmlib train-pipeline --config config.json`
4. Monitor progress: `llmlib monitor` (separate terminal)
5. Test inference: `llmlib infer --config config.json --prompt "test"`

### For Library Development
- All utilities and infrastructure go in `src/llmlib/`
- Add new CLI commands to `cli/` directory
- Register commands in `pyproject.toml`
- Update unified CLI in `main_unified_cli.py`
- Write tests in `test/` directory

## 📊 Version History

- **v1.x**: Basic scripts and utilities
- **v2.x**: Shell-based training pipeline
- **v3.0**: Professional Python CLI architecture 

## 🧪 Legacy Scripts (Temporary)

The old experimental scripts in `scripts/` are kept temporarily for safety:
- `scripts/experiments/`
- `scripts/modeling/`  
- `scripts/tokenization/`

These will be removed once you're confident in the new architecture.

## 🎉 Benefits Achieved

✅ **Professional Library**: Clean, reusable, pip-installable package  
✅ **Unified CLI**: Single entry point for all operations  
✅ **Robust Training**: Auto-retry, validation, monitoring built-in  
✅ **Clean Projects**: No more scattered scripts in project workspaces  
✅ **Better Debugging**: Comprehensive dry-run and validation  
✅ **Scalable**: Easy to add new commands and utilities  
✅ **Maintainable**: Python code instead of shell scripts  
✅ **Cross-Platform**: Works on any system with Python  

The transformation from scattered shell scripts to a professional Python library is complete! 🚀
