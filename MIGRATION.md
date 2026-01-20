# 🔄 Migration Guide: From Shell Scripts to Professional CLI

## Quick Migration Checklist

### ✅ Before You Start
1. **Backup your existing project workspaces** (just in case)
2. **Verify llmlib v3.0 is installed**: `llmlib --version` 
3. **Test the new CLI**: `llmlib --help`

### 🧹 Clean Up Project Workspaces

For each of your learning projects (e.g., `course4_domain_expert_gpt/projects/4_elephant_twicedata/`):

#### Remove Old Scripts
```bash
# Remove any copied shell scripts
rm run_training_pipeline.sh
rm monitor_training.sh
rm validate_setup.py
# etc.
```

#### Keep Only Essential Files
Your project should now only have:
```
my_project/
├── config.json                      # ✅ Keep - configuration
├── data/                            # ✅ Keep - if local data
│   ├── train.txt
│   └── val.txt  
└── results/                         # ✅ Keep - training outputs
    ├── models/
    ├── tokenizers/
    └── logs/
```

### 🔄 Replace Old Commands

#### Training Pipeline
```bash
# OLD: Shell script in project directory  
cd my_project && ./run_training_pipeline.sh

# NEW: Professional CLI from anywhere
llmlib train-pipeline --config my_project/config.json --dry-run  # validate first
llmlib train-pipeline --config my_project/config.json           # then train
```

#### Individual Operations  
```bash
# OLD: Separate commands
train-tokenizer config.json
modern-gpt-train config.json  
modern-gpt-infer config.json

# NEW: Unified interface (same commands still work, plus unified access)
llmlib tokenizer --config config.json
llmlib train --config config.json
llmlib infer --config config.json --prompt "test"
```

#### Monitoring
```bash
# OLD: Manual monitoring with separate scripts
nvidia-smi
htop
tail -f logs/training.log

# NEW: Professional monitoring dashboard
llmlib monitor --interval 30        # Real-time dashboard
llmlib gpu                          # GPU status
llmlib logs                         # Tail recent logs
```

### 🎯 New Recommended Workflow

#### 1. Always Validate First
```bash
cd /path/to/your/learning/projects
llmlib train-pipeline --config 4_elephant_twicedata/config.json --dry-run
```
This replaces the old "manual path checking" and gives you a comprehensive validation.

#### 2. Run Training with Confidence  
```bash
llmlib train-pipeline --config 4_elephant_twicedata/config.json --max-retries 5
```
This replaces the old shell script with robust Python implementation.

#### 3. Monitor in Separate Terminal
```bash
# Terminal 2
llmlib monitor --interval 30
```

#### 4. Test Results
```bash
llmlib infer --config 4_elephant_twicedata/config.json --prompt "What is an elephant?"
```

### 🔧 Configuration Updates (if needed)

Your existing `config.json` files should work as-is, but you might want to verify paths:

```json
{
  "project_metadata": {
    "tokenizer_save_path": "tokenizers/elephant_tokenizer.json",
    "model_save_path": "models/elephant_gpt",
    "data_path": "datasets/elephant_domain",
    "data_file": "train.txt",
    "val_file": "val.txt"
  }
}
```

The new CLI automatically resolves relative paths using:
- `GLOBAL_DATASETS_DIR` for data and tokenizers  
- `GLOBAL_MODELS_DIR` for models (fallback to DATASETS_DIR)

### 🐛 Troubleshooting

#### "Command not found"
```bash
# Reinstall llmlib
cd /path/to/llmlib
pip install -e .
llmlib --version  # Should show 3.0.0+
```

#### "Config file not found"  
```bash
# Use absolute path
llmlib train-pipeline --config /full/path/to/config.json --dry-run
```

#### "Paths not resolving correctly"
```bash
# Check environment variables
echo $GLOBAL_DATASETS_DIR
echo $GLOBAL_MODELS_DIR

# Or run with debug
llmlib train-pipeline --config config.json --dry-run  # Shows all path resolution
```

### 🏆 Benefits You'll Notice

1. **No more script copying** - everything is in llmlib
2. **Better error messages** - clear validation and debugging info
3. **Consistent interface** - same commands across all projects  
4. **Professional monitoring** - real-time dashboards instead of manual checks
5. **Robust training** - auto-retry and timeout handling built-in
6. **Clean projects** - only configs, data, and results

### 📋 Migration Verification

After migrating a project, verify it works:

```bash
# 1. Validate setup
llmlib train-pipeline --config config.json --dry-run

# 2. Quick training test (with auto-confirm to skip prompt)  
llmlib train-pipeline --config config.json --auto-confirm --timeout 1

# 3. Test inference
llmlib infer --config config.json --prompt "test"
```

## 🎊 You're Ready!

Once you've migrated and tested a few projects successfully, you can safely remove the old experimental scripts from `llmlib/scripts/`. The new architecture is production-ready and much more maintainable.

Happy training with your new professional LLM library! 🚀
