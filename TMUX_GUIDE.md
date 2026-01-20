# 🔄 TMux Integration Guide: Long-Running Training Sessions

## 🎯 **The Problem You Identified**

You're absolutely right! For long model training (4-8+ hours), we need proper session management with tmux. The new llmlib v3.1 now includes **professional tmux integration** for persistent training sessions.

## 🚀 **New TMux-Based Training Workflow**

### **1. Quick Start - Long Training in TMux**
```bash
# Start training in persistent tmux session
llmlib tmux start --config config.json --auto-confirm

# Check status of all training sessions
llmlib tmux status

# Attach to training session to see progress
llmlib tmux attach session-name

# Monitor all sessions with live dashboard
llmlib tmux monitor
```

### **2. Complete Workflow Example**
```bash
# Step 1: Validate configuration first (quick check)
llmlib train-pipeline --config config.json --dry-run

# Step 2: Start long training in tmux session 
llmlib tmux start --config config.json --name elephant-training --auto-confirm

# Step 3: Detach and do other work
# (Session continues running in background)

# Step 4: Check status anytime
llmlib tmux list

# Step 5: Attach to see live progress
llmlib tmux attach elephant-training

# Step 6: Monitor multiple sessions
llmlib tmux monitor  # Live dashboard, updates every 5 seconds
```

## 📋 **TMux Commands Reference**

### **Session Management**
```bash
# Start training session
llmlib tmux start --config path/to/config.json [options]

# List all active training sessions  
llmlib tmux list
llmlib tmux status  # Same as list

# Attach to session (interactive or by name)
llmlib tmux attach                    # Choose from list
llmlib tmux attach session-name      # Direct attach

# Monitor all sessions with live updates
llmlib tmux monitor

# Kill sessions
llmlib tmux kill                     # Choose from list  
llmlib tmux kill session-name       # Kill specific session
llmlib tmux kill --all              # Kill all training sessions
```

### **Advanced Start Options**
```bash
# Custom session name
llmlib tmux start --config config.json --name my-experiment

# Dry run in tmux (for testing session setup)  
llmlib tmux start --config config.json --dry-run

# Custom retry and timeout settings
llmlib tmux start --config config.json --max-retries 5 --timeout 12

# Auto-confirm (no interactive prompts)
llmlib tmux start --config config.json --auto-confirm
```

## 🎮 **Session Features**

### **Automatic Session Naming**
Sessions are auto-named with timestamp and config:
```
llmlib-elephant_config-1218_1345    # elephant_config on Dec 18 at 13:45
llmlib-domain_gpt-1218_0930         # domain_gpt on Dec 18 at 09:30
```

### **Session Information Display**
```bash
$ llmlib tmux list

🚀 Active Training Sessions (2):
================================================================================
Session: llmlib-elephant_config-1218_1345
  Status: 🔌 Detached
  Created: 2025-12-18 13:45:23
  Recent: ...✅ Tokenizer training completed at: 2025-12-18 13:46:15

Session: llmlib-domain_gpt-1218_0930  
  Status: 📺 Attached
  Created: 2025-12-18 09:30:12
  Recent: ...🧠 Step 2: Starting robust model training...
```

### **Live Monitoring Dashboard**
```bash
$ llmlib tmux monitor

🖥️  LLMLIB Training Monitor (tmux sessions)
🕒 2025-12-18 13:47:30
================================================================================

🚀 llmlib-elephant_config-1218_1345
   Status: 🔌 Detached
   Created: 2025-12-18 13:45:23
   📝 🧠 Step 2: Starting robust model training...
   📝 💡 Training will auto-retry up to 3 times if it fails
   📝 🔄 Training attempt 1/3...

🚀 llmlib-domain_gpt-1218_0930
   Status: 📺 Attached  
   Created: 2025-12-18 09:30:12
   📝 Epoch 15/20 - Loss: 2.341 - Time: 2h 15m
   📝 🎯 Step 3: Testing inference...
   📝 ✅ Training completed at: 2025-12-18 13:47:12

================================================================================
Press Ctrl+C to stop monitoring
```

## 🔧 **Integration with Existing Workflow**

### **Option 1: Direct Training (Interactive)**
```bash
llmlib train-pipeline --config config.json    # Interactive, runs in foreground
```

### **Option 2: TMux Training (Long-running)**  
```bash
llmlib tmux start --config config.json        # Persistent, runs in background
```

### **Option 3: Hybrid Workflow**
```bash
# Quick validation
llmlib train-pipeline --config config.json --dry-run

# If validation passes, start long training in tmux
llmlib tmux start --config config.json --auto-confirm
```

## 🎯 **Best Practices**

### **For Short Experiments (< 30 minutes)**
```bash
llmlib train-pipeline --config config.json    # Direct training
```

### **For Long Training (> 30 minutes)**
```bash
llmlib tmux start --config config.json --auto-confirm    # TMux session
```

### **For Multiple Experiments**
```bash
# Start multiple sessions with descriptive names
llmlib tmux start --config experiment1.json --name exp1-baseline
llmlib tmux start --config experiment2.json --name exp2-larger-model  
llmlib tmux start --config experiment3.json --name exp3-more-data

# Monitor all at once
llmlib tmux monitor
```

### **For Development/Testing**
```bash
# Test session setup without actual training
llmlib tmux start --config config.json --dry-run --name test-session
```

## 🛠️ **TMux Session Management**

### **Session Persistence**
- Sessions survive SSH disconnections
- Sessions continue after closing terminal
- Sessions persist through system reboots (if tmux server survives)

### **Session Recovery**
```bash
# After reconnecting to server
llmlib tmux list           # See what's still running
llmlib tmux attach         # Reconnect to session
```

### **Working Directory**
Sessions start in the config file's directory, so relative paths work correctly.

### **Error Handling**
- All the robust error handling from `train_pipeline_cli` is preserved
- Auto-retry, timeout, and system resource management still work
- Session output is captured for monitoring

## 🎉 **Summary: Complete Training Solutions**

You now have **three levels** of training commands:

1. **Quick/Interactive**: `llmlib train-pipeline --config config.json`
2. **Long/Persistent**: `llmlib tmux start --config config.json --auto-confirm`  
3. **Monitoring/Management**: `llmlib tmux monitor`

This solves your tmux requirement while maintaining all the robustness, validation, and monitoring features we built! 🚀

### **Perfect for Your Use Case:**
```bash
# Your typical workflow now:
llmlib train-pipeline --config config.json --dry-run    # Quick validation
llmlib tmux start --config config.json --auto-confirm   # Long training in tmux
llmlib tmux monitor                                      # Monitor in another terminal

# Hours later, check results:
llmlib tmux attach session-name                         # See training progress
llmlib infer --config config.json --prompt "test"      # Test the trained model
```

**Problem solved!** 🎯✨
