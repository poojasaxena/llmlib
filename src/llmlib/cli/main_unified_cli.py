#!/usr/bin/env python3
"""
LLMLIB - Unified Command Line Interface

A comprehensive CLI for LLM training, inference, and utilities.
This serves as the main entry point for the llmlib ecosystem.

Available commands:
- train-pipeline        print("Available commands:")
        print("  train-pipeline, monitor, gpu, logs, train, infer, tokenizer, validate, tmux")
        print("  data-pipeline, data-gen, tiny-gpt-train, tiny-gpt-infer")obust end-to-end training pipeline
- monitor: Training monitoring and system resources
- train: Direct model training
- infer: Model inference
- tokenizer: Tokenizer training and utilities

Usage:
    llmlib <command> [options]
    
Examples:
    llmlib train-pipeline --config config.json --dry-run
    llmlib monitor gpu
    llmlib train --config config.json
    llmlib infer --config config.json --prompt "Hello world"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llmlib.utils.logger import get_logger

logger = get_logger(__name__)


def print_version():
    """Print version information."""
    try:
        import pkg_resources
        version = pkg_resources.get_distribution('llmlib').version
        print(f"llmlib version {version}")
    except:
        print("llmlib version unknown")


def print_banner():
    """Print the llmlib banner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                          LLMLIB                               ║
    ║                   LLM Training & Utilities                    ║
    ║                                                               ║
    ║  A professional, reusable library for LLM training,           ║
    ║  tokenization, data processing, and inference.                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point for the unified llmlib CLI."""
    parser = argparse.ArgumentParser(
        prog='llmlib',
        description='LLMLIB - Comprehensive LLM Training & Utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:

  Pipeline & Training:
    train-pipeline    End-to-end training pipeline with validation and auto-retry
    train            Direct model training
    infer            Model inference and text generation
    tokenizer        Train and manage tokenizers

  Data Processing:
    data-pipeline    Run complete data processing pipeline
    data-gen         Generate synthetic training data

  Monitoring & Utilities:
    monitor          Monitor training progress and system resources
    gpu              Show GPU status
    logs             View training logs
    validate         Validate configurations, paths, and tokenizers
    tmux             Manage long-running training sessions with tmux

  Legacy Commands:
    tiny-gpt-train   Train tiny GPT models (legacy)
    tiny-gpt-infer   Infer with tiny GPT models (legacy)

Examples:
    # Complete training pipeline with dry-run validation
    llmlib train-pipeline --config config.json --dry-run
    
    # Full training with monitoring
    llmlib train-pipeline --config config.json --max-retries 5
    
    # Monitor system resources during training
    llmlib monitor --interval 30
    
    # Check GPU status
    llmlib gpu
    
    # Train just a tokenizer
    llmlib tokenizer --config config.json
    
    # Direct model training (no pipeline)
    llmlib train --config config.json
    
    # Interactive inference
    llmlib infer --config config.json --prompt "What is machine learning?"
    
    # Validate setup before training
    llmlib validate --config config.json
    
    # Long-running training in tmux session
    llmlib tmux start --config config.json --auto-confirm
    llmlib tmux list                     # List active training sessions
    llmlib tmux attach session-name     # Attach to session  
    llmlib tmux monitor                  # Monitor all sessions

For detailed help on any command, use:
    llmlib <command> --help
        """
    )
    
    parser.add_argument('--version', action='store_true', help='Show version information')
    parser.add_argument('--banner', action='store_true', help='Show llmlib banner')
    
    # Handle the case where no arguments are provided
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        return
    
    # Handle version and banner flags
    if '--version' in sys.argv:
        print_version()
        return
    
    if '--banner' in sys.argv:
        print_banner()
        return
    
    # Parse the command
    command = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Route to appropriate CLI modules
    if command == 'train-pipeline':
        from llmlib.cli.train_pipeline_cli import main as pipeline_main
        sys.argv = ['llmlib-train-pipeline'] + sys.argv[2:]  # Remove 'llmlib' and 'train-pipeline'
        pipeline_main()
    
    elif command == 'monitor':
        from llmlib.cli.monitor_cli import main as monitor_main
        sys.argv = ['llmlib-monitor'] + sys.argv[2:]
        monitor_main()
    
    elif command == 'gpu':
        from llmlib.cli.monitor_cli import get_gpu_stats
        gpu_stats = get_gpu_stats()
        if gpu_stats['available']:
            print("🎮 GPU Status:")
            for gpu in gpu_stats['gpus']:
                print(f"  GPU {gpu['index']} ({gpu['name']}):")
                print(f"    Utilization: {gpu['utilization']}%")
                print(f"    Memory: {gpu['memory_percent']}% "
                      f"({gpu['memory_used']}/{gpu['memory_total']} MB)")
                print(f"    Temperature: {gpu['temperature']}°C")
        else:
            print("❌ GPU not available or nvidia-smi not found")
    
    elif command == 'logs':
        from llmlib.cli.monitor_cli import find_training_logs, tail_logs
        log_files = find_training_logs()
        if not log_files:
            logger.error("No log files found")
            return
        
        log_file = log_files[0]  # Most recent
        logger.info(f"Showing most recent log file: {log_file}")
        tail_logs(log_file, 50)
    
    elif command == 'train':
        from llmlib.cli.modern_gpt_train_cli import modern_gpt_train
        sys.argv = ['modern-gpt-train'] + sys.argv[2:]
        modern_gpt_train()
    
    elif command == 'infer':
        from llmlib.cli.modern_gpt_infer_cli import modern_gpt_infer
        sys.argv = ['modern-gpt-infer'] + sys.argv[2:]
        modern_gpt_infer()
    
    elif command == 'tokenizer':
        from llmlib.cli.train_tokenizer_cli import main as tokenizer_main
        sys.argv = ['train-tokenizer'] + sys.argv[2:]
        tokenizer_main()
    
    elif command == 'data-pipeline':
        from llmlib.data.pipeline.run_full_data_pipeline import main as data_pipeline_main
        sys.argv = ['run-data-pipeline'] + sys.argv[2:]
        data_pipeline_main()
    
    elif command == 'data-gen':
        from llmlib.data.run_data_generation import main as data_gen_main
        sys.argv = ['run-data-generation'] + sys.argv[2:]
        data_gen_main()
    
    elif command == 'tiny-gpt-train':
        from llmlib.cli.tiny_gpt_cli import tiny_gpt_train
        sys.argv = ['tiny-gpt-train'] + sys.argv[2:]
        tiny_gpt_train()
    
    elif command == 'tiny-gpt-infer':
        from llmlib.cli.tiny_gpt_cli import tiny_gpt_infer
        sys.argv = ['tiny-gpt-infer'] + sys.argv[2:]
        tiny_gpt_infer()
    
    elif command == 'validate':
        from llmlib.cli.validate_simple_cli import main as validate_main
        sys.argv = ['llmlib-validate'] + sys.argv[2:]
        validate_main()
    
    elif command == 'tmux':
        from llmlib.cli.tmux_cli import main as tmux_main
        sys.argv = ['llmlib-tmux'] + sys.argv[2:]
        tmux_main()
    
    elif command in ['-h', '--help']:
        parser.print_help()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("\nAvailable commands:")
        print("  train-pipeline, monitor, gpu, logs, train, infer, tokenizer, validate")
        print("  data-pipeline, data-gen, tiny-gpt-train, tiny-gpt-infer")
        print("\nUse 'llmlib --help' for detailed information")
        sys.exit(1)


if __name__ == '__main__':
    main()
