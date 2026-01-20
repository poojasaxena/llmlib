#!/usr/bin/env python3
"""
Robust LLM Training Pipeline CLI

A comprehensive training pipeline that handles:
- Dry-run validation of all paths and dependencies
- Smart tokenizer training (skips if exists)
- Robust model training with auto-retry
- System resource management
- Comprehensive logging and monitoring
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

from llmlib.utils.logger import get_logger
from llmlib.utils.config_util import load_nested_config

logger = get_logger(__name__)


class TrainingPipelineError(Exception):
    """Custom exception for training pipeline errors."""
    pass


class TrainingPipeline:
    """Robust LLM training pipeline with comprehensive validation and auto-recovery."""
    
    def __init__(self, config_path: Path, dry_run: bool = False, max_retries: int = 3, 
                 timeout_hours: int = 8, auto_confirm: bool = False, skip_sudo: bool = False):
        self.config_path = config_path.resolve()
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.timeout_hours = timeout_hours
        self.auto_confirm = auto_confirm
        self.skip_sudo = skip_sudo
        
        # Load and validate config
        self.config = self._load_config()
        self.project_metadata = self.config.get('project_metadata', {})
        
        # Resolve paths
        self.paths = self._resolve_paths()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        if not self.config_path.exists():
            raise TrainingPipelineError(f"Config file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Config loaded: {self.config_path}")
            return config
        except Exception as e:
            raise TrainingPipelineError(f"Failed to load config: {e}")
    
    def _resolve_paths(self) -> Dict[str, Path]:
        """Resolve and validate all required paths."""
        pm = self.project_metadata
        
        # Get base directories from environment
        datasets_dir = os.environ.get('GLOBAL_DATASETS_DIR', '')
        models_dir = os.environ.get('GLOBAL_MODELS_DIR', datasets_dir)  # Fallback to datasets_dir
        
        def resolve_path(path_str: str, base_dir: str = datasets_dir) -> Path:
            """Resolve a path, handling both absolute and relative paths."""
            if not path_str:
                return Path()
            
            path = Path(path_str)
            if path.is_absolute():
                return path
            elif base_dir:
                return Path(base_dir) / path
            else:
                return Path.cwd() / path
        
        # Resolve key paths
        tokenizer_path = resolve_path(pm.get('tokenizer_save_path', ''), models_dir)
        model_path = resolve_path(pm.get('model_save_path', ''), models_dir)
        data_path = resolve_path(pm.get('data_path', ''))
        
        train_file = data_path / pm.get('data_file', '')
        val_file = data_path / pm.get('val_file', '')
        
        return {
            'tokenizer': tokenizer_path,
            'model': model_path,
            'data': data_path,
            'train_file': train_file,
            'val_file': val_file
        }
    
    def _check_system_info(self):
        """Display system information."""
        import shutil
        
        logger.info("🤖 Starting Robust LLM Training Pipeline")
        logger.info(f"📅 Started at: {datetime.now()}")
        logger.info(f"🖥️  Host: {os.uname().nodename}")
        
        # Check disk space
        total, used, free = shutil.disk_usage(Path.cwd())
        logger.info(f"💾 Available space: {free // (1024**3):.1f} GB")
        
        # Check environment variables
        datasets_dir = os.environ.get('GLOBAL_DATASETS_DIR', 'NOT SET')
        models_dir = os.environ.get('GLOBAL_MODELS_DIR', 'NOT SET')
        logger.info(f"🔧 GLOBAL_DATASETS_DIR: {datasets_dir}")
        logger.info(f"🔧 GLOBAL_MODELS_DIR: {models_dir}")
    
    def _check_gpu_status(self):
        """Check GPU availability and memory."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    used, total = map(int, line.split(', '))
                    logger.info(f"🎮 GPU {i}: {used}/{total} MB used ({used/total*100:.1f}%)")
            else:
                logger.warning("⚠️  Could not query GPU status")
        except FileNotFoundError:
            logger.warning("⚠️  nvidia-smi not available")
    
    def _validate_paths(self) -> bool:
        """Perform comprehensive path validation."""
        logger.info("🔍 === DRY RUN: Validating all paths and dependencies ===")
        
        all_valid = True
        
        # Check config
        logger.info(f"✅ Config file: {self.config_path}")
        
        # Check tokenizer
        tokenizer_path = self.paths['tokenizer']
        if tokenizer_path.exists():
            logger.info(f"✅ Tokenizer EXISTS: {tokenizer_path}")
            self.tokenizer_exists = True
        else:
            logger.info(f"🔧 Tokenizer MISSING (will be trained): {tokenizer_path}")
            logger.info(f"   📁 Parent directory exists: {tokenizer_path.parent.exists()}")
            self.tokenizer_exists = False
        
        # Check model directory
        model_path = self.paths['model']
        if model_path.exists():
            logger.info(f"📁 Model directory EXISTS: {model_path}")
        else:
            logger.info(f"📁 Model directory MISSING (will be created): {model_path}")
            model_path.mkdir(parents=True, exist_ok=True)
        
        # Check training data
        train_file = self.paths['train_file']
        if train_file.exists():
            try:
                with open(train_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                logger.info(f"✅ Training data EXISTS: {train_file} ({line_count:,} lines)")
            except Exception as e:
                logger.error(f"❌ Error reading training data: {e}")
                all_valid = False
        else:
            logger.error(f"❌ Training data MISSING: {train_file}")
            all_valid = False
        
        # Check validation data
        val_file = self.paths['val_file']
        if val_file.exists():
            try:
                with open(val_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                logger.info(f"✅ Validation data EXISTS: {val_file} ({line_count:,} lines)")
            except Exception as e:
                logger.error(f"❌ Error reading validation data: {e}")
                all_valid = False
        else:
            logger.error(f"❌ Validation data MISSING: {val_file}")
            all_valid = False
        
        return all_valid
    
    def _print_execution_plan(self):
        """Print what the pipeline will do."""
        logger.info("")
        logger.info("🎯 Execution Plan:")
        
        if self.tokenizer_exists:
            logger.info("   1️⃣ Skip tokenizer training (already exists)")
        else:
            logger.info(f"   1️⃣ Train new tokenizer → {self.paths['tokenizer']}")
        
        logger.info(f"   2️⃣ Train model → {self.paths['model']}")
        logger.info("   3️⃣ Test inference with sample prompt")
        logger.info("")
        logger.info(f"⏱️  Estimated time: 4-6 hours for model training")
        logger.info(f"🔄 Max retries: {self.max_retries}")
        logger.info(f"⏰ Timeout: {self.timeout_hours} hours")
        logger.info("")
    
    def _get_user_confirmation(self) -> bool:
        """Get user confirmation to proceed."""
        if self.auto_confirm:
            logger.info("🚀 Auto-confirm enabled, proceeding...")
            return True
            
        while True:
            response = input("🤔 Do you want to proceed with training? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                logger.info("🚫 Training cancelled by user")
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def _disable_system_sleep(self):
        """Attempt to disable system sleep during training."""
        try:
            # Check if we should skip sudo commands
            if self.skip_sudo or os.environ.get('TMUX') or not self._can_sudo_without_password():
                logger.info("🔒 Skipping system sleep prevention (skip-sudo flag, tmux session, or sudo requires password)")
                logger.info("💡 You can manually prevent sleep with: sudo systemctl mask sleep.target")
                return
                
            logger.info("🔒 Attempting to prevent system sleep...")
            subprocess.run([
                'sudo', 'systemctl', 'mask', 'sleep.target', 
                'suspend.target', 'hibernate.target', 'hybrid-sleep.target'
            ], check=False, capture_output=True, timeout=5)
            
            # Keep WiFi active
            subprocess.run(['sudo', 'iwconfig', 'wlan0', 'power', 'off'], 
                         check=False, capture_output=True, timeout=5)
            logger.info("✅ System sleep prevention configured")
        except Exception as e:
            logger.warning(f"⚠️  Could not configure sleep prevention: {e}")
    
    def _can_sudo_without_password(self) -> bool:
        """Check if sudo can be used without password prompt."""
        try:
            result = subprocess.run(['sudo', '-n', 'true'], 
                                  capture_output=True, timeout=2)
            return result.returncode == 0
        except:
            return False
    
    def _enable_system_sleep(self):
        """Re-enable system sleep after training."""
        try:
            # Skip if we should skip sudo commands  
            if self.skip_sudo or os.environ.get('TMUX') or not self._can_sudo_without_password():
                logger.info("🔓 Skipping system sleep re-enable (skip-sudo flag, tmux session, or sudo requires password)")
                return
                
            logger.info("🔓 Re-enabling system sleep...")
            subprocess.run([
                'sudo', 'systemctl', 'unmask', 'sleep.target', 
                'suspend.target', 'hibernate.target', 'hybrid-sleep.target'
            ], check=False, capture_output=True)
            logger.info("✅ System sleep re-enabled")
        except Exception as e:
            logger.warning(f"⚠️  Could not re-enable system sleep: {e}")
    
    def _train_tokenizer(self):
        """Train tokenizer if needed."""
        if self.tokenizer_exists:
            logger.info("✅ Tokenizer already exists, skipping training")
            return
        
        logger.info("🔤 Step 1: Training tokenizer...")
        try:
            cmd = ['train-tokenizer', str(self.config_path)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Tokenizer training completed")
            logger.debug(f"Tokenizer output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Tokenizer training failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise TrainingPipelineError("Tokenizer training failed")
    
    def _train_model_robust(self):
        """Train model with robust retry mechanism."""
        logger.info("🧠 Step 2: Starting robust model training...")
        logger.info(f"💡 Training will auto-retry up to {self.max_retries} times if it fails")
        logger.info(f"⏱️  Max training time: {self.timeout_hours} hours with timeout")
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"🔄 Training attempt {attempt}/{self.max_retries}...")
            
            try:
                cmd = ['modern-gpt-train', '--config', str(self.config_path), '--device', 'auto']
                timeout_seconds = self.timeout_hours * 3600
                
                # Stream output instead of capturing to avoid hanging
                result = subprocess.run(
                    cmd, 
                    timeout=timeout_seconds,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("✅ Model training completed successfully!")
                    return
                else:
                    logger.error(f"❌ Training failed with return code {result.returncode}")
                    logger.error("Check the output above for error details")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Training timed out after {self.timeout_hours} hours")
                
            except Exception as e:
                logger.error(f"❌ Training failed with exception: {e}")
            
            # If not the last attempt, wait and retry
            if attempt < self.max_retries:
                logger.info("🔄 Retrying in 30 seconds...")
                time.sleep(30)
        
        raise TrainingPipelineError(f"Model training failed after {self.max_retries} attempts")
    
    def _test_inference(self):
        """Test model inference."""
        logger.info("🎯 Step 3: Testing inference...")
        try:
            cmd = ['modern-gpt-infer', '--config', str(self.config_path)]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True)
            
            stdout, stderr = process.communicate(input="What is an elephant?\n", timeout=60)
            
            if process.returncode == 0:
                logger.info("✅ Inference test completed")
                logger.info(f"Model response: {stdout.strip()}")
            else:
                logger.error(f"❌ Inference test failed: {stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Inference test timed out")
        except Exception as e:
            logger.error(f"❌ Inference test error: {e}")
    
    def run(self):
        """Execute the complete training pipeline."""
        try:
            # System info and checks
            self._check_system_info()
            self._check_gpu_status()
            
            # Validate all paths and dependencies
            if not self._validate_paths():
                raise TrainingPipelineError("Path validation failed")
            
            # Show execution plan
            self._print_execution_plan()
            
            # Dry run exit
            if self.dry_run:
                logger.info("🔍 Dry run completed - all checks passed!")
                return
            
            # Get user confirmation
            if not self._get_user_confirmation():
                return
            
            # Configure system for training
            self._disable_system_sleep()
            
            try:
                # Execute training steps
                logger.info("🚀 Starting training pipeline...")
                logger.info("=" * 50)
                
                self._train_tokenizer()
                self._train_model_robust()
                self._test_inference()
                
                logger.info("✅ Training pipeline completed successfully!")
                logger.info(f"🎉 Finished at: {datetime.now()}")
                
            finally:
                # Always re-enable sleep
                self._enable_system_sleep()
                
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            self._enable_system_sleep()
            sys.exit(1)
        except TrainingPipelineError as e:
            logger.error(f"❌ Pipeline error: {e}")
            self._enable_system_sleep()
            sys.exit(1)
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
            self._enable_system_sleep()
            sys.exit(1)


def main():
    """Main CLI entry point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Robust LLM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to validate everything
    llmlib-train-pipeline --config config.json --dry-run
    
    # Full training with default settings
    llmlib-train-pipeline --config config.json
    
    # Training with custom retry/timeout settings
    llmlib-train-pipeline --config config.json --max-retries 5 --timeout 12 --auto-confirm
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the project configuration JSON file'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Validate all paths and dependencies without running training'
    )
    
    parser.add_argument(
        '--max-retries', 
        type=int, 
        default=3, 
        help='Maximum number of training retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=8, 
        help='Training timeout in hours (default: 8)'
    )
    
    parser.add_argument(
        '--auto-confirm', 
        action='store_true', 
        help='Skip user confirmation prompt'
    )
    
    parser.add_argument(
        '--skip-sudo', 
        action='store_true', 
        help='Skip system sleep prevention (no sudo commands)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        config_path=Path(args.config),
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        timeout_hours=args.timeout,
        auto_confirm=args.auto_confirm,
        skip_sudo=args.skip_sudo
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()
