#!/usr/bin/env python3
"""
Basic Validation CLI

Provides utilities to validate configurations and basic setup without
requiring heavy dependencies like PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llmlib.tokenization.registry import load_tokenizer
from llmlib.utils.logger import get_logger
from llmlib.utils.config_util import load_nested_config

logger = get_logger(__name__)


def validate_tokenizer(tokenizer_path: Path, test_text: str = "Hello world! This is a test.") -> bool:
    """Validate a tokenizer by encoding and decoding test text."""
    try:
        logger.info(f"🔤 Validating tokenizer: {tokenizer_path}")
        
        # Load tokenizer
        tokenizer = load_tokenizer(str(tokenizer_path))
        
        # Test encoding
        tokens = tokenizer.encode(test_text)
        logger.info(f"   Original text: '{test_text}'")
        logger.info(f"   Encoded tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        logger.info(f"   Token count: {len(tokens)}")
        
        # Test decoding
        decoded = tokenizer.decode(tokens)
        logger.info(f"   Decoded text: '{decoded}'")
        
        # Check roundtrip consistency
        if decoded.strip() == test_text.strip():
            logger.info("   ✅ Roundtrip encoding/decoding successful")
            return True
        else:
            logger.warning("   ⚠️  Decoded text differs from original")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Tokenizer validation failed: {e}")
        return False


def validate_paths(config_path: Path) -> bool:
    """Validate that all paths in the config exist and are accessible."""
    try:
        logger.info(f"📂 Validating paths in config: {config_path}")
        
        # Load configuration
        config = load_nested_config(str(config_path), config_path.name)
        project_metadata = config.get('project_metadata', {})
        
        # Resolve paths
        datasets_dir = os.environ.get('GLOBAL_DATASETS_DIR', '')
        models_dir = os.environ.get('GLOBAL_MODELS_DIR', datasets_dir)
        
        def resolve_path(path_str: str, base_dir: str = datasets_dir) -> Path:
            if not path_str:
                return Path()
            path = Path(path_str)
            return path if path.is_absolute() else Path(base_dir) / path
        
        all_valid = True
        
        # Check tokenizer path
        tokenizer_path = resolve_path(project_metadata.get('tokenizer_save_path', ''), models_dir)
        if tokenizer_path and tokenizer_path.exists():
            logger.info(f"   ✅ Tokenizer exists: {tokenizer_path}")
        elif tokenizer_path:
            logger.info(f"   📝 Tokenizer will be created: {tokenizer_path}")
            # Check if parent directory exists
            if not tokenizer_path.parent.exists():
                logger.error(f"   ❌ Tokenizer parent directory missing: {tokenizer_path.parent}")
                all_valid = False
        
        # Check model path
        model_path = resolve_path(project_metadata.get('model_save_path', ''), models_dir)
        if model_path and model_path.exists():
            logger.info(f"   ✅ Model directory exists: {model_path}")
        elif model_path:
            logger.info(f"   📝 Model directory will be created: {model_path}")
            # Ensure parent exists
            model_path.mkdir(parents=True, exist_ok=True)
        
        # Check data paths
        data_path = resolve_path(project_metadata.get('data_path', ''))
        if data_path and data_path.exists():
            logger.info(f"   ✅ Data directory exists: {data_path}")
            
            # Check specific data files
            data_file = project_metadata.get('data_file', '')
            val_file = project_metadata.get('val_file', '')
            
            if data_file:
                train_file_path = data_path / data_file
                if train_file_path.exists():
                    with open(train_file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    logger.info(f"   ✅ Training data: {train_file_path} ({line_count:,} lines)")
                else:
                    logger.error(f"   ❌ Training data missing: {train_file_path}")
                    all_valid = False
            
            if val_file:
                val_file_path = data_path / val_file
                if val_file_path.exists():
                    with open(val_file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    logger.info(f"   ✅ Validation data: {val_file_path} ({line_count:,} lines)")
                else:
                    logger.error(f"   ❌ Validation data missing: {val_file_path}")
                    all_valid = False
                    
        else:
            logger.error(f"   ❌ Data directory missing: {data_path}")
            all_valid = False
        
        return all_valid
        
    except Exception as e:
        logger.error(f"   ❌ Path validation failed: {e}")
        return False


def validate_config(config_path: Path) -> bool:
    """Validate a configuration file structure and contents."""
    try:
        logger.info(f"⚙️  Validating config: {config_path}")
        
        if not config_path.exists():
            logger.error(f"   ❌ Config file not found: {config_path}")
            return False
        
        # Load and parse config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ['model_config', 'training_config', 'project_metadata']
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            logger.error(f"   ❌ Missing config sections: {missing_sections}")
            return False
        
        # Check project metadata
        pm = config['project_metadata']
        required_fields = ['tokenizer_save_path', 'model_save_path', 'data_path']
        missing_fields = []
        
        for field in required_fields:
            if field not in pm:
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"   ❌ Missing project_metadata fields: {missing_fields}")
            return False
        
        logger.info("   ✅ Config structure valid")
        
        # Validate model config
        mc = config['model_config']
        model_fields = ['d_model', 'n_heads', 'n_layers', 'max_position_embeddings']
        
        for field in model_fields:
            if field not in mc:
                logger.warning(f"   ⚠️  Missing model_config field: {field}")
            else:
                value = mc[field]
                if not isinstance(value, int) or value <= 0:
                    logger.error(f"   ❌ Invalid {field}: {value} (must be positive integer)")
                    return False
        
        logger.info("   ✅ Config validation successful")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"   ❌ Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        logger.error(f"   ❌ Config validation failed: {e}")
        return False


def validate_full_setup(config_path: Path) -> bool:
    """Validate entire setup: config and paths."""
    logger.info("🔍 Starting full setup validation...")
    
    all_valid = True
    
    # 1. Validate config structure
    if not validate_config(config_path):
        all_valid = False
    
    # 2. Validate paths
    if not validate_paths(config_path):
        all_valid = False
    
    # 3. Validate tokenizer if it exists
    try:
        config = load_nested_config(str(config_path), config_path.name)
        project_metadata = config.get('project_metadata', {})
        
        models_dir = os.environ.get('GLOBAL_MODELS_DIR', os.environ.get('GLOBAL_DATASETS_DIR', ''))
        tokenizer_path_str = project_metadata.get('tokenizer_save_path', '')
        
        if tokenizer_path_str:
            tokenizer_path = Path(tokenizer_path_str)
            if not tokenizer_path.is_absolute() and models_dir:
                tokenizer_path = Path(models_dir) / tokenizer_path
            
            if tokenizer_path.exists():
                if not validate_tokenizer(tokenizer_path):
                    all_valid = False
    except Exception as e:
        logger.error(f"❌ Error validating tokenizer: {e}")
        all_valid = False
    
    if all_valid:
        logger.info("🎉 Full setup validation successful!")
    else:
        logger.error("❌ Setup validation failed")
    
    return all_valid


def main():
    """Main CLI entry point for validation utilities."""
    parser = argparse.ArgumentParser(
        description="Setup and Configuration Validation Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate entire setup (config + paths + tokenizer if exists)
    llmlib validate --config config.json
    
    # Validate just the configuration file
    llmlib validate --config config.json --config-only
    
    # Validate just paths
    llmlib validate --config config.json --paths-only
    
    # Validate just the tokenizer
    llmlib validate --tokenizer /path/to/tokenizer.json
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to project configuration JSON file'
    )
    
    parser.add_argument(
        '--config-only', 
        action='store_true', 
        help='Validate only the configuration file structure'
    )
    
    parser.add_argument(
        '--paths-only', 
        action='store_true', 
        help='Validate only the paths and file accessibility'
    )
    
    parser.add_argument(
        '--tokenizer', 
        type=str, 
        help='Path to tokenizer file for standalone validation'
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.config:
        config_path = Path(args.config)
        
        if args.config_only:
            success = validate_config(config_path)
        elif args.paths_only:
            success = validate_paths(config_path)
        else:
            success = validate_full_setup(config_path)
    
    elif args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
        success = validate_tokenizer(tokenizer_path)
    
    else:
        parser.print_help()
        return
    
    if success:
        logger.info("✅ All validations passed")
    else:
        logger.error("❌ Validation failed")
        exit(1)


if __name__ == '__main__':
    main()
