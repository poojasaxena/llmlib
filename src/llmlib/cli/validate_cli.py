#!/usr/bin/env python3
"""
Model Validation CLI

Provides utilities to validate trained models, tokenizers, and configurations.
This replaces the scattered sanity check scripts with a professional CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from llmlib.utils.logger import get_logger
from llmlib.utils.config_util import load_nested_config

logger = get_logger(__name__)

# Check if torch is available
def _check_torch():
    try:
        import torch
        return True
    except ImportError:
        return False
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


def validate_model(config_path: Path, test_prompt: str = "What is an elephant?") -> bool:
    """Validate a trained model by running inference."""
    if not HAS_TORCH:
        logger.error("❌ PyTorch not available - cannot validate model")
        return False
        
    try:
        logger.info(f"🧠 Validating model with config: {config_path}")
        
        # Load configuration
        config = load_nested_config(str(config_path), config_path.name)
        project_metadata = config.get('project_metadata', {})
        
        # Resolve paths
        datasets_dir = os.environ.get('GLOBAL_DATASETS_DIR', '')
        models_dir = os.environ.get('GLOBAL_MODELS_DIR', datasets_dir)
        
        def resolve_path(path_str: str, base_dir: str = models_dir) -> Path:
            if not path_str:
                return Path()
            path = Path(path_str)
            return path if path.is_absolute() else Path(base_dir) / path
        
        tokenizer_path = resolve_path(project_metadata.get('tokenizer_save_path', ''))
        model_path = resolve_path(project_metadata.get('model_save_path', ''))
        
        # Validate tokenizer first
        if not tokenizer_path.exists():
            logger.error(f"   ❌ Tokenizer not found: {tokenizer_path}")
            return False
            
        if not validate_tokenizer(tokenizer_path):
            logger.error("   ❌ Tokenizer validation failed")
            return False
        
        # Load tokenizer
        tokenizer = load_tokenizer(str(tokenizer_path))
        
        # Check model directory
        if not model_path.exists():
            logger.error(f"   ❌ Model directory not found: {model_path}")
            return False
        
        model_file = model_path / "model.pt"
        if not model_file.exists():
            logger.error(f"   ❌ Model file not found: {model_file}")
            return False
        
        # Load model configuration
        model_config = config.get('model_config', {})
        
        # Create model
        gpt_config = ModernGPTConfig(
            vocab_size=len(tokenizer.vocab),
            d_model=model_config.get('d_model', 192),
            n_heads=model_config.get('n_heads', 4),
            n_layers=model_config.get('n_layers', 4),
            dropout=model_config.get('dropout', 0.1),
            max_position_embeddings=model_config.get('max_position_embeddings', 512)
        )
        
        model = ModernGPTModel(gpt_config)
        
        # Load trained weights
        logger.info(f"   Loading model weights from: {model_file}")
        checkpoint = torch.load(model_file, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Test inference
        logger.info(f"   Testing inference with prompt: '{test_prompt}'")
        
        with torch.no_grad():
            # Encode prompt
            tokens = tokenizer.encode(test_prompt)
            input_tensor = torch.tensor([tokens], dtype=torch.long)
            
            # Generate
            generated = model.generate(
                input_tensor, 
                max_new_tokens=20, 
                temperature=0.8, 
                top_k=40
            )
            
            # Decode result
            generated_text = tokenizer.decode(generated[0].tolist())
            logger.info(f"   Generated text: '{generated_text}'")
        
        logger.info("   ✅ Model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Model validation failed: {e}")
        import traceback
        logger.debug(f"   Traceback: {traceback.format_exc()}")
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
    """Validate entire setup: config, paths, tokenizer, and model."""
    logger.info("🔍 Starting full setup validation...")
    
    all_valid = True
    
    # 1. Validate config
    if not validate_config(config_path):
        all_valid = False
    
    # 2. Validate model (includes tokenizer validation)
    if not validate_model(config_path):
        all_valid = False
    
    if all_valid:
        logger.info("🎉 Full setup validation successful!")
    else:
        logger.error("❌ Setup validation failed")
    
    return all_valid


def main():
    """Main CLI entry point for validation utilities."""
    parser = argparse.ArgumentParser(
        description="Model and Setup Validation Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate entire setup (config + tokenizer + model)
    llmlib validate --config config.json
    
    # Validate just the configuration file
    llmlib validate --config config.json --config-only
    
    # Validate just the tokenizer
    llmlib validate --tokenizer /path/to/tokenizer.json
    
    # Test model inference with custom prompt
    llmlib validate --config config.json --prompt "Tell me about elephants"
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
        '--tokenizer', 
        type=str, 
        help='Path to tokenizer file for standalone validation'
    )
    
    parser.add_argument(
        '--prompt', 
        type=str, 
        default="What is an elephant?",
        help='Test prompt for model inference validation'
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.config:
        config_path = Path(args.config)
        
        if args.config_only:
            success = validate_config(config_path)
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
