# llmlib/cli/modern_gpt_infer_cli.py

import argparse
from pathlib import Path
import json
import torch

from llmlib.modeling.modern_gpt import ModernGPTConfig, ModernGPTModel

from llmlib.utils.config_util import load_nested_config
from llmlib.utils.checkpoint import load_model
from llmlib.tokenization.byte_bpe_tokenizer import ByteBPETokenizer
from llmlib.tokenization.registry import load_tokenizer

from llmlib.utils.logger import get_logger

logger = get_logger(__name__)
DEBUG = False  # Clean output for production use

# ---------------------------------------------------------
# Inference for ModernGPT
# ---------------------------------------------------------


def sample_from_logits(logits_1d, top_k=None, temperature=1.0, avoid_tokens=None):
    """
    logits_1d: Tensor of shape [vocab_size]
    avoid_tokens: list of token IDs to avoid (like repetitive EOS tokens)
    """
    if torch.isnan(logits_1d).any() or torch.isinf(logits_1d).any():
        print(f"[WARNING] Found NaN or Inf in logits!")
        # Fallback: return a random token
        return torch.randint(0, logits_1d.size(-1), (1,)).item()
    
    # Create a copy to avoid modifying original logits
    modified_logits = logits_1d.clone()
    
    # Strongly penalize problematic tokens (like EOS when we want to force generation)
    if avoid_tokens:
        for token_id in avoid_tokens:
            if 0 <= token_id < modified_logits.size(-1):
                modified_logits[token_id] = -float('inf')  # Completely avoid these tokens
    
    if top_k is not None and top_k > 0:
        top_k = min(top_k, modified_logits.size(-1))
        values, indices = torch.topk(modified_logits, top_k)
        probs = torch.softmax(values / temperature, dim=-1)
        next_idx = torch.multinomial(probs, 1)
        return indices[next_idx].item()
    else:
        probs = torch.softmax(modified_logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()


def _generate_text_modern(
    model: ModernGPTModel,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    avoid_eos_first_n: int) -> str:
    """
    Generate text with top-k / top-p sampling using Byte-level BPE tokenizer.
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        return prompt

    # Remove EOS token from input if present - let model generate it naturally
    eos_id = tokenizer.token_to_id("<eos>")  # Get actual EOS token ID
    pad_id = tokenizer.token_to_id("<pad>")  # Get actual PAD token ID
    
    # Remove trailing EOS token if the tokenizer added it
    if input_ids and input_ids[-1] == eos_id:
        input_ids = input_ids[:-1]
        if DEBUG:
            print(f"[DEBUG] Removed EOS from input tokens")
    
    generated = input_ids.copy()
    
    if DEBUG:
        print(f"[DEBUG] Starting generation with {len(input_ids)} input tokens")
        print(f"[DEBUG] Input tokens: {input_ids[:10]}...")  # Show first 10 tokens
        print(f"[DEBUG] EOS token ID: {eos_id}")

    for i in range(max_new_tokens):
        # Take the last max_seq_len tokens for context
        context = generated[-max_seq_len:] if len(generated) > max_seq_len else generated
        context_tensor = torch.tensor([context], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model(context_tensor)
            
            # Get the logits for the last position
            if logits.dim() == 3:  # (batch, seq, vocab)
                next_logits = logits[0, -1]
            elif logits.dim() == 2:  # (seq, vocab)
                next_logits = logits[-1]
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            # Debug: Check logits statistics
            if DEBUG and i == 0:
                print(f"[DEBUG] Logits shape: {next_logits.shape}")
                print(f"[DEBUG] Logits mean: {next_logits.mean().item():.4f}, std: {next_logits.std().item():.4f}")
                print(f"[DEBUG] Top 5 logits: {torch.topk(next_logits, 5).values.tolist()}")
                print(f"[DEBUG] EOS logit value: {next_logits[eos_id].item():.4f}")

            # Apply temperature and sample
            # Penalize EOS token for first few generations to force content generation
            avoid_tokens = [eos_id] if i < avoid_eos_first_n else []
            next_id = sample_from_logits(
                               next_logits,
                               top_k=top_k,
                               temperature=temperature,
                               avoid_tokens=avoid_tokens
                    )

            
            if DEBUG and i < 5:  # Debug first few generations
                print(f"[DEBUG] Generated token {i+1}: {next_id}")
                if next_id == eos_id:
                    print(f"[DEBUG] Generated EOS token - stopping generation")

        generated.append(next_id)
        
        # Stop generation if EOS token is generated, but only after generating at least one token
        if next_id == eos_id:  # Allow at least one generation step
            if DEBUG:
                print(f"[DEBUG] EOS token generated at position {i+1}, stopping generation")
            break

    if DEBUG:
        print(f"[DEBUG] Final generated tokens: {generated}")
        print(f"[DEBUG] Generated {len(generated) - len(input_ids)} new tokens")

    
    new_tokens = generated[len(input_ids):]
    return tokenizer.decode(new_tokens)

    #full_text = tokenizer.decode(generated)
    #return full_text


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def _select_device(device_arg: str | None) -> str:
    if device_arg in ("cpu", "cuda"):
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def debug_model_generation(model, tokenizer, temperature, device):
    """Debug function to test basic model generation"""
    print("\n=== DEBUG MODEL GENERATION ===")
    
    # Test with simple input
    test_input = [1, 2, 3]  # Simple token IDs
    context_tensor = torch.tensor([test_input], dtype=torch.long, device=device)
    
    print(f"Input tensor shape: {context_tensor.shape}")
    print(f"Input tokens: {test_input}")
    
    with torch.no_grad():
        logits = model(context_tensor)
        print(f"Output logits shape: {logits.shape}")
        
        if logits.dim() == 3:
            next_logits = logits[0, -1]  # Last position of first batch
        else:
            next_logits = logits[-1]  # Last position
            
        print(f"Next token logits shape: {next_logits.shape}")
        print(f"Logits stats - min: {next_logits.min():.3f}, max: {next_logits.max():.3f}, mean: {next_logits.mean():.3f}")
        
        # Get top 10 predictions
        top_values, top_indices = torch.topk(next_logits, 10)
        print(f"Top 10 predictions: {top_indices.tolist()}")
        print(f"Top 10 values: {top_values.tolist()}")
        
        # Sample a token
        probs = torch.softmax(next_logits / temperature, dim=-1)
        sampled_id = torch.multinomial(probs, 1).item()
        print(f"Sampled token ID: {sampled_id}")
    
    print("=== END DEBUG ===\n")


def modern_gpt_infer() -> None:
    """
    Run inference with a saved Modern GPT model.

        modern-gpt-infer --config /path/to/project_config.json [--prompt "hello"] [--device cpu|cuda]

    If --prompt is omitted, it will ask interactively.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with a Modern GPT model."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to project_config.json"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text (if omitted, read from stdin).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use (default: auto)",
    )
    # parser.add_argument(
    #     "--interactive",
    #     action="store_true",
    #     help="Run in interactive prompt mode (REPL).",
    # )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = _select_device(args.device)
    logger.info(f"[modern-gpt-infer] Using device: {device}")

    cfg = load_nested_config(caller_file=str(config_path), config_filename=config_path.name)
    infer_cfg = validate_inference_config(cfg)
    meta_cfg = cfg["project_metadata"]

    max_seq_len = meta_cfg["max_seq_length"]
    max_new_tokens = meta_cfg["max_new_tokens"]

    ## Inference parameters
    temperature = infer_cfg["temperature"]
    top_k = infer_cfg["top_k"]
    avoid_eos_first_n = infer_cfg["avoid_eos_first_n"]
    seed = infer_cfg["seed"]
    if seed is not None:
        import random
        import numpy as np
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Load model using checkpoint utility
    model = load_model(
        model_cls       = ModernGPTModel,
        config_cls      = ModernGPTConfig,
        project_config  = cfg,
        device          = device,
        eval_mode       = True
    )

    if DEBUG:
        print(f"[DEBUG] Model loaded and set to eval mode")
        print(f"[DEBUG] Model config - vocab_size: {model.config.vocab_size}, d_model: {model.config.d_model}")

    # Load Tokenizer using registry (auto-detects type)
    # ------------------------------------------------
    tokenizer = load_tokenizer(cfg)

    # Debug tokenizer info
    if DEBUG:
        print(f"[DEBUG] Tokenizer loaded successfully")
        print(f"[DEBUG] Vocab size from tokenizer: {len(tokenizer.vocab)}")
        print(f"[DEBUG] Special tokens: {tokenizer.special_tokens}")

        # Test tokenizer encoding/decoding
        test_text = "hello world"
        test_tokens = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_tokens)
        print(f"[DEBUG] Test encoding '{test_text}' -> {test_tokens}")
        print(f"[DEBUG] Test decoding {test_tokens} -> '{test_decoded}'")

    # Sanity check
    assert model.config.vocab_size == len(tokenizer.vocab), (
    f"Model vocab={model.config.vocab_size} "
    f"but tokenizer vocab={len(tokenizer.vocab)}"
    )

    def gen(p: str) -> str:
        return _generate_text_modern(
        model=model,
        tokenizer=tokenizer,
        prompt=p,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        avoid_eos_first_n=avoid_eos_first_n,
    )

    # Get model directory for display
    from llmlib.utils.path_util import get_model_dir, short_path
    model_dir = get_model_dir(cfg, create=False)

    # Use home directory as base for shorter, more readable paths
    base_dir = Path.home() / "PoojaVault/Professional/"

    print("\n============================================================")
    print("🧠 Modern GPT Inference")
    print("============================================================")
    print(f"Project config     : ~/{short_path(config_path, base_dir)}")
    print(f"Model directory    : ~/{short_path(model_dir, base_dir)}")
    print(f"Checkpoint         : ~/{short_path(model_dir / 'model.pt', base_dir)}")
    print(f"Model config       : ~/{short_path(model_dir / 'model_config.json', base_dir)}")
    print(f"Tokenizer type     : {type(tokenizer).__name__}")
    print(f"Vocab size         : {len(tokenizer.vocab)}")
    print(f"Special tokens     : {tokenizer.special_tokens}")
    print(f"Device             : {device}")
    print(f"Max sequence len   : {max_seq_len}")
    print(f"Max new tokens     : {max_new_tokens}")
    print("============================================================\n")

    
    # Debug the model generation first
    if DEBUG:
        debug_model_generation(model, tokenizer, temperature, device)

        print("Testing generation:")
        print(f"elephant -> {gen('elephant')}")
        print(f"The elephant is -> {gen('The elephant is')}")
        print(f"In the wild, elephants -> {gen('In the wild, elephants')}")

    # Interactive prompt loop or single generation
    if args.prompt:
        # Single prompt mode
        output = gen(args.prompt)
        print("---")
        print(f"Prompt : {args.prompt}")
        print(f"Output : {output}")
    else:
        # Interactive mode
        print("🤖 Interactive ModernGPT Inference")
        print("Type your prompts below. Type 'quit', 'exit', or press Ctrl+C to stop.\n")

        try:
            while True:
                prompt = input("Enter a prompt: ").strip()

                if not prompt:
                    print("Please enter a non-empty prompt.")
                    continue

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break

                # Generate text
                output = gen(prompt)

                print("---")
                print(f"Prompt : {prompt}")
                print(f"Output : {output}")
                print()  # Add a blank line for readability

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye! 👋")

   

def validate_inference_config(cfg: dict) -> dict:
    if "inference_config" not in cfg:
        raise ValueError("Missing required 'inference_config' in config")

    infer = cfg["inference_config"]

    required = (
        "seed",
        "temperature",
        "top_k",
        "avoid_eos_first_n",
    )

    for k in required:
        if k not in infer:
            raise ValueError(f"Missing inference_config key: '{k}'")

    return infer
