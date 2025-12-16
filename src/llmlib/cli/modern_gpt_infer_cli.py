# llmlib/cli/modern_gpt_infer_cli.py

import argparse
from pathlib import Path
import json
import torch

from llmlib.modeling.modern_gpt import ModernGPTConfig, ModernGPTModel
from llmlib.utils.path_util import get_data_file_path
from llmlib.utils.config_util import load_config
from llmlib.utils.checkpoint import save_model, load_model
from llmlib.utils.path_util import short_path, get_model_dir
from llmlib.tokenization.byte_bpe_tokenizer import ByteBPETokenizer
from llmlib.tokenization.registry import load_tokenizer

import torch.nn.functional as F


from llmlib.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------
# Inference for ModernGPT
# ---------------------------------------------------------


def sample_from_logits(
    logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9
) -> int:
    """
    Sample a token ID from logits with top-k and optional top-p filtering.
    """
    probs = F.softmax(logits, dim=-1)

    # Top-k
    topk_probs, topk_idx = torch.topk(probs, k=top_k)
    topk_probs /= topk_probs.sum()
    next_id = topk_idx[torch.multinomial(topk_probs, 1)].item()

    return next_id



def _generate_text_modern(
    model: ModernGPTModel,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """
    Generate text with top-k / top-p sampling using Byte-level BPE tokenizer.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        return ""

    generated = input_ids.copy()

    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_new_tokens):
        context_ids = torch.tensor([generated[-max_seq_len:]], device=device)
        logits = model(context_ids)
        next_id = sample_from_logits(logits, top_k=50)
        generated.append(next_id)
        if next_id == eos_id:
            break

            # Top-k filtering
            topk_probs, topk_idx = torch.topk(probs, k=top_k)
            topk_probs /= topk_probs.sum()
            next_id = topk_idx[torch.multinomial(topk_probs, 1)].item()

            # # Optional: Top-p (nucleus) filtering
            # sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            # cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            # mask = cumulative_probs <= top_p
            # sorted_probs[~mask] = 0
            # sorted_probs /= sorted_probs.sum()
            # next_id = sorted_idx[torch.multinomial(sorted_probs, 1)].item()

        generated.append(next_id)
        next_char = tokenizer.decode([next_id])
        if next_char in [".", "!", "?"]:
            break

    return tokenizer.decode(generated)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _load_nested_project_config(config_path: Path) -> dict:
    cfg = load_config(caller_file=str(config_path), config_filename=config_path.name)
    for key in ("model_config", "training_config", "project_metadata"):
        if key not in cfg:
            raise ValueError(f"Missing '{key}' in project_config: {config_path}")
    return cfg


def _select_device(device_arg: str | None) -> str:
    if device_arg in ("cpu", "cuda"):
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


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

    cfg = _load_nested_project_config(config_path)
    meta_cfg = cfg["project_metadata"]

    max_seq_len = meta_cfg["max_seq_length"]
    max_new_tokens = meta_cfg.get("max_new_tokens", 40)

    # Load model
    model_cfg = cfg["model_config"]

    config = ModernGPTConfig(
    vocab_size=model_cfg["num_embeddings"],  # ← correct mapping
    d_model=model_cfg["d_model"],
    n_heads=model_cfg["n_heads"],
    n_layers=model_cfg["n_layers"],
    max_position_embeddings=model_cfg["max_position_embeddings"],
    dropout=model_cfg["dropout"])

    model = ModernGPTModel(config)

    model_dir = get_model_dir(cfg, create=False)
    ckpt_path = model_dir / "model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # Prompt
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter a prompt (e.g. 'hello'): ").strip()
        if not prompt:
            prompt = "hello."

    logger.info("\n============================================================")
    logger.info("🧪 modern-gpt-infer")
    logger.info("============================================================")
    logger.info(f"Model Name     : {meta_cfg['model_name']}")
    logger.info(f"Max Seq Length : {max_seq_len}")
    logger.info(f"Max New Tokens : {max_new_tokens}")
    logger.info(f"Device         : {device}")
    logger.info("============================================================\n")

    # Load Byte-level BPE Tokenizer
    # -----------------------------
    tokenizer_path = model_dir / "tokenizer.json"

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer_dict = json.load(f)
        tokenizer = ByteBPETokenizer.from_dict(tokenizer_dict)

    # Sanity check
    assert model_cfg["num_embeddings"] == len(tokenizer.vocab), (
    f"Model vocab={model_cfg['num_embeddings']} "
    f"but tokenizer vocab={len(tokenizer.vocab)}"
    )


    # Print info about the tokenizer
    logger.info("\n[modern-gpt-infer] Tokenizer loaded:")
    logger.info(f"  Type       : {type(tokenizer).__name__}")
    #logger.info(f"  Path       : {tokenizer_path}")
    if hasattr(tokenizer, "vocab_size"):
        logger.info(f"  Vocab Size : {tokenizer.vocab_size}")
    #if "config" in tokenizer_dict:
    #    logger.info(f"  Config keys: {list(tokenizer_dict['config'].keys())}")
    logger.info(f"  Special tokens: {tokenizer.special_tokens}")
    logger.info("============================================================\n")

    # Generate text
    output = _generate_text_modern(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        top_k=50,
        #top_p=0.9
    )

    print("---")
    print(f"Prompt : {prompt}")
    print(f"Output : {output}")
