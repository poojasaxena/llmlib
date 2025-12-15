# llmlib/cli/main_cli.py

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from llmlib.modeling.modern_gpt import ModernGPTConfig, ModernGPTModel
from llmlib.utils.path_util import get_data_file_path
from llmlib.utils.config_util import load_config
from llmlib.utils.checkpoint import save_model, load_model
from llmlib.tokenization.char_tokenizer import CharTokenizer


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


# ---------------------------------------------------------
# Inference for ModernGPT
# ---------------------------------------------------------
def _generate_text_modern(
    model: ModernGPTModel,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    tokenizer = CharTokenizer()
    input_ids = tokenizer.encode(prompt)
    if input_ids.numel() == 0:
        return ""

    input_ids = input_ids.to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        context = generated[-max_seq_len:].unsqueeze(0)  # (1, seq_len)
        with torch.no_grad():
            logits = model(context)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        generated = torch.cat(
            [generated, torch.tensor([next_id], device=device)], dim=0
        )

        next_char = tokenizer.decode(torch.tensor([next_id]))
        if next_char in [".", "?", "!"]:
            break

    return tokenizer.decode(generated.cpu())


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
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = _select_device(args.device)
    print(f"[modern-gpt-infer] Using device: {device}")

    cfg = _load_nested_project_config(config_path)
    meta_cfg = cfg["project_metadata"]

    max_seq_len = meta_cfg["max_seq_length"]
    max_new_tokens = meta_cfg.get("max_new_tokens", 40)

    # Load model
    model = ModernGPTModel(ModernGPTConfig(**cfg["model_config"]))
    ckpt_path = meta_cfg.get("ckpt_path") or f"{meta_cfg['save_dir']}/model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # Prompt
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter a prompt (e.g. 'hello'): ").strip()
        if not prompt:
            prompt = "hello."

    print("\n============================================================")
    print("🧪 modern-gpt-infer")
    print("============================================================")
    print(f"Model Name     : {meta_cfg['model_name']}")
    print(f"Max Seq Length : {max_seq_len}")
    print(f"Max New Tokens : {max_new_tokens}")
    print(f"Device         : {device}")
    print("============================================================\n")

    output = _generate_text_modern(
        model,
        prompt=prompt,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
    )

    print("---")
    print(f"Prompt : {prompt}")
    print(f"Output : {output}")
