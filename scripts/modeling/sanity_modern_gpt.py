## llmlib/scripts/experiments/sanity_modern_gpt.py
## Sanity check script for Modern GPT model
## This script loads a trained Modern GPT model and runs
## a forward pass on a known prompt to verify correctness.

import json
import torch
from pathlib import Path
import os

from llmlib.modeling.modern_gpt import ModernGPTModel, ModernGPTConfig
from llmlib.tokenization.byte_bpe_tokenizer import ByteBPETokenizer

# ------------------------------------------------
# 1) Load tokenizer
# ------------------------------------------------
global_path = os.environ["GLOBAL_MODELS_DIR"]
model_dir = Path(global_path) / "llm/language_models/elephantdomain_gpt/gpt-bpe-v2"

with open(model_dir / "tokenizer.json", "r") as f:
    tokenizer = ByteBPETokenizer.from_dict(json.load(f))

# ------------------------------------------------
# 2) Load model
# ------------------------------------------------
cfg = ModernGPTConfig(
    vocab_size=len(tokenizer.vocab),
    d_model=192,
    n_heads=4,
    n_layers=4,
    max_position_embeddings=128,
    dropout=0.0,
)

model = ModernGPTModel(cfg)
model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
model.eval()

# ------------------------------------------------
# 3) Pick text that EXISTS in training data
# ------------------------------------------------
prompt = "the elephant is the largest land animal"

ids = tokenizer.encode(prompt)
x = torch.tensor([ids], dtype=torch.long)

# ------------------------------------------------
# 4) Forward pass
# ------------------------------------------------
with torch.no_grad():
    logits = model(x)[0, -1]  # [vocab]

# ------------------------------------------------
# 5) Inspect top predictions
# ------------------------------------------------
topk = torch.topk(logits, k=10)
print("\nTop predicted tokens:\n")

for idx in topk.indices.tolist():
    print(f"{idx:>5} → {repr(tokenizer.decode([idx]))}")
