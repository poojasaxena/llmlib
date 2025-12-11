# ## llm/dataset/split_corpus.py
from pathlib import Path
import random
DATA_ROOT = Path("$GLOBAL_DATASETS_DIR/llm/mixed_text_v2".replace("$GLOBAL_DATASETS_DIR", str(Path.home() / "PoojaVault/Professional/Workbench/Datasets")))
src = DATA_ROOT / "corpus.dedup.txt"
lines = [l.rstrip() for l in src.open(encoding="utf-8") if l.strip()]
random.shuffle(lines)
n = len(lines)
n_val = max(1, int(n * 0.01))
n_test = max(1, int(n * 0.01))
n_train = n - n_val - n_test

train = lines[:n_train]
val = lines[n_train:n_train+n_val]
test = lines[n_train+n_val:]

(DATA_ROOT / "train.txt").write_text("\n".join(train)+"\n", encoding="utf-8")
(DATA_ROOT / "val.txt").write_text("\n".join(val)+"\n", encoding="utf-8")
(DATA_ROOT / "test.txt").write_text("\n".join(test)+"\n", encoding="utf-8")

print("Counts: train", len(train), "val", len(val), "test", len(test))
print("Chars train:", sum(len(l) for l in train))
