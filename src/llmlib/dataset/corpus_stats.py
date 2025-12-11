#llmlib/dataset/corpus_stats.py
from pathlib import Path
from math import ceil
DATA_ROOT = Path("$GLOBAL_DATASETS_DIR/llm/mixed_text_v2".replace("$GLOBAL_DATASETS_DIR", str(Path.home() / "PoojaVault/Professional/Workbench/Datasets")))
for name in ["train.txt","val.txt","test.txt"]:
    p = DATA_ROOT / name
    lines = [l.rstrip() for l in p.open(encoding="utf-8") if l.strip()]
    chars = sum(len(l) for l in lines)
    words = sum(len(l.split()) for l in lines)
    print(name, "lines", len(lines), "chars", chars, "words", words)
