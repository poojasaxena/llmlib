# ## llm/dataset/synth_expand.py
from pathlib import Path
DATA_ROOT = Path("$GLOBAL_DATASETS_DIR/llm/mixed_text_v2".replace("$GLOBAL_DATASETS_DIR", str(Path.home() / "PoojaVault/Professional/Workbench/Datasets")))
src = DATA_ROOT / "train.txt"
out = DATA_ROOT / "synthetic_generated.txt"
lines = [l.rstrip() for l in src.open(encoding="utf-8") if l.strip()]
new = []
for i,l in enumerate(lines):
    # simple transformations:
    new.append(l)
    new.append(l.replace("I ", "I often ").replace(" you ", " you sometimes "))
    if i % 3 == 0:
        new.append("Question: " + l)
        new.append("Answer: " + l)
out.write_text("\n".join(new[:20000]) + "\n", encoding="utf-8")  # limit size
print("Wrote synthetic:", out, "lines:", len(new[:20000]))
