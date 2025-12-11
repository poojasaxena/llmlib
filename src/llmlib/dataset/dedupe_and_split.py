## llm/dataset/dedupe_and_split.py

DATA_ROOT="$GLOBAL_DATASETS_DIR/llm/mixed_text_v2"
python - <<'PY'
from pathlib import Path
p = Path("$DATA_ROOT/corpus.txt")
lines = [l.rstrip() for l in p.open(encoding="utf-8") if l.strip()]
# dedupe (preserve order)
seen = set()
uniq = []
for l in lines:
    if l in seen:
        continue
    seen.add(l)
    uniq.append(l)
# extra filters: remove lines that are too short or too long
filtered = [l for l in uniq if 10 <= len(l) <= 500]
# write
Path("$DATA_ROOT/corpus.dedup.txt").write_text("\n".join(filtered)+"\n", encoding="utf-8")
print("Orig:", len(lines), "Unique:", len(uniq), "Filtered:", len(filtered))
PY
