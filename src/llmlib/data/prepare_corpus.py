from pathlib import Path
import re
import argparse
import random
import os

DATASET_DIR = os.environ["GLOBAL_DATASETS_DIR"]
DATA_ROOT = Path(DATASET_DIR) / "llm/mixed_text/"
RAW = DATA_ROOT / "raw"
OUT = DATA_ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

RE_WHITESPACE = re.compile(r"\s+")


def clean_line(s: str, min_len: int = 5, max_len: int = 800) -> str:
    s = s.strip()
    s = RE_WHITESPACE.sub(" ", s)
    s = "".join(ch for ch in s if ord(ch) >= 32)
    if len(s) < min_len or len(s) > max_len:
        return ""
    return s


def prepare(
    raw_dir: Path = RAW,
    out_dir: Path = OUT,
    file_out: str = "elephant_human_90_10_corpus.txt",
    shuffle: bool = True,
    dedupe: bool = True,
) -> dict:
    files = sorted(raw_dir.glob("**/*.txt"))
    print("Found raw files:", files)

    lines = []
    for f in files:
        for ln in f.open(encoding="utf-8"):
            c = clean_line(ln)
            if c:
                lines.append(c)

    if dedupe:
        seen = set()
        uniq_lines = []
        for l in lines:
            if l not in seen:
                seen.add(l)
                uniq_lines.append(l)
        lines = uniq_lines

    if shuffle:
        random.shuffle(lines)

    out_path = out_dir / file_out
    with out_path.open("w", encoding="utf-8") as fw:
        for ln in lines:
            fw.write(ln + "\n")

    print(
        f"Wrote {out_path} | lines: {len(lines)}, chars: {sum(len(l) for l in lines)}"
    )
    return {"lines": len(lines), "chars": sum(len(l) for l in lines)}
