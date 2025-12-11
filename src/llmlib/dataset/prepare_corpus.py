## llm/dataset/prepare_corpus.py
from pathlib import Path
import re
import argparse
import random

DATA_ROOT = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text_v2"
RAW = DATA_ROOT / "raw"
OUT = DATA_ROOT
OUT.mkdir(parents=True, exist_ok=True)

RE_WHITESPACE = re.compile(r'\s+')

def clean_line(s: str) -> str:
    s = s.strip()
    # normalize whitespace
    s = RE_WHITESPACE.sub(' ', s)
    # remove control characters
    s = ''.join(ch for ch in s if ord(ch) >= 32)
    # basic length filter
    if len(s) < 5 or len(s) > 800:
        return ""
    return s

def main(shuffle: bool = True):
    files = sorted(RAW.glob("**/*.txt"))
    print("Found raw files:", files)
    lines = []
    for f in files:
        for ln in f.open(encoding="utf-8"):
            c = clean_line(ln)
            if c:
                lines.append(c)

    if shuffle:
        random.shuffle(lines)

    out = OUT / "corpus.txt"
    with out.open("w", encoding="utf-8") as fw:
        for ln in lines:
            fw.write(ln + "\n")
    print("Wrote", out, "lines:", len(lines))
    print("Chars:", sum(len(l) for l in lines))

if __name__ == "__main__":
    main()
