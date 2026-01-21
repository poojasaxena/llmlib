
# High-level plan (one glance)
* Collect sources (local + curated public + synthetic) into Datasets/llm/mixed_text_v2/raw/
* Merge & clean → Datasets/llm/mixed_text_v2/corpus.txt
* Deduplicate, filter length, normalize → corpus.cleaned.txt
* Split train/val/test (e.g., 98%/1%/1%)
* (Optional) Retrain BPE tokenizer on train (vocab 8k–12k)
* Train model on train; monitor val perplexity.

Target: ~1,000,000 characters (about 150k–250k tokens depending on tokenizer). That’s a sweet spot for your 256-dim model.

## Where files will live (paths you can use)

(Assuming your GLOBAL_DATASETS_DIR env is set; if not replace with full path.)
```
$GLOBAL_DATASETS_DIR/llm/mixed_text_v2/
├── raw/                     # raw source files you will collect/paste
│   ├── your_journal.txt
│   ├── conversations.txt
│   ├── wiki_snippets.txt
│   └── synthetic_generated.txt
├── corpus.txt               # merged raw
├── corpus.cleaned.txt       # cleaned + normalized
├── corpus.dedup.txt         # deduped
├── train.txt
├── val.txt
└── test.txt
```

## Step 0 — rough sizing guidance
* Characters per English sentence ≈ 60–100.
* 1M chars ≈ 10k–16k sentences (depends).
* If average sentence length 70 chars → 1,000,000 / 70 ≈ 14,285 lines.
Aim for 12k–20k high-quality lines.


## Step 1 — Collect sources (what to put in raw/)
* You want a mix — diversity + consistent style. Prioritize quality over sheer quantity.

> Suggested sources (practical and legal):
* Your own notes / journal / dialogues (copy/paste) — very high value. (your_journal.txt)
* Short dialogues / chat snippets you write (polite Q&A). (conversations.txt)
* Wikipedia short paragraphs (Simple English Wikipedia is great) — extract 3–5k lines. (wiki_snippets.txt)
* Public domain short texts (Project Gutenberg short works, but be careful) — small excerpts only.
* Synthetic expansions produced by the model/ChatGPT — ask it to generate small dialogues or paraphrases. (synthetic_generated.txt)
* Shorthowtos / tips / instructions you write — good structure examples.
* Non-sensitive multilingual greetings (Spanish, German) — small set.
Practical note: start with what you already have. Paste them into files under raw/.


## Step 2 — Merge & clean script (copy/paste)
> llm/data/prepare_corpus.py

## Step 3 — Deduplicate and final clean + length filters
> ## llm/dataset/dedupe_and_split.py
> Run
```
bash llmlib/data/dedupe_and_split.sh
```

## Step 4 — Split into train/val/test
* Use an 98/1/1 split (or 96/2/2 if dataset larger). Script:
* llm/dataset/split_corpus.py
* Run:
  ```
  python llmlib/data/split_corpus.py

  ```

## Step 5 — Synthetic expansion (if you need to reach 1M chars quickly)
If you’re short, generate synthetic paraphrases via the model or ChatGPT. Keep them diverse and short. Example: use your model to paraphrase each line once with temperature/ sampling. Or use a tiny script to duplicate lines with simple template variations.

Quick synthetic generator (local, cheap):  `llm/dataset/synth_expand.py`
> This is quick and dirty — better to get real content, but it helps bootstrap.


## Step 6 — Stats and sanity checks
* Use this script to check tokens/characters and estimated token counts for BPE/vocab planning:
`llmlib/dataset/corpus_stats.py`
* run:
  `python projects/utils/corpus_stats.py`
* If total chars < 1,000,000, repeat collection or generate more synthetic samples until target reached.

## Step 7 — Tokenizer decision & recommendations
* When you have ~1M chars:
    * Retrain tokenizer on train.txt (not whole corpus if you want strict eval). Use ByteBPETokenizer for robustness.
* Vocab size recommendation:
    *  Small local: 4k–8k → faster, smaller embedding
    *  Better for 256-dim: 8k–12k (balanced)
    *  If you want GPT-like: 30k–50k (heavy)
* Since you already have ByteBPETokenizer and CharBPETokenizer, pick byte_bpe for best multilingual/emoji handling.
* Example call (with your trainer CLI):
```
python -m llmlib.cli.train_tokenizer_cli --config projects/4_tiny_gpt_bpe_v2/project_config.json --tokenizer-type byte_bpe --vocab-size 8000
```
> Make sure project_config.json tokenizer_config block exists or pass --vocab-size.


## Step 8 — Training settings for model after dataset
Given d_model=256, n_layers=8:
* Batch size: 8–16 (CPU may require 1–4)
* Sequence length: project_metadata.max_seq_length = 128 (as you used)
* Train steps: depends on dataset size. If you have N_train lines and average tokens ~30 per line, you can compute steps:

Estimate:
* If train tokens ≈ 150k tokens, and batch_size=16, seq_len=128 → tokens per step ≈ 16*128 = 2048 tokens/step
* epochs_needed ~ few passes. Start with train_steps = 5000 and monitor val loss. Increase to 10k–20k if needed.
Use validation perplexity as stopping criterion.


# Quick example commands to run end-to-end

1. Create raw dir and paste files.

2. Merge & clean:
`python projects/utils/prepare_corpus.py`

3. Dedup:
`bash projects/utils/dedupe_and_split.sh`

4. Split:
`python projects/utils/split_corpus.py`

5. Stats:
`python projects/utils/corpus_stats.py`

6. Retrain tokenizer (example):
` python -m llmlib.cli.train_tokenizer_cli --config projects/4_tiny_gpt_bpe_v2/project_config.json --tokenizer-type byte_bpe --vocab-size 8000`

7. Train model:
`python projects/4_tiny_gpt_bpe_v2/train_bpe_gpt_v2.py`

## how to run full pipeline:
```
this is the statistics I have:
$ cd $LLMLIB_ROOT
$ python src/llmlib/data/pipeline/run_full_data_pipeline.py       
=== Step 1: Prepare raw corpus ===
Found raw files: [PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/QA.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/african.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/asian.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/domain.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/fun_info.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/national_day.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/paraphrases.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/reasoning.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/synthetic.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/human/conversations.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/human/synthetic_smalltalk.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/human/user_writing.txt')]
Wrote /home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/out/elephant_human_90_10_corpus.txt | lines: 3851, chars: 491931
=== Step 2: Dedupe and filter corpus ===
Orig: 3851 | Unique: 3851 | Filtered: 3486
=== Step 3: Split corpus ===
Split statistics:
train: {'lines': 3418, 'chars': 309551, 'words': 52407}
val: {'lines': 34, 'chars': 3221, 'words': 552}
test: {'lines': 34, 'chars': 2846, 'words': 492}
=== Step 4: Generate synthetic expansions ===
Wrote 9116 synthetic lines to /home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/out/synthetic_generated.txt
=== Step 5: Corpus statistics ===
train.txt: lines 3418, chars 309551, words 52407
val.txt: lines 34, chars 3221, words 552
test.txt: lines 34, chars 2846, words 492
train.txt: {'lines': 3418, 'chars': 309551, 'words': 52407}
val.txt: {'lines': 34, 'chars': 3221, 'words': 552}
test.txt: {'lines': 34, 'chars': 2846, 'words': 492}
```