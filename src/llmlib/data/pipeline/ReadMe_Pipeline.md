## How to run
1. As code from python project:
```
from llmlib.data.pipeline.elephant_gpt_pipeline import main as run_pipeline
run_pipeline()
```

2. via Terminal
```
$ cd $LLMLIB_ROOT
$ python src/llmlib/data/pipeline/run_full_data_pipeline.py               
=== Step 1: Prepare raw corpus ===
Found raw files: [PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/QA.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/african.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/asian.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/domain.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/fun_info.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/national_day.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/paraphrases.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/reasoning.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/elephant/synthetic.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/human/conversations.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/human/synthetic_smalltalk.txt'), PosixPath('/home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/human/user_writing.txt')]
Wrote /home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/out/elephant_human_90_10_corpus.txt | lines: 5081, chars: 553742
=== Step 2: Generate synthetic expansions ===
Wrote 13550 synthetic lines to /home/pooja-saxena/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/out/synthetic_generated.txt
=== Step 3: Combine human + synthetic corpus ===
=== Step 4: Dedupe and filter combined corpus ===
Orig: 18631 | Unique: 7340 | Filtered: 6767
=== Step 5: Split corpus ===
Split statistics:
train: {'lines': 5415, 'chars': 503828, 'words': 85800}
val: {'lines': 676, 'chars': 66018, 'words': 11298}
test: {'lines': 676, 'chars': 61040, 'words': 10436}
=== Step 6: Corpus statistics ===
train.txt: lines 5415, chars 503828, words 85800
val.txt: lines 676, chars 66018, words 11298
test.txt: lines 676, chars 61040, words 10436
train.txt: {'lines': 5415, 'chars': 503828, 'words': 85800}
val.txt: {'lines': 676, 'chars': 66018, 'words': 11298}
test.txt: {'lines': 676, 'chars': 61040, 'words': 10436}
```
3. Make it executable
    * I have made it executable via console sript entry in pyproject.toml
```
[tool.poetry.scripts]
run-data-pipeline = "llmlib.data.pipeline.run_full_data_pipeline:main"
```
Then one can run from terminal:
```
$ run-data-pipeline

```