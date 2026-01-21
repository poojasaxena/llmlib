# Tokenization Sanity Scripts

These scripts are **manual sanity checks**, not automated tests.

Use them when:
- developing a tokenizer
- debugging training issues
- visually inspecting encode/decode behavior

They are NOT run by pytest.

# Example Test

### Testing of BPE byte Tokenizer
```
$ python -m scripts.tokenization.sanity_byte_bpe_tokenizer                         
Vocab size: 280
----
INPUT   : hello
IDS     : [2, 270, 3]
DECODED : hello
----
INPUT   : hello elephants
IDS     : [2, 270, 36, 267, 3]
DECODED : hello elephants
----
INPUT   : where elephants live?
IDS     : [2, 274, 36, 267, 36, 279, 3]
DECODED : where elephants live?
----
INPUT   : Elephants live in Africa and Asia.
IDS     : [2, 73, 112, 105, 116, 108, 101, 114, 120, 119, 36, 278, 36, 109, 114, 36, 69, 106, 118, 109, 103, 101, 36, 101, 114, 104, 36, 69, 119, 109, 101, 50, 3]
DECODED : Elephants live in Africa and Asia.
```
### Testing of bpe Training
```
$ python -m scripts.tokenization.sanity_bpe_training
Text      : Hello elephants!
Token IDs : [2, 1599, 36, 329, 37, 3]
Decoded   : Hello elephants!
```