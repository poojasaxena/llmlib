## llmlib/test/tokenization/test_byte_bpe_tokenizer.py
"""
Manual sanity check for ByteBPETokenizer.
Run directly, NOT via pytest.
"""
from llmlib.tokenization.byte_bpe_tokenizer import ByteBPETokenizer

# Load or train tokenizer
TEXT = "hello elephants\nwhere do elephants live?"

tok = ByteBPETokenizer.train(
    text=TEXT,
    vocab_size=3072,
    min_freq=1,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
)

print("Vocab size:", len(tok.vocab))

tests = [
    "hello",
    "hello elephants",
    "where elephants live?",
    "Elephants live in Africa and Asia.",
]

for t in tests:
    ids = tok.encode(t)
    decoded = tok.decode(ids)
    print("----")
    print("INPUT   :", t)
    print("IDS     :", ids)
    print("DECODED :", decoded)
