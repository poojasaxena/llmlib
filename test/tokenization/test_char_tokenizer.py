# test/test_tokenizer.py

from llmlib.tokenization.char_tokenizer import VOCAB_SIZE, CharTokenizer

def test_encode_decode_roundtrip():

    text = "hello world!"
    tokenizer = CharTokenizer()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    # Not all characters may be in vocab; we only check that
    # decoding reverts what we actually encoded.
    reencoded = tokenizer.encode(decoded)
    assert (ids == reencoded)
    assert isinstance(VOCAB_SIZE, int)
    assert VOCAB_SIZE > 0
