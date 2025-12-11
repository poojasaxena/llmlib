# test/test_tokenizer.py

from llmlib.tokenization.tokenizer import encode, decode, VOCAB_SIZE


def test_encode_decode_roundtrip():
    text = "hello world!"
    ids = encode(text)
    decoded = decode(ids)
    # Not all characters may be in vocab; we only check that
    # decoding reverts what we actually encoded.
    reencoded = encode(decoded)
    assert (ids == reencoded).all()
    assert isinstance(VOCAB_SIZE, int)
    assert VOCAB_SIZE > 0
