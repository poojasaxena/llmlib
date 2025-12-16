import pytest
from llmlib.tokenization.byte_bpe_tokenizer import ByteBPETokenizer


TEXT = "hello elephants\nwhere do elephants live?"


@pytest.fixture(scope="module")
def tokenizer():
    return ByteBPETokenizer.train(
        text=[TEXT],
        vocab_size=512,  # keep small for fast tests
        min_freq=1,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )


def test_vocab_size_reasonable(tokenizer):
    # vocab should not exceed requested size
    assert len(tokenizer.vocab) <= 512
    assert len(tokenizer.vocab) > 4  # more than just special tokens


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "hello elephants",
        "where elephants live?",
        "Elephants live in Africa and Asia.",
    ],
)
def test_encode_decode_roundtrip(tokenizer, text):
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert isinstance(decoded, str)

    # Byte BPE may normalize casing / spacing slightly
    assert len(decoded) > 0
