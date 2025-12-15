from llmlib.tokenization.training.bpe_training import train_and_save_tokenizer

cfg = {
    "model_config": {
        "d_model": 192,
        "n_heads": 4,
        "n_layers": 4,
        "max_position_embeddings": 256,
        "dropout": 0.1,
        "num_embeddings": 5096,
    },
    "training_config": {
        "batch_size": 8,
        "block_size": 128,
        "learning_rate": 0.0005,
        "train_steps": 3000,
        "num_epochs": 5,
        "eval_interval": 100,
        "eval_iters": 200,
    },
    "project_metadata": {
        "model_name": "gpt-bpe-v2",
        "model_save_path": "llm/language_models/elephantdomain_gpt/",
        "tokenizer_save_path": "llm/tokenizers/bpe-elephant/v2/tokenizer.json",
        "data_path": "llm/mixed_text/out",
        "data_file": "train.txt",
        "max_seq_length": 128,
        "max_new_tokens": 80,
    },
    "data": {
        "root_path": "llm/splits",
        "train_file": "train.txt",
        "val_file": "val.txt",
        "test_file": "test.txt",
    },
    "tokenizer_config": {
        "type": "byte_bpe",
        "vocab_size": 5096,
        "min_freq": 2,
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
    },
}
# load your config.json as a dict

tokenizer, _ = train_and_save_tokenizer(
    cfg, tokenizer_type="byte_bpe", vocab_size=5096, min_freq=2, save=False
)

# Test encoding/decoding
text = "Hello elephants!"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print("Text      :", text)
print("Token IDs :", ids)
print("Decoded   :", decoded)
