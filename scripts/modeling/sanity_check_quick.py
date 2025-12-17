model.eval()
ids = torch.tensor([[tokenizer.token_to_id("<bos>")]], device=device)
with torch.no_grad():
    logits = model(ids)

top = torch.topk(logits[0, -1], 10)
for i in top.indices.tolist():
    print(i, repr(tokenizer.id_to_token(i)))
