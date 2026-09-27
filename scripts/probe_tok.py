from localagent.model.tokenizer import load_tokenizer
tok = load_tokenizer("bpe", "data/tokenizer-h100-16k.json")
print(type(tok).__name__)
print([a for a in dir(tok) if not a.startswith("_")][:25])
for attr in ("vocab", "get_vocab", "token_to_id", "tok", "encoder"):
    print(attr, hasattr(tok, attr))
