from tokenizers import Tokenizer

def load_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer
