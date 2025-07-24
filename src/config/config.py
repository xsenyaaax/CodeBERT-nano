# some default values

import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path("models/model_v3.pt")
TOKENIZER_PATH = Path("tokenizer/code_tokenizer_metaspace.json")

labels = ['c', 'cpp', 'css', 'html', 'java', 'javascript', 'python', 'r', 'sqlite']

language_logos = {
    "c": "static/logos/c.png",
    "cpp": "static/logos/cpp.png",
    "css": "static/logos/css.png",
    "html": "static/logos/html.png",
    "java": "static/logos/java.png",
    "javascript": "static/logos/js.png",
    "python": "static/logos/python.png",
    "r": "static/logos/r.png",
    "sqlite": "static/logos/sqlite.png"
}
