# prediction and classification code

import torch
from src.model.model import TransformerClassifierV3
from src.utils.tokenizer_utils import load_tokenizer
from src.config.config import device, MODEL_PATH, TOKENIZER_PATH, labels, language_logos
from bertviz import head_view
import pandas as pd
from gradio import Error

tokenizer = load_tokenizer(str(TOKENIZER_PATH))
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = TransformerClassifierV3(
    vocab_size=checkpoint['vocab_size'],
    n_embd=checkpoint['embedding_dim'],
    num_classes=checkpoint['num_classes'],
    block_size=checkpoint['block_size']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


@torch.no_grad()
def predict(text: str, return_attention=False, max_tokens_in_attn=None):
    ret = {}
    model.eval()
    
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor(encoded.ids, device=device).unsqueeze(0) # (1, T)
    attention_mask = torch.tensor(encoded.attention_mask, device=device).unsqueeze(0) # (1, T)

    logits, attention = model(input_ids, attention_mask, return_attention=True)
    
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    predicted_label = labels[pred]
    ret['label'] = predicted_label
    ret['probs'] = probs.squeeze(0).tolist() 
    
    if return_attention:
        first_input_ids = input_ids[0]
        tokens = [tokenizer.id_to_token(id.item()) for id in first_input_ids]
        attention_matrix = attention[0].cpu() # (T, T)
        
        #N = 15
        if max_tokens_in_attn:
            tokens = tokens[:max_tokens_in_attn]
            attention_matrix = attention_matrix[:max_tokens_in_attn, :max_tokens_in_attn]
        attention_matrix = attention_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attention_viz = head_view(attention=[attention_matrix], tokens=tokens, sentence_b_start=None, html_action='return') 
        ret['attention_html'] = attention_viz.data
        with open("tmp.html", 'w') as f:
            f.write(attention_viz.data)
        
    return ret
    
def classify_code_no_attention(code: str):    
    if not code.strip():
        return Error("Please enter some code to classify.")
    
    ret = predict(code)
    df = pd.DataFrame({
        "Language": labels,
        "Probability": ret['probs']
    })
    
    logo_path = language_logos.get(ret['label'])    

    return logo_path, df
