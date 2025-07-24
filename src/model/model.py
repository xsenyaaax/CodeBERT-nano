# Model definition taken the from notebook/transformer.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, head_size, embedding_dim, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None, return_attention=False):
        B, T, C = x.shape 
        
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) * (B, head_size, T) -> (B, T, T)
        
        if attention_mask is not None:
            mask = attention_mask[:, None, :]
            wei = wei.masked_fill(mask == 0, float('-inf'))
        
        wei_soft = F.softmax(wei, dim=-1)
        wei_drop = self.dropout(wei_soft)
        
        out = wei_drop @ v # (B, T, T) * (B, T, head_size) -> (B, T, head_size) 
        return (out, wei_soft) if return_attention else out
        
class TransformerClassifierV3(nn.Module):
    def __init__(self, vocab_size, n_embd, num_classes, block_size, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attn_head = Head(n_embd, n_embd, dropout, block_size)
        self.classifier = nn.Linear(n_embd, num_classes)

    def forward(self, x, attn_masks=None, return_attention=False):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        
        out = tok_emb + pos_emb  # (B, T, C)
        
        if return_attention:
            out, attention = self.attn_head(out, attn_masks, return_attention=True)  # (B, T, C)
        else:
            out = self.attn_head(out, attn_masks)
            attention = None
                
        cls_token = out[:, 0, :]
        logits = self.classifier(cls_token)
        return (logits, attention) if return_attention else logits