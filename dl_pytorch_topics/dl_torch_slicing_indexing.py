import torch
import torch.nn as nn



alphas = torch.tensor([0.9, 0.8, 0.7, 0.6])  # shape: [4]
t = torch.tensor([0, 2, 1])  # shape: [3]

selected = alphas[t]  # selects alphas[0], alphas[2], alphas[1]
print(selected)  # tensor([0.9, 0.7, 0.8])



B, L, V = 8, 12, 100
llm_logits = torch.randn(B, L, V)

llm_logit = llm_logits[:, -1, :]
print(f'shape of llm_logits[:, -1, :] {llm_logit.shape}')

llm_logit = llm_logits[:, 2:3, :]
print(f'shape of llm_logits[:, 2:3, :] {llm_logit.shape}')


B, L = 1, 1
tokens = torch.randn(B, L)
print(f'shape of tokens[0, 0] {tokens[0, 0].shape}')
print(f'shape of tokens[:, 0] {tokens[:, 0].shape}')
