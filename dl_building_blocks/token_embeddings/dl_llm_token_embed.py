import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use open-access LLaMA-2 compatible variant
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Access token embedding layer
embedding_layer = model.model.embed_tokens  # nn.Embedding
embedding_weights = embedding_layer.weight  # Tensor of shape (vocab_size, hidden_dim)

print(f'Embedding layer shape: {embedding_layer.weight.shape}')

# Example: retrieve embedding for token
token = "hello"
token_id = tokenizer.convert_tokens_to_ids(token)
token_embedding = embedding_weights[token_id]

print(f"Token ID: {token_id}")
print(f"Embedding shape: {token_embedding.shape}")
print(f"Embedding (first 5 values): {token_embedding[:5]}")
