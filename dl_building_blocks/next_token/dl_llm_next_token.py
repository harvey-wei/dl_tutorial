import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name: open-access LLaMA-2 variant
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Load tokenizer and model with float16 precision
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically maps model to GPU if available
)

# Get embedding layer
embedding_layer = model.model.embed_tokens
embedding_weights = embedding_layer.weight  # Shape: (vocab_size, hidden_dim)
print(f"Embedding layer shape: {embedding_weights.shape}")

# Input string and tokenization
text = "Have se"
inputs = tokenizer(text, return_tensors="pt").to(model.device)  # move to GPU
token_id = inputs.input_ids[0, 0].item()
print(f"Token: '{text}' → Token ID: {token_id}")

# Get embedding for the first token
token_embedding = embedding_weights[token_id]
print(f"Embedding shape: {token_embedding.shape}")
print(f"Embedding (first 5 values): {token_embedding[:5]}")

# Generate next token
with torch.no_grad():
    output = model.generate(inputs.input_ids, max_new_tokens=1)  # appends 1 token

next_token_id = output[0, -1].item()
next_token = tokenizer.decode([next_token_id])
next_token_embedding = embedding_weights[next_token_id]

print(f"Next Token: {next_token} (ID: {next_token_id})")
print(f"Next Token Embedding (first 5 values): {next_token_embedding[:5]}")
