import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model + tokenizer (LLaMA-2 chat variant)
model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# User input text
text = "I would like invite you to dinner."
inputs = tokenizer(text, return_tensors="pt")
inputs = inputs.to(model.device)

# Get token ID of first token in input
input_token_ids = inputs.input_ids
first_token_id = input_token_ids[0, 0].item()

# Print original token info
print(f"Input text: '{text}'")
print(f"First token ID: {first_token_id}")
print(f"First token: {tokenizer.decode([first_token_id])}")

# Access token embedding
embedding_weights = model.model.embed_tokens.weight
input_embedding = embedding_weights[first_token_id]
print(f"Input token embedding shape: {input_embedding.shape}")
print(f"Input embedding (first 5 values): {input_embedding[:5].cpu()}")

# Generate next N tokens
N = 20
max_len = input_token_ids.shape[1] + N

with torch.no_grad():
    generated = model.generate(
        input_token_ids,
        # max_new_tokens=N,
        max_length=max_len,
        do_sample=True,         # change to True for sampling
        temperature=1.0,         # lower (<1) = more confident, higher (>1) = more random
    )

# Extract only newly generated tokens (excluding input)
new_token_ids = generated[0, input_token_ids.shape[1]:]
new_tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
print(f"\nGenerated {N} tokens:")
for i, (tid, tok) in enumerate(zip(new_token_ids, new_tokens)):
    embedding = embedding_weights[tid]
    print(f"\nToken {i+1}: '{tokenizer.decode([tid])}' (ID: {tid.item()})")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5 values): {embedding[:5].cpu()}")

print(f"\nGenerated text: '{tokenizer.decode(new_token_ids)}'")
