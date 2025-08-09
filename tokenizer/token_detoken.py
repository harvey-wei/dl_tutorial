
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Text to token IDs
text = "Transformers are powerful ML"
input_ids = tokenizer.encode(text, return_tensors="pt")  # shape: [1, seq_len]
print(f'Shape of input_ids {input_ids.size()}')

# 3. Token IDs to embeddings (implicitly handled by model)
# model.transformer.wte = word token embedding
# model.transformer.wpe = positional embedding
# embeddings = model.transformer.wte(input_ids)

# 4. Pass through model
outputs = model(input_ids)
logits = outputs.logits  # shape: [1, seq_len, vocab_size]

print(f'Shape of logits {logits.size()}')

# 5. Generate next token
next_token_id = torch.argmax(logits[:, -1, :], dim=-1)  # greedy decoding
generated_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
print(f'Generate_ids shape {generated_ids.shape}')

# 6. Decode token IDs back to text
decoded_output = tokenizer.decode(generated_ids[0])
print(decoded_output)  # e.g., "Transformers are powerful models"
