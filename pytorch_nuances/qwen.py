from transformers import AutoTokenizer, AutoModelForCausalLM

# import QWen2VL-2B Instruction model architecture
from transformers import Qwen2VLForConditionalGeneration

# Load pretrain model
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2VL-2B-Instruct")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2VL-2B-Instruct")

# Load image
# image = Image.open("image.jpg")

