import torch
from transformers import T5EncoderModel

device = torch.device("cuda")

# Load T5 encoder
t5 = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16,
).to(device)
t5.eval()

# Disable gradients
for param in t5.parameters():
    param.requires_grad = False


# Wrapper function for clean tracing
def encode_text(input_ids, attention_mask):
    with torch.no_grad():
        return t5(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


# Fixed sequence length for FLUX
max_length = 512
batch_size = 1

# Create example inputs
example_input_ids = torch.randint(0, 32128, (batch_size, max_length), device=device)
example_attention_mask = torch.ones(batch_size, max_length, dtype=torch.long, device=device)

# Trace
with torch.no_grad():
    traced_t5 = torch.jit.trace(encode_text, (example_input_ids, example_attention_mask))

traced_t5.save("/tmp/flux_t5_encoder.pt")
print("T5 exported")

# Test
t5_model = torch.jit.load("/tmp/flux_t5_encoder.pt", map_location="cuda")
test_ids = torch.randint(0, 32128, (1, max_length), device=device)
test_mask = torch.ones(1, max_length, dtype=torch.long, device=device)
output = t5_model(test_ids, test_mask)
print(f"T5 output shape: {output.shape}")  # Should be [1, 512, 4096]
