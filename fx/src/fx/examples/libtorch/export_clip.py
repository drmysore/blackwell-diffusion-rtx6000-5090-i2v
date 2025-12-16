# export_clip.py
import torch
from transformers import CLIPTextModelWithProjection

device = torch.device("cuda")

# Load CLIP
clip = CLIPTextModelWithProjection.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
).to(device)
clip.eval()

for param in clip.parameters():
    param.requires_grad = False


def encode_clip(input_ids, attention_mask):
    with torch.no_grad():
        outputs = clip(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.text_embeds, outputs.last_hidden_state


# CLIP uses shorter sequences
max_length = 77
example_input_ids = torch.randint(0, 49408, (1, max_length), device=device)
example_attention_mask = torch.ones(1, max_length, dtype=torch.long, device=device)

with torch.no_grad():
    traced_clip = torch.jit.trace(encode_clip, (example_input_ids, example_attention_mask))

traced_clip.save("/tmp/flux_clip_encoder.pt")
print("CLIP exported")

# Test
clip_model = torch.jit.load("/tmp/flux_clip_encoder.pt", map_location="cuda")
pooled, hidden = clip_model(example_input_ids, example_attention_mask)
print(f"CLIP pooled: {pooled.shape}, hidden: {hidden.shape}")
