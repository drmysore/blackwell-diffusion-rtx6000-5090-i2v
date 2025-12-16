# export_dit.py
import torch
from diffusers import FluxTransformer2DModel

device = torch.device("cuda")

transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
).to(device)
transformer.eval()

for param in transformer.parameters():
    param.requires_grad = False


def forward_dit(
    hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids
):
    with torch.no_grad():
        return transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]


batch_size = 1
height = 128
width = 128
channels = 64
txt_seq_len = 512

# Latents
example_img_latents = torch.randn(
    batch_size, height * width, channels, dtype=torch.bfloat16, device=device
)
example_txt_latents = torch.randn(
    batch_size, txt_seq_len, channels, dtype=torch.bfloat16, device=device
)
example_hidden = torch.cat([example_img_latents, example_txt_latents], dim=1)

# Text encoder output - needs to match TOTAL sequence length for joint attention
# Pad or repeat to match combined length
example_encoder_hidden = torch.randn(
    batch_size, height * width + txt_seq_len, 4096, dtype=torch.bfloat16, device=device
)

# CLIP pooled
example_pooled = torch.randn(batch_size, 768, dtype=torch.bfloat16, device=device)

# Timestep
example_timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=device)

# Position IDs - 2D not 3D
# Image IDs: [height*width, 3]
example_img_ids = torch.zeros(height * width, 3, dtype=torch.bfloat16, device=device)
for i in range(height):
    for j in range(width):
        idx = i * width + j
        example_img_ids[idx, 0] = i
        example_img_ids[idx, 1] = j

# Text IDs: [seq_len, 3]
example_txt_ids = torch.zeros(txt_seq_len, 3, dtype=torch.bfloat16, device=device)

# Trace
with torch.no_grad():
    traced_dit = torch.jit.trace(
        forward_dit,
        (
            example_hidden,
            example_encoder_hidden,
            example_pooled,
            example_timestep,
            example_img_ids,
            example_txt_ids,
        ),
    )

traced_dit.save("/tmp/flux_transformer.pt")
print("DiT exported")

# Test
dit_model = torch.jit.load("/tmp/flux_transformer.pt", map_location="cuda")
output = dit_model(
    example_hidden,
    example_encoder_hidden,
    example_pooled,
    example_timestep,
    example_img_ids,
    example_txt_ids,
)
print(f"DiT output shape: {output.shape}")
