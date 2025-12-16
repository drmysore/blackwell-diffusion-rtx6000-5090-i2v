import torch
from diffusers import AutoencoderKL

device = torch.device("cuda")
vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", subfolder="vae", torch_dtype=torch.bfloat16
).to(device)
vae.eval()

for param in vae.parameters():
    param.requires_grad = False


def encode_fn(x):
    with torch.no_grad():
        return vae.encode(x).latent_dist.sample()


def decode_fn(z):
    with torch.no_grad():
        return vae.decode(z).sample


example_image = torch.randn(1, 3, 1024, 1024, dtype=torch.bfloat16, device=device)
example_latent = torch.randn(1, 16, 128, 128, dtype=torch.bfloat16, device=device)

with torch.no_grad():
    traced_encoder = torch.jit.trace(encode_fn, example_image)
    traced_decoder = torch.jit.trace(decode_fn, example_latent)

traced_encoder.save("/tmp/flux_vae_encoder.pt")
traced_decoder.save("/tmp/flux_vae_decoder.pt")

print("VAE exported")

encoder = torch.jit.load("/tmp/flux_vae_encoder.pt", map_location="cuda")
decoder = torch.jit.load("/tmp/flux_vae_decoder.pt", map_location="cuda")

test_img = torch.randn(1, 3, 1024, 1024, dtype=torch.bfloat16, device="cuda")
latent = encoder(test_img)
reconstructed = decoder(latent)
print(f"Latent: {latent.shape}, Reconstructed: {reconstructed.shape}")
