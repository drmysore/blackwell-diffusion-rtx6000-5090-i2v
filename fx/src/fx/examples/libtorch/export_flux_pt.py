import torch
from diffusers import FluxPipeline

device = torch.device("cuda")
dtype = torch.bfloat16

# Load pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=dtype).to(
    device
)

# Set eval
for model in [pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

with torch.no_grad():
    # VAE functions
    def vae_encode(x):
        return pipe.vae.encode(x).latent_dist.sample()

    def vae_decode(z):
        return pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample

    # Text encoder functions
    def t5_encode(input_ids, attention_mask):
        return pipe.text_encoder_2(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    def clip_encode(input_ids, attention_mask):
        return pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    # Create examples
    example_image = torch.randn(1, 3, 1024, 1024, dtype=dtype, device=device)
    example_latent = torch.randn(1, 16, 128, 128, dtype=dtype, device=device)
    example_clip_ids = torch.randint(0, 49408, (1, 77), device=device)
    example_clip_mask = torch.ones(1, 77, dtype=torch.long, device=device)
    example_t5_ids = torch.randint(0, 32128, (1, 512), device=device)
    example_t5_mask = torch.ones(1, 512, dtype=torch.long, device=device)

    # Trace the working components
    print("Tracing VAE encoder...")
    traced_vae_encoder = torch.jit.trace(vae_encode, example_image)
    traced_vae_encoder.save("/tmp/flux_vae_encoder.pt")

    print("Tracing VAE decoder...")
    traced_vae_decoder = torch.jit.trace(vae_decode, example_latent)
    traced_vae_decoder.save("/tmp/flux_vae_decoder.pt")

    print("Tracing T5...")
    traced_t5 = torch.jit.trace(t5_encode, (example_t5_ids, example_t5_mask))
    traced_t5.save("/tmp/flux_t5.pt")

    print("Tracing CLIP...")
    traced_clip = torch.jit.trace(clip_encode, (example_clip_ids, example_clip_mask))
    traced_clip.save("/tmp/flux_clip.pt")

    print("\nSuccessfully exported:")
    print("- VAE encoder/decoder")
    print("- T5 text encoder")
    print("- CLIP text encoder")
    print("\nTransformer export skipped due to RoPE complexity.")
    print("For transformer, consider using ONNX or the original PyTorch model directly.")
