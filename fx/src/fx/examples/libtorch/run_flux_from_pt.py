# run_flux_from_pt_working.py
import torch
from diffusers import FluxPipeline
from PIL import Image


class FluxPipelineFromPT:
    def __init__(self, model_dir="/tmp"):
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        # Load traced models
        print("Loading traced models...")
        self.vae_decoder = torch.jit.load(f"{model_dir}/flux_vae_decoder.pt").to(self.device)
        self.t5_encoder = torch.jit.load(f"{model_dir}/flux_t5.pt").to(self.device)
        self.clip_encoder = torch.jit.load(f"{model_dir}/flux_clip.pt").to(self.device)

        # Load reference pipeline
        print("Loading reference pipeline...")
        self.ref_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=self.dtype
        ).to(self.device)

        self.tokenizer = self.ref_pipe.tokenizer
        self.tokenizer_2 = self.ref_pipe.tokenizer_2

    @torch.no_grad()
    def __call__(self, prompt, height=1024, width=1024, num_inference_steps=4):
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(self.device)

        text_inputs_2 = self.tokenizer_2(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Use traced encoders
        pooled_prompt_embeds = self.clip_encoder(text_inputs.input_ids, text_inputs.attention_mask)
        prompt_embeds = self.t5_encoder(text_inputs_2.input_ids, text_inputs_2.attention_mask)

        # Run the pipeline with latent output
        outputs = self.ref_pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            output_type="latent",
        )

        # Get latents - they're in packed format [1, 4096, 64]
        packed_latents = outputs.images
        print(f"Packed latent shape: {packed_latents.shape}")

        # Unpack using the pipeline's method
        latents = self.ref_pipe._unpack_latents(
            packed_latents, height=height // 8, width=width // 8, vae_scale_factor=8
        )
        print(f"Unpacked latent shape: {latents.shape}")

        # Decode with traced VAE decoder
        image = self.vae_decoder(latents)
        print(f"Decoded image shape: {image.shape}")

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")

        return Image.fromarray(image[0])


# Test
pipe = FluxPipelineFromPT()
image = pipe("A majestic lion in the savanna", num_inference_steps=4)
image.save("flux_output.png")
print(f"Saved flux_output.png with size {image.size}")
