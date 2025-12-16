from pathlib import Path

import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel


local_model_path = Path.expanduser("~/flux.1-schnell-nunchaku/flux.1-schnell")

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    local_model_path,
    local_files_only=True,
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda")

image = pipeline(
    "A cat holding a sign that says hello world",
    width=1024,
    height=1024,
    num_inference_steps=4,
    guidance_scale=0,
).images[0]

image.save("flux.1-schnell-local.png")
