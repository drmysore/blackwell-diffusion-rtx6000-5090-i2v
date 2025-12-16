import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision


transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
)

pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

image = load_image("/home/b7r6/IMG_20250909_122139_359.jpg").convert("RGB")

prompt = "Keep the face the same but give it glowing blue eyes like Dune"
image = pipeline(image=image, prompt=prompt, guidance_scale=2.5).images[0]
image.save("flux-kontext-dev.png")
