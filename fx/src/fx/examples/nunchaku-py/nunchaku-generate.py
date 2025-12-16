import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.utils import get_precision


precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev-colossus/svdq-{precision}_r32-flux.1-dev-colossusv12.safetensors"
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

composed_lora = compose_lora(
    [
        ("Shakker-Labs/FLUX.1-dev-LoRA-add-details/FLUX-dev-lora-add_details.safetensors", 0.6),
        ("alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors", 0.2),
    ]
)  # set your lora strengths here when using composed lora

transformer.update_lora_params(composed_lora)
### End of LoRA Related Code ###

image = pipeline(
    "a handsome older wise-looking guy with stubble in a sky ping summer suit standing in the middle of a busy street in manhattan holding a sign that says 'we tried building it wrong, time to build it fast...'",
    num_inference_steps=30,
    guidance_scale=6.5,
).images[0]

image.save(f"flux.1-dev-colossus-{precision}.png")
