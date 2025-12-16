import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision


scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

num_inference_steps = 4
rank = 32
model_paths = {
    4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
    8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
}

transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[num_inference_steps])
pipe = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
).to("cuda")


prompt = """Highly photorealistic selfie of a beautiful brunette woman in a red summer dress holding a handwritten cardboard sign that reads "qwen is at least as fast as FLUX.1 dev..." in Times Square at night. Sharp detailed background showing Broadway billboards with clear readable text "PHANTOM OF THE OPERA", "COCA-COLA", "NASDAQ". LED tickers displaying "BREAKING NEWS" and stock prices. She's smiling at the camera, sign perfectly in focus. Crisp details: reflections on wet pavement, sharp city lights, clear storefront signs reading "FOREVER 21", "PLANET HOLLYWOOD". Every text element crystal clear and legible."""

negative_prompt = "blurry, low quality, out of focus, soft focus, depth of field, bokeh, unclear text, illegible signs, fuzzy background"

for idx in range(5):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=1.0,
    ).images[0]
    image.save(f"qwen-generate_r{rank}-{idx}.png")
