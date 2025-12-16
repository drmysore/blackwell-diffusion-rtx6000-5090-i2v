import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from diffusers.utils import load_image
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision


# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

num_inference_steps = 4  # you can also use the 8-step model to improve the quality
rank = 32  # you can also use the rank=128 model to improve the quality

model_paths = {
    4: f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit-lightningv1.0-4steps.safetensors",
    8: f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit-lightningv1.0-8steps.safetensors",
}


transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[num_inference_steps])
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
).to("cuda")


image = load_image(
    "https://media.wired.com/photos/68ba2ddb68c1027d7bc61668/master/w_2240,c_limit/Tech-CEOs-Fawn-Over-Trump-Politics-2233060770.jpg",
).convert("RGB")

prompt = "make the woman's white dress navy blue and change nothing else"
inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 1,
    "negative_prompt": " ",
    "num_inference_steps": num_inference_steps,
}
output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"qwen-image-edit-lightning-r{rank}-{num_inference_steps}steps-0x01.png")

prompt = "make the woman's white dress pastel pink and change nothing else"
inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 1,
    "negative_prompt": " ",
    "num_inference_steps": num_inference_steps,
}
output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"qwen-image-edit-lightning-r{rank}-{num_inference_steps}steps-0x02.png")

prompt = "make the woman look to her right and change nothing else"
inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 1,
    "negative_prompt": " ",
    "num_inference_steps": num_inference_steps,
}
output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"qwen-image-edit-lightning-r{rank}-{num_inference_steps}steps-0x03.png")

prompt = "remove the bald man from the photo and change nothing else"
inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 1,
    "negative_prompt": " ",
    "num_inference_steps": num_inference_steps,
}
output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"qwen-image-edit-lightning-r{rank}-{num_inference_steps}steps-0x04.png")
