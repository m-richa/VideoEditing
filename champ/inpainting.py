import inspect, os
from typing import List, Optional, Union
from omegaconf import OmegaConf
import numpy as np
import torch
import argparse
import PIL

from diffusers import StableDiffusionInpaintPipeline


def main(cfg):
    
    device = "cuda"
    model_path = "runwayml/stable-diffusion-inpainting"
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    
    image = PIL.Image.open(cfg.inpaint_image_path).convert("RGB")
    mask = PIL.Image.open(cfg.mask_path).convert("RGB")
    
    prompt = cfg.prompt
    print(prompt)

    guidance_scale=7.5
    num_samples = 1
    generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results
    
    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images
    
    images[0].save(os.path.join(cfg.ref_img_path, "test.png"))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference_inpaint.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    main(cfg)