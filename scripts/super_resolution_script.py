# Super Resolution Script using stable diffusion
"""
!pip install git+https://github.com/huggingface/diffusers.git
"""

import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
device = "cuda" if torch.cuda.is_available() else "cpu"

low_res_img = Image.open("/kaggle/input/super-image-resolution/Data/LR/25.png")

def super_res_enhance(low_res_img, num_inference_steps=100, eta=1):
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    upscaled_image = pipeline(
        low_res_img, num_inference_steps=num_inference_steps, eta=eta).images[0]
    upscaled_image.save("ldm_generated_image.png")
    return upscaled_image

upscaled_image = super_res_enhance(low_res_img)
