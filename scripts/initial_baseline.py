"""
!pip install git+https://github.com/huggingface/diffusers.git -q
!pip install stitching -q
!pip install transformers -q
"""

import os
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import stitching
device = "cuda" if torch.cuda.is_available() else "cpu"


def img_res_enhancer(low_res_img, num_inference_steps=100, eta=1):
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    upscaled_image = pipeline(
        low_res_img, num_inference_steps=num_inference_steps, eta=eta).images[0]
    return upscaled_image


def img_stitcher(path1, path2, path3, confidence_threshold=0.2):
    stitcher = stitching.Stitcher()
    settings = {"detector": "sift",
                "confidence_threshold": confidence_threshold}
    panorama = stitcher.stitch([path1, path2, path3])
    return panorama


if __name__ == "main":
    low_res_img_1 = Image.open(
        "/kaggle/input/image-stitching-from-drone-capture-opencv/drone/image_0081.jpg").resize((128, 128))
    low_res_img_2 = Image.open(
        "/kaggle/input/image-stitching-from-drone-capture-opencv/drone/image_0091.jpg").resize((128, 128))
    low_res_img_3 = Image.open(
        "/kaggle/input/image-stitching-from-drone-capture-opencv/drone/image_0101.jpg").resize((128, 128))
    upscaled_image_1 = img_res_enhancer(low_res_img_1, num_inference_steps=100, eta=1)
    upscaled_image_2 = img_res_enhancer(low_res_img_2, num_inference_steps=100, eta=1)
    upscaled_image_3 = img_res_enhancer(low_res_img_3, num_inference_steps=100, eta=1)
    upscaled_image_1.save("ldm_generated_image_1.png")
    upscaled_image_2.save("ldm_generated_image_2.png")
    upscaled_image_3.save("ldm_generated_image_3.png")
    path1 = "/kaggle/working/ldm_generated_image_1.png"
    path2 = "/kaggle/working/ldm_generated_image_3.png"
    path3 = "/kaggle/working/ldm_generated_image_2.png"
    panorama = img_stitcher(path1, path2, path3)
    panorama.save("panorama.png")
