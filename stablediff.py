import cv2
import torch
import numpy as np
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import make_image_grid

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",  torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                         controlnet=controlnet,
                                                         torch_dtype=torch.float16)
#use GPU
pipe.to("cuda")

# Load image
sample_image = Image.open("letter/A.jpg")

# Make canny image
canny_image = np.array(sample_image)
low_thresh = 100
high_thresh = 200
canny_image = cv2.Canny(canny_image, low_thresh, high_thresh)
canny_image = Image.fromarray(canny_image)

# Image generation
prompt = "(flowers, plants), digital art, flat, simplistic, cartoon"

negative_prompt = "(nsfw)"

# specify Stable Diffusion pipeline parameters
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=500,
    height=500,
    image=canny_image,
    guidance_scale=5,
    num_images_per_prompt=1, # number of images to generate
    controlnet_conditioning_scale=0.35, # contribution of controlnet
    num_inference_steps=30,
    generator=torch.manual_seed(311)).images

image[0].save('output/output6.jpg', 'JPEG')
