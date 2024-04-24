import torch
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

pipe = pipe.to("cuda")


def create_letter(text):
    font = ImageFont.truetype('Helvetica', 500)
    # Create a new image with a white background
    image = Image.new("RGB", (512, 512), (255, 255, 255))

    # Get the size of the text and calculate its position in the center of the image
    text_size = font.getbbox(text)
    text_x = (image.width - text_size[2]) / 2
    text_y = (image.height - text_size[3]) / 2

    # Draw the text onto the image multiple times with a slight offset to create a shadow effect
    draw = ImageDraw.Draw(image)

    # Draw the final text layer on top of the shadows
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    return image


def create_word(word: str):
    font = ImageFont.truetype('arialbd.ttf', 500)
    tw, th = font.getbbox(word)[2:]
    tw -= int(tw % 8)
    th -= int(th % 8)

    image = Image.new("RGB", (tw, th), 'white')

    draw = ImageDraw.Draw(image)
    draw.text((0, -25), word, font=font, fill='black')

    return image

def create_mask(image):
    image = image.filter(ImageFilter.GaussianBlur(radius=3))
    gray_image = image.convert('L')
    gray_array = np.array(gray_image)
    threshold = 200
    mask_array = np.where(gray_array > threshold, 0, 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_array)
    return mask_image


def safety_checker(images, clip_input):
    return images, False


def create_image(prompt, negative_prompt, image, mask, seed, filename):
    w, h = image.size
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=20,
        num_images_per_prompt=1,
        width=w,
        height=h,
        generator=torch.manual_seed(seed)
    ).images[0]
    output.save(filename + '.png')


letter = 'Sound'
partial = ''
if partial == '':
    partial = letter

prompt = f'(({partial})), digital art, wallpaper art'
neg_prompt = 'negative space, white space, text'
seed_base = 100

steps = 4
image = create_word(letter)
image.save(f'letters/{letter}0.png')
for i in range(1, steps + 1):
    mask = create_mask(image)
    create_image(prompt=prompt, negative_prompt=neg_prompt,
                 image=image, mask=mask, seed=(seed_base+i), filename=f'letters/{letter}{i}')
    print('a')
    image = Image.open(f'letters/{letter}{i}.png')
