import cv2
import os
import svgcon
import numpy as np
from PIL import Image, ImageFilter
import svg_stack as ss
from rembg import remove

# out = Image.open('b1.png')
# out = remove(out)
#
# canny_image = np.array(out)
# low_thresh = 100
# high_thresh = 200
# canny_image = cv2.Canny(canny_image, low_thresh, high_thresh)
# canny_image = Image.fromarray(canny_image)
# canny_image.show()
# letter = Image.open('letters/H3.png')
# letter = letter.filter(ImageFilter.GaussianBlur(radius=2))
# letter = letter.convert('L')
# letter = np.array(letter)
# letter = np.where(letter > 200, 255, 0).astype(np.uint8)
# letter = Image.fromarray(letter)
# letter.show()


def create_binary(filename):
    image = Image.open(filename)
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    gray_image = image.convert('L')
    gray_array = np.array(gray_image)
    binary_array = np.where(gray_array > 225, 255, 0).astype(np.uint8)
    binary_image = Image.fromarray(binary_array)
    return binary_image


# def create_binary(filename):
#     image = cv2.imread(filename)
#     kernel = np.ones((4,4), np.uint8)
#     erode_image = cv2.erode(image, kernel, iterations=1)
#     dil_image = cv2.dilate(erode_image, kernel, iterations=1)
#     dil_image = np.array(dil_image)
#     return Image.fromarray(dil_image)


drc_in = 'letters'
drc_b = 'letters_binary'
drc_svg = 'letters_svg'
for filename in os.scandir(drc_in):
    # image = Image.open(filename.path)
    binary_image = create_binary(filename.path)
    binary_image.save(f'{drc_b}/B_{filename.name}', 'PNG')

for binary in os.scandir(drc_b):
    svgcon.file_to_svg(filename=binary.path, directory=drc_svg)

doc = ss.Document()
layout = ss.VBoxLayout()
for svg in os.scandir(drc_svg):
    svg_file = svg.path
    layout.addSVG(svg_file, alignment=ss.AlignCenter)

doc.setLayout(layout)
doc.save('Sound_100.svg')
