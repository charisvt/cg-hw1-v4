import numpy as np
import cv2
from functions import render_img

# Load input data
data = np.load("hw1.npy", allow_pickle=True).item()
vertices = data["v_pos2d"]
uvs = data["v_uvs"]
vcolors = data["v_clr"]
faces = data["t_pos_idx"]
depth = data["depth"]

# Render image with flat shading
img = render_img(faces, vertices, vcolors, None, depth, shading="f", textImg=None)

# Save the rendered image
cv2.imwrite("flat_shading.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))