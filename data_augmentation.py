import os
from PIL import Image, ImageEnhance
import random
import torchvision.transforms as T
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# --- CONFIG ---
project_dir = "./handwriting_dataset"
images_dir = os.path.join(project_dir, "images")
labels_file = os.path.join(project_dir, "labels.txt")
num_augmentations = 3  # number of augmented copies per image

# --- Load existing labels ---
with open(labels_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines()]

# Parse existing data
existing_images = []
existing_texts = []
for line in lines:
    img_name, text = line.split("|", 1)
    existing_images.append(img_name)
    existing_texts.append(text)

# --- Helper functions ---
def add_noise(img, amount=0.02):
    np_img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.randn(*np_img.shape) * amount
    np_img = np.clip(np_img + noise, 0, 1)
    return Image.fromarray((np_img * 255).astype(np.uint8))

def elastic_transform(img, alpha=34, sigma=4):
    # Convert to numpy
    img_np = np.array(img)
    random_state = np.random.RandomState(None)
    shape = img_np.shape

    dx = (random_state.rand(*shape) * 2 - 1) * alpha
    dy = (random_state.rand(*shape) * 2 - 1) * alpha

    dx = gaussian_filter(dx, sigma, mode="constant", cval=0)
    dy = gaussian_filter(dy, sigma, mode="constant", cval=0)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape)
    return Image.fromarray(distorted)

# --- Determine starting index ---
existing_numbers = [int(os.path.splitext(f)[0]) for f in os.listdir(images_dir) if f.endswith(".png") and f.split(".")[0].isdigit()]
counter = max(existing_numbers)+1 if existing_numbers else 1

# --- Augmentation pipeline ---
for img_name, text in zip(existing_images, existing_texts):
    img_path = os.path.join(images_dir, img_name)
    img = Image.open(img_path).convert("RGB")

    for i in range(num_augmentations):
        aug_img = img.copy()

        # --- Random transformations ---
        # Rotation ±10°
        angle = random.uniform(-10, 10)
        aug_img = aug_img.rotate(angle, expand=True, fillcolor=(255,255,255))

        # Scaling 90%-110%
        scale = random.uniform(0.9, 1.1)
        w, h = aug_img.size
        aug_img = aug_img.resize((int(w*scale), int(h*scale)))

        # Brightness / Contrast
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(aug_img)
        aug_img = enhancer.enhance(random.uniform(0.8, 1.2))

        # Optional: Gaussian noise
        if random.random() < 0.5:
            aug_img = add_noise(aug_img, amount=0.02)

        # Optional: Elastic distortion
        if random.random() < 0.3:
            try:
                aug_img = elastic_transform(aug_img)
            except:
                pass  # fallback if scipy fails

        # --- Save augmented image ---
        new_name = f"{counter}.png"
        aug_img.save(os.path.join(images_dir, new_name))

        # --- Append label ---
        with open(labels_file, "a", encoding="utf-8") as f:
            f.write(f"{new_name}|{text}\n")

        counter += 1

print(f"Completed.")
