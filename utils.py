from PIL import Image
import numpy as np
import os


def reshape_image(image):
    image = image.resize((256, 256))

    return np.array(image)


def list_images_files(directory_path):
    return [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if '.png' in filename]


def load_image(path):
    return Image.open(path)


def load_images(path):
    files_paths = list_images_files(path)
    images = []
    for path in files_paths:
        image = load_image(path)
        reshaped = reshape_image(image)
        images.append(reshaped)

    return images
