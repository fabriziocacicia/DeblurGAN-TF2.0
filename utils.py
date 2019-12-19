from PIL import Image
import numpy as np
import os
import tensorflow as tf


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


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    return tf.image.resize(img, [256, 256])


def process_dataset(blur_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(blur_path)
    img = decode_img(img)

    return img

