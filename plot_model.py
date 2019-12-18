from network.deblurgan import DeblurGAN
from tensorflow import keras
import os

LOGS_PATH = './logs/'

if __name__ == '__main__':
    deblurgan = DeblurGAN()

    # Makes sure that the logs directory exists
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    keras.utils.plot_model(model=deblurgan.generator, to_file=LOGS_PATH+"generator.png", show_shapes=True)
    keras.utils.plot_model(model=deblurgan.discriminator, to_file=LOGS_PATH+"discriminator.png", show_shapes=True)