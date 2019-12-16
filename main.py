from network.discriminator import Discriminator
from network.generator import Generator
from tensorflow import keras

if __name__ == '__main__':
    generator = Generator.create_model()
    discriminator = Discriminator.create_model()

    keras.utils.plot_model(model=generator, to_file="generator.png", show_shapes=True)
    keras.utils.plot_model(model=discriminator, to_file="discriminator.png", show_shapes=True)
