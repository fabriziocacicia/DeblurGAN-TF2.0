from network.generator import Generator
from tensorflow import keras

if __name__ == '__main__':
    generator = Generator.generate_model()

    keras.utils.plot_model(generator, "generator.png")
