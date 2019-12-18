from network.discriminator import Discriminator
from network.generator import Generator


class DeblurGAN:
    def __init__(self):
        generator = Generator()
        self.generator = generator.model

        discriminator = Discriminator()
        self.discriminator = discriminator.model
