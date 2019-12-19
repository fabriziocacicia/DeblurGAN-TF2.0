from network.discriminator import Discriminator
from network.generator import Generator


class DeblurGAN:
    def __init__(self):
        self.generator = Generator()
        self. discriminator = Discriminator()
