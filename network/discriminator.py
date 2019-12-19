from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


class Discriminator:
    def __init__(self):
        self.model = self.create_model()
        self.optimizer = Adam(lr=0.0001, beta_1=0.5)

    @staticmethod
    def strided_conv_block(output_dim: int, normalized: bool):
        model = Sequential(name="conv_{}_block".format(output_dim))
        model.add(Conv2D(filters=output_dim, kernel_size=4, strides=2, padding="same"))
        if normalized:
            model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        return model

    @staticmethod
    def second_last_conv_block():
        model = Sequential(name="second_last_conv_block")
        model.add(Conv2D(filters=512, kernel_size=4, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        return model

    @staticmethod
    def output_block():
        model = Sequential(name="discriminator_output")
        model.add(Conv2D(filters=1, kernel_size=4, padding="same"))
        model.add(Activation("sigmoid"))
        model.add(Flatten())
        model.add(Dense(1))

        return model

    @staticmethod
    def create_model():
        inputs = Input(shape=[256, 256, 3], name="input_image")
        x = inputs

        x = Discriminator.strided_conv_block(output_dim=64, normalized=False)(x)
        x = Discriminator.strided_conv_block(output_dim=128, normalized=True)(x)
        x = Discriminator.strided_conv_block(output_dim=256, normalized=True)(x)
        x = Discriminator.strided_conv_block(output_dim=512, normalized=True)(x)

        x = Discriminator.second_last_conv_block()(x)

        outputs = Discriminator.output_block()(x)

        return Model(inputs=inputs, outputs=outputs)
