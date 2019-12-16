from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import Sequential
from tensorflow.keras import Model


class Discriminator:
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

        return model

    @staticmethod
    def create_model():
        input_image = Input(shape=[256, 256, 3], name="input_image")
        target_image = Input(shape=[256, 256, 3], name="target_image")

        x = Concatenate(name="discriminator_merged_input")([input_image, target_image])

        x = Discriminator.strided_conv_block(output_dim=64, normalized=False)(x)
        x = Discriminator.strided_conv_block(output_dim=128, normalized=True)(x)
        x = Discriminator.strided_conv_block(output_dim=256, normalized=True)(x)
        x = Discriminator.strided_conv_block(output_dim=512, normalized=True)(x)

        x = Discriminator.second_last_conv_block()(x)

        output = Discriminator.output_block()(x)

        return Model([input_fake, input_real], output)
