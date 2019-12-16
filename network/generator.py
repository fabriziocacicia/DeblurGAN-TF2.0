from tensorflow import keras


class Generator:
    @staticmethod
    def input_layer():
        return keras.layers.Input(shape=[256, 256, 3], name="generator_input")

    @staticmethod
    def initial_block():
        model = keras.Sequential(name="initial_block")
        model.add(keras.layers.Conv2D(filters=64, kernel_size=7, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())

        return model

    @staticmethod
    def strided_conv_block(out_dim: int):
        model = keras.Sequential(name='strided_conv_{}_block'.format(out_dim))
        model.add(keras.layers.Conv2D(filters=out_dim, kernel_size=3, strides=2, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())

        return model

    @staticmethod
    def res_block(name):
        model = keras.Sequential(name='res_block_{}'.format(name))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())
        model.add(keras.layers.Dropout(rate=0.5))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(keras.layers.BatchNormalization())

        return model

    @staticmethod
    def strided_deconv_block(out_dim: int):
        model = keras.Sequential(name='strided_deconv_{}_block'.format(out_dim))
        model.add(keras.layers.Conv2DTranspose(filters=out_dim, kernel_size=3, strides=2, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())

        return model

    @staticmethod
    def final_block():
        model = keras.Sequential(name='final_block')
        model.add(keras.layers.Conv2D(filters=3, kernel_size=7, activation='tanh', padding="same"))

        return model

    @staticmethod
    def create_model():
        inputs = Generator.input_layer()

        x = inputs

        x = Generator.initial_block()(x)

        x = Generator.strided_conv_block(128)(x)
        x = Generator.strided_conv_block(256)(x)

        for index, _ in enumerate(range(9)):
            x = keras.layers.Add()([x, Generator.res_block(index)(x)])

        x = Generator.strided_deconv_block(128)(x)
        x = Generator.strided_deconv_block(64)(x)

        x = Generator.final_block()(x)
        
        outputs = keras.layers.Add(name="generator_output")([inputs, x])

        return keras.Model(inputs=inputs, outputs=outputs)



