import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import mean_squared_error

vgg19 = VGG19(include_top=False, weights='imagenet')
vgg19.trainable = False


def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_pred) - backend.mean(y_true)


def gradient_penalty(real, generated, discriminator):
    # alpha = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=1)
    alpha = backend.random_uniform(shape=(1, 1))

    sampled_point = (alpha * real) + ((1 - alpha) * generated)

    disc_output = discriminator(sampled_point)

    gradient = backend.gradients(disc_output, sampled_point)[0]

    # Compute the l2 norm
    import numpy as np
    square = backend.square(gradient)
    sum_over_rows = backend.sum(square, axis=np.arange(1, len(square.shape)))
    l2_norm = backend.sqrt(sum_over_rows)

    _lambda = 10
    return _lambda * backend.mean(backend.square(l2_norm - 1))


def adversarial_loss(blurred, generated_sharp, discriminator):
    blurred_critic = discriminator(blurred)
    generated_sharp_critic = discriminator(generated_sharp)

    return wasserstein_loss(blurred_critic, generated_sharp_critic) + gradient_penalty(blurred, generated_sharp,
                                                                                       discriminator)


class PerceptualLoss:
    def __init__(self):
        inputs = vgg19.input

        self.vgg_conv3_3 = Model(inputs=inputs, outputs=vgg19.layers[9].output)

    @tf.function
    def get_loss(self, real, generated):
        real_feature_map = self.vgg_conv3_3(real)
        gen_feature_map = self.vgg_conv3_3(generated)

        return mean_squared_error(real_feature_map, gen_feature_map)
