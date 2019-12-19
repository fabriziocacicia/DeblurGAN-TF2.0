import tensorflow as tf

import utils
from losses import adversarial_loss, PerceptualLoss, wasserstein_loss
from network.deblurgan import DeblurGAN

import datetime
import os

LOGS_PATH = './logs/'
BLUR_PATH = './dataset/blur'
SHARP_PATH = './dataset/sharp'

BATCH_SIZE = 1


class Trainer:
    def __init__(self, gan: DeblurGAN):
        self.generator = gan.generator
        self.discriminator = gan.discriminator

        self.perceptual_loss = PerceptualLoss()

        self.summary_writer = tf.summary.create_file_writer(
            LOGS_PATH + "train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    @tf.function
    def training_step(self, blurred_image, sharp_image, epoch):
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            fake_sharp_image = self.generator.model(blurred_image)

            adv_loss = tf.constant(0, dtype=tf.float32)
            for _ in tf.range(5):
                adv_loss = adversarial_loss(blurred_image, fake_sharp_image, self.discriminator.model)

                discriminator_gradients = disc_tape.gradient(adv_loss,
                                                             self.discriminator.model.trainable_variables)

                self.discriminator.optimizer.apply_gradients(zip(discriminator_gradients, 
                                                                 self.discriminator.model.trainable_variables))

            content_loss = self.perceptual_loss.get_loss(sharp_image, fake_sharp_image)

            total_loss = adv_loss + content_loss
            print("adv_loss", adv_loss.shape)
            print("content_loss", content_loss)
            print("total_loss", total_loss)

            generator_gradients = gen_tape.gradient(content_loss, self.generator.model.trainable_variables)

            self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.model.trainable_variables))

            with self.summary_writer.as_default():
                tf.summary.scalar('adv_loss', adv_loss, step=epoch)
                #tf.summary.scalar('content_loss', content_loss, step=epoch)
                #tf.summary.scalar('total_loss', total_loss, step=epoch)


if __name__ == '__main__':
    deblurgan = DeblurGAN()

    trainer = Trainer(deblurgan)

    # Makes sure that the logs directory exists
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    generator = deblurgan.generator
    discriminator = deblurgan.generator

    blur_dataset = tf.data.Dataset.list_files(str('./dataset/blur/*.png'))
    blur_dataset = blur_dataset.map(utils.process_dataset)
    sharp_dataset = tf.data.Dataset.list_files(str('./dataset/sharp/*.png'))
    sharp_dataset = sharp_dataset.map(utils.process_dataset)

    for blurred_img, sharp_img in tf.data.Dataset.zip((blur_dataset, sharp_dataset)).take(1):
        tf.summary.trace_on(graph=True, profiler=True)

        trainer.training_step(blurred_image=tf.expand_dims(blurred_img, 0),
                              sharp_image=tf.expand_dims(sharp_img, 0), epoch=1)

        with trainer.summary_writer.as_default():
            tf.summary.trace_export(
                name="training_step",
                step=0,
                profiler_outdir=LOGS_PATH)

