from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt


# np.ndarray, (60000, 28, 28), (10000, 28, 28), mim=0, max=255
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

# @staticmethod
# from_tensor_slices(tensors)
# Creates a Dataset whose elements are slices of the given tensors.
# The given tensors are sliced along their first dimension. This operation preserves the structure of the input tensors,
#   removing the first dimension of each tensor and using it as the dataset dimension. All input tensors must have the
#   same size in their first dimensions.

# batch: Combines consecutive elements of this dataset into batches, still returns a dataset

# from np.ndarray to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    # Encoder
    self.inference_net = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        # Input: 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".
        # Output: 4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        #         rows and cols values might have changed due to padding.
        # padding='valid'(default) - valid padding means no padding.
        # padding='same' - tries to pad evenly left and right, but if the amount of columns to be added is odd, it
        # will add the extra column to the right.
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
        tf.keras.layers.Flatten(),
        # No activation
        # Encoding to a diagonal lognormal distribution with mean and logvar.
        tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )
    # Decoder
    self.generative_net = tf.keras.Sequential(
      [
        # (latent_dim,) is a tuple
        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
        tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
        tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
        # Express the convolution operation by means of a convolution matrix.
        tf.keras.layers.Conv2DTranspose( filters=64, kernel_size=3, strides=(2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="same", activation='relu'),
        # No activation
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="same"),
      ]
    )

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    # return z
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x):
  # log likelihood(x) - D[encoder(x) || posterior(z|x)] = E_{z~encoder(x)}[log(p(x|z))] - D[encoder(x) || prior(z)]
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)  # z ~ encoder(x)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  # Let x_prob = model.decode(z, apply_sigmoid=True). Then log(p(x|z)) = ... somehow is related to logpx_z.
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  # prior(z) ~ N(0, I). logpz = log_normal_pdf(z, 0., 0.).
  # D[encoder(x) || prior(z)] = E_{z~encoder(x)}[logqz_x - logpz].
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  # max log likelihood(x) ~=
  # max E_{z~encoder(x)}[log(p(x|z))] - D[encoder(x) || prior(z)] =
  # max {logpx_z - (logqz_x - logpz)} =
  # -min {logpx_z + logpz - logqz_x}
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

latent_dim = 50
model = CVAE(latent_dim)
# keeping the random vector constant for generation so it will be easier to see the improvement.
num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

def generate_and_save_images(model, epoch, input_vector_for_generation):
  predictions = model.decode(input_vector_for_generation, apply_sigmoid=True)
  fig = plt.figure(figsize=(4,4))
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

generate_and_save_images(model, 0, random_vector_for_generation)

epochs = 5
for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    compute_apply_gradients(model, train_x, optimizer)
  end_time = time.time()
  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch {}'.format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, random_vector_for_generation)
