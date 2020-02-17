from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_probability as tfp

import datetime
import matplotlib.pyplot as plt


# np.ndarray, (60000, 28, 28), (10000, 28, 28), mim=0, max=255
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def noisify(images, destruction_proportion=0.5, seed=123):
  noise = tfp.distributions.Bernoulli(probs=destruction_proportion).sample(sample_shape=images.shape, seed=seed).numpy()
  return images * noise

train_images_noisy = noisify(train_images)
test_images_noisy = noisify(test_images)

def munge(images):
  images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
  images /= 255.
  images[images >= .5] = 1.
  images[images < .5] = 0.
  return images

train_images = munge(train_images)
train_images_noisy = munge(train_images_noisy)
test_images = munge(test_images)
test_images_noisy = munge(test_images_noisy)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2, padding='same'),
  tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2, padding='same'),
  tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
  tf.keras.layers.UpSampling2D((2, 2)),
  tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
  tf.keras.layers.UpSampling2D((2, 2)),
  tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same'),
])

model.compile(
  optimizer=tf.keras.optimizers.Adam(1e-4),
  loss=tf.keras.losses.BinaryCrossentropy()
)

log_dir = "logs/denoising_ae/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

model.fit(train_images_noisy, train_images, epochs=10, batch_size=100, shuffle=True,
          validation_data=(test_images_noisy, test_images),
          callbacks=[tensorboard_callback])

test_images_reconstruction = model.predict(test_images_noisy)

n = 20
plt.figure(figsize=(40, 6))
for i in range(1, n+1):
  # display original
  ax = plt.subplot(3, n, i)
  plt.imshow(test_images[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display noisy
  ax = plt.subplot(3, n, i+n)
  plt.imshow(test_images_noisy[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(3, n, i+2*n)
  plt.imshow(test_images_reconstruction[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.savefig('three_way_comparison.png')
plt.show()
