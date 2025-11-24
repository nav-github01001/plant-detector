import numpy as np 
import os
import PIL 
import PIL.Image
import tensorflow as tf
import pathlib

data_dir = pathlib.Path("./pathogen").with_suffix("")



b_size = 16
img_height = 256
img_width = 256




train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=b_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=b_size)

print(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache("C:\\Users\\Owner\\Documents\\Project NDNP\\cache_test").shuffle(buffer_size=20).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache("C:\\Users\\Owner\\Documents\\Project NDNP\\cache_test").shuffle(buffer_size=20).prefetch(buffer_size=AUTOTUNE)


num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)

model.save('my_model.keras')