import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Path of train and test data
train_dir = "./seg_train/"
test_dir = "./seg_test/"

# Data configs
batch_size = 32
img_height = 150
img_width = 150

# Load train data
train_ds = tf.keras.utils.image_dataset_from_directory(
	train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Load test data
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Training the MASTER Model - using Transfer Learning
# Here we are using ImageNet pre-trained model weights
base_model = keras.applications.EfficientNetB1(
		weights='imagenet',  # Load weights pre-trained on ImageNet.
		input_shape=(img_height, img_width, 3),
		include_top=False)  # Do not include the ImageNet classifier at the top.
base_model.trainable = False
inputs = keras.Input(shape=(img_height, img_width, 3))


# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
# A Dense classifier with a single unit (binary classification)
x = keras.layers.Dense(30)(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(6)(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

epochs = 20
model.fit(train_ds, epochs=epochs)

# Generate results on test data
results = model.evaluate(test_ds)
print(f"Test accuracy with trained teacher model:{results[1]*100 :.2f} %")