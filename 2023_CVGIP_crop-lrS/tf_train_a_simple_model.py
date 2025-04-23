# [TWCC]

# This toys model Resource exhausted: OOM when allocating tensor with shape[32,16,180,180]
# 
# james@nchc
# 2021-04-17

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)



for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# 自動調節tf.data管道
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)




#normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
#
#
#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
## Notice the pixels values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))


num_classes = 5

#model = Sequential([
#  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#  layers.Conv2D(16, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(32, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(64, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Flatten(),
#  layers.Dense(128, activation='relu'),
#  layers.Dense(num_classes)
#])



model_name = 'EfficientNetB0'

# dropout rate
dropout_rate = 0.4
# Fine-tune from this layer onwards
fine_tune_at = 15

base_model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                   weights='imagenet')
# Freeze the pretrained weights
base_model.trainable = False
# Rebuild top
gap2d = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
BNL = tf.keras.layers.BatchNormalization()(gap2d)
dropout = tf.keras.layers.Dropout(dropout_rate)(BNL)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(dropout)

model = tf.keras.Model(base_model.input, outputs, name=model_name)

# unfreeze the top #fine_tune_at# layers while leaving BatchNorm layers frozen
for layer in model.layers[-fine_tune_at:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
      layer.trainable = True

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# plot acc/val
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("training_acc.png")


#save model
#t = time.time()
#current_model_name = f'a_simple_model_{int(t)}.h'
model.save('a_simple_model.h5')

#Loading the model back:
#model_back = tf.keras.models.load_model(current_model_name)
reloaded = tf.keras.models.load_model('a_simple_model.h5')



#pred
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = reloaded.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
