import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 32
img_height = 180
img_width = 180
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

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
    "\n This image most likely belongs to {} with a {:.2f} percent confidence. \n"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

