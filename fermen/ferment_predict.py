# 2022-01-22 NCHC
# for_API_sample_code

import tensorflow as tf
import random
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = tf.keras.models.load_model('ferment.h5')

img_height=img_width=512
TYPE_name=['toast','croissant']
FERMEN_name=['NG','OK']

# img_path = "./toast-01_0911.jpg"
img_path = "./croissant-01_0024.jpg"

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch


predictions = model(img_array)
score_t, score_f = predictions[0], predictions[1]

print(f'[TYPE   ]:\t{TYPE_name[np.argmax(score_t)]}    \t{(100 * np.max(score_t)):.2f}% confidence')
print(f'[Ferment]:\t{FERMEN_name[np.argmax(score_f)]}\t\t{(100 * np.max(score_f)):.2f}% confidence')
