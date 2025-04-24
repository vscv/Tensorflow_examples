# export TFHUB_CACHE_DIR=/tmp/tfhub_modules to cache the hub model
# Hub ViT are working with tf 260, 270 not the 250!!!!

from tensorflow import keras


import tensorflow as tf

## TF2 version (not work)
#import tensorflow.compat.v2 as tf

import tensorflow_hub as hub

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import matplotlib.pyplot as plt
import numpy as np



def get_model(
    handle="https://tfhub.dev/sayakpaul/vit_b8_fe/1",
    num_classes=5,
):
    hub_layer = hub.KerasLayer(handle, trainable=True)

    model = keras.Sequential(
        [
            keras.layers.InputLayer((224, 224, 3)),
            hub_layer,
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model
    
print(tf.__version__)
print(tf.__version__)
print(hub.__version__)
print(hub.__version__)

#get_model().summary()



#classifier = hub.KerasLayer('https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2')
#
#
#model = keras.Sequential(
#    [classifier,])
#model.summary()




# mbnetv2 ok to load
"""
Model: "sequential" #if donot give a name to it.
Model: "mobilenet_v2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
keras_layer (KerasLayer)     (None, 1280)              706224
_________________________________________________________________
dense (Dense)                (None, 5)                 6405
=================================================================
Total params: 712,629
Trainable params: 6,405
Non-trainable params: 706,224
_________________________________________________________________
"""
#num_classes=5
#m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/5", trainable=False), tf.keras.layers.Dense(num_classes, activation='softmax')
#], name="mobilenet_v2")
#m.build([None, 96, 96, 3])# Batch input shape.
#m.summary()





#tf.saved_model.load()
"""  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/saved_model/load.py", line 554, in _recreate
    raise ValueError("Unknown SavedObject type: %r" % kind)
ValueError: Unknown SavedObject type: None

跟直接用hub_layer = hub.KerasLayer(handle, trainable=True)報錯內容相同
可能是要tf2.7
"""

m=tf.saved_model.load("vit_b8_fe_1/")



