"""
ValueError: Unknown layer: KerasLayer. Please ensure this object is passed to the `custom_objects` argument. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.

https://stackoverflow.com/questions/61814614/unknown-layer-keraslayer-when-i-try-to-load-model

best_model_reload = tf.keras.models.load_model(best_model_name, custom_objects={'KerasLayer':hub.KerasLayer})
"""


import os
# set log level should be before import tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import cv2
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

import json
import errno
import seaborn as sns

from tqdm import tqdm
from cycler import cycler
from datetime import datetime

# albumentations
from functools import partial
import albumentations as A

# RandAugment, AutoAugment, augment.py from TF github's augment.py
from augment import RandAugment,AutoAugment


from pytictoc import TicToc


# custom leaf tk #
from LeafTK import build_efn_model, WarmUpCosine, CosineDecayCLRWarmUpLSW, PrintLR, count_model_trainOrNot_layers
# custom leaf tk new tf hub #
from LeafTK import build_tf_hub_models, tf_hub_dict



best_model_name ="./TrainSaveDir-2022-02-12_EfficientNetV2L_hub_crop_plateau/ft_EfficientNetV2L_imagenet_crop_512x512_plateau_RA_bs32_best_val_accuracy.h5"
print("best_model_name = ", best_model_name)



best_model_reload = tf.keras.models.load_model(best_model_name, custom_objects={'KerasLayer':hub.KerasLayer})
