#!/usr/bin/env python
# coding: utf-8
# %%

# ### Leaf benchmark for SOTA models and search better ensamble 
# 
#     2021-12-01
#     This nb is adopt by KPTK_build_models_BaseModelsBuilder.ipynb and tf.data_Loadandpreprocessdata_image_tf2.3to2.2_twcc_clean_GPUS-Temp-Leaf-EFN-2021-11-18-forReviewOnlyTwcc.ipynb
#     
#     (1) Rewrite tfds creation to read from CSV rather than from the name of directories. [12/03 done]
#     (2)
# 
# 12/08
# VGG16/16 seems not works? train fail at step-2/epoch-1, need use tf 21.08 docker.
# 
# 
# ## 12/27
# ####    rewrite ipynb to scrip version for production run. 
#         1. convert to py
#             restart kernel and clean all output.
#             jupyter nbconvert --to script 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.ipynb 
#             
#         2. parameterize the ds, lr, aug, models and witght selection.
#             LeafTK.py : repository for custom funtion. 
#             Train_Leaf.py : train loop for benchmark.
#             
#             
#         
#         
#         
#"""
#$python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 7 imagenet1k crop plateau RA 512 512 TrainSaveDir-1227-toPytest
#
#somehow clear dir fist for save_best_model without cache!
#
#
#python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 0 22 imagenet1k crop plateau RA 512 512 TrainSaveDir-1228_0-22
#
#"""


# 2022-01-13
# To reload save models and eval the test acc, ensemble models prediction.
#
# 1. keep all argv input to reproduce training parameters.
# 2. remove or comment the training parts.
# 3. add model reload, test ds predict.
#
#
# [Usage] ** Please just copy cmd from tf.notify message then replace the 2022_XXXX_-reloadModelTestAcc.Py.
# python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-reloadModelTestAcc.py 0 22 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-01-14_0-1_WCD_None_e20
# python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-reloadModelTestAcc.py 0 1 imagenet1k crop WCD None 512 512 TrainSaveDir-2022-01-14_0-1_WCD_None_e20


"""
when use:

import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()

https://issueexplorer.com/issue/tensorflow/tensorflow/52607

Exception ignored in: <function Pool.__del__ at 0x7f1ce204d430>
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/pool.py", line 268, in __del__
  File "/usr/lib/python3.8/multiprocessing/queues.py", line 362, in put
AttributeError: 'NoneType' object has no attribute 'dumps'

maybe it fixed at tf 260.

"""

"""2022-02-12
Test the 1 epoch training is really done the learning?

    $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 28 29 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2L_hub_crop_plateau
    [614.0859096050262] of epoch 1 to 0.86008
    [193.26175022125244] of epoch 2 val_loss: nan
    loss: nan
        
    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-reloadModelTestAcc.py 28 29 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2L_hub_crop_plateau

    valid:
        acc_count = 2760   (if score == label[i] then count one.)
        accuracy  =  0.8600810221252727 % is correct with val_acc in training.
    test:
        acc_count = 2767   (if score == label[i] then count one.)
        accuracy  =  0.86226238703646 %
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




t = TicToc() #create instance of class

t.tic() #Start timer


# Check pkg version #
#print(f'tf: {tf.__version__} \ncv2: {cv2.__version__} \nnp: {np.__version__} \npd: {pd.__version__} \nmatplotlib: {matplotlib.__version__}')
t.toc() #End timer




# Path to save #
#log_dir_name = "TrainSaveDir-1227-toPytest" # Put all results in same dir with different file name.
"""Ex: best_model_name: ./TrainSaveDir/ft_RRNET_imagenet1k_resize_120x120_CDCLR_AA_bs32_best_val_accuracy.h5"""
log_dir_name = sys.argv[9]

# Model pick up #
"""for Model_List = Model_List[m_start:m_end]"""
m_start=int(sys.argv[1])
m_end=int(sys.argv[2])



"""
$python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 7 imagenet1k crop plateau RA 512 512 TrainSaveDir-1227-toPytest

somehow clear dir fist for save_best_model without cache!


python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 0 22 imagenet1k crop plateau RA 512 512 TrainSaveDir-1228_0-22

"""

#
# hyper setting
#

#weight="imagenet1k" #random, maybe just 1k is enough
#crop= "crop" #"resize" #crop=center crop
#lr_name= 'plateau' #'plateau' #'WCD' # 'CDR'  #'fixed' #'lrdump' #"WCD" # WCD, WCDC, lrdump, platrure
#augment= 'RA' #None # None, 'AA', 'RA', 'NoisyStudent', 'all'

pretrain_weight=sys.argv[3]
if pretrain_weight=='imagenet1k':
    weight="imagenet"
if pretrain_weight=='imagenet21k':
    weight="imagenet21k"
if pretrain_weight=="None":
    weight=None


crop=sys.argv[4]
lr_name=sys.argv[5]
augment=sys.argv[6]

# hyper models #
EPOCHS=20 #20, vit50fortest.

"""#less dp rate, say 0.1, train_loss will lower than val_loss # f
or flood 0.2 is ok. for leaf 0.4 is better. for foot 0.8 is fine."""
top_dropout_rate = 0.4

""" #for efnetBx only This parameter serves as a toggle for extra
regularization in finetuning, but does not affect loaded weights."""
drop_connect_rate = 0.9

"""# classes of 5"""
outputnum = 5

"""save best val_acc model"""
monitor = 'val_accuracy' #'val_loss' 'val_accuracy' if use ed_loss it still the loss here.


# Image size #
BATCH_SIZE = 4 #32#4 #2 # 8# 32 #64 #64:512*8 OOM, B7+bs8:RecvAsync is cancelled
#img_height = 512 #600 #512 #120
#img_width = 512 #600 #512 #120
img_height = int(sys.argv[7])
img_width = int(sys.argv[8])

patience_1 = 3
patience_2 = 50#5


# automatic tuning the pipeline of tf.data #
AUTOTUNE = tf.data.experimental.AUTOTUNE
print('AUTOTUNE=', AUTOTUNE)

# visible/logical device (able to be used) #
print(tf.config.experimental.list_physical_devices('GPU'),'\n')
print(tf.config.experimental.list_logical_devices('GPU'),'\n')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num logical_gpus  :", len(tf.config.experimental.list_logical_devices('GPU')))

# tf MirroredStrategy setting #
strategy = tf.distribute.MirroredStrategy()
REPLICAS = strategy.num_replicas_in_sync
print('\nNumber of REPLICAS: {}\n'.format(REPLICAS))

# batch size for multi gpu #
#MULTI_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
MULTI_BATCH_SIZE = 4 * 8 # set 32 for eval only.
print('BATCH_SIZE: {}, MULTI_BATCH_SIZE: {}'.format(BATCH_SIZE, MULTI_BATCH_SIZE))


""" terminal check """
print(f'* * * Pre-train weight: {weight}')
print(f'\n * * * model_start:model_end: [{m_start} to {m_end}]\n')
fpn= f'_{log_dir_name}/th_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_'
print(f'* * * Field parameter name:\n {fpn} \n')
print(f'\n * * * EPOCHS {EPOCHS}\n top_dropout_rate {top_dropout_rate}\n drop_connect_rate {drop_connect_rate}\n outputnum {outputnum}\n monitor {monitor}\n * * *')


#
# hyper setting
#




# ### 2. Dataset (DS)
# #### <font color=orange>[DS] Create the training dataset W/ croped</font>
# #### <font color=#00FF00>[DS] Create the training dataset W/ croped</font>
# 
#     label_num_to_disease_map.json    {
#     "0": "Cassava Bacterial Blight (CBB)", 
#     "1": "Cassava Brown Streak Disease (CBSD)", 
#     "2": "Cassava Green Mottle (CGM)", 
#     "3": "Cassava Mosaic Disease (CMD)", 
#     "4": "Healthy"}

CLASSES = ['CBB',
           'CBSD', 
           'CGM', 
           'CMD', 
           'Healthy']
           
LABELS = {"0": "CBB", 
          "1": "CBSD", 
          "2": "CGM", 
          "3": "CMD", 
          "4": "Healthy"}

data_dir = '/home/uu/.keras/datasets/leaf/'
leaf_dir = '/home/uu/.keras/datasets/leaf/train_images/'

df_train = pd.read_csv(data_dir + '/train.csv')

# check labels #
for i in range(5):
    print(i, CLASSES[i])

## check labels #
#print([(i,l) for i,l in zip(LABELS.keys(), LABELS.values())])
#
## check labels #
#for i,l in zip(LABELS.keys(), LABELS.values()):
#    print(i,l)
#
#print(f'*** Total Image of training set: {len(df_train)}')
#print(f'*** Fist 5 csv data: \n {df_train[:5]}')
#
#
## Shuffle and reset index #
## fixed shuffle for compare later, random_state=42
#df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
#
#print(f'*** Now the ds was shuffled:: \n {df_train[:5]}')
#print(f'*** Check train image again: \n {df_train.count()}')
#
#
#freq = df_train.groupby(['label']).count()
#print(f"Number image of each Label: \n{freq}")
#
#
## get no key freq #
#freq = df_train['label'].value_counts()
#ax = freq.plot.bar(x='image_id', y='label', rot=0) #no key so x y no matter.
#print("No key freq: \n", freq)



## check image size #
#import imageio
#for tmp in ['827159844.jpg', '795383461.jpg', '851791464.jpg', '923880010.jpg']:
#    tmp_img = imageio.imread(leaf_dir + '/' + tmp)
#    print(type(tmp_img), tmp_img.shape, tmp_img.dtype)



## [DS] Create tf.dataset (DS) ##
# from dataframe
list_ds = tf.data.Dataset.from_tensor_slices((df_train['image_id'], df_train['label']))

# create a Python iterator
it_list_ds = iter(list_ds) # Make sure iter ds only once.

# using iter and consuming its elements using next: every print different image name.
print(f'* * * Check image_id and label.')
for i in range(4):
    image_id, label = next(it_list_ds)
    print(image_id.numpy(), label.numpy())


#
# map list to ds.
#
def process_path_label(image_id, label):
    file_path = leaf_dir + image_id
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)#can read the byte string paths b'image_0001.png'
    img = tf.io.decode_jpeg(img, channels=3)
    
    if crop == "resize":
        print("--resize")
        img = tf.cast(tf.image.resize(img, [img_height, img_width]), tf.uint8) # for resize the training image for faster checing! tf.image.resize return a float!
    if crop == "crop":
        print("--crop")
        # crop the toe from top-left corner [image, offset_height y1, offset_width x1, target_height, target_width]
        y1=(600-img_height)/2;    x1=(800-img_width)/2;    h=img_height;    w=img_width # not the pp location
        img = tf.image.crop_to_bounding_box(img, int(y1), int(x1), h, w)

    return img, label


# Leaf train ds
train_ds_map = list_ds.map(process_path_label, num_parallel_calls=AUTOTUNE)


## [DS] Split TVT ##
"""split TVT train/val/test in 7 1.5 1.5"""
val_size = int(tf.data.experimental.cardinality(train_ds_map).numpy() * 0.15)
# val_size = int(tf.data.experimental.cardinality(train_ds_map_toe).numpy() * 0.1)#no help

print("val size:", val_size)

train_ds_map_s = train_ds_map.skip(val_size+val_size)
temp_s = train_ds_map.take(val_size+val_size)

valid_ds_map_s = temp_s.take(val_size)
test_ds_map_s = temp_s.skip(val_size)

print("total size:", len(train_ds_map))
print("\ntrain", tf.data.experimental.cardinality(train_ds_map_s).numpy())
print("valid", tf.data.experimental.cardinality(valid_ds_map_s).numpy())
#print("test", tf.data.experimental.cardinality(test_ds_map_s).numpy())
test_samples=tf.data.experimental.cardinality(test_ds_map_s).numpy()
print("test", test_samples)



## [DS] Augmentation and performance cache pipeline ##

## AA, auto aug test
def AA(image, label):
    Auto_Aug = AutoAugment()
    return Auto_Aug.distort(image), label

## RA, is upgraded version of AA.
def RA(image, label):
    Rand_Aug = RandAugment()
    return Rand_Aug.distort(image), label


## DS performance cache ##

def configure_for_performance_cache(ds, cache=True, augment=None):  
    """#TODO: need to check the parse logic of ds.cache.
    if cache:
        print("Check cache-f1 to file:", cache)
        if isinstance(cache, str):
            ds = ds.cache(cache)
            print("Check cache-f2 to file:", cache)
    else:
        ds = ds.cache()
        print("Check cache in memory:", cache)
    """    
    if cache:
        ds = ds.cache()
        print("Check cache in memory:Y", cache)
    else:
        print("Check cache in memory:N", cache)

        
#     if augment== "Albu":
#         ds = ds.map(data_augment, num_parallel_calls=AUTOTUNE)        
    if augment=="AA":
        ds = ds.map(AA, num_parallel_calls=AUTOTUNE)
        print("Check augment :Y", augment)
    if augment=="RA":
        ds = ds.map(RA, num_parallel_calls=AUTOTUNE)
        print("Check augment :Y", augment)
    if augment==None:
        print("Check augment :N", augment)
    
    #ds = ds.repeat()#TODO:2020-12-14: test
    ds = ds.shuffle(buffer_size=MULTI_BATCH_SIZE, reshuffle_each_iteration=True) #buffer_size=MULTI_BATCH_SIZE*2 10sec. # (buffer_size=MULTI_BATCH_SIZE*5) ~10sec,buffer_size=1000 take few sec. or buffer_size=image_count <- take too long # each take ds take 30~45 sec, TODO!!
    """Note: While large buffer_sizes shuffle more thoroughly, they can take a lot of memory, and 
        significant time to fill. Consider using Dataset.interleave across files if this becomes a problem."""
    
    ds = ds.batch(MULTI_BATCH_SIZE)#MULTI_BATCH_SIZE for multi-GPUs
    ds = ds.prefetch(buffer_size=AUTOTUNE) #buffer_size=AUTOTUNE seem no speed improve
    
    print("Check ds cache[{}] and augment[{}]".format(cache, augment))
    
    return ds


# give augment type to train_ds_pre #
train_ds_pre = configure_for_performance_cache(train_ds_map_s, cache=True, augment=augment)
valid_ds_pre = configure_for_performance_cache(valid_ds_map_s)
test_ds_pre = configure_for_performance_cache(test_ds_map_s)



#
# 
## [Models] ##
#
#

#
#
## [Models] ##
#
#

## Pick a model #
#Model_List = [
## GoogleNet 0,1
#"InceptionV3", "Xception",
#
## ResNet 2,3,4
#"ResNet50", "ResNet101", "ResNet152",
#
## Mobilenet 56,78
#"MobileNet", "MobileNetV2",
#"MobileNetV3Small",
#"MobileNetV3Large",
#
## Densenet 9,10,11
#"DenseNet121","DenseNet169","DenseNet201",
#
## NASNet 12,13
#"NASNetMobile","NASNetLarge", # hard code of size!224 331! #12 NaN
#
## EffNet 14 15 16 17 18 19 20
#"EfficientNetB0", #14 NaN
#"EfficientNetB1", #15 NaN
#"EfficientNetB2", #16 NaN
#"EfficientNetB3",
#"EfficientNetB4",
#"EfficientNetB5", #19
#"EfficientNetB6", #20 NaN
#"EfficientNetB7", #21
#
## VGG 22  23
#'VGG16', #  #at leass twcc21.08
#'VGG19', #
#
##ViT [24,25] [26,27], vit-keras only has b16/32,l16/32.
##'ViT-B/8', # Vision Transformer with vit-keras, or tf.hub with tf21.11.
#'ViT-B16',
#'ViT-B32',
#'ViT-L16',
#'ViT-L32',
#
## Mixer
#'Mixer-B/16', #MLP-Mixer (Mixer for short)
#'Mixer-L/16',
#
## EA
#'EANet',
#
## C-Mixer
#'ConvMixer', #
#
#'BiT', # BigTransfer
#]

Model_List=list(tf_hub_dict)
#Model_List = Model_List[5:6] #[6:7] then MobileNetV2 train OK, but [5:6] MobileNetV2 will nan
Model_List = Model_List[m_start:m_end]
print(f"Model list: \n {Model_List}")







#
#
#
#""" Blocked for eval only """
#""" Blocked for eval only """
#""" Blocked for eval only """
#""" Blocked for eval only """
#""" Blocked for eval only """
#
#
##
##
### [Models] ##
##
##
#
##
##
### [Models] ##
##
##
#
#
#
#
#
#
## [Models] Train misc. #
#
#def mk_log_dir(log_dir_name):
#    try:
#        os.makedirs(log_dir_name)
#    except OSError as e:
#        print("This log dir exist.")
#        if e.errno != errno.EEXIST:
#            raise ValueError("we got problem.")
#
#def get_best_model_name(th):
#    return './' + log_dir_name + '/' + th + '_' + model_name + '_bs' + str(BATCH_SIZE) + '_w' + str(img_width) + '_best_' + monitor + '.h5'
#
#
#
#def get_best_model_name_bench():
#    return f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.h5'
#
#
#
## th = "stage1"
#th = "ft" # fine tune only without transfer learning.
#
#
##  Tensorboard Log dir name #
## use once at the time
## log_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S")
## log_dir_name = "TrainSaveDir"
#
#mk_log_dir(log_dir_name)
#logdir = log_dir_name + "/logs/toe/"
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#
#
#
## [Models] Learning Rate Scheduler #
#
#
## [WCD] StepWise warmup cosine decay learning rate [optimazer]
## for some models need very small lr.
#INIT_LR = 0.00001 #1e-5
#WAMRUP_LR = 0.0000001 #1e-7
#WARMUP_STEPS = 5
#
#WCD = WarmUpCosine(
#    learning_rate_base=INIT_LR,
#    total_steps=EPOCHS,
#    warmup_learning_rate=WAMRUP_LR,
#    warmup_steps=WARMUP_STEPS,
#)
#
#
## [CDR ]
#"""warm Cosine Decay Restart"""
#CDR = CosineDecayCLRWarmUpLSW
#
## [Plateau]
#"""Callback lr """
#
## Select LR_SCHEDULER [move inside the build_efn()] #
#if lr_name=='WCD':
#    # tk.optimizer
#    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(WCD)
#    # tk.callback
##     scheduled_lr = tf.keras.callbacks.LearningRateScheduler(fixed_WCD)
#
#if lr_name=='CDR':
#    # tk.optimizer
#    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(CDR)
#
#if lr_name=='plateau':
#    #learning_rate=0.0001 in the tf.keras.optimizers.Adam()
#    scheduled_lr = tf.keras.callbacks.ReduceLROnPlateau(
#        monitor=monitor, factor=0.5, patience=1, verbose=1,
#        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0)
#
#if lr_name=='fixed':
#    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(fixed_scheduler)
#    print(fixed_scheduler(1))
#
#if lr_name=='lrdump':
#    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(lrdump)
#
#if lr_name=='WCDCR': # warmup cosin decay with cycle
#    print("No implemented yet.")
#    pass
#
#print(f'Set scheduler LR : {lr_name}')
#
#
#
#callback_lr_time = PrintLR() #return a object of callback, not use the Classs PrintLR.
#
#
#
#
##tt = 0
##nt = 0
#
#
#
#
## #### <font color="yellow"> [Models] Train top layers (transfer learning)</font>
## # fit the model on all data
## history_toe = model_toe.fit(train_ds_pre,
##                       verbose=1,
##                       epochs=5, #ep_num_transf,
##                       validation_data=valid_ds_pre,
##                       callbacks=callbacks)#, validation_split=0.1)
#
#
#
## 5. Fine tune #
##
## #### <font color="yellow"> [Models] Train bench models (Fine tune)</font>
##
#
## def unfreeze_model(model, base_model):
## #     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
## #     for layer in model.layers[-20:]:
## #         if not isinstance(layer, layers.BatchNormalization):
## #             layer.trainable = True
#
## #     model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),#RMSprop , Adam, SGD Adadelta(learning_rate=0.001), if set lr_callback the learning_rate=0.001 will not effeced.
#
#
#
## [Models] Train bench models (Fine tune) #
##
## copy from K10L195
#
## use dict of dict to store hist
#history_toe_finetune = {}
#
#
#
#
##
## [Models] Train bench models (Fine tune) #
##
#
##
## [Models] Train bench models (Fine tune) #
##
#
#for model_name in Model_List:
#    print("\n \n K = ", model_name, "\n")
#    best_model_name = get_best_model_name_bench()
#    best_model_save = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_name,
#                                 save_best_only = True,
#                                 save_weights_only = False,
#                                 monitor = monitor,
#                                 mode = 'auto', verbose = 1)
#
#    callbacks_s2 = [
#                #     tensorboard_callback,
#                    best_model_save,
#                    tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience_2), #patience=step_size or ep_num
#                #     lr_reduceonplateau,
##                     tf.keras.callbacks.LearningRateScheduler(lrdump),#lrdump, decay or lrfn or lrfn2. clr
#                    scheduled_lr,
#                    callback_lr_time,
#                #     tensorboard_callback,
#                ]
#    print('best_model_name:', best_model_name)
#
#
#    with strategy.scope():
#        model_toe, base_model = build_efn_model(weight, model_name, outputnum, img_height, img_width, top_dropout_rate, drop_connect_rate)
#
##     # Train K-Model with transfer learning # IF HAVE!
##     hist = model_toe.fit(train_ds_pre_toe_s,
##                           verbose=1,
##                           epochs=ep_num_transf,
##                           validation_data=valid_ds_pre_toe_s,
##                           callbacks=callbacks_toe_tl)#, validation_split=0.1)
##     history_toe.append(hist)
#
#
#    # Train K-Model with fine tune #
#
#    # bench models, FT
##     unfreeze_model(model_toe,base_model) # skip the TL so unfreeze when build_EFN()
#    count_model_trainOrNot_layers(model_toe)
#    print(model_toe.summary()) ## print will added a 'None' of end of summary().
#    # fit the model on all data
#    hist = model_toe.fit(train_ds_pre,
#                          verbose=2, #'auto' or 0 1 2 , 0 = silent, 1 = progress bar, 2 = one line per epoch.
#                          epochs=EPOCHS,
#                          validation_data=valid_ds_pre,
#                          callbacks=callbacks_s2)#, validation_split=0.1)
#    # add the epoch timeing
#    hist.history['epoch_time_secs'] = callback_lr_time.times
#    history_toe_finetune[model_name] = hist.history #hist # what if use hist.history
#
##
## [Models] Train bench models (Fine tune) #
##
#
##
## [Models] Train bench models (Fine tune) #
##
#
#
#
#
#
#
#
#
##ED sum
#def get_valloss(his_v_l):
#    return np.min(his_v_l), np.argmin(his_v_l)
#
#t_vl = []
## h_vl = []
#for k in Model_List:
#    print(f'K:{k}')
#    t_v, _ = get_valloss(history_toe_finetune[k]['val_loss'])
##     h_v, _ = get_valloss(history_heel_finetune[k].history['val_loss'])
#
#    t_vl.append(t_v)
##     h_vl.append(h_v)
#
## t_vl = np.mean(t_vl, axis=0)
## h_vl = np.mean(h_vl, axis=0)
## print(f'{round(t_vl,5)} + {round(h_vl,5)} = {round(t_vl + h_vl,5)}')
#
#print(f'Minimal losses: {t_vl}')
#
#
#
#
#
#
## [Models] Result Ploting #
#
## Save hist to np #
#bench_log_name = f'./{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.npy'
#np.save(bench_log_name, history_toe_finetune)
#
## reload hist from npy
#history_np_load = np.load(bench_log_name, allow_pickle='TRUE').item()
#hisnp = history_np_load.copy()
#
#
## Reload np to hist #
## draft reload his from saved np
#bench_log_name = f'./{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.npy'
#
## Note that, if change bs4 to bs32, when load hist by one gpu container #
## bench_log_name = './TrainSaveDir-1223-CDR/ft_bench_imagenet1k_crop_512x512_CDR_RA_bs32_best_val_accuracy.npy'
#
#
## reload hist from npy #
#history_np_load = np.load(bench_log_name, allow_pickle='TRUE').item()
#hisnp = history_np_load.copy()
#
#handles = [handle for handle in Model_List]
#print(f'*** reload model handles list: \n {handles}')
#
#
## plotting train log #
#"""Rewrite to a function
#plot_save_bench_val_loss(Model_List, hisnp, dpi=300, png_name=png)
#"""
#
#plt.figure(figsize=(25, 10))
#
#for k in Model_List:
#
#    x = len(hisnp[k]['loss']) + 1
#    x = range(1,x,1)
#    y = hisnp[k]['loss']
#    #y = [f'{z:.4f}' for z in y]
#    plt.plot(x, y, label=f'{k}_loss')
#
#
#    for a,b in zip(x, y):
#        #plt.text(a, b, str(b))
#        plt.scatter(a,b, color='black', alpha=0.2)
#        plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#    y = hisnp[k]['val_loss']
#    plt.plot(x, y, label=f'{k}_val_loss')
#
#
#plt.title('K-model ed loss toe-TL')
#plt.ylabel('ed loss'), plt.ylim(0, 2)# for too large loss
#plt.xlabel('epoch')
#plt.legend(title='Nets')
#
## save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
#dpi=600 #300 for quickly check, 600up to 800 for journal paper.
#pgn=f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_loss.png'
#plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
#print(f'Save to {pgn} \n')
#
#
#
## naive check #
#for k in Model_List:
#    print(f"val_acc of {k}: {hisnp[k]['val_accuracy']}")
#
#
## plotting train log #
#"""Rewrite to a function
#plot_save_bench_val_accuracy(Model_List, hisnp, dpi=300, png_name=png)
#"""
#plt.figure(figsize=(25, 10))
#
#for k in Model_List:
#
#    x = len(hisnp[k]['val_accuracy']) + 1
#    x = range(1,x,1)
#    y = hisnp[k]['val_accuracy']
#    #y = [f'{z:.4f}' for z in y]
#    plt.plot(x, y, label=f'{k}_val_accuracy')
#
#
#    for a,b in zip(x, y):
#        #plt.text(a, b, str(b))
#        plt.scatter(a,b, color='black', alpha=0.2)
#        plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#    #plt.plot(hisnp[k]['val_accuracy'])
#
#
#plt.title('K-model-TL val_accuracy', fontsize='xx-large')
#plt.ylabel('val_accuracy'), plt.ylim(0.1, 0.9)# for too large loss
#plt.xlabel('epoch')
#plt.legend(title='Nets:', title_fontsize='x-large', fontsize='large')
#"""title_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
#"""
#
## save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
#dpi=300
#pgn= f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc.png'
#print(f'Save to {pgn} \n')
#plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
#
#
## naive check #
#for k in Model_List:
#    print(f" {k}: {hisnp[k]['val_accuracy']}")
#
#
#
#
## plotting train log with lr #
#""" Rewrite to def fun()"""
#
#"""# ax1 for val_accuracy #
## nice to have this colorful tip."""
#fig, ax1 = plt.subplots(figsize=(25, 10))
#color = 'tab:red'
#ax1.set_title('K-model-TL val_accuracy with lr', fontsize='xx-large')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('val_accuracy', color=color)
#
#for k in Model_List:
#
#    x = len(hisnp[k]['val_accuracy']) + 1
#    x = range(1,x,1)
#    y = hisnp[k]['val_accuracy']
#    #y = [f'{z:.4f}' for z in y]
#    plt.plot(x, y, label=f'{k}_val_accuracy')
#
#
#    for a,b in zip(x, y):
#        #plt.text(a, b, str(b))
#        plt.scatter(a,b, color='black', alpha=0.2)
#        plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#ax1.tick_params(axis='y', labelcolor=color)
#ax1.legend(title='Nets:', title_fontsize='x-large', fontsize='large')
#"""title_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}"""
#
#
## ax2 for learning rate #
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#color = 'tab:green'
#ax2.set_ylabel('learning rate', color=color)
## ax2.plot(hisnp[k]['lr'], color=color)
#x = len(hisnp[k]['lr']) + 1
#x = range(1,x,1)
#y = hisnp[k]['lr']
##y = [f'{z:.4f}' for z in y]
#plt.plot(x, y, color='green', label=f'learning rate')
#
#for a,b in zip(x, y):
#    #plt.text(a, b, str(b))
#    plt.scatter(a,b, color='green', alpha=0.2)
#    plt.annotate(f'{b:.7f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#ax2.tick_params(axis='y', labelcolor=color)
#ax2.legend(fontsize='large', loc='upper center')
#
## save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
#dpi=300
#pgn= f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc_lr.png'
#print(f'Save to {pgn} \n')
#plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
#
#
#
#
### with cycle color/marker ##
### with cycle color/marker ##
#
#matplotlib.rcParams['lines.linewidth'] = 1.5
#matplotlib.rcParams["markers.fillstyle"] = 'left' # 'full', 'left', 'right', 'bottom', 'top', 'none'
#
#
#auto_custom_cycler_01 = (cycler(color=[plt.get_cmap('jet')(i/13) for i in range(24)]) + # 24 colors
#                  cycler(linestyle=['-', '--', ':', '-.'] * 6) + # [4]*6 = 24
#                    cycler(marker=['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D'] * 2)) # [12]*2 = 24
#
#
## ploting train log with lr
## plt.figure(figsize=(25, 10))
#
## for different scales (different Y-axes)
## fig, ax1 = plt.subplots()
#fig, ax1 = plt.subplots(figsize=(25, 15))
#
#
#ax1.set_prop_cycle(auto_custom_cycler_01) # set to use custom_cycler
#
#
## ax1 for val_accuracy #
## nice to have this colorful tip.
#color = 'tab:red'
#ax1.set_title('K-model-TL val_accuracy with lr', fontsize='xx-large')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('val_accuracy', color=color)
#
#for k in Model_List:
#
#    x = len(hisnp[k]['val_accuracy']) + 1
#    x = range(1,x,1)
#    y = hisnp[k]['val_accuracy']
#    #y = [f'{z:.4f}' for z in y]
#    plt.plot(x, y, label=f'{k}_val_accuracy')
#
##     # v_a value #
##     for a,b in zip(x, y):
##         #plt.text(a, b, str(b))
##         plt.scatter(a,b, color='black', alpha=0.2)
##         plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#
#ax1.tick_params(axis='y', labelcolor=color)
## ax1.legend(title='Nets:', title_fontsize='x-large', fontsize='large')
#"""title_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}"""
#ax1.legend(title='Nets:', title_fontsize='x-large', fontsize='large', bbox_to_anchor=(1.23, 1.0))
#
#
#
## ax2 for learning rate #
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#color = 'tab:green'
#ax2.set_ylabel('learning rate', color=color)
## ax2.plot(hisnp[k]['lr'], color=color)
#x = len(hisnp[k]['lr']) + 1
#x = range(1,x,1)
#y = hisnp[k]['lr']
##y = [f'{z:.4f}' for z in y]
#plt.plot(x, y, color='green', label=f'learning rate')
#
## lr value #
## for a,b in zip(x, y):
##     #plt.text(a, b, str(b))
##     plt.scatter(a,b, color='green', alpha=0.2)
##     plt.annotate(f'{b:.7f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#ax2.tick_params(axis='y', labelcolor=color)
#ax2.legend(fontsize='large', loc='lower right', bbox_to_anchor=(1.15, .4))
#
#
## save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
#dpi=300
#pgn= f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc_lr_cyc.png'
#print(f'Save to {pgn} \n')
#plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
#
### with cycle color/marker ##
### with cycle color/marker ##
#
#
#
## best val_acc and the epoch #
#
#print("[max val_acc]")
#tmp_acc_his = []
#for k in Model_List:
#    v_a = hisnp[k]['val_accuracy']
#    b_e = np.argmax(v_a)
#    print(f'  {k}-------------------------:\n\t       {v_a[b_e]} Epoch@P{b_e}')
#
#    tmp_acc_his.append(v_a[b_e])
#
#print('\n[Best model and val_acc]')
#b_m_e = np.argmax(tmp_acc_his)
#print(f'  {Model_List[b_m_e]} : {tmp_acc_his[b_m_e]}')
#
#
#
#""" Blocked for eval only """
#""" Blocked for eval only """
#""" Blocked for eval only """
#""" Blocked for eval only """
#""" Blocked for eval only """

















# ## Evaluate the valid and test accuracy
# 
#     for DenseNet201_imagenet1k_crop_512x512_CDR_RA_bs32
#      the eval/test accuracy is match the training result.
# 
#     803/803 [==============================] - 45s 56ms/step - loss: 0.3456 - accuracy: 0.8962
#     Test accuracy : 0.8962293863296509
#     803/803 [==============================] - 31s 39ms/step - loss: 0.3273 - accuracy: 0.8965
#     Test accuracy : 0.8965409994125366

# %%


# evl_m_path = log_dir_name + "/" + "ft_DenseNet201_imagenet1k_crop_512x512_CDR_RA_bs32_best_val_accuracy.h5"
# evl_model = tf.keras.models.load_model(evl_m_path)


# %%


# loss, accuracy = evl_model.evaluate(valid_ds_pre)
# print('Test accuracy :', accuracy)

# # print("count roughly ds size: ", tf.data.experimental.cardinality(valid_ds_pre).numpy() * BATCH_SIZE)


# %%


# loss, accuracy = evl_model.evaluate(test_ds_pre)
# print('Test accuracy :', accuracy)

# # print("count roughly ds size: ", tf.data.experimental.cardinality(val_ds_pre).numpy() * BATCH_SIZE)



# ### confusion matrix move to finianl script
# 
#     * model-eval_acc_confusion.ipynb
# 
#









#
# [Models] Train bench models (Fine tune) #
#


def mk_log_dir(log_dir_name):
    try:
        os.makedirs(log_dir_name)
    except OSError as e:
        print("This log dir exist.")
        if e.errno != errno.EEXIST:
            raise ValueError("we got problem.")

def get_best_model_name(th):
    return './' + log_dir_name + '/' + th + '_' + model_name + '_bs' + str(BATCH_SIZE) + '_w' + str(img_width) + '_best_' + monitor + '.h5'



def get_best_model_name_bench():
    return f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.h5'

th = "ft" # fine tune only without transfer learning.


for model_name in Model_List:
    print("\n \n K = ", model_name, "\n")
    best_model_name = get_best_model_name_bench()
    print("best_model_name = ", best_model_name)

    #Loading the model back:
    try:
        """ tf250 trained model seems can not be reload with tf260!"""
        best_model_reload = tf.keras.models.load_model(best_model_name, custom_objects={'KerasLayer':hub.KerasLayer})
        print("K reload = ", best_model_name)
#        # eval
#        loss, accuracy = best_model_reload.evaluate(valid_ds_pre)
#        print('Valid accuracy :', accuracy)
#        loss, accuracy = best_model_reload.evaluate(test_ds_pre)
#        print('Test accuracy :', accuracy)

    except:
        print("No this kind of model!!", model_name)



print('End time of evaluation: ', t.tocvalue())
t.toc() #





# pred

#batch_n = 0
#acc_count= 0

label_true_all = []
label_pred_all = []

# rewrite to prediction with batch of ds, replace the list of file to speed up.
# todo: Checked right!:model_back.predict_on_batch [OK done 20200904]
def pred_on_batch(model_back, val_ds_pre):

    batch_n = 0
    acc_count= 0

    for image_batch, label_batch in tqdm(val_ds_pre): #ds set to repeat forever
        batch_n += 1
        pred_max = []
        pred = model_back.predict_on_batch(image_batch)
        
        label_batch_np = label_batch.numpy()
        label_true_all.extend(label_batch_np)
        #print('label_batch_np = ',label_batch_np)
        
        for i in range(MULTI_BATCH_SIZE):#BATCH_SIZE to MULTI_BATCH_SIZE if used Multi-GPU training
    #         print(i)
            try:
                score = tf.nn.softmax(pred[i])
                label_pred = np.argmax(score)
                pred_max.append(label_pred)
                
    #             print('label_batch_np[i] = ', label_batch_np[i])
                
                if label_batch_np[i] == label_pred:
                    acc_count += 1
            except IndexError:
                #print("End of batch")
                pass
                
        label_pred_all.extend(pred_max)
        
        #print("pred =", pred_max)
    print("acc_count =", acc_count, "  (if score == label[i] then count one.)")
    print("accuracy  = ", acc_count/test_samples, "%")
    print("Number of batch used = ",batch_n)


# conf matrix 1
def plot_confusion_matrix_with_seaborn():
    # Plot confusion matrix with seaborn

    cm = tf.math.confusion_matrix(label_true_all, label_pred_all, num_classes=outputnum)
    print("abstract of cm: \n", cm)

    classes = CLASSES
    print("number of classes: ", len(CLASSES))

    # acc in %. Comment this line to be number of images.
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    print("abstract of cm in %: \n", cm)

    #set inner text scale
    # sns.set(font_scale=1.2)

    # Let label of xticks go to top
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    #set inner text scale, for inner of inner digits
    #sns.set(font_scale=1.2)

    sns.heatmap(
        cm, annot=True,
        fmt='.2f',
        cmap=plt.cm.Blues,
        vmin=0, vmax=1,
        xticklabels=classes,
        yticklabels=classes)
    plt.title('Confusion Matrix', fontsize='x-large')
#    plt.xlabel("Predicted class\n",fontsize=14, fontweight='bold')
#    plt.ylabel("True class",fontsize=14,fontweight='bold')
    plt.xlabel("Predicted class\n")
    plt.ylabel("True class")

    # Let y-label also matching matplotlib
    plt.yticks(rotation=0)
    # plt.title('Confusion Matrix', fontsize=20)
    
    dpi=100
    pgn=f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_confusion_matrix_seaborn.png'
    print(f'Save to {pgn} \n')
    plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)


def plot_confusion_matrix_with_seaborn_diag():
    # Plot confusion matrix with seaborn

    cm = tf.math.confusion_matrix(label_true_all, label_pred_all, num_classes=outputnum)
    print("abstract of cm: \n", cm)

    classes = CLASSES
    print("number of classes: ", len(CLASSES))

    # acc in %. Comment this line to be number of images.
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    print("abstract of cm in %: \n", cm)

    
    # diag of cm
    mask = np.ones_like(cm)
    mask[np.diag_indices_from(mask)] = False #True


    #set inner text scale
    # sns.set(font_scale=1.2)

    # Let label of xticks go to top
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    #set inner text scale, for inner of inner digits
    #sns.set(font_scale=1.2)

    # Outline frame of cm
    ax.axhline(y=0, color='k',linewidth=1)
    ax.axhline(y=cm.shape[1], color='k',linewidth=2)
    ax.axvline(x=0, color='k',linewidth=1)
    ax.axvline(x=cm.shape[0], color='k',linewidth=2)
    
    sns.heatmap(
        cm, annot=True,
        fmt='.2f',
        mask=mask,
        cmap=plt.cm.Blues,
        vmin=0, vmax=1,
        xticklabels=classes,
        yticklabels=classes)
    plt.title('Confusion Matrix', fontsize='x-large')
#    plt.xlabel("Predicted class\n",fontsize=14, fontweight='bold')
#    plt.ylabel("True class",fontsize=14,fontweight='bold')
    plt.xlabel("Predicted class\n")
    plt.ylabel("True class")

    # Let y-label also matching matplotlib
    plt.yticks(rotation=0)
    # plt.title('Confusion Matrix', fontsize=20)
    
    dpi=100
    pgn=f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_confusion_matrix_seaborn_diag.png'
    print(f'Save to {pgn} \n')
    plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
    
    
# conf matrix 2
def plot_confusion_matrix_with_pyplot():
    cm = tf.math.confusion_matrix(label_true_all, label_pred_all, num_classes=outputnum)
    print("abstract of cm: \n", cm)
    
    classes = CLASSES
    print("number of classes: ", len(CLASSES))
    
    # acc in %. Comment this line to be number of images.
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    print("abstract of cm in %: \n", cm)
       
    # Let label of xticks go to top
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) #plt.cm.Blues plt.cm.winter
    plt.title('Confusion Matrix', fontsize='x-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    print(tick_marks)
    plt.xticks(tick_marks, classes)#, rotation=-45)
    plt.yticks(tick_marks, classes)


    iters = [[i,j] for i in range(len(classes)) for j in range(len(classes))]
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    #plt.show()
    
    dpi=100
    pgn=f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_confusion_matrix_plot.png'
    print(f'Save to {pgn} \n')
    plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
    

# Roll out for validation set
def roll_out_valid():
    """ Somehow, switch order of plot cm1 and cm2, the cm2 is correct plot.
    
    # pred
    pred_on_batch(best_model_reload, test_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    
    """
    # pred
    pred_on_batch(best_model_reload, valid_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    plot_confusion_matrix_with_seaborn_diag()
    
    
# Roll out final test data
def roll_out():
    """ Somehow, switch order of plot cm1 and cm2, the cm2 is correct plot.
    
    # pred
    pred_on_batch(best_model_reload, test_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    
    """
    # pred
    pred_on_batch(best_model_reload, test_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    plot_confusion_matrix_with_seaborn_diag()

roll_out_valid()
roll_out()



