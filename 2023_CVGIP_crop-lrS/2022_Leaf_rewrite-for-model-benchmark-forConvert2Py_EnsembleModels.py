#!/usr/bin/env python
# coding: utf-8

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
"""
EnsembleTopModelNs_stacking_13s_c5 1287 種組合 2022-03-04
EnsembleTopModelNs_stacking_13s_c4 715 種組合 2022-03-04
EnsembleTopModelNs_stacking_13s_c3 286 種組合 2022-03-04
13s_c2 78種組合 2022-03-04 OK
"""

"""
13s_c2 best 78種組合
EfficientNetV2M+BiTSR50x3    0.8965409994125370
InceptionV3+ViT-B8    0.8956061005592350
MobileNetV3Small+ViT-B8    0.8943595886230470
EfficientNetB1+ViT-B8    0.8946712613105770
"""

"""Pre load SavedModel to Dict
InceptionV3
Elapsed time is 8.152319 seconds.
ResNet50
Elapsed time is 7.018549 seconds.
MobileNetV2
Elapsed time is 9.598717 seconds.
MobileNetV3Small
Elapsed time is 8.347718 seconds.
DenseNet121
Elapsed time is 23.653240 seconds.
EfficientNetB1
Elapsed time is 18.379433 seconds.
EfficientNetB7
Elapsed time is 50.469212 seconds.
EfficientNetV2B1
Elapsed time is 10.983292 seconds.
EfficientNetV2M
Elapsed time is 27.130764 seconds.
VGG16
Elapsed time is 2.191221 seconds.
ViT-B8
Elapsed time is 15.773300 seconds.
Mixer-B16
Elapsed time is 10.292532 seconds.
BiTSR50x3
Elapsed time is 23.527117 seconds.

somehow twcc will held the message of eval phase, when fine csv was written, the std just released.

"""


"""
cd tf.ds.pipeline; sh install_env.sh;
python3
2022_Leaf_rewrite-for-model-benchmark-forConvert2Py_EnsembleModels.py 30 31 imagenet1k crop plateau RA 512 512 1
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
# finetune leaf SavedModel #
from LeafTK import ensemble_model_dict

from itertools import permutations, combinations

import csv


## Set if memory growth should be enabled (not workable when model size still bigger.)
#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#for gpu in gpus:
#    print(f"Set set_memory_growth GPU:{gpu}")
#    tf.config.experimental.set_memory_growth(gpu, True)


t = TicToc() #create instance of class

t.tic() #Start timer


# Check pkg version #
#print(f'tf: {tf.__version__} \ncv2: {cv2.__version__} \nnp: {np.__version__} \npd: {pd.__version__} \nmatplotlib: {matplotlib.__version__}')
t.toc() #End timer


# Model pick up #
"""for Model_List = Model_List[m_start:m_end]"""
m_start=int(sys.argv[1])
m_end=int(sys.argv[2])

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
EPOCHS=20 #20 # vit50fortest.

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
BATCH_SIZE = 4
#combination test acc: ['ViT-B8', 'Mixer-B16']
#bs4:92 0.886 for Ensemble test acc can be larger!
#bs32:109
#bs16:100
#bs1:96
#bs1x8:45s
#bs4x8:51


img_height = int(sys.argv[7])
img_width = int(sys.argv[8])

# N round of fine tune, N=5
n_round = int(sys.argv[9])
N=n_round

patience_1 = 3
patience_2 = 5


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
MULTI_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print('BATCH_SIZE: {}, MULTI_BATCH_SIZE: {}'.format(BATCH_SIZE, MULTI_BATCH_SIZE))


# Path to save #
#log_dir_name = "TrainSaveDir-1227-toPytest" # Put all results in same dir with different file name.
"""[2022-02-15 remove fixed log dir name, use train parameter instead.]
log_dir_name = ./imagenet1k_crop_plateau_RA_512_512/
model SavedModel = log_dir_name/model_name/N/
"""
#log_dir_name = sys.argv[9]
log_dir_name=f'{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}'



""" terminal check """
print(f'* * * Pre-train weight: {weight}')
print(f'\n * * * model_start:model_end: [{m_start} to {m_end}]\n')
fpn= f'_{log_dir_name}/th_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_'
print(f'* * * Field parameter name:\n {fpn} \n')
print(f'* * * Field parameter name:\n {log_dir_name} \n')
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

data_dir = '/home/u3148947/.keras/datasets/leaf/'
leaf_dir = '/home/u3148947/.keras/datasets/leaf/train_images/'

df_train = pd.read_csv(data_dir + '/train.csv') #21398 images
#df_train = pd.read_csv(data_dir + '/train_crop.csv') #2499 images


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
print("test", tf.data.experimental.cardinality(test_ds_map_s).numpy())




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


# give augment type to train_ds_pre # # WE do need the train/val ds here! #
#train_ds_pre = configure_for_performance_cache(train_ds_map_s, cache=True, augment=augment)
#valid_ds_pre = configure_for_performance_cache(valid_ds_map_s)
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

Model_List=list(tf_hub_dict)
#Model_List = Model_List[5:6] #[6:7] then MobileNetV2 train OK, but [5:6] MobileNetV2 will nan
Model_List = Model_List[m_start:m_end]
print(Model_List)

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



# [Models] Train misc. #

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

def get_best_model_name_bench_SavedModel_Nround(N=1):
    return f'./{log_dir_name}/{model_name}/{N}/'

# th = "stage1"
th = "ft" # fine tune only without transfer learning.


#  Tensorboard Log dir name #
# use once at the time
# log_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir_name = "TrainSaveDir"



mk_log_dir(log_dir_name)



logdir = log_dir_name + "/logs/toe/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)



# [Models] Learning Rate Scheduler #


# [WCD] StepWise warmup cosine decay learning rate [optimazer] for some models need very small lr.
# [plateau] initial lr is setting in the tf.keras.optimizers.Adam(0.0001) not 0.00001 1e-5
# keep same as WCD and CDR.
INIT_LR = 0.00001 #1e-5
WAMRUP_LR = 0.0000001 #1e-7
WARMUP_STEPS = 5

WCD = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=EPOCHS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)

# [CDR ]
"""warm Cosine Decay Restart"""
CDR = CosineDecayCLRWarmUpLSW

# [Plateau]
"""Callback lr """


# Select LR_SCHEDULER [move inside the build_efn()] #
if lr_name=='WCD':
    # tk.optimizer
    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(WCD)
    # tk.callback
#     scheduled_lr = tf.keras.callbacks.LearningRateScheduler(fixed_WCD)

if lr_name=='CDR':
    # tk.optimizer
    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(CDR)

if lr_name=='plateau':
    #learning_rate=0.0001 in the tf.keras.optimizers.Adam()
    scheduled_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=0.5, patience=1, verbose=1,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0)
        
if lr_name=='fixed':
    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(fixed_scheduler)
    print(fixed_scheduler(1))
    
if lr_name=='lrdump':
    scheduled_lr = tf.keras.callbacks.LearningRateScheduler(lrdump)
    
if lr_name=='WCDCR': # warmup cosin decay with cycle
    print("No implemented yet.")
    pass

print(f'Set scheduler LR : {lr_name}')


        
callback_lr_time = PrintLR() #return a object of callback, not use the Classs PrintLR.











############################################
#                                          #
#                                          #
#  Ensemble Models for better performance  #
#                                          #
#                                          #
############################################

""" [2022-02-28]
Model list of best finetune round.

usage:

python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py_EnsembleModels.py 30 31 imagenet1k crop plateau RA 512 512 1

"""

"""
average : [InceptionV3_out, ViTB8_out, EfficientNetV2M_out]
Testing the accuracy of test DS...
803/803 [==============================] - 149s 135ms/step - loss: 0.5410 - accuracy: 0.8969
 Test dataset accuracy: 0.8968526124954224
 
3s average:(0.889996 - 0.89685)*100 = 0.6854 % improvement
4s average:(0.889996 - 0.89841)*100 = 0.8414 % improvement
"""




#en_dir="EnsembleTopModel3s/"
#en_dir="EnsembleTopModel4s/"
#en_dir="EnsembleTopModel4s_dense/"
en_dir="EnsembleTopModelNs_stacking_13s_c4/"
mk_log_dir(en_dir)
print("\n ensemble list: \n", list(ensemble_model_dict))

# Combination of pair number c# #
comb_num=4

T_ACC_csv = en_dir + "/" + "EnsembleTopModelNs_stacking_13s_c4.csv"



## tf.keras.models.load_model
#print('\n Loading SavedModels...\n')
#InceptionV3 = tf.keras.models.load_model(ensemble_model_dict["InceptionV3"])
#print('\n Loading SavedModels...\n')
#ViTB8 = tf.keras.models.load_model(ensemble_model_dict["ViT-B8"])
#print('\n Loading SavedModels...\n')
#EfficientNetV2M = tf.keras.models.load_model(ensemble_model_dict["EfficientNetV2M"])
#print('\n Loading SavedModels...\n')
#
#BiTSR50x3 = tf.keras.models.load_model(ensemble_model_dict["BiTSR50x3"])
#print('\n Loading SavedModels ok...\n')



#InceptionV3.trainable = False
#ViTB8.trainable = False
#EfficientNetV2M.trainable = False
#BiTSR50x3.trainable = False

# move to Top
#inputs = tf.keras.Input(shape=(img_height, img_width, 3)) #shape=(120, 120, 3), img_height, img_width
#
#InceptionV3_out = InceptionV3(inputs, training=False)
#ViTB8_out = ViTB8(inputs, training=False)
#EfficientNetV2M_out = EfficientNetV2M(inputs, training=False)
#BiTSR50x3_out = BiTSR50x3(inputs, training=False)



# Switch to task #
simple_stacking=True
dense_training=False


############################################
#                                          #
#  Train a dense for ensemble models       #
#                                          #
############################################
if dense_training:
    print('\n Train a dense for ensemble models ...\n')
    
    len_of_merge = len(ensemble_model_dict)
    hidden_num=len_of_merge*outputnum
    
    
    
    # Hand write ensemble # workable but duty.
#    with strategy.scope():
#        """
#        ValueError: The name "input_first" is used 4 times in the model. All layer names should be unique.
#        ValueError: The name "top_output" is used 4 times in the model. All layer names should be unique.
#
#
#         LookupError: No gradient defined for operation 'IdentityN' (op type: IdentityN)
#             Custom gradients can be saved to SavedModel by using the option tf.saved_model.SaveOptions(experimental_custom_gradients=True).
#             https://github.com/tensorflow/tensorflow/issues/40166
#             https://www.tensorflow.org/guide/advanced_autodiff#custom_gradients_in_savedmodel
#
#             some model can load SavedModel and retrain (InceptionV3), but some is not (ViTb8)
#        """
#        # input layer name "input_first" can not repeat in ensemble models
#        # (4) tf.keras.backend.reset_uids() #Totally not work for this issue.
#        # (3) re-set the first layer's name
#
#        # (5) hand write ensemble model
#        # still LookupError: No gradient defined for operation 'IdentityN' (op type: IdentityN)
#        # what if only ViTB8: it happen 'IdentityN'.
#        # InceptionV3 only: it can train!!!
#        # BiTSR50x3 only: it can train!!!
#        # [InceptionV3_out, BiTSR50x3_out]:Test dataset accuracy: 0.889373
#
#        inputs = tf.keras.Input(shape=(img_height, img_width, 3)) #shape=(120, 120, 3), img_height, img_width
#        # tf.keras.models.load_model
#        """need in side the scope
#        """
#        print('\n Loading SavedModels...\n')
#        InceptionV3 = tf.keras.models.load_model(ensemble_model_dict["InceptionV3"])
##        print('\n Loading SavedModels...\n')
#        #ViTB8 = tf.keras.models.load_model(ensemble_model_dict["ViT-B8"])
#        BiTSR50x3 = tf.keras.models.load_model(ensemble_model_dict["BiTSR50x3"])
#        print('\n Loading SavedModels ok...\n')
#
#        InceptionV3.trainable = False
#        #ViTB8.trainable = False
#        BiTSR50x3.trainable = False
#
#        InceptionV3_out = InceptionV3(inputs, training=False)
#        #ViTB8_out = ViTB8(inputs, training=False)
#        BiTSR50x3_out = BiTSR50x3(inputs, training=False)
#
#        en_output = [InceptionV3_out, BiTSR50x3_out] #[BiTSR50x3_out] #[InceptionV3_out] #[ViTB8_out] #[InceptionV3_out, ViTB8_out] #, EfficientNetV2M_out, BiTSR50x3_out]
#        # Average
#        #outputs = tf.keras.layers.average(en_output)
#
#
#        merge = tf.keras.layers.concatenate(en_output)
#        #hidden = tf.keras.layers.Dense(hidden_num, activation='relu')(merge)
#        output = tf.keras.layers.Dense(outputnum, activation='softmax')(merge)
#
#        model = tf.keras.Model(inputs=inputs, outputs=output) # single input
#
#        # ensemble model were freeze weight, and need larger lr to train.
#        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
#                        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
#                         metrics=['accuracy'])
#
#
#    # plot graph of ensemble
#    print(model.summary())
#    tf.keras.utils.plot_model(model, en_dir + "EnsembleTopModel4s_dense_20220301_1630_hand.png", show_shapes=True, show_dtype=True, expand_nested=True, rankdir='TB') #rankdir='TB' or 'LR'
#
#
#    # Build and Train
#
#
#    best_model_save = tf.keras.callbacks.ModelCheckpoint(filepath=en_dir + "/model/",
#                             save_best_only = True,
#                             save_weights_only = False,
#                             monitor = monitor,
#                             options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
#                             mode = 'auto', verbose = 1)
#
#    callbacks_s2 = [
#                #     tensorboard_callback,
#                    best_model_save,
#                    tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience_2),
#                    scheduled_lr,
#                    callback_lr_time,
#                ]
#
#
#
#    hist = model.fit(train_ds_pre,
#                      verbose=1, #'auto' or 0 1 2 , 0 = silent, 1 = progress bar, 2 = one line per epoch.
#                      epochs=EPOCHS,
#                      validation_data=valid_ds_pre,
#                      callbacks=callbacks_s2)#, validation_split=0.1)
#
#    # Check Test ds acc
#        # tf.keras.models.load_model
#    evl_model = tf.keras.models.load_model(en_dir + "/model/")
#
#    # verbose 0 = silent, 1 = progress bar.
#    loss, accuracy = evl_model.evaluate(test_ds_pre, verbose=1)
#    print("\n Test dataset accuracy:", accuracy, "\n")
    
    
    
    
    # Auto configure dense ensemble #
    with strategy.scope():
        """
        Checked: The transformer like model not works: that is ViTs and Mixers.
        
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py_EnsembleModels.py 30 31 imagenet1k crop plateau RA 512 512 1
        
        
        ensemble list: [InceptionV3_out, BiTSR50x3_out]
        Total params: 233,017,889
        Trainable params: 55

        # [InceptionV3_out, BiTSR50x3_out] lr plateau 0.001 :Test dataset accuracy: 0.8887503743171692 e6
        # [InceptionV3_out, BiTSR50x3_out] lr plateau  0.01 :Test dataset accuracy: 0.8890620470046997 e3
        
        # [InceptionV3_out, BiTSR50x3_out] lr plateau 0.1 :Test dataset accuracy: 0.8899968862533569 e1 best
        # [InceptionV3_out, BiTSR50x3_out] lr plateau 0.9 :Test dataset accuracy: 0.8893736600875854 e13
        # [InceptionV3_out, BiTSR50x3_out] lr plateau   9 :Test dataset accuracy: 0.8790900707244873 e1
        
        ensemble list: ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3']
        Total params: 286,174,707
        Trainable params: 80
        lr plateau 0.1 : Test dataset accuracy: 0.8909317851066589 e4
        
        ensemble list:  ['InceptionV3', 'ViT-B8', 'EfficientNetV2M', 'BiTSR50x3']
        Total params: 286,326,033
        Trainable params: 105
            LookupError: No gradient defined for operation 'IdentityN' (op type: IdentityN) <--- 'ViT-B8' cause it!
        
        [2022-03-02]
        ensemble list: (note that MobileNetV3Small #84.4500)
        ['InceptionV3', 'MobileNetV3Small', 'EfficientNetV2M', 'BiTSR50x3'] dense5
        Test dataset accuracy: 0.8881271481513977 e5 , less than EfficientNetV2M#88.9997
        
        ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3'] dense5 plateau0.01
        Test dataset accuracy: 0.8934247493743896                           better ********************
                
        ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3'] dense5 plateau0.1
        Test dataset accuracy: 0.8928015232086182 (batter than with dense1024relu) ********************
                
        ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3'] dense5 plateau0.4
        Test dataset accuracy: 0.8915550112724304 e3
                
                
        ['InceptionV3', 'MobileNetV3Small', 'EfficientNetV2M', 'BiTSR50x3'] dense1024relu->dense5
        Test dataset accuracy: 0.887815535068512 e10
        
        ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3'] dense1024relu->dense5
        Test dataset accuracy: 0.8918666243553162 e11   better
        
        ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3'] dense102linear->dense5
        Test dataset accuracy: 0.8915550112724304 e2    better
        
        ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3'] dense102lsoftmax->dense5
        Test dataset accuracy: 0.8834527730941772 e3
        
        
        [2022-03-03]
        ensemble list:
        ['InceptionV3', 'DenseNet121', 'EfficientNetB7', 'EfficientNetV2M', 'BiTSR50x3'] dense5 plateau0.01
        Total params: 357,327,878
        Trainable params: 130
        [1066.026839017868] of epoch 1
        [474.96363735198975] of epoch 2
        Test dataset accuracy: 0.8875039219856262 e4 to 0.89062va       lower than 3s
        
        
        """
        # one input head
        inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="input_org")
        # load top models
        members=[tf.keras.models.load_model(ensemble_model_dict[key]) for key in ensemble_model_dict]
        
        
        # define multi-headed input (our case is one head, so leave this away.)
        #ensemble_visible = [model.input for model in members]


        """
        ValueError: The name "input_first" is used 4 times in the model. All layer names should be unique.
        ValueError: The name "top_output" is used 4 times in the model. All layer names should be unique.


         LookupError: No gradient defined for operation 'IdentityN' (op type: IdentityN)

        """
        # input layer name "input_first" can not repeat in ensemble models
        # (4) tf.keras.backend.reset_uids() #Totally not work for this issue.
        # (3) re-set the first layer's name

        members_re_in_name=[]
        for model in members:
            #tf.keras.backend.reset_uids()
#            model.layers[0]._name = model.name + "_in"
#            model.layers[-1]._name = model.name + "_ot"
            model.trainable = False

            #model=tf.keras.Sequential(model)#model.layers[1:-2]
            #model.build(input_shape=(img_height, img_width, 3))
            #print(model.summary())

            for layer in model.layers:
                # rename to avoid 'unique layer name' issue (multiple head?) [No need for our one input head!]
                # for i, model in enumerate(models):
                #   layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
                #   or layer.name = "Flatten_{}".format( i )
                print(layer._name)

            new_model = tf.keras.Sequential([inputs, model])

            members_re_in_name.append(new_model)
            print(f"Check layer name: {model.layers[0]._name} {model.name}")
            print(new_model.summary())

        #members_re_in_name=[model.layers[0]._name for model in members]

        #ensemble_visible = [model.input for model in members_re_in_name]

        # input layer name "input_first" can not repeat in ensemble models
    #    # (2) old for
    #    members=[]
    #    ensemble_visible=[]
    #    for key in ensemble_model_dict:
    #        members.append(tf.keras.models.load_model(ensemble_model_dict[key]))
    #
    #    i=0
    #    for model in members:
    #        ensemble_visible.append(model.input(name=str(i)))
    #        i+=1

        # input layer name "input_first" can not repeat in ensemble model
        # (1) print what it is
    #    i=0
    #    for input in ensemble_visible:
    #        input(name=str(i))
    #        i+=1
    #        print(input)
        #ensemble_visible = [print(input) for input in ensemble_visible]

        # concatenate merge output from each model
        #ensemble_outputs = [model.output for model in members]
        ensemble_outputs = [model.output for model in members_re_in_name] # finial dense 5 output
#        ensemble_outputs = [model.layers[-2] for model in members_re_in_name] # keras_layer output
        merge = tf.keras.layers.concatenate(ensemble_outputs)

        len_of_merge = len(ensemble_model_dict)
        print(f"len_of_merge : {len_of_merge}")
        hidden_num=len_of_merge*outputnum
        #hidden = tf.keras.layers.Dense(hidden_num, activation='relu')(merge)
#        hidden = tf.keras.layers.Dense(1024, activation='softmax')(merge)
        output = tf.keras.layers.Dense(outputnum, activation='softmax')(merge)

        #model = tf.keras.Model(inputs=ensemble_visible, outputs=output) # list input
        model = tf.keras.Model(inputs=inputs, outputs=output) # single input

        # default 0.00001 too small for ensemble, 0.01 or 0.1 mm 0.9 better
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
                        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                         metrics=['accuracy'])


    # plot graph of ensemble
    print(model.summary())
    tf.keras.utils.plot_model(model, en_dir + "EnsembleTopModel4s_dense_20220301_1400_confense.png", show_shapes=True, show_dtype=True, expand_nested=True, rankdir='TB') #rankdir='TB' or 'LR'


    # Build and Train


    best_model_save = tf.keras.callbacks.ModelCheckpoint(filepath=en_dir + "/model/",
                             save_best_only = True,
                             save_weights_only = False,
                             monitor = monitor,
                             options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
                             mode = 'auto', verbose = 1)

    callbacks_s2 = [
                #     tensorboard_callback,
                    best_model_save, # if comment, default is SavedModel (model.save()), then will save to ../model_name/xx.pb
                    tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience_1), #patience=step_size or ep_num
                #     lr_reduceonplateau,
#                     tf.keras.callbacks.LearningRateScheduler(lrdump),#lrdump, decay or lrfn or lrfn2. clr
                    scheduled_lr,
                    callback_lr_time,
                #     tensorboard_callback,
                ]



    hist = model.fit(train_ds_pre,
                      verbose=1, #'auto' or 0 1 2 , 0 = silent, 1 = progress bar, 2 = one line per epoch.
                      epochs=EPOCHS,
                      validation_data=valid_ds_pre,
                      callbacks=callbacks_s2)#, validation_split=0.1)
                      
                      
                      
    # Check Test ds acc
    # tf.keras.models.load_model
    evl_model = tf.keras.models.load_model(en_dir + "/model/")

    # verbose 0 = silent, 1 = progress bar.
    loss, accuracy = evl_model.evaluate(test_ds_pre, verbose=1)
    print("\n Test dataset accuracy:", accuracy, "\n")





                      
    
############################################
#                                          #
#  Stacking a simple outputs of models     #
#                                          #
############################################
if simple_stacking:
    """
    [2022-03-03]
    Somehow train the new dense of top of ensemble models can not achieve the t_acc of Stacking.
    
    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py_EnsembleModels.py 30 31 imagenet1k crop plateau RA 512 512 1
    
    GPU*8
    ['InceptionV3', 'EfficientNetV2M', 'BiTSR50x3'] average:
    101/101 [==============================] - 135s 329ms/step - loss: 0.5538 - accuracy: 0.8947
    Test dataset accuracy: 0.8946712613105774
    
    GPU*1
    803/803 [==============================] - 205s 194ms/step - loss: 0.5538 - accuracy: 0.8947
    Test dataset accuracy: 0.8946712613105774
    
    GPU*1
    ['InceptionV3', 'EfficientNetV2M', 'ViT-B8', 'BiTSR50x3'] average:
    803/803 [==============================] - 249s 259ms/step - loss: 0.5375 - accuracy: 0.8984
    Test dataset accuracy: 0.8984107375144958                                           **** best ****
    
    GPU*1
    ['InceptionV3', 'DenseNet121', 'EfficientNetB7', 'EfficientNetV2B1', 'EfficientNetV2M', 'ViT-B8', 'BiTSR50x3'] average:
    803/803 [==============================] - 370s 356ms/step - loss: 0.5356 - accuracy: 0.8959
    Test dataset accuracy: 0.8959177136421204                                               lower
    
    CropNet
    ['InceptionV3', 'EfficientNetV2M', 'ViT-B8', 'BiTSR50x3', 'CropNet']
    
    """

    print('\n Stacking a simple outputs of models ...\n')
    len_of_merge = len(ensemble_model_dict)
    hidden_num=len_of_merge*outputnum
    

    csv_comb_model_list=[]
    csv_comb_testacc_list=[]

    # Auto configure stacking ensemble #
    with strategy.scope():
    
        # [InceptionV3_out, ViTB8_out, EfficientNetV2M_out, BiTSR50x3_out] 0.8984107375144958 **** best ****
        
        """
        [2022-03-03]
        can we load all models at once to reduce the repeat loading time? YES, 1 gpu ccs can.
        ['InceptionV3', 'ResNet50', 'MobileNetV2', 'MobileNetV3Small', 'DenseNet121', 'EfficientNetB1', 'EfficientNetB7', 'EfficientNetV2B1', 'EfficientNetV2M', 'VGG16', 'ViT-B8', 'BiTSR50x3']
        Total params: 413,090,002
        803/803 [==============================] - 462s 425ms/step - loss: 0.7981 - accuracy: 0.8950
        Test dataset accuracy: 0.8949828743934631
        
        
        """
        # Combination list for ensemble #
        model_list = list(ensemble_model_dict)
        #model_list = model_list[7:10]
        #7:10 no-preload real    4m6.549s
        #7:10 preload  real    3m35.067s the neck is in initial the graph of model, preload save a little time.
        print(len(model_list), model_list, "\n")
        combinations_list=list(combinations(model_list,comb_num))
        print(len(combinations_list), combinations_list)
        
        
        # Pre load SavedModel to Dict #
        print("\n Pre load SavedModel to Dict \n")
        saved_model_dict={}
        
        for name in model_list:
            t.tic()
            print(name)
            model=tf.keras.models.load_model(ensemble_model_dict[name])
            saved_model_dict[str(name)]=model
            t.toc()
        
        
        # Write comb test acc to file #
        
        #header=['Models', log_dir_name, "MB"]
        header=['Comb-Models']
        header.append('Test Accuracy')
        print(header)
        
        with open(T_ACC_csv, 'w', encoding='UTF8', newline='') as csv_f:
            """ Parse the test accuracy from bench models."""
            # CSV
            writer = csv.writer(csv_f)
            # write the header
            writer.writerow(header)
        
        
            # use pre-load to combine ensemble #
            for comb in tqdm(combinations_list):
            
                """
                # With `clear_session()` called at the beginning,
                # Keras starts with a blank state at each iteration
                # and memory consumption is constant over time.
                """
                tf.keras.backend.clear_session()
                
                #for comb in combinations_list:
                t.tic()
                comb=list(comb)
                print(f"\n combination test acc: {comb}")


                # one input head
                inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="input_org")
                # load top models
                #members=[tf.keras.models.load_model(ensemble_model_dict[key]) for key in comb]
                members=[saved_model_dict[name] for name in comb]
                
                
                # Ensemble output
                print('\nMerging outputs...\n')
                en_output = [model(inputs) for model in members] #[model.output for model in members]

                # Average
                outputs = tf.keras.layers.average(en_output) #3s:Test dataset accuracy: 0.8968526124954224, 4s:Test dataset accuracy: 0.8984107375144958 **** best ****

                # build single model
                ensemble_model = tf.keras.Model(inputs, outputs, name="EnsembleTopModelNsStacking")

        #        print(ensemble_model.summary())
        #        tf.keras.utils.plot_model(ensemble_model, en_dir + "EnsembleTopModel3s_Stacking_20220303_1100.png", show_shapes=True, show_dtype=True, expand_nested=True, rankdir='TB') #rankdir='TB' or 'LR'


                # You must compile your model before training/testing.
                print('\nYep we compile en model again...\n')
                ensemble_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
                        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                         metrics=['accuracy'])
                t.toc()

                t.tic()
                # verbose 0 = silent, 1 = progress bar.
                """Model.evaluate is not yet supported with tf.distribute.experimental.ParameterServerStrategy."""
                print('\nTesting the accuracy of test DS...\n')
                loss, accuracy = ensemble_model.evaluate(test_ds_pre, verbose=0)
                print("\n Test dataset accuracy:", accuracy, "\n")
                t.toc()

                #accuracy=0.7788
                #csv_comb_model_list.append(comb)
                #csv_comb_testacc_list.append(accuracy)
                
                
                # write the data
                #csv_record_m_a = [ writer.writerow([mo, acc]) for mo, acc in list(zip(csv_comb_model_list, csv_comb_testacc_list)) ]
                #csv_record_m_a = [ writer.writerow(["+".join(mo), acc]) for mo, acc in list(zip(comb, str(accuracy))) ]
                writer.writerow(["+".join(comb), accuracy])


#    # Write comb test acc to file #
#
#    #header=['Models', log_dir_name, "MB"]
#    header=['Comb-Models']
#    header.append('Test Accuracy')
#    print(header)
#
#    #csv_record_m_a = [ list(item) for item in list(zip(csv_comb_model_list, csv_comb_testacc_list)) ]
#    #csv_record_m_a = [ list([mo, acc]) for mo, acc in list(zip(csv_comb_model_list, csv_comb_testacc_list)) ]
#    with open(T_ACC_csv, 'w', encoding='UTF8', newline='') as csv_f:
#        """ Parse the test accuracy from bench models."""
#        # CSV
#        writer = csv.writer(csv_f)
#        # write the header
#        writer.writerow(header)
#        # write the data
#        #csv_record_m_a = [ writer.writerow([mo, acc]) for mo, acc in list(zip(csv_comb_model_list, csv_comb_testacc_list)) ]
#        csv_record_m_a = [ writer.writerow(["+".join(mo), acc]) for mo, acc in list(zip(csv_comb_model_list, csv_comb_testacc_list)) ]
##        for mo, acc in zip(csv_comb_model_list, csv_comb_testacc_list):
##            writer.writerow([mo, acc])
#        #writer.writerow([csv_record_m_a])
        
        
        
######################################################
        

        # pre setting model list ensemble #
#        # one input head
#        inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="input_org")
#        # load top models
#        members=[tf.keras.models.load_model(ensemble_model_dict[key]) for key in ensemble_model_dict]
#
##        # CropNet #
##        CropNet_hub=hub.KerasLayer("https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2")
##        scale_layer=tf.keras.layers.Rescaling(1./255)
##        resize_layer_224 = tf.keras.layers.Resizing(224, 224)
##        CropNet = tf.keras.Sequential([inputs, scale_layer, resize_layer_224, CropNet_hub, )
##        members.append(CropNet)# 6 classes output shape not fit out 5 !
#
#
##        members_re_in_name=[]
##        for model in members:
##            model.trainable = False
##            for layer in model.layers:
##                print(layer._name)
##
##            new_model = tf.keras.Sequential([inputs, model])
##
##            members_re_in_name.append(new_model)
##            print(f"Check layer name: {model.layers[0]._name} {model.name}")
##            print(new_model.summary())
#
#        # Ensemble output
#        print('\nMerging outputs...\n')
#
#        en_output = [model(inputs) for model in members] #[model.output for model in members]
#
#        #en_output = [InceptionV3_out, ViTB8_out, EfficientNetV2M_out, BiTSR50x3_out]
#        #outputs = tf.keras.layers.Dense(outputnum, activation="softmax")(en_output)
#
#        # Average
#        outputs = tf.keras.layers.average(en_output) #3s:Test dataset accuracy: 0.8968526124954224, 4s:Test dataset accuracy: 0.8984107375144958 **** best ****
#
#        # add + average
#        #add = tf.keras.layers.Add()([InceptionV3_out, ViTB8_out])
#        #outputs = tf.keras.layers.average([add, EfficientNetV2M_out]) #Test dataset accuracy: 0.8968526124954224  same ??? why?
#
#        #outputs = EfficientNetV2M_out #Test dataset accuracy: 0.8899968862533569 is correct of test in local.
#
#        # average + add
#        #average = tf.keras.layers.average([InceptionV3_out, ViTB8_out])
#        #outputs = tf.keras.layers.Add()([average, EfficientNetV2M_out])#Test dataset accuracy: 0.8956061005592346 lower?
#
#
#
#        ##
#        ## if train a dense for ensemble models
#        ##
#        #merge = tf.keras.layers.concatenate(en_output)
#
#
#        # build single model
#        ensemble_model = tf.keras.Model(inputs, outputs, name="EnsembleTopModelNsStacking")
#
#        print(ensemble_model.summary())
#        tf.keras.utils.plot_model(ensemble_model, en_dir + "EnsembleTopModel3s_Stacking_20220303_1100.png", show_shapes=True, show_dtype=True, expand_nested=True, rankdir='TB') #rankdir='TB' or 'LR'
#
#
#        # You must compile your model before training/testing.
#        print('\nYep we compile en model again...\n')
#        ensemble_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
#                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
#                 metrics=['accuracy'])
#
#
#    # verbose 0 = silent, 1 = progress bar.
#    print('\nTesting the accuracy of test DS...\n')
#    loss, accuracy = ensemble_model.evaluate(test_ds_pre, verbose=1)
#    print("\n Test dataset accuracy:", accuracy, "\n")



#if simple_stacking:
#
#    # Ensemble output
#    print('\nMerging outputs...\n')
#    en_output = [InceptionV3_out, ViTB8_out, EfficientNetV2M_out, BiTSR50x3_out] 0.8984107375144958 **** best ****
#    #outputs = tf.keras.layers.Dense(outputnum, activation="softmax")(en_output)
#
#    # Average
#    outputs = tf.keras.layers.average(en_output) #3s:Test dataset accuracy: 0.8968526124954224, 4s:Test dataset accuracy: 0.8984107375144958 **** best ****
#
#    # add + average
#    #add = tf.keras.layers.Add()([InceptionV3_out, ViTB8_out])
#    #outputs = tf.keras.layers.average([add, EfficientNetV2M_out]) #Test dataset accuracy: 0.8968526124954224  same ??? why?
#
#    #outputs = EfficientNetV2M_out #Test dataset accuracy: 0.8899968862533569 is correct of test in local.
#
#    # average + add
#    #average = tf.keras.layers.average([InceptionV3_out, ViTB8_out])
#    #outputs = tf.keras.layers.Add()([average, EfficientNetV2M_out])#Test dataset accuracy: 0.8956061005592346 lower?
#
#
#
#    ##
#    ## if train a dense for ensemble models
#    ##
#    #merge = tf.keras.layers.concatenate(en_output)
#
#
#    # build single model
#    ensemble_model = tf.keras.Model(inputs, outputs, name="EnsembleTopModel3s")
#
#    print(ensemble_model.summary())
#    tf.keras.utils.plot_model(ensemble_model, en_dir + "EnsembleTopModel3s_20220301_1000.png", show_shapes=True, show_dtype=True, expand_nested=True, rankdir='TB') #rankdir='TB' or 'LR'
#
#
#    # You must compile your model before training/testing.
#    print('\nYep we compile en model again...\n')
#    ensemble_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
#            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
#             metrics=['accuracy'])
#
#
#    # verbose 0 = silent, 1 = progress bar.
#    print('\nTesting the accuracy of test DS...\n')
#    loss, accuracy = ensemble_model.evaluate(test_ds_pre, verbose=1)
#    print("\n Test dataset accuracy:", accuracy, "\n")


#############################################
##                                          #
##                                          #
##  Leave this block to take next round     #
##                                          #
##                                          #
#############################################
#exit()










############################################
#                                          #
#                                          #
#  Leave this block to take next round     #
#                                          #
#                                          #
############################################
exit()


#tt = 0
#nt = 0




# #### <font color="yellow"> [Models] Train top layers (transfer learning)</font>
# # fit the model on all data
# history_toe = model_toe.fit(train_ds_pre, 
#                       verbose=1, 
#                       epochs=5, #ep_num_transf, 
#                       validation_data=valid_ds_pre, 
#                       callbacks=callbacks)#, validation_split=0.1)



# 5. Fine tune #
# 
# #### <font color="yellow"> [Models] Train bench models (Fine tune)</font>
#

# def unfreeze_model(model, base_model):
# #     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
# #     for layer in model.layers[-20:]:
# #         if not isinstance(layer, layers.BatchNormalization):
# #             layer.trainable = True

# #     model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),#RMSprop , Adam, SGD Adadelta(learning_rate=0.001), if set lr_callback the learning_rate=0.001 will not effeced.



# [Models] Train bench models (Fine tune) #
# 
# copy from K10L195

# use dict of dict to store hist
history_toe_finetune = {}




#
# [Models] Train bench models (Fine tune) #
#

#
# [Models] Train bench models (Fine tune) #
#

for model_name in Model_List:
    npydir = log_dir_name + "/" + model_name + "/npy/"
    mk_log_dir(npydir)
    
    for N in range(n_round):
        print("\n \n K model= ", model_name)
        print("\n \n N round= ", N, "\n") # [0,1,2,3,4]
        #best_model_name = get_best_model_name_bench() # keras hdf5
        best_model_name_dir = get_best_model_name_bench_SavedModel_Nround(N) # SavedModel dir, also with N=5 round
        best_model_name = best_model_name_dir
        
        best_model_save = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_name,
                                     save_best_only = True,
                                     save_weights_only = False,
                                     monitor = monitor,
                                     mode = 'auto', verbose = 1)

        callbacks_s2 = [
                    #     tensorboard_callback,
                        best_model_save, # if comment, default is SavedModel (model.save()), then will save to ../model_name/xx.pb
                        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience_2), #patience=step_size or ep_num
                    #     lr_reduceonplateau,
    #                     tf.keras.callbacks.LearningRateScheduler(lrdump),#lrdump, decay or lrfn or lrfn2. clr
                        scheduled_lr,
                        callback_lr_time,
                    #     tensorboard_callback,
                    ]
        print('best_model_name:', best_model_name)


        """
        # With `clear_session()` called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        """
        tf.keras.backend.clear_session()
  
        with strategy.scope():
            """ Need switch k.apps and t.hub build model. When some model not supported with tf260+tf.hub.
            
                if model name in k.apps_list:
                    build_efn_model()
                else:
                    build_tf_hub_models()
            """
            if model_name in ['MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large', 'DenseNet121', 'DenseNet169', 'DenseNet201',
                'NASNetMobile', 'NASNetLarge',
                "__",
                'EfficientNetB0',
                'EfficientNetB1',
                'EfficientNetB2',
                'EfficientNetB3',
                'EfficientNetB4',
                'EfficientNetB5',
                'EfficientNetB6',
                'EfficientNetB7',
                'VGG16', 'VGG19']:
                print('\n\n* * * Building keras.apps...\n\n')
                model_toe, base_model = build_efn_model(weight, model_name, outputnum, img_height, img_width, top_dropout_rate, drop_connect_rate)
            else:
                print('\n\n* * * Building tf.hub...\n\n')
                model_toe, base_model = build_tf_hub_models(model_name, outputnum, img_height, img_width)
        
    #     # Train K-Model with transfer learning # IF HAVE!
    #     hist = model_toe.fit(train_ds_pre_toe_s,
    #                           verbose=1,
    #                           epochs=ep_num_transf,
    #                           validation_data=valid_ds_pre_toe_s,
    #                           callbacks=callbacks_toe_tl)#, validation_split=0.1)
    #     history_toe.append(hist)
        
        
        # Train K-Model with fine tune #
        
        # bench models, FT
    #     unfreeze_model(model_toe,base_model) # skip the TL so unfreeze when build_EFN()
        #count_model_trainOrNot_layers(model_toe)
        print(model_toe.summary()) ## print will added a 'None' of end of summary().
        # fit the model on all data
        hist = model_toe.fit(train_ds_pre,
                              verbose=2, #'auto' or 0 1 2 , 0 = silent, 1 = progress bar, 2 = one line per epoch.
                              epochs=EPOCHS,
                              validation_data=valid_ds_pre,
                              callbacks=callbacks_s2)#, validation_split=0.1)
        # add the epoch timeing
        hist.history['epoch_time_secs'] = callback_lr_time.times
        # for each N round
        n_round_np_name = f'{model_name}_{N}'
        history_toe_finetune[n_round_np_name] = hist.history #hist # what if use hist.history


        # SavedModel : backup a SavedModel to compare the size of hdf5
        # Note that: this save is not the best model but the latest model.
        # the SavedModel size is same as hdf5 in VGG16/19
        # VGG16 hdf5 176.7MB SaveModel 177.3MB
        # VGG19 hdf5 240.55MB SavedModel 241.2MB
        # EfficientNetV2B2 hdf5 105.9MB SavedModel 112.5MB
        # NASNetMobile hdf5 54.0MB SavedModel 78.7MB
    #    SavedModel_save_path=f'./{log_dir_name}/{model_name}/{N}_bk/'
    #    tf.keras.models.save_model(model_toe, SavedModel_save_path)
    
    
#
# [Models] Train bench models (Fine tune) #
#

#
# [Models] Train bench models (Fine tune) #
#








        #ED sum
        def get_valloss(his_v_l):
            return np.min(his_v_l), np.argmin(his_v_l)

        t_vl = []
        if n_round_np_name:
            print(f'n_round_np_name:{n_round_np_name}')
            t_v, _ = get_valloss(history_toe_finetune[n_round_np_name]['val_loss'])
            
#        # h_vl = []
#        for k in Model_List:
#            print(f'K:{k}')
#            t_v, _ = get_valloss(history_toe_finetune[k]['val_loss'])
#        #     h_v, _ = get_valloss(history_heel_finetune[k].history['val_loss'])
            
            t_vl.append(t_v)
        #     h_vl.append(h_v)

        # t_vl = np.mean(t_vl, axis=0)
        # h_vl = np.mean(h_vl, axis=0)
        # print(f'{round(t_vl,5)} + {round(h_vl,5)} = {round(t_vl + h_vl,5)}')

        print(f'Minimal losses: {t_vl}')






        # [Models] Result Ploting #

        # Save hist to np #
        #bench_log_name = f'./{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.npy'
        bench_log_name = f'./{log_dir_name}/{model_name}/npy/{model_name}_{N}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.npy'
        np.save(bench_log_name, history_toe_finetune)

        # reload hist from npy
        history_np_load = np.load(bench_log_name, allow_pickle='TRUE').item()
        hisnp = history_np_load.copy()


        # Reload np to hist #
        # draft reload his from saved np
        #bench_log_name = f'./{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.npy'
        bench_log_name = f'./{log_dir_name}/{model_name}/npy/{model_name}_{N}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.npy'
        # Note that, if change bs4 to bs32, when load hist by one gpu container #
        # bench_log_name = './TrainSaveDir-1223-CDR/ft_bench_imagenet1k_crop_512x512_CDR_RA_bs32_best_val_accuracy.npy'


        # reload hist from npy #
        history_np_load = np.load(bench_log_name, allow_pickle='TRUE').item()
        hisnp = history_np_load.copy()

        handles = [handle for handle in Model_List]
        print(f'*** reload model handles list: \n {handles}')


        # plotting train log #
        """Rewrite to a function
        plot_save_bench_val_loss(Model_List, hisnp, dpi=300, png_name=png)
        """

        plt.figure(figsize=(25, 10))
        """
        RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
        
        plt.cla()
        plt.clf()
        plt.close()
         choose add plt.cla() after plt.savefig
        """
        
        #for k in Model_List:
        if n_round_np_name:
            x = len(hisnp[n_round_np_name]['loss']) + 1
            x = range(1,x,1)
            y = hisnp[n_round_np_name]['loss']
            #y = [f'{z:.4f}' for z in y]
            plt.plot(x, y, label=f'{n_round_np_name}_loss')
            
            
            for a,b in zip(x, y):
                #plt.text(a, b, str(b))
                plt.scatter(a,b, color='black', alpha=0.2)
                plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
            
            y = hisnp[n_round_np_name]['val_loss']
            plt.plot(x, y, label=f'{n_round_np_name}_val_loss')

            
        plt.title('K-model ed loss toe-TL')
        plt.ylabel('ed loss'), plt.ylim(0, 2)# for too large loss
        plt.xlabel('epoch')
        plt.legend(title='Nets')

        # save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
        dpi=600 #300 for quickly check, 600up to 800 for journal paper.
        #pgn=f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_loss.png'
        pgn=f'./{log_dir_name}/{model_name}/npy/{model_name}_{N}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_loss.png'
        plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
        plt.cla()
        print(f'Save to {pgn} \n')



        # naive check #
        #for k in Model_List:
        if n_round_np_name:
            print(f"val_acc of {n_round_np_name}: {hisnp[n_round_np_name]['val_accuracy']}")


        # plotting train log #
        """Rewrite to a function
        plot_save_bench_val_accuracy(Model_List, hisnp, dpi=300, png_name=png)
        """
        plt.figure(figsize=(25, 10))

        #for k in Model_List:
        if n_round_np_name:
            x = len(hisnp[n_round_np_name]['val_accuracy']) + 1
            x = range(1,x,1)
            y = hisnp[n_round_np_name]['val_accuracy']
            #y = [f'{z:.4f}' for z in y]
            plt.plot(x, y, label=f'{n_round_np_name}_val_accuracy')
            
            
            for a,b in zip(x, y):
                #plt.text(a, b, str(b))
                plt.scatter(a,b, color='black', alpha=0.2)
                plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
            
            #plt.plot(hisnp[k]['val_accuracy'])

            
        plt.title('K-model-TL val_accuracy', fontsize='xx-large')
        plt.ylabel('val_accuracy'), plt.ylim(0.1, 0.9)# for too large loss
        plt.xlabel('epoch')
        plt.legend(title='Nets:', title_fontsize='x-large', fontsize='large')
        """title_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        """

        # save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
        dpi=300
        #pgn= f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc.png'
        pgn=f'./{log_dir_name}/{model_name}/npy/{model_name}_{N}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc.png'
        print(f'Save to {pgn} \n')
        plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
        plt.cla()

        # naive check #
        #for k in Model_List:
        if n_round_np_name:
            print(f" {n_round_np_name}: {hisnp[n_round_np_name]['val_accuracy']}")




        # plotting train log with lr #
        """ Rewrite to def fun()"""

        """# ax1 for val_accuracy #
        # nice to have this colorful tip."""
        fig, ax1 = plt.subplots(figsize=(25, 10))
        color = 'tab:red'
        ax1.set_title('K-model-TL val_accuracy with lr', fontsize='xx-large')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('val_accuracy', color=color)

        #for k in Model_List:
        if n_round_np_name:
            x = len(hisnp[n_round_np_name]['val_accuracy']) + 1
            x = range(1,x,1)
            y = hisnp[n_round_np_name]['val_accuracy']
            #y = [f'{z:.4f}' for z in y]
            plt.plot(x, y, label=f'{n_round_np_name}_val_accuracy')
            
            
            for a,b in zip(x, y):
                #plt.text(a, b, str(b))
                plt.scatter(a,b, color='black', alpha=0.2)
                plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(title='Nets:', title_fontsize='x-large', fontsize='large')
        """title_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}"""


        # ax2 for learning rate #
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('learning rate', color=color)
        # ax2.plot(hisnp[k]['lr'], color=color)
        x = len(hisnp[n_round_np_name]['lr']) + 1
        x = range(1,x,1)
        y = hisnp[n_round_np_name]['lr']
        #y = [f'{z:.4f}' for z in y]
        plt.plot(x, y, color='green', label=f'learning rate')

        for a,b in zip(x, y):
            #plt.text(a, b, str(b))
            plt.scatter(a,b, color='green', alpha=0.2)
            plt.annotate(f'{b:.7f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'

        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(fontsize='large', loc='upper center')

        # save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
        dpi=300
        #pgn= f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc_lr.png'
        pgn=f'./{log_dir_name}/{model_name}/npy/{model_name}_{N}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc_lr.png'
        print(f'Save to {pgn} \n')
        plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
        plt.cla()



        ## with cycle color/marker ##
        ## with cycle color/marker ##

        matplotlib.rcParams['lines.linewidth'] = 1.5
        matplotlib.rcParams["markers.fillstyle"] = 'left' # 'full', 'left', 'right', 'bottom', 'top', 'none'


        auto_custom_cycler_01 = (cycler(color=[plt.get_cmap('jet')(i/13) for i in range(24)]) + # 24 colors
                          cycler(linestyle=['-', '--', ':', '-.'] * 6) + # [4]*6 = 24
                            cycler(marker=['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D'] * 2)) # [12]*2 = 24


        # ploting train log with lr
        # plt.figure(figsize=(25, 10))

        # for different scales (different Y-axes)
        # fig, ax1 = plt.subplots()
        fig, ax1 = plt.subplots(figsize=(25, 15))


        ax1.set_prop_cycle(auto_custom_cycler_01) # set to use custom_cycler


        # ax1 for val_accuracy #
        # nice to have this colorful tip.
        color = 'tab:red'
        ax1.set_title('K-model-TL val_accuracy with lr', fontsize='xx-large')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('val_accuracy', color=color)

        #for k in Model_List:
        if n_round_np_name:
            x = len(hisnp[n_round_np_name]['val_accuracy']) + 1
            x = range(1,x,1)
            y = hisnp[n_round_np_name]['val_accuracy']
            #y = [f'{z:.4f}' for z in y]
            plt.plot(x, y, label=f'{n_round_np_name}_val_accuracy')
            
        #     # v_a value #
        #     for a,b in zip(x, y):
        #         #plt.text(a, b, str(b))
        #         plt.scatter(a,b, color='black', alpha=0.2)
        #         plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'

                
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.legend(title='Nets:', title_fontsize='x-large', fontsize='large')
        """title_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}"""
        ax1.legend(title='Nets:', title_fontsize='x-large', fontsize='large', bbox_to_anchor=(1.23, 1.0))



        # ax2 for learning rate #
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('learning rate', color=color)
        # ax2.plot(hisnp[k]['lr'], color=color)
        x = len(hisnp[n_round_np_name]['lr']) + 1
        x = range(1,x,1)
        y = hisnp[n_round_np_name]['lr']
        #y = [f'{z:.4f}' for z in y]
        plt.plot(x, y, color='green', label=f'learning rate')

        # lr value #
        # for a,b in zip(x, y):
        #     #plt.text(a, b, str(b))
        #     plt.scatter(a,b, color='green', alpha=0.2)
        #     plt.annotate(f'{b:.7f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'

        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(fontsize='large', loc='lower right', bbox_to_anchor=(1.15, .4))


        # save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
        dpi=300
        #pgn= f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc_lr_cyc.png'
        pgn=f'./{log_dir_name}/{model_name}/npy/{model_name}_{N}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_val_acc_lr_cyc.png'
        
        print(f'Save to {pgn} \n')
        plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
        plt.cla()

        ## with cycle color/marker ##
        ## with cycle color/marker ##



        # Show best val_acc and the epoch #

        print("[max val_acc]")
        tmp_acc_his = []
        #for k in Model_List:
        if n_round_np_name:
            v_a = hisnp[n_round_np_name]['val_accuracy']
            b_e = np.argmax(v_a)
            print(f'  {n_round_np_name} :\t       {v_a[b_e]} Epoch@P{b_e}')
            
            tmp_acc_his.append(v_a[b_e])

        print('\n[Best model and val_acc] [name need to fix]')
        b_m_e = np.argmax(tmp_acc_his)
        print(f'  {Model_List[b_m_e]} : {tmp_acc_his[b_m_e]}')



        # Save a hit note on the model_name/N/ dir to quickly display #

        max_val_acc_name=f'./{log_dir_name}/{model_name}/npy/{model_name}_{v_a[b_e]}_{N}_{b_e}e.vacc'
        with open(max_val_acc_name, 'w') as f:
            f.write(max_val_acc_name)


    # [2022-02-15]
    # Best One type of SavedModel reload, and test dataset evaluation.
    # under for model_name in Model_List:
    """
    $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py_testDS_SavedModel.py 5 6 imagenet1k crop plateau RA 512 512 4
    
    """
    
    
    
    ###########################################
    # Test DS accuracy in N rounds per models #
    ###########################################
    
    """2022-02-26
    K model= Mixer-L16
    N round= 4
    tensorflow.python.framework.errors_impl.InternalError: Failed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY: out of memory
    """
    #for model_name in Model_List:
    test_acc_list = []
    for N in range(n_round):
    
        """
        # With `clear_session()` called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        """
        tf.keras.backend.clear_session()
        
        
        print("\n \n K model= ", model_name)
        print("\n \n N round= ", N, "\n")
        #best_model_name = get_best_model_name_bench() # keras hdf5
        
        best_model_name_dir = get_best_model_name_bench_SavedModel_Nround(N) # SavedModel dir, also with N=5 round
        best_model_name = best_model_name_dir
        print("\n \n N round= ", N, best_model_name, "\n")
                
        # saved_model.load
        #evl_model = tf.saved_model.load(best_model_name)
        
        # tf.keras.models.load_model
        evl_model = tf.keras.models.load_model(best_model_name)
        
        # verbose 0 = silent, 1 = progress bar.
        loss, accuracy = evl_model.evaluate(test_ds_pre, verbose=0)
        test_acc_list.append(accuracy)
        print("\n Test dataset accuracy:", accuracy, "\n")
        
    
    # Save test dataset accuracy for each SavedModel #
    
    b_test_acc_arg = np.argmax(test_acc_list)
    b_test_acc = test_acc_list[b_test_acc_arg]
    max_test_acc_name=f'./{log_dir_name}/{model_name}/{model_name}_test_{b_test_acc}_{b_test_acc_arg}.vacc'
    print("test_acc_list:", test_acc_list)
    with open(max_test_acc_name, 'w') as f:
        f.write(str(test_acc_list))
    
    
    
    
    
    

# ## Evaluate the valid and test accuracy
# 
#     for DenseNet201_imagenet1k_crop_512x512_CDR_RA_bs32
#      the eval/test accuracy is match the training result.
# 
#     803/803 [==============================] - 45s 56ms/step - loss: 0.3456 - accuracy: 0.8962
#     Test accuracy : 0.8962293863296509
#     803/803 [==============================] - 31s 39ms/step - loss: 0.3273 - accuracy: 0.8965
#     Test accuracy : 0.8965409994125366

# In[ ]:


# evl_m_path = log_dir_name + "/" + "ft_DenseNet201_imagenet1k_crop_512x512_CDR_RA_bs32_best_val_accuracy.h5"
# evl_model = tf.keras.models.load_model(evl_m_path)


# In[ ]:


# loss, accuracy = evl_model.evaluate(valid_ds_pre)
# print('Test accuracy :', accuracy)

# # print("count roughly ds size: ", tf.data.experimental.cardinality(valid_ds_pre).numpy() * BATCH_SIZE)


# In[ ]:


# loss, accuracy = evl_model.evaluate(test_ds_pre)
# print('Test accuracy :', accuracy)

# # print("count roughly ds size: ", tf.data.experimental.cardinality(val_ds_pre).numpy() * BATCH_SIZE)


# ### confusion matrix move to finianl script
# 
#     * model-eval_acc_confusion.ipynb
# 
#








print('End time of training: ',t.toc())
t.toc() #
