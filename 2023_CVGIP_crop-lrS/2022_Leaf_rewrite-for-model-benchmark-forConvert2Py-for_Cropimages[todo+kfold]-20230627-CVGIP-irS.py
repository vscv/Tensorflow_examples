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


# 2022-05-09
# for Crop land image classifying
# just re use the training code.
"""任務型容器詳細資料
映像檔
tensorflow-21.11-tf2-py3
命令
cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh 1>&- 2>&-; sh run_training_croplandimage.sh;

    [For example: in the run_training_croplandimage.sh] -> agv[10] = n ruound = 1
    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo].py 5 8 imagenet1k resize plateau RA 512 512 1
    
    
cropland001
echo "cropland001";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_training_croplandimage.sh;

    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo].py 0 15 imagenet1k resize plateau RA 512 512 3

cropland002
echo "cropland002";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_training_croplandimage_002.sh;

    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo].py 15 25 imagenet1k resize plateau RA 512 512 3

cropland003
echo "cropland003";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_training_croplandimage_003.sh;

    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo].py 25 35 imagenet1k resize plateau RA 512 512 3
    
cropland004
echo "cropland004";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_training_croplandimage_004.sh;

    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo].py 35 43 imagenet1k resize plateau RA 512 512 3
"""
   
   
   
   
   
"""
2022-05-12

K Fold CV with EfficientNetV2B3, now the N-round is K-fold!!!! k=5=[0 1 2 3 4]

croplandk0
echo "croplandk0";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_train_croplandimage_k0.sh;

python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 512 512 0
    
croplandk1
echo "croplandk1";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_train_croplandimage_k1.sh;

python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 512 512 1

croplandk2
echo "croplandk2";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_train_croplandimage_k2.sh;

python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 512 512 2

croplandk3
echo "croplandk3";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_train_croplandimage_k3.sh;

python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 512 512 3

croplandk4
echo "croplandk4";cd ~/tf.ds.pipeline/CropLandClassify/; sh install_env.sh; sh run_train_croplandimage_k4.sh;

python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 512 512 4
    
    
"""


"""
$python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 7 imagenet1k crop plateau RA 512 512 TrainSaveDir-1227-toPytest

somehow clear dir fist for save_best_model without cache!


python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 0 22 imagenet1k crop plateau RA 512 512 TrainSaveDir-1228_0-22

$python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 1 2 imagenet1k crop plateau RA 512 512 TrainSaveDir-0124_Xception_keras_plateau_rescale


[2022-02-15]
    1. Add N round fine tune
    2. Using SavedModel rather than keras hdf5
    3. Resize_layer_224 for some tf260 hub model has fixed inputs

    # In case N=3, for 3 repeat round of train
    
    $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 0 8 imagenet1k crop plateau RA 512 512 3
    
        ['InceptionV3', 'InceptionV4', 'ResNet50', 'ResNet101', 'ResNet152', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small']
        
    $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 7 imagenet1k crop plateau RA 512 512 3
    
        ['MobileNet', 'MobileNetV2']
        
    $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 8 imagenet1k crop plateau RA 512 512 3
    
        ['MobileNet', 'MobileNetV2', 'MobileNetV3Small']
    
    
"""


""" 2022-05-16 [ Test tfds.cache to dir/ or to /file, bcs cache in memory in host with big dataset is not good idea. ]


"""



import os
# set log level should be before import tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import cv2
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf




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


# Set if memory growth should be enabled
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    print(f"Set set_memory_growth GPU:{gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)


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
EPOCHS=40 #40 for 768x768 longer #20 # vit50fortest.

figsize=(12, 6) #(25,15) #too big, sometime over the 2^16.
dpi=100 #600 #300 for quickly check, 600up to 800 for journal paper.
"""ValueError: Image size of 4035x172641 pixels is too large. It must be less than 2^16 (~65,536) in each direction.
2022-05-11 maybe is the  x,y value too large when gradient explode .
Let's try EfficientNetB1 for check. dpi=100, figsize=(10,6)
python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo].py 15 16 imagenet1k resize plateau RA 512 512 3
"""



"""#less dp rate, say 0.1, train_loss will lower than val_loss # f
or flood 0.2 is ok. for leaf 0.4 is better. for foot 0.8 is fine."""
top_dropout_rate = 0.4

""" #for efnetBx only This parameter serves as a toggle for extra
regularization in finetuning, but does not affect loaded weights."""
drop_connect_rate = 0.9

"""# classes of 5"""
outputnum = 14 #5

"""save best val_acc model"""
monitor = 'val_accuracy' #'val_loss' 'val_accuracy' if use ed_loss it still the loss here.




# Image size #
BATCH_SIZE = 11 #2 #6 #4 #32#4 #2 # 8# 32 #64 #64:512*8 OOM, B7+bs8:RecvAsync is cancelled



#img_height = 512 #600 #512 #120
#img_width = 512 #600 #512 #120
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

#data_dir = '/home/uu/.keras/datasets/leaf/'
#leaf_dir = '/home/uu/.keras/datasets/leaf/train_images/'
#
#df_train = pd.read_csv(data_dir + '/train.csv')


data_dir = "/home/uu/data/CropLandImage/"
leaf_dir = "/home/uu/data/CropLandImage/img_org/"

#df_train = pd.read_csv(data_dir + '/train_CropLand_label2int_list_.csv') # using label to int.
df_train = pd.read_csv(data_dir + '/train_CropLand_label2int_list_img800x.csv') # using label to int.
#df_train = pd.read_csv(data_dir + '/train_CropLand_label2int_list_img1024x.csv') # using label to int.
# df_train = pd.read_csv(data_dir + '/train_CropLand_label2int_list_img1024x1024.csv') # using label to int.
#df_train = pd.read_csv(data_dir + '/train_CropLand_label2int_list_img1360x1360.csv') # using label to int.
#df_train = pd.read_csv(data_dir + '/train_CropLand_label2int_list_img1800x1800.csv') # using label to int.




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


# Shuffle and reset index #
# fixed shuffle for compare later, random_state=42
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

#df_train = df_train[:2000] # just take 300 element for faster checking.


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
list_ds = tf.data.Dataset.from_tensor_slices((df_train['image_filename'], df_train['label']))

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
#    file_path = leaf_dir + image_id
    file_path = image_id
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


##########
#        #
# K-Fold # 2022-05-12 NEW add for crop land.
#        #
##########
def get_KFold_ds(x, K=0):
        
    k = K
    # may skip twicce to perform kflod
    # train ds
    t_take = x.take(k*val_size)
    t_skip = x.skip(k*val_size+val_size)
    k_train = t_take.concatenate(t_skip)
    # val ds
    v_skip = x.skip(k*val_size)
    k_valid = v_skip.take(val_size)

    return k_train, k_valid

## [DS] Split TVT ##
"""split TVT train/val/test in 7 1.5 1.5"""
val_size = int(tf.data.experimental.cardinality(train_ds_map).numpy() * 0.2)
# val_size = int(tf.data.experimental.cardinality(train_ds_map_toe).numpy() * 0.1)#no help

print("val size:", val_size)


#train_ds_map_s = train_ds_map.skip(val_size)
##temp_s = train_ds_map.take(val_size+val_size)
#
#print("\nvalid_ds == test_ds\n")
#valid_ds_map_s = train_ds_map.take(val_size)
#test_ds_map_s = valid_ds_map_s
##valid_ds_map_s = temp_s.take(val_size)
##test_ds_map_s = temp_s.skip(val_size)
#
#print("total size:", len(train_ds_map))
#print("\ntrain", tf.data.experimental.cardinality(train_ds_map_s).numpy())
#print("valid", tf.data.experimental.cardinality(valid_ds_map_s).numpy())
#print("test", tf.data.experimental.cardinality(test_ds_map_s).numpy())




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
""" 2022-05-16 tf.ds.cache testing. train image 8W org 6k,4k with 180GB, but resize to 1800x1800.

So, it cache a ONE large file. So, the better way is save to TFrecord with few shard. The shard can be randomize and speedup IO than a single large cache file.

time python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 1800 1800 1
    1. cache=False :  K4 [3201.733945131302] of epoch 1, ~54 min. [3033.5913565158844] of epoch 3 no cache effect. (First run seem no disk I/O warmup.)
                   K0 [2945.9191575050354] of epoch 1  [2808.0731234550476] of epoch 3 (other run seems take disk warmup.)
                   K3 [3047.8622081279755] of epoch 1 [2937.5690021514893] of epoch 3
                   K2 [3096.966369867325] of epoch 1 [3008.1399075984955] of epoch 3
                   K1 [2995.913995027542] of epoch 1 [2849.2167665958405] of epoch 3
                   

    2. cache="./tfdscache/" dir:
        52G May 16 10:38 _0.data-00000-of-00001.tempstate16115470749020310010 The size (~582G) will increases until end of training in epoch 1. YES cache ~582G, just 180GB image.
        22  May 16 10:33 _0.lockfile
        -->
            582G May 16 11:19 .data-00000-of-00001
            4.0M May 16 11:19 .index
           
        23088MiB / 32510MiB (use set_memory_growth, V2B3 1800x1800 bs2 GPU memory usage. maybe we can set bs3.)
        [3037.877233028412] of epoch 1
        [2763.240327358246] of epoch 2
        [2741.44286775589] of epoch 3


    3. cache="./tfds.cache" file:

#=================================================================================#

            CropLandImage 80000張, 4k~8k, ~180GB, tf.data.cache 約需15分鐘。

With V2B3_1204x1024_bs11_BS88
    time python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 1024 1024 1
    
    sudo apt-get install htop #Similar to the top but with more information. As you can, it got the command column, which is handy to identify the process path. And also it is colorful.
    
    top  : check the memory usage. with MiB.
    htop : check the memory usage. with color and readable GB.
    atop : can log to file.
    
    0. cache=True :
        host Mem (htop)
            [  376G/754G] after epoch 1.
        GPU Mem (nvidia-smi)
            23084MiB / 32510MiB ~ 32330MiB / 32510MiB (final step when save) almost explode.
        ep time 730/730 steps / epoch
        [1173.9842166900635] of epoch 1 ~20 min
        [864.0467283725739] of epoch 2  ~14.4 min
        [894.4732885360718] of epoch 3
        [873.7898225784302] of epoch 4
        
    1. cache=False :
        host Mem (htop)
            [  55.5G/754G] after epoch 1.
        GPU Mem (nvidia-smi)
            32330MiB / 32510MiB
        ep time 730/730 steps / epoch
            [1163.0222661495209] of epoch 1
            [940.6910030841827] of epoch 2
            [939.5078680515289] of epoch 3  ~15.6 min
            [944.3968679904938] of epoch 4
            
            
            
    2. cache="./tfdscache/" dir: (must remove last .data-00000-of-00001 nor the shape error.)
        "./tfdscache/"
            189G May 16 16:07 .data-00000-of-00001
            4.0M May 16 16:07 .index
        host Mem (htop)
            [  56.5G/754G] after epoch 1.
        GPU Mem (nvidia-smi)
            32330MiB / 32510MiB
        ep time 730/730 steps / epoch
            [1165.618730545044] of epoch 1 cache到檔案比跟完全沒cache快40秒，但比cache memory約慢30~40秒而已。
            [904.7576906681061] of epoch 2
            [902.4967408180237] of epoch 3 ~15 min
            [891.332617521286] of epoch 4
            
    2.2 (tf.ds.shard) cache="./tfdscache/" dir: (tf.data.shard 好像不用預先存成Tfrecord也可以有shard多檔分散)
    
    
    2.3 tf.data.experimental.save(dataset, path) tf.data.experimental.load(path)
            1. tfdsshard_train = "./tfdsshard/train/" 僅產生一個shard 202GB ~15min
            
            2. tfdsshard_train = "./tfdsshard/train/ds_train_shard" 同上 202GB ~15min
            兩種路徑都相同，都會產生sub dir來存放shard/00000000.snapshot

            3. with shard_func: 先用tf.data.shard產生多個 shard，再分別save。目前測試可行。
            分shard存檔非常地慢....每個shard都要15分鐘，跟save整個ds依樣時間，但需重複N次。[應該是最尾端train_ds_pre才分shard，需要把所有的map, shfulle都執行過一遍造成。但改去前端k-Fold就也需要改了。]
            shard=10 每個20GB，相符於是存單一檔的十分之一。Elapsed time is 8789.916376 seconds. ~2.5hrs.
            73/73 [314.8084669113159] of epoch 1 數量不對 少太多 僅load到一個shard的數量
                因為snapshot.metadata 裡面只存放最後一份製作的shard資料夾名稱，這是因為我們預先用tfds.shard做了N份，然而save是需要一次分完所有的element然後給予特定的
                1. 若把所有000000xx.snapshot放進snapshot.metadata 記載的最後一個資料夾中呢？#不行僅讀一份
                
                2. 把#.shard/00000000.snapshot 改成#.shard/0000000#.snapshot #看來路徑被寫死，要真的去改snapshot.metadata才行啊
                
                 (0) Not found:  ./tfdsshard/train/ds_train_shard/6545271800692428971/00000007.shard/00000000.snapshot; No such file or directory

                3.直接改snapshot.metadata
                tensorflow.python.framework.errors_impl.OutOfRangeError: Read less bytes than requested [Op:LoadDataset]
            
            4. 更新shard_func，利用ds.enumerate()增加一個index用來作為shard，繞過label.numpy()的坑。
                dataset = ...
                dataset = dataset.enumerate()
                dataset = dataset.apply(tf.data.experimental.snapshot("/path/to/snapshot/dir",
                    shard_func=lambda x, y: x % NUM_SHARDS, ...))
                dataset = dataset.map(lambda x, y: y) （使用時再把index消掉）
                但，無法取出index
                    shard_func=lambda index, img, label: index % NUM_SHARDS
                    TypeError: <lambda>() missing 1 required positional argument: 'label'
                    因為沒有index所以會少讀到變成第三位的label!!
                    可能是已經做batch或其他map造成吧 若取消這些 #沒用 以list_ds測試依樣沒有index
            
            
            [２０２２－０５－１８ 先停止測試]
                
    3. cache="./tfds.cache" file:



"""

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
        #ds = ds.cache()
        ds = ds.cache("./tfdscache/")
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


def configure_for_performance_cache_split_train_val(ds, cache=True, split=None, augment=None):
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
        #ds = ds.cache()
        if split=='train':
            ds = ds.cache("./tfdscache/train/")
            print(f"Check cache in memory: {cache} {split}" )
        elif split=='val':
            ds = ds.cache("./tfdscache/val/")
            print(f"Check cache in memory: {cache} {split}")
        else:
            print("\n Please set the correct cache [split] for path !!!!! \n")
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


## give augment type to train_ds_pre #
#train_ds_pre = configure_for_performance_cache(train_ds_map_s, cache=True, augment=augment)
#valid_ds_pre = configure_for_performance_cache(valid_ds_map_s)
#test_ds_pre = configure_for_performance_cache(test_ds_map_s)



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
#"InceptionV3",
##"Xception",
#"InceptionV4", # replace X to V4 (a.k.a InceptionResNetV2)
#
## ResNet V2 2,3,4
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
## TF hub Vit [28]
##python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 28 29 imagenet1k resize plateau RA 224 224 TrainSaveDir-2022-01-18_28-29_WCD_None_e20
#
#"vit_b8_fe",
#
#
## Mixer 29 30
#'Mixer-B/16', #MLP-Mixer (Mixer for short)
#'Mixer-L/16',
#
## EA 31
#'EANet',
#
## C-Mixer 32
#'ConvMixer', #
#
#'BiT', # 33 BigTransfer
#]

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

""" 2022-05-12
    We do a trick that using N-round as K-fold to get the best+model+name and then add get_KFold_ds(N-round) to spare k tf.ds.
    So if everything OK we do-not need to rewrite too many code.
    ...
    
    2022-05-12
    K Fold CV with EfficientNetV2B3, now the N-round is K-fold!!!! k=5=[0 1 2 3 4]

    python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold].py 25 26 imagenet1k resize plateau RA 512 512 0
    
"""
count_element=0
        
for model_name in Model_List:
    npydir = log_dir_name + "/" + model_name + "/npy/"
    mk_log_dir(npydir)
    
    #for N in range(n_round):
    print("\n \n N round= ", n_round,  "model_name", model_name,"\n") # [0,1,2,3,4]
    for N in [n_round]:
        print("NNNN", N)
    for N in [n_round]: # just give a one number not the range(). to keep the code structure.
    
        ##########
        #        #
        # K-Fold # 2022-05-12 NEW add for crop land. NOW N == K.
        #        #
        ##########
        train_ds_map_s, valid_ds_map_s = get_KFold_ds(train_ds_map, K=N)

        #configure_for_performance_cache_split_train_val()
        train_ds_pre = configure_for_performance_cache_split_train_val(train_ds_map_s, split='train', augment=augment)
        valid_ds_pre = configure_for_performance_cache_split_train_val(valid_ds_map_s, split='val')
        test_ds_pre = valid_ds_pre
        
#        train_ds_pre = configure_for_performance_cache(train_ds_map_s, augment=augment)
#        valid_ds_pre = configure_for_performance_cache(valid_ds_map_s) #it may cache the same trin_ds so let the val_cc ~100%.
#        test_ds_pre = valid_ds_pre
   
   
   
   
#        ##  Test tf.data.experimental.save with shard : combo tfds.sahrd and save ##
        #train_ds_pre_for_shard = configure_for_performance_cache(train_ds_map_s, augment=augment)
#        train_ds_pre_for_shard = list_ds
        #df_train
#        tfdsshard_train = "./tfdsshard/train/ds_train_shard_enumerate/"
#        print("\n split few shards...")
#        t.tic()
#        NUM_SHARDS = 10
#        shard_0 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=0)
#        shard_1 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=1)
#        shard_2 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=2)
#        shard_3 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=3)
#        shard_4 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=4)
#        shard_5 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=5)
#        shard_6 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=6)
#        shard_7 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=7)
#        shard_8 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=8)
#        shard_9 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=9)
#        #shard_10 = train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=10)
#
#        t.toc()
#
#        shard_list = [shard_0, shard_1, shard_2, shard_3, shard_4, shard_5, shard_6, shard_7, shard_8, shard_9]
        #todo
        """
        shard_list = [train_ds_pre_for_shard.shard(num_shards=NUM_SHARDS, index=index) for index in range(NUM_SHARDS) ]
        """
        
        
#        # Do not run again, it take 2-3 hours to save 15minX10. Just reload the saved shard next time.
#        print("\n save splied shards...")
#        t.tic()
#        for shard_n in range(NUM_SHARDS):
#            def custom_shard_func(img, label):
#                print(f"\n working on the shard num : {shard_n}")
#                return tf.constant(shard_n, dtype=tf.int64) #`shard_func` must return a scalar int64.
#
#            tf.data.experimental.save(
#            shard_list[shard_n], tfdsshard_train, compression=None, shard_func=custom_shard_func
#            )
#        t.toc()
#
#
#        # Do not run again, it take 2-3 hours to save 15minX10. Just reload the saved shard next time.
#        print("\n save split shards...")
#        t.tic()
#
#        train_ds_pre_for_shard = train_ds_pre_for_shard.enumerate()
#
#        def custom_shard_func(num, img, label):
#            #print(f"\n working on the shard num : {shard_n}")
#            #return tf.constant(label, dtype=tf.int64) #`shard_func` must return a scalar int64.
#            shard_n = num % NUM_SHARDS
#            print(f"\n working on the shard num : {shard_n}")
#            return tf.constant(shard_n, dtype=tf.int64)
#
#        tf.data.experimental.save(
#                train_ds_pre_for_shard, tfdsshard_train, compression=None, shard_func=lambda index, img, label: index % NUM_SHARDS
#        )
#        t.toc()
#
#
#        # then re-load the saved cached shards
#        train_ds_pre = tf.data.experimental.load(tfdsshard_train)
        
        
        
#        ##  Test tf.data.experimental.save with shard  ##
#        print('Test tf.ds.save....')
#        t.tic()
#        #tfdsshard_train = "./tfdsshard/train/"
#        tfdsshard_train = "./tfdsshard/train/ds_train_shard"
#
#        NUM_SHARDS = 20
#        #shard_func=lambda x, y:x % NUM_SHARDS
#
#
#        def custom_shard_func(img, label):
#            global count_element
#            count_element += 1
#            ns = count_element % NUM_SHARDS
#            #print(f"\n {count_element}, {ns}, {tf.keras.backend.get_value(label)} \n")
#            print("len of tf.print label", tf.print(label))
#            return 2
#            #tf.keras.backend.get_value(label) #AttributeError: 'Tensor' object has no attribute 'numpy
#            #label = Tensor("args_1:0", shape=(None,), dtype=int64)
#            #tf.get_static_value(label) #None
#            #label.numpy() #AttributeError: 'Tensor' object has no attribute 'numpy'
#            #tf.constant(label.eval(), dtype=tf.int64) #No default session is registered.
#            #tf.constant(label.numpy(), dtype=tf.int64) #AttributeError: 'Tensor' object has no attribute 'numpy'
#            #tf.constant(tf.get_static_value(label), dtype=tf.int64)
#            #tf.constant(ns, dtype=tf.int64) create all ds in shard1.
#
#        tf.data.experimental.save(
#            train_ds_pre, tfdsshard_train, compression=None, shard_func=custom_shard_func
#        )
#
#        exit('Get out the tf ds shard test save....')
#        t.toc()
   
   
        
        print("\n \n K model= ", model_name)
        print("\n \n N round= ", N, "\n") # [0,1,2,3,4]
        #best_model_name = get_best_model_name_bench() # keras hdf5
        best_model_name_dir = get_best_model_name_bench_SavedModel_Nround(N) # SavedModel dir, also with N=5 round
        best_model_name = best_model_name_dir
        
        best_model_save = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_name,
                                     save_best_only = True,
                                     save_weights_only = False,
                                     monitor = monitor,
                                     options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
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
               
        
        # Train K-Model with fine tune #
        print(model_toe.summary()) ## print will added a 'None' of end of summary().
        # fit the model on all data
        hist = model_toe.fit(train_ds_pre,
                              verbose=1, #'auto' or 0 1 2 , 0 = silent, 1 = progress bar, 2 = one line per epoch.
                              epochs=EPOCHS,
                              validation_data=valid_ds_pre,
                              callbacks=callbacks_s2)#, validation_split=0.1)
        # add the epoch timeing
        hist.history['epoch_time_secs'] = callback_lr_time.times
        # for each N round
        n_round_np_name = f'{model_name}_{N}'
        history_toe_finetune[n_round_np_name] = hist.history #hist # what if use hist.history



    
    
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

#        plt.figure(figsize=figsize)
#        """
#        RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
#
#        plt.cla()
#        plt.clf()
#        plt.close()
#         choose add plt.cla() after plt.savefig
#        """
#
#        #for k in Model_List:
#        if n_round_np_name:
#            x = len(hisnp[n_round_np_name]['loss']) + 1
#            x = range(1,x,1)
#            y = hisnp[n_round_np_name]['loss']
#            #y = [f'{z:.4f}' for z in y]
#            plt.plot(x, y, label=f'{n_round_np_name}_loss')
#
#
#            for a,b in zip(x, y):
#                #plt.text(a, b, str(b))
#                plt.scatter(a,b, color='black', alpha=0.2)
#                plt.annotate(f'{b:.3f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#            y = hisnp[n_round_np_name]['val_loss']
#            plt.plot(x, y, label=f'{n_round_np_name}_val_loss')
#
#
#        plt.title('K-model ed loss toe-TL')
#        plt.ylabel('ed loss'), plt.ylim(0, 2)# for too large loss
#        plt.xlabel('epoch')
#        plt.legend(title='Nets')
#
#        # save plot : comment plo.show in jupyter notebook.  dpi=600 is good for journal.
#        #dpi=50 #600 #300 for quickly check, 600up to 800 for journal paper.
#        #pgn=f'{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_loss.png'
#        pgn=f'./{log_dir_name}/{model_name}/npy/{model_name}_{N}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_hisnpTL_loss.png'
#        plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
#        plt.cla()
#        print(f'Save to {pgn} \n')
        """ValueError: Image size of 4035x172641 pixels is too large. It must be less than 2^16 in each direction.
        2022-05-11 maybe is the  x,y value too large when gradient explode .
        Let's try EfficientNetB1 for check. dpi=100, figsize=(10,6) <--- not about fig/plt setting, is x,y!
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo].py 15 16 imagenet1k resize plateau RA 512 512 3
        """


        # naive check #
        #for k in Model_List:
        if n_round_np_name:
            print(f"val_acc of {n_round_np_name}: {hisnp[n_round_np_name]['val_accuracy']}")


        # plotting train log #
        """Rewrite to a function
        plot_save_bench_val_accuracy(Model_List, hisnp, dpi=300, png_name=png)
        """
        plt.figure(figsize=figsize)

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
        #dpi=300
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
        fig, ax1 = plt.subplots(figsize=figsize)
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
        #dpi=300
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
        fig, ax1 = plt.subplots(figsize=figsize) # in inches


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
        #dpi=300
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








print('End time of training: ',t.toc())
t.toc() #
