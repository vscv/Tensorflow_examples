#!/usr/bin/env python
# coding: utf-8
# %%


   
   
   
"""
2022-05-13 GET the CSV from merge K folds prediction.

1. remove the N-round training.
2. add CSV merge.
3. mean the CSVs.
4. vote the CSVs.

K Fold CV with EfficientNetV2B3, now the N-round is K-fold!!!! k=5=[0 1 2 3 4]

MULTI_BATCH_SIZE = 128
python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold+CSV].py 25 26 imagenet1k resize plateau RA 512 512 5
***** the TEST image size will resize to 512 too *****
    
    
    
MULTI_BATCH_SIZE = 80
python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold+CSV].py 25 26 imagenet1k resize plateau RA 1024 1024 5
***** the TEST image size will resize to 1024 too *****

ULTI_BATCH_SIZE = 48
python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-for_Cropimages[todo+kfold+CSV].py 25 26 imagenet1k resize plateau RA 1360 1360 5



WahtIf do not resize TEST let model do it in input layer??

"""




import os
# set log level should be before import tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

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
BATCH_SIZE = 16 #4 #32#4 #2 # 8# 32 #64 #64:512*8 OOM, B7+bs8:RecvAsync is cancelled
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
MULTI_BATCH_SIZE = 48 # 80 #128 #BATCH_SIZE * strategy.num_replicas_in_sync
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




CLASSES = ['banana',
         'bareland',
         'carrot',
         'corn',
         'dragonfruit',
         'garlic',
         'guava',
         'peanut',
         'pineapple',
         'pumpkin',
         'rice',
         'soybean',
         'sugarcane',
         'tomato']
         
LABELS = {0: 'banana',
         1: 'bareland',
         2: 'carrot',
         3: 'corn',
         4: 'dragonfruit',
         5: 'garlic',
         6: 'guava',
         7: 'peanut',
         8: 'pineapple',
         9: 'pumpkin',
         10: 'rice',
         11: 'soybean',
         12: 'sugarcane',
         13: 'tomato'}


#data_dir = "/home/uu/data/CropLandImage/"
#leaf_dir = "/home/uu/data/CropLandImage/img_org/"

data_dir = "/home/uu/data/CropLandImage/"
test_dir = "/home/uu/data/CropLandImage/testset/"

#df_train = pd.read_csv(data_dir + '/train_CropLand_label2int_list_.csv') # using label to int.
df_test = pd.read_csv(data_dir + '/submission_example.csv') # using label to int.
#df_test = df_test[:10] # for fastest check predict flow.


# check labels #
for i in range(outputnum):
    print(i, CLASSES[i])

# check labels #
print([(i,l) for i,l in zip(LABELS.keys(), LABELS.values())])

# check labels #
for i,l in zip(LABELS.keys(), LABELS.values()):
    print(i,l)

print(f'*** Total Image of training set: {len(df_test)}')
print(f'*** Fist 5 csv data: \n {df_test[:5]}')



## [DS] Create tf.dataset (DS) ##
# from dataframe
list_ds_test = tf.data.Dataset.from_tensor_slices(df_test['image_filename'])
print(f'\n [ list_ds_test image ]  {len(list_ds_test)} \n')

# create a Python iterator
#it_list_ds_test = iter(list_ds_test) # Make sure iter ds only once.

# using iter and consuming its elements using next: every print different image name.
print(f'* * * Check image_id and label.')
for f in list_ds_test.take(5):
    print(f'take test sample: {f}')


#
# map list to ds.
#
def process_path_label(image_filename):
    file_path = test_dir + image_filename
    #file_path = image_id
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

    return img, image_filename


# Leaf train ds
#train_ds_map = list_ds.map(process_path_label, num_parallel_calls=AUTOTUNE)
test_ds_map = list_ds_test.map(process_path_label, num_parallel_calls=AUTOTUNE)


for img, file_name in test_ds_map.take(5):
    print(f'\n take test_ds_map sample: {img.shape} {file_name}')

#exit('\n * * * exit  * * * \n')



###########
##        #
## K-Fold # 2022-05-12 NEW add for crop land.
##        #
###########
#def get_KFold_ds(x, K=0):
#
#    k = K
#    # may skip twicce to perform kflod
#    # train ds
#    t_take = x.take(k*val_size)
#    t_skip = x.skip(k*val_size+val_size)
#    k_train = t_take.concatenate(t_skip)
#    # val ds
#    v_skip = x.skip(k*val_size)
#    k_valid = v_skip.take(val_size)
#
#    return k_train, k_valid
#
### [DS] Split TVT ##
#"""split TVT train/val/test in 7 1.5 1.5"""
#val_size = int(tf.data.experimental.cardinality(train_ds_map).numpy() * 0.2)
## val_size = int(tf.data.experimental.cardinality(train_ds_map_toe).numpy() * 0.1)#no help
#
#print("val size:", val_size)


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
    #ds = ds.shuffle(buffer_size=MULTI_BATCH_SIZE, reshuffle_each_iteration=True) #buffer_size=MULTI_BATCH_SIZE*2 10sec. # (buffer_size=MULTI_BATCH_SIZE*5) ~10sec,buffer_size=1000 take few sec. or buffer_size=image_count <- take too long # each take ds take 30~45 sec, TODO!!
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
# prepare TEST_ds_pre
#
# TEST CropLandImage ds_pre
test_ds_pre = configure_for_performance_cache(test_ds_map)

#exit('\n * * * exit  * * * \n')


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





# use dict of dict to store hist
history_toe_finetune = {}




#                              #
# [ Load K-models for TTA CSV] #
#                              #

#                              #
# [ Load K-models for TTA CSV] #
#                              #

""" 2022-05-13 rewrite to KFold/TTA CSV generator.
   
"""
""" [2022-05-13]
predict() 返回值是數值，表示樣本屬於每一個類別的概率。[3.7297e-11 1.312e-11 1.0244e-13 1.000e+00...]
    Generates output predictions for the input samples.
    Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.
    自動batch

predict_classes() 返回的是類別的索引，即該樣本所屬的類別標籤。 NO THIS FUN
predict_on_batch() 需手動batch
        
"""

predictions = []

for model_name in Model_List:

    ##########
    #        #
    # K Fold #
    #        #
    ##########
        
    for N in range(n_round):
        print("\n \n N round= ", n_round,  "model_name", model_name,"\n") # [0,1,2,3,4]
        

        
    #for K in [n_round]:
    for K in range(n_round):
        print("K-K-K-K", K)
        
    #for K in [n_round]: # just give a one number not the range(). to keep the code structure.
    for K in range(n_round):
    
            ##########
            #        #
            #  TTA   # TODO: change this cell to TTA.
            #        #
            ##########
            
            #train_ds_map_s, valid_ds_map_s = get_TTA_ds(train_ds_map, K=K)
            #train_ds_pre = configure_for_performance_cache(train_ds_map_s, cache=True, augment=augment)
            #valid_ds_pre = configure_for_performance_cache(valid_ds_map_s)
            #test_ds_pre = valid_ds_pre
   

        best_model_name_dir = get_best_model_name_bench_SavedModel_Nround(K)
        best_model_name = best_model_name_dir
        print(f"\n\n Load k models: {K} {best_model_name}")
        
        t.tic()
        # Loading the model back
        try:
            """ tf250 trained model seems can not be reload with tf260!"""
            best_model_reload = tf.keras.models.load_model(best_model_name)#,custom_objects={'KerasLayer':hub.KerasLayer})
            print("K reload = ", best_model_name)
    #        # eval
    #        loss, accuracy = best_model_reload.evaluate(valid_ds_pre)
    #        print('Valid accuracy :', accuracy)
    #        loss, accuracy = best_model_reload.evaluate(test_ds_pre)
    #        print('Test accuracy :', accuracy)

        except:
            print("\n\n ********** No this kind of model!! ********** \n\n", model_name)
        t.toc()
        

        # Predict
        print(f'Predict.... take long time 7min')
        t.tic()
        pred_k = best_model_reload.predict(test_ds_pre)
        t.toc()
        predictions.append(pred_k)

        #print(f'pred_k {pred_k[0][:]}')



#####################################
#                                   #
# mean or vote the predictions[K=5] #
#                                   #
#####################################

# check predictions
print(f'\n predictions np shape : {np.shape(predictions)}')

# mean the k-predictions
k_predictions = np.mean(predictions, axis=0)
print(f'\n k_predictions(mean) np shape : {np.shape(k_predictions)}')


test_pred_int = np.argmax(k_predictions , axis=-1)
print(f'\n test_pred_int (argmax) np shape : {np.shape(test_pred_int)}')
print(f'test_pred_int[:10] : {test_pred_int}')

test_pred_label = [ LABELS[pred_int] for pred_int in test_pred_int ]
test_pred_label = np.expand_dims(test_pred_label, axis=1)  # #(10,) --> (10, 1)
print(f'\n test_pred_label (class) np shape : {np.shape(test_pred_label)}')
print(f'test_pred_label[:10] : {test_pred_label}')
        
#exit('\n * * * exit  * * * \n')
    
    
    
image_filename_np = np.expand_dims(df_test['image_filename'], axis=1)
print(f'\n image_filename_np : {image_filename_np.shape}')
#(10, 1)
    
predictions_merge = np.append(image_filename_np, test_pred_label, axis=1)#左右接
print(f'\n predictions_merge : {predictions_merge.shape}')


df_submission = pd.DataFrame(predictions_merge)
df_submission.columns = ['image_filename','label']

CSVNAME = "V2B3K5e40lr1e-4_"
submi_name = CSVNAME + log_dir_name +'.csv'

df_submission.to_csv(submi_name, index=False)
print('Save {} as submission CSV.'.format(submi_name))


#exit('\n * * * exit  * * * \n')

#
# [Models] Train bench models (Fine tune) #
#

#
# [Models] Train bench models (Fine tune) #
#


    
#    ###########################################
#    # Test DS accuracy in N rounds per models #
#    ###########################################
#
#    """2022-02-26
#    K model= Mixer-L16
#    N round= 4
#    tensorflow.python.framework.errors_impl.InternalError: Failed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY: out of memory
#    """
#    #for model_name in Model_List:
#    test_acc_list = []
#    for N in range(n_round):
#
#        """
#        # With `clear_session()` called at the beginning,
#        # Keras starts with a blank state at each iteration
#        # and memory consumption is constant over time.
#        """
#        tf.keras.backend.clear_session()
#
#
#        print("\n \n K model= ", model_name)
#        print("\n \n N round= ", N, "\n")
#        #best_model_name = get_best_model_name_bench() # keras hdf5
#
#        best_model_name_dir = get_best_model_name_bench_SavedModel_Nround(N) # SavedModel dir, also with N=5 round
#        best_model_name = best_model_name_dir
#        print("\n \n N round= ", N, best_model_name, "\n")
#
#        # saved_model.load
#        #evl_model = tf.saved_model.load(best_model_name)
#
#        # tf.keras.models.load_model
#        evl_model = tf.keras.models.load_model(best_model_name)
#
#        # verbose 0 = silent, 1 = progress bar.
#        loss, accuracy = evl_model.evaluate(test_ds_pre, verbose=0)
#        test_acc_list.append(accuracy)
#        print("\n Test dataset accuracy:", accuracy, "\n")
#
#
#    # Save test dataset accuracy for each SavedModel #
#
#    b_test_acc_arg = np.argmax(test_acc_list)
#    b_test_acc = test_acc_list[b_test_acc_arg]
#    max_test_acc_name=f'./{log_dir_name}/{model_name}/{model_name}_test_{b_test_acc}_{b_test_acc_arg}.vacc'
#    print("test_acc_list:", test_acc_list)
#    with open(max_test_acc_name, 'w') as f:
#        f.write(str(test_acc_list))
        
    
    

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
