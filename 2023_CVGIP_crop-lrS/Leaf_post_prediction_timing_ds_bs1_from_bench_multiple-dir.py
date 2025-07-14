# 2022-03-10
#
# Timing the inference time with different batch size 32, 16, 8, 4, 1.
# only with plateau AA models, it's not about the accuracy just a timing inference.
#
# ==============================================================================
"""

$time python3 Leaf_post_prediction_timing_ds_bs32_from_bench_multiple-dir.py



total size: 21397

train 14979
valid 3209
test 3209




[each sub-job bs take about ?? hours.]

timingbs32

cd ~/tf.ds.pipeline;
sh install_env.sh 1>&- 2>&-;
time python3 Leaf_post_prediction_timing_ds_bs32_from_bench_multiple-dir.py;

timingbs16

cd ~/tf.ds.pipeline;
sh install_env.sh 1>&- 2>&-;
time python3 Leaf_post_prediction_timing_ds_bs16_from_bench_multiple-dir.py;

timingbs8

cd ~/tf.ds.pipeline;
sh install_env.sh 1>&- 2>&-;
time python3 Leaf_post_prediction_timing_ds_bs8_from_bench_multiple-dir.py;


timingbs4

cd ~/tf.ds.pipeline;
sh install_env.sh 1>&- 2>&-;
time python3 Leaf_post_prediction_timing_ds_bs4_from_bench_multiple-dir.py;

timingbs1

cd ~/tf.ds.pipeline;
sh install_env.sh 1>&- 2>&-;
time python3 Leaf_post_prediction_timing_ds_bs1_from_bench_multiple-dir.py;

"""

import os
# set log level should be before import tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import sys
import csv
import pandas as pd
import tensorflow as tf

import glob
import numpy as np

from tqdm import tqdm
#from pytictoc import TicToc

from LeafTK import tf_hub_dict

from LeafTK import pred_on_batch


# Get size of SavedMode
from LeafTK import get_dir_size


from pytictoc import TicToc
t = TicToc() #create instance of class
#t.tic() #Start timer




#T_ACC_csv=log_dir_name + "/" + "ALL_T_ACC.csv"
T_ACC_csv="ALL_model_timing_bs1.csv"
# Image size #
BATCH_SIZE = 1 #4 #2 # 8# 32 #64 #64:512*8 OOM, B7+bs8:RecvAsync is cancelled


# Model pick up #
"""for Model_List = Model_List[m_start:m_end]"""
#m_start=int(sys.argv[1])
#m_end=int(sys.argv[2])


#
# hyper setting
#

# Model pick up #
"""for Model_List = Model_List[m_start:m_end]"""
#m_start=int(sys.argv[1])
#m_end=int(sys.argv[2])
#
#pretrain_weight=sys.argv[3]
#if pretrain_weight=='imagenet1k':
#    weight="imagenet"
#if pretrain_weight=='imagenet21k':
#    weight="imagenet21k"
#if pretrain_weight=="None":
#    weight=None
    
weight="imagenet"

#crop=sys.argv[4]
#lr_name=sys.argv[5]
#augment=sys.argv[6]
crop="crop"
lr_name="WCD"
augment="AA"

# hyper models #
EPOCHS=2 #20, vit50fortest.


""" #for efnetBx only This parameter serves as a toggle for extra
regularization in finetuning, but does not affect loaded weights."""
drop_connect_rate = 0.9

"""# classes of 5"""
outputnum = 5

"""save best val_acc model"""
monitor = 'val_accuracy' #'val_loss' 'val_accuracy' if use ed_loss it still the loss here.


#img_height = 512 #600 #512 #120
#img_width = 512 #600 #512 #120
img_height = 512 #int(sys.argv[7])
img_width = 512 # int(sys.argv[8])

# N round of fine tune, N=5
n_round = 5 #int(sys.argv[9])
N=n_round

patience_1 = 3
patience_2 = 5


# automatic tuning the pipeline of tf.data #
AUTOTUNE = tf.data.experimental.AUTOTUNE




#pretrain_weight=sys.argv[3]
#if pretrain_weight=='imagenet1k':
#    weight="imagenet"
#if pretrain_weight=='imagenet21k':
#    weight="imagenet21k"
#if pretrain_weight=="None":
#    weight=None


#crop=sys.argv[4]
#lr_name=sys.argv[5]
#augment=sys.argv[6]

## hyper models #
#EPOCHS=2 #20, vit50fortest.
#
#"""#less dp rate, say 0.1, train_loss will lower than val_loss # f
#or flood 0.2 is ok. for leaf 0.4 is better. for foot 0.8 is fine."""
#top_dropout_rate = 0.4
#
#""" #for efnetBx only This parameter serves as a toggle for extra
#regularization in finetuning, but does not affect loaded weights."""
#drop_connect_rate = 0.9
#
#"""# classes of 5"""
#outputnum = 5
#
#"""save best val_acc model"""
#monitor = 'val_accuracy' #'val_loss' 'val_accuracy' if use ed_loss it still the loss here.


# Image size #
#BATCH_SIZE = 32#4 #2 # 8# 32 #64 #64:512*8 OOM, B7+bs8:RecvAsync is cancelled
img_height = 512 #600 #512 #120
img_width = 512 #600 #512 #120
#img_height = int(sys.argv[7])
#img_width = int(sys.argv[8])

# N round of fine tune, N=5
#n_round = int(sys.argv[9])
#N=n_round

#patience_1 = 3
#patience_2 = 5

# batch size for multi gpu #
MULTI_BATCH_SIZE = BATCH_SIZE  # * strategy.num_replicas_in_sync

data_dir = '/home/uu/.keras/datasets/leaf/'
leaf_dir = '/home/uu/.keras/datasets/leaf/train_images/'

df_train = pd.read_csv(data_dir + '/train.csv')


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
test_samples=tf.data.experimental.cardinality(test_ds_map_s).numpy()



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
#train_ds_pre = configure_for_performance_cache(train_ds_map_s, cache=True, augment=augment)
#valid_ds_pre = configure_for_performance_cache(valid_ds_map_s)
test_ds_pre = configure_for_performance_cache(test_ds_map_s)



#log_dir_name=f'{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}'
#print(f'* * * log_dir_name:\n {log_dir_name} \n')


Model_List=list(tf_hub_dict)
#Model_List = Model_List[5:6]
print(Model_List)

def get_best_model_name_bench_vacc():
    return f'./{log_dir_name}/{model_name}/'
    
    

#
# parse multiple directory: [logdir_name*]/[model_name]/model_name_test_*_N.vacc
#

pretrain_weight=['imagenet'] #['imagenet','imagenet21k',"None"]
crop_list=["crop"] #["crop","resize","None"]
lr_name_list=['plateau']
augment_list=['AA']

log_dir_name_list=[]
for weight in pretrain_weight:
    for crop in crop_list:
        for augment in augment_list:
            for lr_name in lr_name_list:
                log_dir_name=f'{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs32'
                # NOTE: for bs 16 8 4 1, we force set to 32 to load the correct dir name.
                print(f'* * * log_dir_name:\n {log_dir_name} \n')
                log_dir_name_list.append(log_dir_name)

print(log_dir_name_list)


#
# [Models] Train bench models (Fine tune) #
#




#header=['Models', log_dir_name, "MB"]
header=['Models name']
header.extend(log_dir_name_list)
header.append('time of inference')

print(header)

with open(T_ACC_csv, 'w', encoding='UTF8', newline='') as csv_f:
    """ Timing the inference of test ds. """
    # CSV
    writer = csv.writer(csv_f)
    # write the header
    writer.writerow(header)
    
    for model_name in Model_List:
        #for model_name in Model_List:
        test_acc_list = []

        print("\n \n K model= ", model_name)

        
        t_acc_list=[]
        for log_dir_name in tqdm(log_dir_name_list):

            vacc_file_dir = get_best_model_name_bench_vacc()

            isdir=os.path.isdir(vacc_file_dir)
            print(f"isdir:{isdir} {vacc_file_dir}")



            if isdir:
        #        vacc_name = [f for f in os.listdir(vacc_file_dir) if f.endswith('.vacc')][0]
                #for f in os.listdir(vacc_file_dir + "*"):
                for f in glob.glob(vacc_file_dir + "*.vacc"):
                    print(f)
                    if f.endswith('.vacc'):
                        vacc_name=f.split("/",3)[3]
                    else:
                        vacc_name="__"

                isfile=os.path.isfile(vacc_file_dir + vacc_name)
                print(f"isfile:{isfile} {vacc_name}")
                if isfile:
                    #print(f"vacc name: {vacc_name}")
                    t_acc=vacc_name.split("_",3)[2]
                    #print(f"t_acc: {t_acc}")
                    print(f"t_acc roundf.4: {float(t_acc)*100:.4f}")
                    # to 100%
                    # round the float as XX.XXXX by t_acc=f'{float(t_acc)*100:.4f}'
                    t_acc=f'{float(t_acc)*100:.4f}'
                    
                    best_round_t_acc=vacc_name.split("_",3)[3]
                    best_round_t_acc=best_round_t_acc.split(".",1)[0]
                    print(f"best_round_t_acc=[ {best_round_t_acc} ]")


                    """
                    # With `clear_session()` called at the beginning,
                    # Keras starts with a blank state at each iteration
                    # and memory consumption is constant over time.
                    """
                    tf.keras.backend.clear_session()
                    

                                        
                    # tf.keras.models.load_model
                    best_model= tf.keras.models.load_model(vacc_file_dir + "/" + best_round_t_acc)
                    
                    
                    # (1) model.evaluate
                    #loss, accuracy = best_model.evaluate(test_ds_pre, verbose=0)# verbose 0 = silent, 1 = progress bar.
                    #print("\n Test dataset accuracy:", accuracy, "\n")


                    # N round timing #
                    spam=[]
                    for N in range(5):
                    
                        t.tic() #Start timer
                        
                        # (2) model..predict_on_batch
                        label_pred_all, label_true_all = pred_on_batch(best_model, test_ds_pre, MULTI_BATCH_SIZE, test_samples)
                        #np.save(vacc_file_dir + "/npy/" + "best_pred.npy", label_pred_all)
                        
                        spam.append(t.tocvalue())
                    
                    
                    
    #                # (2-1) predict by post result (OK, works same as (2) pred_on_batch(), then save to npy for reuse in ensemble.)
    #                acc_count= 0
    #                for i in range(len(label_true_all)):
    #
    #                    try:
    #                        if label_true_all[i] == label_pred_all[i]:
    #                            acc_count += 1
    #                    except IndexError:
    #                        #print("End of batch")
    #
    #                        pass
    #                print("model eval accuracy  = ", t_acc)
    #                print("pred again accuracy  = ", acc_count/test_samples)
    #                np.save(vacc_file_dir + "/npy/" + "best_pred.npy", label_pred_all)
                    
                    
    #                # (2-2) Check reload pred npy
    #                pred_np_reload = np.load(vacc_file_dir + "/npy/" + "best_pred.npy") #, allow_pickle='TRUE').item()
    #                acc_count= 0
    #                for i in range(len(label_true_all)):
    #                    try:
    #                        if label_true_all[i] == pred_np_reload[i]:
    #                            acc_count += 1
    #                    except IndexError:
    #                        #print("End of batch")
    #
    #                        pass
    #                print("pred reload again accuracy  = ", acc_count/test_samples)

                    
                    
    #                # Get model parameters
    #                # save summary to file
    #                # Open the file
    #                with open(vacc_file_dir + "/npy/" + "model_summary.txt",'w') as fh:
    #                    # Pass the file handle in as a lambda function to make it callable
    #                    best_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    #
    #                stringlist = []
    #                best_model.summary(print_fn=lambda x: stringlist.append(x))
    #                short_model_summary = "\n".join(stringlist)
    #                print(short_model_summary)
    #
    #                for line in short_model_summary.split('\n'):
    #                    if line.startswith('Total params:'):
    #                        print(line)
    #                        total_params=int(line.split(" ",2)[2].replace(',' , ''))
    #                        total_params_M=total_params/1000000
    #                        print("Total params:", total_params, "<--->", total_params_M , "M", ", int =", int(total_params_M), "M" )


                    # Get model size
                    #model_size=get_dir_size(vacc_file_dir + "/" + best_round_t_acc)
                    #print(f"model_size: {model_size} bytes, {int(round(model_size/1024/1024,0))} MB")
                    

                else:
                    print(f"f na")
                    t_acc="f na"
            else:
                print(f"d na")
                t_acc="d na"

        # write the data
        #data=[model_name, t_acc, "MB"]
        data=[model_name]
        #data.append(spam)
        data.append(test_samples)
        data.extend(spam)
        writer.writerow(data)
        
        # check spam timing
        print(f"\n{model_name}@{spam}\n")

# Check T_ACC results
file = T_ACC_csv
df = pd.read_csv(file)
pd.options.display.max_columns = len(df.columns)
print(df)

print("The inference time only refer to inference test-ds in bs32 without the loading phase, ms per image (Millisecond equals 1/1000 of a second).")
