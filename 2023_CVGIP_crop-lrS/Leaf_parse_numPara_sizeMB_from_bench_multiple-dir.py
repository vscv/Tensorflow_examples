# 2022-03-10
#
# parse total parameter and model size from bench round training and save to CSV or tab formate.
# this is it.
#
#
#
#
#
# ==============================================================================
"""


time python3 Leaf_parse_numPara_sizeMB_from_bench_multiple-dir.py


save total parameter and model size in to CSV.

"""

import os
import sys
import csv
import pandas as pd
import tensorflow as tf

import glob

from tqdm import tqdm
#from pytictoc import TicToc

from LeafTK import tf_hub_dict

# Get size of SavedMode
from LeafTK import get_dir_size

from pytictoc import TicToc
t = TicToc() #create instance of class
#t.tic() #Start timer


# Model pick up #
"""for Model_List = Model_List[m_start:m_end]"""
#m_start=int(sys.argv[1])
#m_end=int(sys.argv[2])


#
# hyper setting
#


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
BATCH_SIZE = 4 #32#4 #2 # 8# 32 #64 #64:512*8 OOM, B7+bs8:RecvAsync is cancelled
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
MULTI_BATCH_SIZE = BATCH_SIZE * 8 # * strategy.num_replicas_in_sync

#log_dir_name=f'{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}'
#print(f'* * * log_dir_name:\n {log_dir_name} \n')


Model_List=list(tf_hub_dict)
#Model_List=Model_List[0:6]
#Model_List = Model_List[m_start:m_end]
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
                log_dir_name=f'{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}'
                print(f'* * * log_dir_name:\n {log_dir_name} \n')
                log_dir_name_list.append(log_dir_name)

print(log_dir_name_list)


#
# [Models] Train bench models (Fine tune) #
#


#T_ACC_csv=log_dir_name + "/" + "ALL_T_ACC.csv"
T_ACC_csv="ALL_model_parameter_and_seize_round4.csv"

#header=['Models', log_dir_name, "MB"]
header=['Models name']
header.append('Para# (M)')
header.append('mosel size (MB)')
header.extend(log_dir_name_list)
print(header)


with open(T_ACC_csv, 'w', encoding='UTF8', newline='') as csv_f:
    """ Parse the test accuracy from bench models."""
    # CSV
    writer = csv.writer(csv_f)
    # write the header
    writer.writerow(header)

    #
    # parse multiple directory: [logdir_name*]/[model_name]/model_name_test_*_N.vacc
    #
    for model_name in Model_List:
        #for model_name in Model_List:
        #test_acc_list = []

        print("\n \n K model= ", model_name)

        
        t_acc_list=[]
        for log_dir_name in log_dir_name_list:

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
                    print(f"vacc name: {vacc_name}")
                    t_acc=vacc_name.split("_",3)[2]
                    print(f"t_acc: {t_acc}")
                    print(f"t_acc round5: {round(float(t_acc)*100,3)}")
                    # to 100%
                    # round the float as XX.XXXX by t_acc=f'{float(t_acc)*100:.4f}'
                    t_acc=f'{float(t_acc)*100:.4f}'
                    
                    
                    """
                    # With `clear_session()` called at the beginning,
                    # Keras starts with a blank state at each iteration
                    # and memory consumption is constant over time.
                    """
                    tf.keras.backend.clear_session()
                    
            
                    t.tic() #Start timer
                    
                    # tf.keras.models.load_model
                    best_round_t_acc="0" # just take any one of fold to load the model
                    best_model= tf.keras.models.load_model(vacc_file_dir + "/" + best_round_t_acc)
                    
                    spam = t.tocvalue()
                    
                    
                    # Get model parameters
                    stringlist = []
                    best_model.summary(print_fn=lambda x: stringlist.append(x))
                    short_model_summary = "\n".join(stringlist)
                    print(short_model_summary)

                    for line in short_model_summary.split('\n'):
                        if line.startswith('Total params:'):
                            print(line)
                            total_params=int(line.split(" ",2)[2].replace(',' , ''))
                            total_params_M=total_params/1000000
                            print("Total params:", total_params, "<--->", total_params_M , "M", ", int_round0 =", int(round(total_params_M, 0)), "M" )
                    
                    
                    # Get model size
                    model_size=get_dir_size(vacc_file_dir + "/" + best_round_t_acc)
                    model_size_MB=model_size/1024/1024 #int(round(model_size/1024/1024,0))
                    print(f"model_size: {model_size} bytes, {model_size_MB} MB")
                    
                else:
                    print(f"f na")
                    t_acc="f na"
            else:
                print(f"d na")
                t_acc="d na"
            
            #test_acc_list.append(t_acc)
            
        # write the data
        #data=[model_name, t_acc, "MB"]
        data=[model_name]
        data.append(total_params_M)
        data.append(model_size_MB)
        #data.extend(spam)
        data.append(spam)
        writer.writerow(data)


# Check T_ACC results
file = T_ACC_csv
df = pd.read_csv(file)
pd.options.display.max_columns = len(df.columns)
print(df)

print("Note that last item is the spam time sec. of loading the SavedModel which is cold loading time (without warmup). It's a time to load model into memory unconcerned with create the computing graph or the inference dataset. The inference time may refer to time of test test-ds in bs32, image per ms (Millisecond equals 1/1000 of a second).")
print("42 models within plateau_AA_bs32/ take about 15m30s.")








#with open(T_ACC_csv, 'w', encoding='UTF8', newline='') as csv_f:
#    """ Parse the test accuracy from bench models."""
#    # CSV
#    writer = csv.writer(csv_f)
#    # write the header
#    writer.writerow(header)
#
#    #
#    # parse multiple directory: [logdir_name*]/[model_name]/model_name_test_*_N.vacc
#    #
#    for model_name in Model_List:
#        #for model_name in Model_List:
#        test_acc_list = []
#
#        print("\n \n K model= ", model_name)
#
#        vacc_file_dir = get_best_model_name_bench_vacc()
#
#        isdir=os.path.isdir(vacc_file_dir)
#        print(f"isdir:{isdir} {vacc_file_dir}")
#
#
#
#        if isdir:
#    #        vacc_name = [f for f in os.listdir(vacc_file_dir) if f.endswith('.vacc')][0]
#            #for f in os.listdir(vacc_file_dir + "*"):
#            for f in glob.glob(vacc_file_dir + "*.vacc"):
#                print(f)
#                if f.endswith('.vacc'):
#                    vacc_name=f.split("/",3)[3]
#                else:
#                    vacc_name="__"
#
#            isfile=os.path.isfile(vacc_file_dir + vacc_name)
#            print(f"isfile:{isfile} {vacc_name}")
#            if isfile:
#                print(f"vacc name: {vacc_name}")
#                t_acc=vacc_name.split("_",3)[2]
#                print(f"t_acc: {t_acc}")
#                print(f"t_acc round5: {round(float(t_acc)*100,3)}")
#                # to 100%
#                # round the float as XX.XXXX by t_acc=f'{float(t_acc)*100:.4f}'
#                t_acc=f'{float(t_acc)*100:.4f}'
#
#            else:
#                print(f"f na")
#                t_acc="f na"
#        else:
#            print(f"d na")
#            t_acc="d na"
#
#        # write the data
#        data=[model_name, t_acc, "MB"]
#        writer.writerow(data)
#
#
## Check T_ACC results
#file = T_ACC_csv
#df = pd.read_csv(file)
#pd.options.display.max_columns = len(df.columns)
#print(df)









###################################################################################################################################
        
#        else:
#            print(f"tac: NA")
#    else:
#        print(f"tac: NA")
    
#    for N in range(1,n_round,1):
#        print("\n \n K model= ", model_name)
#        print("\n \n N round= ", N, "\n")
#        #best_model_name = get_best_model_name_bench() # keras hd5f
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
#
#        loss, accuracy = evl_model.evaluate(test_ds_pre)
#        test_acc_list.append(accuracy)
#        print("\n accuracy:", accuracy, "\n")
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



#t.toc() #End timer
#print('End time of training: ',t.toc())
#t.toc() #
###################################################################################################################################
