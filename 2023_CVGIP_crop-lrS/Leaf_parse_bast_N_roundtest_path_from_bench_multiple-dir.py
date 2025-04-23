# 2022-03-11
#
# Leaf_parse_bast_N_roundtest_path_from_bench_multiple-dir.
# this is it.
#

#
#
# ==============================================================================
"""
python3 Leaf_parse_bast_N_roundtest_path_from_bench_multiple-dir.py


Same as Leaf_parse_test_acc_from_bench_multiple-dir.py but to get the ../model_path/{N}/ and keep as CSV,

(1) Calculate best parameter model from 9 set.
Models          Parameters * 9
InceptionV3     argmax([0:9]) and check the value/position of max acc is match the CSV table.
InceptionV4
ResNet50
ResNet101
.
.
.

(X) First, create a path list of 'best N-round' of every hyper parameters.
Models          PATH
InceptionV3     model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N}
InceptionV4     model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N}
ResNet50        model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N}
ResNet101       model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N} model_path/{N}
.
.
.

(2) then, keep only 'best parameter' accuracy path to convert to dict for ensemble.
model_path = argmax([0:9])
N_round = get_max_acc(model_path)

combination_dict={
"InceptionV3":  "model_path/1/.."  ("..npy/best_pred.npy" <-- postfix is fixed.)
"InceptionV4":  "model_path/4/.."
"ResNet50"   :  "model_path/0/.."
"ResNet101"  "  "model_path/3/.."
.
.
.}
"""

import os
import sys
import csv
import pandas as pd

import glob

from tqdm import tqdm
#from pytictoc import TicToc

from LeafTK import tf_hub_dict


#t = TicToc() #create instance of class
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
#Model_List = Model_List[m_start:m_end]
print(Model_List)

def get_best_model_name_bench_vacc():
    return f'./{log_dir_name}/{model_name}/'
    
    

#
# parse multiple directory: [logdir_name*]/[model_name]/model_name_test_*_N.vacc
#

pretrain_weight=['imagenet'] #['imagenet','imagenet21k',"None"]
crop_list=["crop"] #["crop","resize","None"]
lr_name_list=['plateau','WCD','CDR']
augment_list=['AA','RA',"None"]

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
T_ACC_csv="ALL_T_ACC.csv"

#header=['Models', log_dir_name, "MB"]
header=['Models']
header.append('Para#')
header.append('MB')
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
        test_acc_list = []

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

                else:
                    print(f"f na")
                    t_acc="f na"
            else:
                print(f"d na")
                t_acc="d na"
            
            test_acc_list.append(t_acc)
            
        # write the data
        #data=[model_name, t_acc, "MB"]
        data=[model_name]
        data.append('1M')
        data.append('22')
        data.extend(test_acc_list)
        writer.writerow(data)


# Check T_ACC results
file = T_ACC_csv
df = pd.read_csv(file)
pd.options.display.max_columns = len(df.columns)
print(df)











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
