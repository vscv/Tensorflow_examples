# From the tensorflow/models/research/ directory
# faster_rcnn_resnet101_kitti2pet
# faster_rcnn_resnet101_coco2pet
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_lsw.config"
MODEL_DIR="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/save_models/faster_rcnn_resnet101_coco2pet/"
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
#model_main.py訓練到evl時即u8中斷




'''
# remove --sample_1_of_n_eval_examples , 沒有影響仍是u8中斷
same Fatal Python error: Aborted
I0103 16:32:01.380961 140573968140096 saver.py:1280] Restoring parameters from /home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/save_models/faster_rcnn_resnet101_coco2pet/model.ckpt-330
after read temp ckpt always fail.
'''
python object_detection/model_main.py \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--model_dir=${MODEL_DIR} \
--num_train_steps=${NUM_TRAIN_STEPS} \
--alsologtostderr

--------------------------------------------------------------------------------------------------------

20200106
#### PEP dataset training
### 使用舊版train.py測試，tf不建議使用此版，但此版有寫分散訓練
#初期會慢很多檢查位動作No whitelist ops found, nothing to do, Converted 0/893 nodes to float16等等
#大約十分鐘後 訓練就正常，兩小時後每global step時間下降到0.3~0.5 sec/step了。前十分鐘後大約一秒一步。
#同樣，tfboard log只記錄前十分鐘左右就停止紀錄了。
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configuring-a-training-pipeline
usage: python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=${PIPELINE_CONFIG_PATH}



#注意model.ckpt-32703的寫法
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=${PIPELINE_CONFIG_PATH}
TRAINED_CKPT_PREFIX=training/model.ckpt-32703
EXPORT_DIR=export_models/
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
    
除了pb模型，另外以輸出一次pipeline.config，原本訓練pipeline的相同但寫法與浮點值進位有差異
I0106 14:17:57.648298 139835459643200 builder_impl.py:421] SavedModel written to: export_models/saved_model/saved_model.pb
I0106 14:17:57.699470 139835459643200 config_util.py:190] Writing pipeline config file to export_models/pipeline.config





2020-01-08 [2020-01-11 fix classes problem at label.pbtxt so retrain again]
#### ivs train public dataset training

#Train
-->faster_rcnn_resnet101_pets_ivs_lsw.config
預設套件
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

預設路徑
2020-01-11 go
#faster_rcnn_resnet101_pets_ivs_lsw.config
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step1500K"

2020-01-11 go
#faster_rcnn_resnet101_pets_ivs_lsw_step-fast_100k.config
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_step-fast_100k.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step100K"


2020-01-11 try another efficeny way
#faster_rcnn_resnet101_pets_ivs_lsw_aug.config
#PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug.config"
#output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step1500K_aug"

2020-01-11 20:50 bs16 lr:0.0001, .00001, .000001 @ 30k 60k endat 90k, aug,
##faster_rcnn_resnet101_pets_ivs_lsw_aug.config
#PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug.config"
#output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step90K_bs16_lr1_aug"
Killed OOM

2020-01-11 20:50 bs16 lr:0.0001, .00001, .000001 @ 30k 60k endat 90k, aug,
##faster_rcnn_resnet101_pets_ivs_lsw_aug.config
#PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug-fast_50k_bs10_lr.3.config"
#output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step50K_bs10_lr.3_aug"
Killed OOM

2020-01-12 14:25 bs:4 lr:0.0003, .00003, .000003 @ 20k 40k endat 50k, aug,
#faster_rcnn_resnet101_pets_ivs_lsw_aug-fast_50k_bs4_lr.3.config
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug-fast_50k_bs4_lr.3.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step50K_bs4_lr.3_aug"

2020-01-12 14:25 bs:6 lr:0.0003, .00003, .000003 @ 20k 40k endat 50k, aug,
#faster_rcnn_resnet101_pets_ivs_lsw_aug-fast_50k_bs4_lr.3.config
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug-fast_50k_bs6_lr.3.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step50K_bs6_lr.3_aug"

2020-01-12 14:25 bs:8 lr:0.0003, .00003, .000003 @ 20k 40k endat 50k, aug,
#faster_rcnn_resnet101_pets_ivs_lsw_aug-fast_50k_bs4_lr.3.config
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug-fast_50k_bs8_lr.3.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step50K_bs8_lr.3_aug"

----

2020-01-15 23:00 bs:4 lr:0.0003, .00003, .000003 @ 200k 400k endat 500k, aug, ->>_ breakoff@30k steps
2020-01-16 09:00 bs:4 lr:0.0003, .00003, .000003 @ 200k 400k endat 500k, aug
#faster_rcnn_resnet101_pets_ivs_lsw_aug_500k_bs4_lr.3.config
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug_500k_bs4_lr.3.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step500K_bs4_lr.3_aug"

2020-01-15 23:00 bs:4 lr:0.0001, .00001, .000001 @ 200k 400k endat 500k, aug, ->> _ breakoff@28k steps
2020-01-16 09:00 bs:4 lr:0.0001, .00001, .000001 @ 200k 400k endat 500k, aug,
#faster_rcnn_resnet101_pets_ivs_lsw_aug_500k_bs4_lr.1.config
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug_500k_bs4_lr.1.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step500K_bs4_lr.1_aug"

bs 4以上均會中斷，唯一次bs4到50k正常train結束


----
2020-01-16 15:40 500k_bs4_lr.1_retry
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_aug_500k_bs4_lr.1_retry.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step500K_bs4_lr.1_aug_retry"
36k中斷，且screen 無法接回，跟網頁開screen相同！！

-----
2020-01-31 23:21 test the multi GPU models
PIPELINE_CONFIG_PATH="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/pipeline/faster_rcnn_resnet101_pets_ivs_lsw_multiGPUs_v1.config"
output_train_dir="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/faster_rcnn_resnet101_pets_ivs_lsw_multiGPUs_v1_step1M"




#sending TRAIN
999999999999 run train 9999999999999999
python train.py --logtostderr --train_dir=${output_train_dir}/ --pipeline_config_path=${PIPELINE_CONFIG_PATH}

#sending TRAIN with nohup, loging to file.
$nohup python train.py --logtostderr --train_dir=${output_train_dir}/ --pipeline_config_path=${PIPELINE_CONFIG_PATH} &> log_step500K_bs4_lr.1_aug_retry.txt &

#sending Multi-GPU TRAIN wiht 1M step
python train.py --logtostderr --train_dir=${output_train_dir}/ --pipeline_config_path=${PIPELINE_CONFIG_PATH} --worker_replicas=2 --num_clones=2 --ps_tasks=1
new batch_size = num_gpu * batch per gpu, 2 for 2 gpu = 1 per gpu!


# watch nohup log file.
$watch -n 2 tail log_step500K_bs4_lr.1_aug_retry.txt


#TensorBoard
tensorboard --logdir=${output_train_dir}/ --port=5000


#export
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=${PIPELINE_CONFIG_PATH}
TRAINED_CKPT_PREFIX="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/training/ivs_frres101_coco_step50K_bs4_lr.3_aug/model.ckpt-50000"
EXPORT_DIR="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/od_comp_2020_01_30/save_models/pb_ivs_frres101_coco_step50K_bs4_lr.3_aug/"

python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
    
save_models/pb_ivs_frres101_coco_step100K//
-rw-r--r--  1 TOPATH   77 Jan  8 15:03 checkpoint
-rw-r--r--  1 TOPATH 182M Jan  8 15:03 frozen_inference_graph.pb
-rw-r--r--  1 TOPATH 239M Jan  8 15:03 model.ckpt.data-00000-of-00001
-rw-r--r--  1 TOPATH  26K Jan  8 15:03 model.ckpt.index
-rw-r--r--  1 TOPATH 2.6M Jan  8 15:03 model.ckpt.meta
-rw-r--r--  1 TOPATH 3.6K Jan  8 15:03 pipeline.config
drwxr-xr-x  3 TOPATH 4.0K Jan  8 15:03 saved_model/



#Eval
# in /work////save_models$ <模型路徑是手動寫在od_infer_dir.py中>改了-m : export_dir路徑
Usage:
$time python od_infer_dir.py -i ../data/ivslab_test_public/JPEGImages/All/ -m pb_faster_rcnn_resnet101_pets_ivs_lsw_ckpt-30000 -s 0.05

$time python od_infer_dir.py -i ../data/ivslab_test_public/JPEGImages/All/ -m pb_ivs_frres101_coco_step100K -s 0.0

--------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------
[2020-01-30] 第一階段資料 mAP
$time python od_infer_dir.py -i ../data/ivslab_test_public/JPEGImages/All/ -m pb_ivs_frres101_coco_step100K -s 0.7



--------------------------------------------------------------------------------------------------------
[2020-01-30] 第二階段資料 mAP
Train stage-2 ivslab_test_qualification
新的測試集總共有3700張，包含第一階段釋放的 1000張 Public 測試資料(ivslab_test_public.tar)，以及2700張 Private 測試資料，請將對這3700張進行推論，答案依照原來上傳的地方進行上傳(上傳 -> 上傳成果)，上傳目前無限制，在第一階段比賽結束前只會公佈 Public 排行榜，等第一階段比賽結束，也就是 1/31 將會公布 Private 排行榜，若還有問題請在討論區發問，或者寄信到admin.AIdea@itri.org.tw

#CSV submission name. --> s2_
csvFileName= "submit_s2_" + model_dir + '_thres' + threshold + '.csv'
#where to save image with bbox.
outPath = "output_s2_" + model_dir + "_thres" + threshold
:save_models$time python od_infer_dir.py -i ../data/ivslab_test_qualification/JPEGImages/All/ -m pb_ivs_frres101_coco_step100K -s 0.0
real    6m46.543s (無繪圖存圖)
user    8m5.943s
sys     1m17.776s
submit_s2_pb_ivs_frres101_coco_step100K_thres0.0.csv 48.9MB
:save_models$time python od_infer_dir.py -i ../data/ivslab_test_qualification/JPEGImages/All/ -m pb_ivs_frres101_coco_step100K -s 0.1
real    7m55.938s (有繪圖存圖)
user    11m36.072s
sys     1m25.968s
submit_s2_pb_ivs_frres101_coco_step100K_thres0.1.csv 1.7MB
:save_models$time python od_infer_dir.py -i ../data/ivslab_test_qualification/JPEGImages/All/ -m pb_ivs_frres101_coco_step100K -s 0.5
real    8m20.458s (有繪圖存圖)
user    11m34.478s
sys     1m26.515s
:save_models$time python od_infer_dir.py -i ../data/ivslab_test_qualification/JPEGImages/All/ -m pb_ivs_frres101_coco_step100K -s 0.05
real    8m28.255s (有繪圖存圖)
user    11m44.072s
sys     1m25.127s
:save_models$time python od_infer_dir.py -i ../data/ivslab_test_qualification/JPEGImages/All/ -m pb_ivs_frres101_coco_step100K -s 0.01
real    6m39.802s
user    8m1.614s
sys     1m19.877s


--------------------------------------------------------------------------------------------------------





--------------------------------------------------------------------------------------------------------

2020-01-07
VOC data 測試 在轉成isv
因為原始範例路徑在home下面太佔空間，為統一放在/work/od_xx/data下，由增加路徑變數去執行。

Generating the Oxford-IIIT Pet TFRecord files.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md

od_path="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/models/research/object_detection/"
${od_path}

# From tensorflow/models/research/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

#J#
od_path="/home/TOPATH/twcc_gpfs/Object_Detection_with_TensorFlow2/models/research/object_detection/"
${od_path}

python ${od_path}/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=${od_path}/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
    
python ${od_path}/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=${od_path}/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record

-rw-r--r-- 1 TOPATH 638M Jan  7 09:57 pascal_train.record
-rw-r--r-- 1 TOPATH 648M Jan  7 09:59 pascal_val.record

--------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------
2020-01-08 轉成isv
# isvlab dataset, in /work/xx/ivslab_train/
cp ivslab_label_map.pbtxt ${od_path}/data/
cp create_ivslab_tf_record.py ${od_path}/dataset_tools/;

time python ${od_path}/dataset_tools/create_ivslab_tf_record.py \
    --label_map_path=${od_path}/data/ivslab_label_map.pbtxt \
    --data_dir=./ --year=All --set=train \
    --output_path=pascal_train.record
    
#上面分行符號不能跑
ivslab_train$cp create_ivslab_tf_record.py ${od_path}/dataset_tools/;time python ${od_path}/dataset_tools/create_ivslab_tf_record.py     --label_map_path=${od_path}/data/ivslab_label_map.pbtxt     --data_dir=./ --year=All    --set=All     --output_path=ivslab_train_public.record

2020-01-08 轉成isv but remove bbox has NaN problem's from All.txt to 'All_rm_boxNan.txt' to All.txt
# 懶得再去改裡面 year set對應路徑問題，直接複製新的All.txt to 'All_rm_boxNan.txt'所以指令全部不用更改！！！！！
ivslab_train$cp create_ivslab_tf_record.py ${od_path}/dataset_tools/;python ${od_path}/dataset_tools/create_ivslab_tf_record.py     --label_map_path=${od_path}/data/ivslab_label_map.pbtxt     --data_dir=./ --year=All    --set=All     --output_path=ivslab_train_public.record

--------------------------------------------------------------------------------------------------------
