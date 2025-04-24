# 2021-02-24 updated
# 2021-12-06 run_taskCCS_20211206.sh
# 2021-12-23 env install replace by install_env.sh, use post "1>&- 2>&-" to silence the task-CCS logging.
# 2022-01-14 python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 7 imagenet1k crop plateau RA 512 512 ${savedir} | tee ${savedir}/2022_Leaf_rewrite-for-model-benchmark-forConvert2Py_log
#            to echo ${train_cmd}



#
# hyper setting
#
m_s=6
m_e=7
weight="imagenet1k" # "random" "imagenet1k" "imagenet21k"
crop="crop"         # "crop" "resize" #crop=center crop
lr_name='plateau'       # 'plateau' 'WCD' 'CDR'
augment="RA"      # "None" 'AA' 'RA'


echo "==============================================================="
echo "[Install env]"
date +"%Y-%m-%d %H:%M:%S"
cd ~/tf.ds.pipeline
# Send notify to IM
sh line_notify_start.sh "[Install env]"
# mute everything : 1>&- 2>&-
sh install_env.sh #1>&- 2>&-
date +"%Y-%m-%d %H:%M:%S"
echo "[Install done]"
echo "==============================================================="



# run jupyter notebook on the fly
echo ""
echo ""
echo ""
echo "==============================================================="
echo "###############################################################"
echo "==============================================================="
echo ""
echo "[Start]"
echo ""
date +"%Y-%m-%d %H:%M:%S"
echo ""

cd ~/tf.ds.pipeline
pwd
today=`date +"%Y-%m-%d"`
savedir=`echo "TrainSaveDir-${today}_${m_s}-${m_e}_${lr_name}_${augment}_e20"`
mkdir ${savedir}
echo " * * Save to --> \"${savedir}\" * *"
echo ""

train_cmd="python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py ${m_s} ${m_e} ${weight} ${crop} ${lr_name} ${augment} 512 512 ${savedir} | tee ${savedir}/${savedir}_log.txt"

echo ${train_cmd}


# Send notify to IM
sh line_notify_start.sh "${train_cmd}"

#python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 2 3 imagenet1k crop plateau RA 512 512 TrainSaveDir-0124_ResNet50_hub_plateau_rescale

# run training cmd
eval ${train_cmd}


# Send notify to IM
sh line_notify_end.sh

echo ""
date +"%Y-%m-%d %H:%M:%S"
echo ""
echo ""
echo "[End]"
echo ""
echo "==============================================================="
echo "###############################################################"
echo "==============================================================="


