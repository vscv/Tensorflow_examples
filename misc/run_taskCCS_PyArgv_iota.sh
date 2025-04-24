# 2021-02-24 updated
# 2021-12-06 run_taskCCS_20211206.sh
# 2021-12-23 env install replace by install_env.sh, use post "1>&- 2>&-" to silence the task-CCS logging.
# 2022-01-14 python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 7 imagenet1k crop plateau RA 512 512 ${savedir} | tee ${savedir}/2022_Leaf_rewrite-for-model-benchmark-forConvert2Py_log
#            to echo ${train_cmd}

# 2022-02-01
# runtaskccsbench
# sh tf.ds.pipeline/run_taskCCS_PyArgv_20211228.sh

# 2022-02-15
# nv2111tf260iota nv2111tf260kappa nv2111tf260Eta
# echo "nv2111tf260iota";
# sh tf.ds.pipeline/run_taskCCS_PyArgv_iota.sh
#

# 2022-02-17
# Sending iota
# nv2111tf260iota
#echo "nv2111tf260iota";sh tf.ds.pipeline/run_taskCCS_PyArgv_iota.sh



#
# hyper setting
#
m_s=16 #0
m_e=17 #14
weight="imagenet1k" # "random" "imagenet1k" "imagenet21k"
crop="crop"         # "crop" "resize" #crop=center crop
lr_name='WCD'       # 'plateau' 'WCD' 'CDR'
augment="RA"      # "None" 'AA' 'RA'


# message on/offf
message_on="True" #"True"  or "False"
install_on="True"

echo "==============================================================="
echo "[Install env]"
date +"%Y-%m-%d %H:%M:%S"
cd ~/tf.ds.pipeline

# Send notify to IM
if [ ${message_on} = "True" ]
then
    sh line_notify_start.sh "[Install env Iota]"
fi

# mute everything : 1>&- 2>&-
if [ ${install_on} = "True" ]
then
    sh install_env.sh 1>&- 2>&-
fi

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
#pwd
#today=`date +"%Y-%m-%d"`
#savedir=`echo "TrainSaveDir-${today}_${m_s}-${m_e}_${lr_name}_${augment}_e20"`
#mkdir ${savedir}
#echo " * * Save to --> \"${savedir}\" * *"
echo ""

#train_cmd="python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py ${m_s} ${m_e} ${weight} ${crop} ${lr_name} ${augment} 512 512 ${savedir} | tee ${savedir}/${savedir}_log.txt"
# easy way
train_cmd="python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py ${m_s} ${m_e} ${weight} ${crop} ${lr_name} ${augment} 512 512 5"

echo " * * Execute --> Iota ${train_cmd} * *"
echo ""
echo ""

# Send notify to IM
if [ ${message_on} = "True" ]
then
    sh line_notify_start.sh "Iota ${train_cmd}"
fi

#python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 2 3 imagenet1k crop plateau RA 512 512 TrainSaveDir-0124_ResNet50_hub_plateau_rescale

# run training cmd
eval ${train_cmd}


# Send notify to IM
if [ ${message_on} = "True" ]
then
    #sh line_notify_end.sh
    sh line_notify_start.sh "[DONE] Iota ${train_cmd}"
fi

echo ""
date +"%Y-%m-%d %H:%M:%S"
echo ""
echo ""
echo "[End]"
echo ""
echo "==============================================================="
echo "###############################################################"
echo "==============================================================="


