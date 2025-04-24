# 2022-02-25
# Sending Mu
# nv2111tf260Mu
# echo "nv2111tf260Mu";sh tf.ds.pipeline/run_taskCCS_PyArgv_mu.sh



#
# hyper setting
#
m_s=24 #36  ###### N round = 5
m_e=25
weight="imagenet1k" # "random" "imagenet1k" "imagenet21k"
crop="crop"         # "crop" "resize" #crop=center crop
lr_name='WCD'       # 'plateau' 'WCD' 'CDR'
augment='AA'      # None 'AA' 'RA'


# message on/offf
message_on="True" #"True"  or "False"
install_on="True"

echo "==============================================================="
echo "[Install env Mu]"
date +"%Y-%m-%d %H:%M:%S"
cd ~/tf.ds.pipeline

# Send notify to IM
if [ ${message_on} = "True" ]
then
    sh line_notify_start.sh "[Install env Mu]"
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

echo " * * Execute --> Mu ${train_cmd} * *"
echo ""
echo ""

# Send notify to IM
if [ ${message_on} = "True" ]
then
    sh line_notify_start.sh "Mu ${train_cmd}"
fi

#python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 2 3 imagenet1k crop plateau RA 512 512 TrainSaveDir-0124_ResNet50_hub_plateau_rescale

# run training cmd
eval ${train_cmd}


# Send notify to IM
if [ ${message_on} = "True" ]
then
    #sh line_notify_end.sh
    sh line_notify_start.sh "[DONE] Mu ${train_cmd}"
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


