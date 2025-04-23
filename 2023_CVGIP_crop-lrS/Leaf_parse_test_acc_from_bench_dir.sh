# 2022-02-19
# parse the test ds accuracy from bench round training and save to CSV or tab formate.
# nv2111tf260Zeta
# echo "nv2111testacc"; cd tf.ds.pipeline; sh Leaf_parse_test_acc_from_bench_dir.sh;



#
# hyper setting
#
m_s=0  ###### N round = 10
m_e=50
weight="imagenet1k" # "random" "imagenet1k" "imagenet21k"
crop="crop"         # "crop" "resize" #crop=center crop
lr_name='plateau'       # 'plateau' 'WCD' 'CDR'
augment="RA"      # "None" 'AA' 'RA'


# message on/offf
message_on="False" #"True"  or "False"
install_on="False"

echo "==============================================================="
echo "[Install env]"
date +"%Y-%m-%d %H:%M:%S"
cd ~/tf.ds.pipeline

# Send notify to IM
if [ ${message_on} = "True" ]
then
    sh line_notify_start.sh "[Install env]"
fi

# mute everything : 1>&- 2>&-
if [ ${install_on} = "True" ]
then
    sh install_env.sh #1>&- 2>&-
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
#train_cmd="python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py ${m_s} ${m_e} ${weight} ${crop} ${lr_name} ${augment} 512 512 10"

train_cmd="python3 Leaf_parse_test_acc_from_bench_dir.py ${m_s} ${m_e} ${weight} ${crop} ${lr_name} ${augment} 512 512"
echo " * * Execute --> ${train_cmd} * *"
echo ""
echo ""

# Send notify to IM
if [ ${message_on} = "True" ]
then
    sh line_notify_start.sh "${train_cmd}"
fi

#python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 2 3 imagenet1k crop plateau RA 512 512 TrainSaveDir-0124_ResNet50_hub_plateau_rescale

# run training cmd
eval ${train_cmd}


# Send notify to IM
if [ ${message_on} = "True" ]
then
    sh line_notify_end.sh
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


