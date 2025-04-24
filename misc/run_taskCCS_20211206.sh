# 2021-02-24 updated
# 2021-12-06 run_taskCCS_20211206.sh
# 2021-12-23 env install replace by install_env.sh, use post "1>&- 2>&-" to silence the task-CCS logging.


echo ""
echo "[Install env]"
date +"%Y-%m-%d %H:%M:%S"

cd ~/tf.ds.pipeline
sh install_env.sh 1>&- 2>&-

echo ""
date +"%Y-%m-%d %H:%M:%S"
echo "[Install done]"


# run jupyter notebook on the fly
echo ""
echo ""
echo ""
echo "==============================================================="
echo "###############################################################"
echo "==============================================================="
echo ""
echo ""
echo ""
echo "[Start]"

date +"%Y-%m-%d %H:%M:%S"

cd ~/tf.ds.pipeline
pwd
jupyter nbconvert --to notebook --execute "2022_Leaf_rewrite-for-model-benchmark.ipynb"

date +"%Y-%m-%d %H:%M:%S"
echo "[End]"
