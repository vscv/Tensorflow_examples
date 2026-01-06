[2023-07-31]
yolov8n.pt 6.23MB
yolov8s.pt 21.5MB
模型     尺寸   mAPval 50-95 (ms) (ms) params(M) FLOPs(B)
YOLOv8n    640    37.3    80.4    0.99    3.2    8.7
YOLOv8s    640    44.9    128.4    1.20    11.2    28.6
YOLOv8m    640    50.2    234.7    1.83    25.9    78.9
YOLOv8l    640    52.9    375.2    2.39    43.7    165.2
YOLOv8x    640    53.9    479.1    3.53    68.2    257.8

                mAPval
RT-DETR-X  640  54.8
RT-DETR-Swin    640 56.2
RT-DETR-FocalNet    640 56.9
""" 槳槳的RT-DETR比yolov8還強大
https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr

"""

安裝在虛擬空間
/home/u3148947/2023-07-31_PaddleDetection
python3 -m venv ppdet
source ppdet/bin/activate

用最新cuda版https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html

python -m pip install paddlepaddle-gpu==2.5.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

#bug
'ImportError: libssl.so.1.1: cannot open shared object file: No such file or directory'
This fixes it (a problem with packaging in 22.04):
wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb


#check
$python -c "import paddle; print(paddle.__version__)"
2.5.1

#check2
$python3
import paddle
paddle.utils.run_check()

    Running verify PaddlePaddle program ...
    I0731 14:47:40.759037  3822 interpretercore.cc:237] New Executor is Running.
    W0731 14:47:40.760149  3822 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 12.1, Runtime API Version: 12.0
    W0731 14:47:40.761473  3822 gpu_resources.cc:149] device: 0, cuDNN Version: 8.9.
    I0731 14:47:41.140825  3822 interpreter_util.cc:518] Standalone Executor is Used.
    PaddlePaddle works well on 1 GPU.
    ======================= Modified FLAGS detected =======================
    FLAGS(name='FLAGS_selected_gpus', current_value='0', default_value='')
    =======================================================================
    I0731 14:47:42.358402  3849 tcp_utils.cc:181] The server starts to listen on IP_ANY:43718
    I0731 14:47:42.358594  3849 tcp_utils.cc:130] Successfully connected to 127.0.0.1:43718
    ======================= Modified FLAGS detected =======================
    FLAGS(name='FLAGS_selected_gpus', current_value='1', default_value='')
    =======================================================================
    I0731 14:47:42.369797  3850 tcp_utils.cc:130] Successfully connected to 127.0.0.1:43718
    W0731 14:47:42.629009  3849 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 12.1, Runtime API Version: 12.0
    W0731 14:47:42.629746  3849 gpu_resources.cc:149] device: 0, cuDNN Version: 8.9.
    W0731 14:47:42.709048  3850 gpu_resources.cc:119] Please NOTE: device: 1, GPU Compute Capability: 7.0, Driver API Version: 12.1, Runtime API Version: 12.0
    W0731 14:47:42.710631  3850 gpu_resources.cc:149] device: 1, cuDNN Version: 8.9.
    I0731 14:47:45.223748  3885 tcp_store.cc:273] receive shutdown event and so quit from MasterDaemon run loop
    PaddlePaddle works well on 2 GPUs.
    PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
    '讚喔

#注意
#如果您希望在多卡环境下使用PaddleDetection，请首先安装NCCL (使用TWCC TF23.5 CCS繞過這個工作！  )


2. 安装PaddleDetection 注意： pip安装方式只支持Python3
    # 克隆PaddleDetection仓库
    cd <path/to/clone/PaddleDetection>
    git clone https://github.com/PaddlePaddle/PaddleDetection.git

    # 安装其他依赖
    cd PaddleDetection
    pip install -r requirements.txt
    pip install numba

    # 编译安装paddledet
    python setup.py install

    #安装后确认测试通过：/home/USER/2023-07-31_PaddleDetection/PaddleDetection/ppdet 不是自裝的虛擬venv喔！!

    python ppdet/modeling/tests/test_architectures.py
    测试通过后会提示如下信息：

    .......
    ----------------------------------------------------------------------
    Ran 7 tests in 12.816s
    OK

#快速体验
PaddleDetection$cd dataset/coco/
coco$python download_coco.py
先下好cococ ds

AttributeError: 'ImageDraw' object has no attribute 'textsize'
This is a known problem caused by a change in Pillow,
python3 -c "import PIL;print(PIL.__version__)"
If it's 10.0, then this is the problem. The solution is to downgrade your copy:

python3 -m pip install Pillow==9.5.0


# 在GPU上预测一张图片
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
#会在output文件夹下生成一个画有预测结果的同名图像。

# model pp cache
~/.cache/PP....
[07/31 15:26:56] ppdet.utils.checkpoint INFO: Finish loading model weights: /home/USER/.cache/paddle/weights/ppyolo_r50vd_dcn_1x_coco.pdparams


# ============================================================ #

"PAIR-LITEON_fisheye_1st_data"
# ~/data/PAIR-LITEON_fisheye_1st_data/

"FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection"
# [todo]

# ============================================================ #



# ============================================================ #
https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md

'准备数据
目前PaddleDetection支持：COCO VOC WiderFace, MOT四种数据格式。

總共有5個YML設定檔要修改
yolov3_mobilenet_v1_roadsign 文件入口

roadsign_voc 主要说明了训练数据和验证数据的路径

runtime.yml 主要说明了公共的运行参数，比如说是否使用GPU、每多少个epoch存储checkpoint等

optimizer_40e.yml 主要说明了学习率和优化器的配置。

ppyolov2_r50vd_dcn.yml 主要说明模型、和主干网络的情况。

ppyolov2_reader.yml 主要说明数据读取器配置，如batch size，并发加载子进程数等，同时包含读取后预处理操作，如resize、数据增强等等
# ============================================================ #
