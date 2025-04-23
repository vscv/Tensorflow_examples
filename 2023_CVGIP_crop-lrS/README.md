2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-20230626-CVGIP-lrS.ipynb
2023-06-26

主題：在單一模型下提升辨識效能的嘗試

探討不同LR排程對於提升卷積神經網路在crop疾病辨識的有效性

'投稿CVGIP版本 僅討論不同lr，限定在低功耗裝置，因此僅能用小模型mbnet, efficentNet等，aug僅使用AutoAug一種以示公平。

2023-06-28

原本規劃題目是 單一模型下討論不同的Aug方法對效能的幫助 但要做K-fold 5 x aug 2~4種 訓練時間太久，原本的k-fold沒找到變成要全部重寫。所以改題目，使用先前的數據來寫，遍歷42種模型效能的測試實驗。 Comprehensive Evaluation of 42 State-of-the-Art Deep Neural Networks for Crop Disease Classification Comprehensive Evaluation of State-of-the-Art Deep Neural Networks for Crop Disease Classification


```
# 評比模型

1 InceptionV3
2 InceptionV4
3 ResNet50
4 ResNet101
5 ResNet152
6 MobileNet
7 MobileNetV2
8 MobileNetV3Small
9 MobileNetV3Large
10 DenseNet121
11 DenseNet169
12 DenseNet201
13 NASNetMobile
14 NASNetLarge
15 EfficientNetB0
16 EfficientNetB1
17 EfficientNetB2
18 EfficientNetB3
19 EfficientNetB4
20 EfficientNetB5
21 EfficientNetB6
22 EfficientNetB7
23 EfficientNetV2B0
24 EfficientNetV2B1
25 EfficientNetV2B2
26 EfficientNetV2B3
27 EfficientNetV2S
28 EfficientNetV2M
29 EfficientNetV2L
30 VGG16
31 VGG19
32 ViT-B8
33 ViT-B16
34 ViT-B32
35 ViT-S16
36 ViT-L16
37 Mixer-B16
38 Mixer-L16
39 BiT-S-R50x1
40 BiT-S-R50x3
41 BiT-S-R101x1
42 BiT-S-R101x3
43 ConvMixer-1024-20
44 ConvMixer-768-32
45 ConvMixer-1536-20
```



```
# hyper setting
weight="imagenet1k" #random, maybe just 1k is enough
crop= "crop" #"resize" #crop=center crop
lr_name= 'CDR' #'fixed' # 'CDR' #'WCD' #'plateau'#'lrdump' #"WCD" # WCD, WCDC, lrdump, platrure
augment= 'RA' #None # None, 'AA', 'RA', 'NoisyStudent', 'all'


# hyper models
top_dropout_rate = 0.4 #less dp rate, say 0.1, train_loss will lower than val_loss # for flood 0.2 is ok. for leaf 0.4 is better. for foot 0.8 is fine.
drop_connect_rate = 0.9 #for efnet This parameter serves as a toggle for extra regularization in finetuning, but does not affect loaded weights.
outputnum = 5 # classes of 5


# Image size
BATCH_SIZE = 16 #4 #32#4 #2 # 8# 32 #64 #64:512*8 OOM, B7+bs8:RecvAsync is cancelled
img_height = 384 #512 #600 #512 #120
img_width = 384 #512 #600 #512 #120

patience_1 = 3
patience_2 = 5
```
