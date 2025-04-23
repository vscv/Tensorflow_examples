# 2022-02-07
# ß
# the tf_hub_dict is used for model name and its hub url.



#
#
## [Models] ##
#
#
"""
aps: module 'tensorflow.keras.applications' from '/usr/local/lib/python3.8/dist-packages/tensorflow/keras/applications/__init__.py
vim /usr/local/lib/python3.8/dist-packages/tensorflow/keras/applications/__init__.py

可以實現由動態字串變數載入特定的基本模型<但是太多處理>還不如直接表列每項寫出來的清楚簡單!!!!

Model_List = ["Xception", "ResNet50", "ResNet101", "ResNet152", "InceptionV3", "MobileNet", "MobileNetV2",
"DenseNet121","DenseNet169","DenseNet201",
"NASNetMobile","NASNetLarge",
"EfficientNetB0",
"EfficientNetB1", #13
"EfficientNetB3",
"EfficientNetB5", #15
"EfficientNetB7",
]
"""

# TF HUB models
tf_hub_dict = {
    # GoogleNet 0,1
    "InceptionV3":"https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5", #plateau v_a~8.65
    #"Xception":"", # not in the tfhub.
    "InceptionV4":"https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",

    #ResNet V2 ResNet 2,3,4
    "ResNet50":"https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
    "ResNet101":"https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5",
    "ResNet152":"https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5",

    # Mobilenet 56,78
    #""" With with a depth multiplier of 1.0 (100 or 100%) """
    "MobileNet":"https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5",
    "MobileNetV2":"https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
    "MobileNetV3Small":"https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
    "MobileNetV3Large":"https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",

    # Densenet 9,10,11
    "DenseNet121":"",# not in the tfhub.
    "DenseNet169":"",# not in the tfhub.
    "DenseNet201":"",# not in the tfhub.

    # NASNet 12,13
    "NASNetMobile":"https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/5",# hard code of size!224 331! #12 NaN
    "NASNetLarge":"https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/5", # hard code of size!224 331! #12 NaN

#    # PNASNet "12 decoy" "Progressive Neural Architecture Search", from NAS. For this model, the size of the input images is fixed to height x width = 331 x 331 pixels.
#    "PNASNet":"https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/5",
    
    # EfficientNet V1 B0~B7: 14 15 16 17 18 19 20 21, switch to k.apps has better train outcome.
    "EfficientNetB0":"https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", #14 NaN
    "EfficientNetB1":"https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1", #15 NaN
    "EfficientNetB2":"https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1", #16 NaN
    "EfficientNetB3":"https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
    "EfficientNetB4":"https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
    "EfficientNetB5":"https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1", #19
    "EfficientNetB6":"https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1", #20 NaN
    "EfficientNetB7":"https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1", #21

    # EfficientNet V2 B0123, S,M,L: # 22 -> 28
    "EfficientNetV2B0":"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
    "EfficientNetV2B1":"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
    "EfficientNetV2B2":"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
    "EfficientNetV2B3":"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
    "EfficientNetV2S":"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
    "EfficientNetV2M":"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
    "EfficientNetV2L":"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",

    
        
    # VGG 29  30, only exist in keras.apps, and work with tf250 not 260!
    'VGG16':"", #  #at leass twcc21.08
    'VGG19':"", #



    #ViT [31,32,33] [34,35], vit-keras only has b16/32,l16/32.
    'ViT-B8':"https://tfhub.dev/sayakpaul/vit_b8_fe/1", # Vision Transformer with vit-keras, or tf.hub with tf21.11.
    'ViT-B16':"https://tfhub.dev/sayakpaul/vit_b16_fe/1",
    'ViT-B32':"https://tfhub.dev/sayakpaul/vit_b32_fe/1",
    'ViT-S16':"https://tfhub.dev/sayakpaul/vit_s16_fe/1",
    'ViT-L16':"https://tfhub.dev/sayakpaul/vit_l16_fe/1",


#    # EA, (External Attention Transformer) 2021, no hub no apps
#    'EANet':"",
    
    
    # Mixer 36 37 : MLP-Mixer B-16 feature extractor fine-tuned on ImageNet-1k.:
    'Mixer-B16':"https://tfhub.dev/sayakpaul/mixer_b16_i1k_fe/1", #MLP-Mixer (Mixer for short) https://tfhub.dev/sayakpaul/collections/mlp-mixer/1
    'Mixer-L16':"https://tfhub.dev/sayakpaul/mixer_l16_i1k_fe/1",

    # Big Transfer 38 39 40 41 42:s-imagenet1k, m-imagenet21k, l-JTF-300M. R50x1 R50x3 R101x1 R101x3 R152x4
    'BiT-S-R50x1':"https://tfhub.dev/google/bit/s-r50x1/1",
    'BiT-S-R50x3':"https://tfhub.dev/google/bit/s-r50x3/1",
    'BiT-S-R101x1':"https://tfhub.dev/google/bit/s-r101x1/1",
    'BiT-S-R101x3':"https://tfhub.dev/google/bit/s-r101x3/1",
    #'BiT-S-R152x4':"https://tfhub.dev/google/bit/s-r152x4/1", #OOM, else bs2/16
    
    # C-Mixer #https://tfhub.dev/rishit-dagli/collections/convmixer/1
    'ConvMixer-1024-20':"https://tfhub.dev/rishit-dagli/convmixer-1024-20-fe/1", #i1k 76.94 87MB
    'ConvMixer-768-32':"https://tfhub.dev/rishit-dagli/convmixer-768-32-fe/1", #i1k 80.16 75MB
    'ConvMixer-1536-20':"https://tfhub.dev/rishit-dagli/convmixer-1536-20-fe/1", #i1k 81.37 184MB

    }


## check hub dict
#model_name='BiT'
#print(f"[Check tf hub dcit] model:{model_name} hub:{tf_hub_dict[model_name]}")
#print(f"[Check tf hub dcit] model:{model_name} hub:{tf_hub_dict.get(model_name)}")



#2022-03-02 merge all model dict in this file

ensemble_model_dict={
    "InceptionV3":"imagenet_crop_512x512_plateau_RA_bs32/InceptionV3/3/",           #T88.0337

    "ResNet50":"imagenet_crop_512x512_plateau_RA_bs32/ResNet50/1/",                 #87.7220
    "MobileNetV2":"imagenet_crop_512x512_plateau_RA_bs32/MobileNetV2/4/",#87.1299
    "MobileNetV3Small":"imagenet_crop_512x512_plateau_RA_bs32/MobileNetV3Small/1/",    #84.4500
    "DenseNet121":"imagenet_crop_512x512_plateau_RA_bs32/DenseNet121/2/",#88.1271
    "EfficientNetB1":"imagenet_crop_512x512_plateau_RA_bs32/EfficientNetB1/0/",#87.7532
    "EfficientNetB7":"imagenet_crop_512x512_plateau_RA_bs32/EfficientNetB7/3/",#88.1271
    "EfficientNetV2B1":"imagenet_crop_512x512_plateau_RA_bs32/EfficientNetV2B1/1/",#88.3141
    "EfficientNetV2M":"imagenet_crop_512x512_plateau_RA_bs32/EfficientNetV2M/1/",#88.9997
    "VGG16":"imagenet_crop_512x512_plateau_RA_bs32/VGG16/4/",#87.4416


    "ViT-B8":"imagenet_crop_512x512_plateau_RA_bs32/ViT-B8/2/",                     #T89.1243 (No gradient defined for operation 'IdentityN') but stacking is fine.
    "Mixer-B16":"imagenet_crop_512x512_plateau_RA_bs32/Mixer-B16/2/",               #86.1016 (No gradient defined for operation 'IdentityN')


    "BiTSR50x3":"imagenet_crop_512x512_plateau_RA_bs32/BiT-S-R50x3/4/",             #T88.875037
    
    # Extra dataset # 6 cassava disease classes: and need KerasLayer loader.
#    "CropNet":"https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2",
}





"""

from tf_hub_dict import tf_hub_dict
print(tf_hub_dict)

print(list(tf_hub_dict))


$py tf_hub_dict_test.py
{'InceptionV3': 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5', 'Xception': '', 'InceptionV4': 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5', 'ResNet50': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5', 'ResNet101': 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5', 'ResNet152': 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5', 'MobileNet': 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5', 'MobileNetV2': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5', 'MobileNetV3Small': 'https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5', 'MobileNetV3Large': 'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5', 'DenseNet121': '', 'DenseNet169': '', 'DenseNet201': '', 'NASNetMobile': '', 'NASNetLarge': '', 'EfficientNetB0': '', 'EfficientNetB1': '', 'EfficientNetB2': '', 'EfficientNetB3': '', 'EfficientNetB4': '', 'EfficientNetB5': '', 'EfficientNetB6': '', 'EfficientNetB7': '', 'VGG16': '', 'VGG19': '', 'ViT-B16': '', 'ViT-B32': '', 'ViT-L16': '', 'ViT-L32': '', 'vit_b8_fe': '', 'Mixer-B/16': '', 'Mixer-L/16': '', 'EANet': '', 'ConvMixer': '', 'BiT': 'https://tfhub.dev/google/bit/s-r50x1/1'}

['InceptionV3', 'Xception', 'InceptionV4', 'ResNet50', 'ResNet101', 'ResNet152', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'VGG16', 'VGG19', 'ViT-B16', 'ViT-B32', 'ViT-L16', 'ViT-L32', 'vit_b8_fe', 'Mixer-B/16', 'Mixer-L/16', 'EANet', 'ConvMixer', 'BiT']
"""


