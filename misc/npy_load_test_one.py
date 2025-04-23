

import numpy as np

Model_List = ["Xception", "ResNet50", "ResNet101", "ResNet152", "InceptionV3", "MobileNet", "MobileNetV2", # 0-6
 "DenseNet121","DenseNet169","DenseNet201", # 7 8 9
 "NASNetMobile","NASNetLarge", # 10 11 (hard code of size!224 331!)
 "EfficientNetB0", #12
 "EfficientNetB1", #13
 "EfficientNetB3",
 "EfficientNetB5", #15
 "EfficientNetB7", #16
 'VGG16', # 17 #at leass twcc21.11
 'VGG19', # 18
 'ViT_b8', #19 Vision Transformer
 'ViT_s16',
 'EANet',
 'ConvMixer', #
 'BiT', # BigTransfer
]

Model_List = Model_List[:19]


#bench_log_name='TrainSaveDir/ft_bench_imagenet1k_crop_512x512_WCD_RA_bs32_best_val_accuracy有三個模型異常值.npy'
#bench_log_name='TrainSaveDir/ft_bench_imagenet1k_crop_512x512_WCD_RA_bs32_best_val_accuracy.npy'
bench_log_name='imagenet_crop_512x512_plateau_RA_bs32/MobileNet/MobileNet_1_imagenet_crop_512x512_plateau_RA_bs32_best_val_accuracy.npy'


# draft reload his from saved np
#bench_log_name = f'./{log_dir_name}/{th}_bench_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}.npy'
# reload hist from npy
history_np_load = np.load(bench_log_name, allow_pickle='TRUE').item()
hisnp = history_np_load.copy()

#handles = [handle for handle in Model_List]
#print(handles)
print("--")
print(hisnp)
print("--")

#for m in handles:
m=handle="MobileNet_1"
if handle:
    v_a=hisnp[m]['val_accuracy']
    #print(f'{m}:  {v_a} \n')
    print(f'{m}\N{TAB}:\N{TAB}  {np.max(v_a)} @{np.argmax(v_a)}\n')



##check one of m
#m='NASNetLarge'
#v_a=hisnp[m]['val_accuracy']
#print(f'{m}:\N{TAB}  {v_a}\n')
#

#check one of m
#m='InceptionV3'
v_a=hisnp[m]['val_accuracy']
print(f'{m}:\N{TAB}  {v_a}\n')
