#
# Leaf ToolKits
#

# [2021-12-27]
#

import tensorflow as tf
import tensorflow_hub as hub
import importlib
import numpy as np
import time
import os


from tqdm import tqdm
from datetime import datetime

from tf_hub_dict import tf_hub_dict, ensemble_model_dict

#EPOCHS=20 # only set to run.py

# [WCD] StepWise warmup cosine decay learning rate [optimazer] for some models need very small lr.
# [plateau] initial lr is setting in the tf.keras.optimizers.Adam(0.0001) not 0.00001 1e-5
# keep same as WCD and CDR.
INIT_LR = 0.00001 #1e-5
WAMRUP_LR = 0.0000001 #1e-7
WARMUP_STEPS = 5



#
#
## [Models] ##
#
#


###################################
#                                 #
#                                 #
#  tf HUB build as the K.Layers   #
#                                 #
#                                 #
###################################

def build_tf_hub_models(model_name, outputnum, img_height, img_width):
    """ Move keras.applications to tf hub pre-trained models.
    
    th hub seems expected to have color values in the range [0,1]!
    
    inception_v3.preprocess_input will scale input pixels between -1 and 1. ???? not the 0 an 1???
    
    [2022-01-26]
        add customize tf.keras.layers.Rescaling(1./255) convert 0,255 to 0,1.
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling
    
    [ rescale layer for pixel value ]
    tf.keras.layers.Rescaling(scale, offset=0.0, **kwargs)
    For instance:
        To rescale an input in the [0, 255] range to be in the [0, 1] range, you would pass scale=1./255.

        To rescale an input in the [0, 255] range to be in the [-1, 1] range, you would pass scale=1./127.5, offset=-1.
        
    #  resize layer: only exist with and after tf260 #
    #        resize_layer = tf.keras.layers.Resizing(224, 224)
    #        resize = resize_layer(inputs)
    #        rescaling_input=scale_layer(resize)
    """
    
    # move input layer to the Top
    inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="input_first") #shape=(120, 120, 3), img_height, img_width shape=(img_height, img_width, 3)
    
    # input with [0,1] for tf hub only, otherwise use models.preprocess_input(inputs)
    scale_layer=tf.keras.layers.Rescaling(1./255)
    
    # [-1, 1]
    scale_layer_2=tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
   
    # Resize layers: for some model has fixed input size.
    resize_layer_224 = tf.keras.layers.Resizing(224, 224)
    resize_layer_331 = tf.keras.layers.Resizing(331, 331)
   
   
    # InceptionV3 # 0
    if model_name.startswith('InceptionV3'):
        # Load model into KerasLayer
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0-1
        rescaling_input=scale_layer(inputs)

    # InceptionV4 # 1
    if model_name.startswith('InceptionV4'):
        """ [2022-02-07] Replace Xception by InceptionV4 (a.k.a InceptionResNetV2)
        
        Xception was published after InceptionV3 and InceptionV4, so the performance is slightly higher than V3 but the model size is similar (V3 92MB, Xception 88MB). But the paper does not compare with V4 (215MB), it is cause that the V4 model is already twice as large.
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 1 2 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_InceptionV4_hub_crop_plateau
        
        # InceptionV4 crop 512 512 bs32 gpux1 #
        [498.8356809616089] of epoch 1 to 0.83422
        [399.1370131969452] of epoch 2 to 0.84637
        [409.3563859462738] of epoch 3 to 0.85946
        
        # InceptionV4 crop 512 512 bs256 gpux8 # largest bs
        [391.20265674591064] of epoch 1 to 0.75039
        [62.11274170875549] of epoch 2 to 0.81147
        ...
        [60.1460440158844] of epoch 6 to 0.85728
        
        # InceptionV4 crop 512 512 bs32 gpux8 # more gpu the initial time more longer (~4min.).
        [418.08811116218567] of epoch 1 to 0.82113
        [132.8441607952118] of epoch 2 to 0.85136
        [142.58988881111145] of epoch 3 to 0.85198
        ...
        [129.73995447158813] of epoch 6 to 0.86600
        """
        # Load model into KerasLayer
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0-1
        rescaling_input=scale_layer(inputs)
    
    
    # Xception # 1 [maybe move to the laster order eg. 43]
    if model_name.startswith('Xception'):
        """
        When run in "tf" mode it actually expect the input to be uint8 between 0 and 255 and scales it to the range from -1.0 to 1.0.
        Check the docstring and the source code.""" #NOT TRUE
        """ For Xception, call tf.keras.applications.xception.preprocess_input on your inputs before passing them to the model.
        xception.preprocess_input will scale input pixels between -1 and 1.
        
        
        [2022-01-24]
            Note: current tf260 hub not include Xception.
                  current tf260 with keras.app h5: TypeError: Layer tf.nn.convolution was passed non-JSON-serializable arguments.
                  
                 build_efn_model() or build_tf_hub_models() seems has same tf260 issue! but both work with tf250. And tf250 very quick to acc 0.75 in 1 epoch, v_acc 0.8709878325462341 Epoch@P12.
        
        [2022-01-25]
            tf260 can not save keras Xception model (.h5). TypeError: 'Not JSON Serializable:'
            
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 1 2 imagenet1k crop plateau RA 512 512 TrainSaveDir-0124_Xception_keras_plateau_rescale
            
            build_efn_model() or build_tf_hub_models() seems has same tf260 issue! but both work with tf250.
            and tf250 very quick to acc 0.75 in 1 epoch, v_acc 0.8709878325462341 Epoch@P12
            
            may be a solution: save_weights_only=True may will work fine? https://stackoverflow.com/questions/68319579/tfa-optimizers-multioptimizer-typeerror-not-json-serializable
            
            
        [TODO] use tf260 k.apps + SaveModel, works for other fixed inputs and json serial error issue.
        
        
        
        """
        # Load keras applications
        BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.xception"), model_name)
        # For go around!!
        weight='imagenet'
        base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        # input with -1.0 ~ 1.0
        rescaling_input = tf.keras.applications.xception.preprocess_input(inputs)
        

    # ResNet V2: ResNet50V2 ResNet101V2 ResNet152V2 # 2,3,4
    if model_name.startswith('ResNet'):
        """
        
        From keras application:
        Model    Size (MB)    Top-1 Accuracy
        ResNet50    98    0.749
        ResNet101    171    0.764
        ResNet152    232    0.766
        ResNet50V2    98    0.760
        ResNet101V2    171    0.772
        ResNet152V2    232    0.780
        
        However, in re-trained th.hub
        https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5
        https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5
        https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5
        save models as
        ResNet50V2  283MB
        ResNet101V2 512MB
        ResNet152V2 701MB
        

        
        * ResNet V2 The key difference compared to ResNet V1 is the use of batch normalization before every weight layer.
        
        * The input images are expected to have color values in the range [0,1], following the common image input conventions. The expected size of the input images is height x width = 224 x 224 pixels by default, but other input sizes are possible (within limits).
        
        * Can not use resnet.preprocess_input, it will convert the input images from RGB to BGR, then will zero-center each color channel. However, use scale_layer() works!
        
        
        # All ResNet is ResNet V2 #
        
        #ResNet50 crop 512 512 bs32 gpu1
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 2 3 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_ResNet50_hub_crop_plateau
        [270.4963104724884] of epoch 1 to 0.82331 val_accuracy
        [214.477201461792] of epoch 2 to 0.83920
        [229.24656748771667] of epoch 3 to 0.85385
        [217.6792070865631] of epoch 4 to 0.85541
        
        #ResNet101 crop 512 512 bs32 gpu1
        [359.4189579486847] of epoch 1 to 0.83297 val_accuracy
        [306.66890120506287] of epoch 2 to 0.84762
        [331.3009672164917] of epoch 3 from 0.84762
        [305.9286458492279] of epoch 4 0.86444
        
        #ResNet152 crop 512 512 bs32 gpu1
        [540.2421028614044] of epoch 1 to 0.81334
        [450.7571494579315] of epoch 2 to 0.83328
        [455.8971347808838] of epoch 3 to 0.85665
        [448.9527268409729] of epoch 4 from 0.85665
        ...
        [450.3447439670563] of epoch 14 to 0.86912
        """
        # Load model into KerasLayer
        print(f'hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with -1,1
#        rescaling_input = tf.keras.applications.resnet.preprocess_input(inputs)

        # input with 0,1
        rescaling_input=scale_layer(inputs)
        
        
        
        
    # HUB MobileNet # v1 5
    if model_name.endswith('MobileNet'):
        """ For HUB MobileNet V1,
        
        (tfhub260)
        B0:
            For this module, the size of the input image is flexible, but it would be best to match the model training input, which is height x width = 224 x 224 pixels for this model. The input images are expected to have color values in the range [0,1], following the common image input conventions.
            
        (keras.apps)The input images are expected to have color values in the range [0,1], following the common image input conventions. The expected size of the input images is height x width = 224 x 224 pixels by default, but other input sizes are possible (within limits).
        
        #MobileNet V1 crop 512 512 bs32 gpu1
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 5 6 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_MobileNet_hub_crop_plateau
        [278.9528772830963] of epoch 1 to 0.80524
        [203.69077324867249] of epoch 2 to 0.81801
        [218.37954449653625] of epoch 3 to 0.84107
        [209.13313031196594] of epoch 4 to 0.84512
        [204.52017521858215] of epoch 5 to 0.85728
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0,1
        rescaling_input=scale_layer(inputs)
        
        
    # HUB MobileNetV2 # v2 6
    if model_name.startswith('MobileNetV2'):
        """ For HUB MobileNetV2, "***** fixed *****" to height x width = 224 x 224 pixels.
        
        The input images are expected to have color values in the range [0,1], following the common image input conventions. For this model, the size of the input images is "***** fixed *****" to height x width = 224 x 224 pixels.
        
        The momentum (a.k.a. decay coefficient) of batch norm's exponential moving averages defaults to 0.99 for this model, in order to accelerate training on small datasets (or with huge batch sizes). Advanced users can set another value (say, 0.997) by loading this model like
                
                
                
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 6 7 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_MobileNetV2_hub_crop_plateau
        
        [When input is not the 224x224 wtih tf260 hub.]
            ValueError: Could not find matching function to call loaded from the SavedModel. Got:
              Positional arguments (4 total):
                * Tensor("inputs:0", shape=(None, 512, 512, 3), dtype=float32)
        
        [When input is not the 224x224, however, k.apps.mbnetv1 can train with different size. but with 224 weight.]
                
        [max val_acc]  crop_224x224_plateau_RA_bs32 may too small ROI of target!
          MobileNetV2-------------------------:
               0.8242443203926086 Epoch@P17
               
        [max val_acc]  resize_224x224_plateau_RA_bs32  may lose lot of detail/resolution!
          MobileNetV2-------------------------:
               0.8161420822143555 Epoch@P13
                maybe fist crop to 600x600, then resize to 224 will be better.
        
        [keras] switch to build_efn_model()
        What if use Keras.apps.mbnetV2
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 6 7 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_MobileNetV2_keras_crop_plateau
        [258.2454090118408] of epoch 1 to 0.72920
        [211.49663257598877] of epoch 2 to 0.78623
        [219.79729437828064] of epoch 3 to 0.82362
        [206.5822296142578] of epoch 4 to 0.83297
        
        [211.5231373310089] of epoch 16 from 0.85042
        [max val_acc]
          MobileNetV2-------------------------:
           0.8510439395904541 Epoch@P16 (actually is epoch 17:Epoch 00017: val_accuracy improved from 0.85042 to 0.85104,)
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0,1
        rescaling_input=scale_layer(inputs)
 
 
    # HUB MobileNetV3Small  # V3 7 8 switch to keras
    if model_name.startswith('MobileNetV3Small'):
        """ HUB MobileNet V3 (all use input images sized 224x224)
        The input images are expected to have color values in the range [0,1], following the common image input conventions. For this model, the size of the input images is fixed to height x width = 224 x 224 pixels.

        crop_224x224_plateau_RA_bs32
            loss: nan
            
        
        [260 hub: fixed size 224x224,] 2022-02-08
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 7 8 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_MobileNetV3Small_hub_crop_plateau
        
        
            ValueError: Could not find matching function to call loaded from the SavedModel. Got:
                  Positional arguments (4 total):
                    * Tensor("inputs:0", shape=(None, 512, 512, 3), dtype=float32)
                    * True
                    * False
                    * 0.997
                  Keyword arguments: {}
                  
                  
        [keras] switch to build_efn_model() [change within 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py]

        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
            # input with 0,1
        rescaling_input=scale_layer(inputs)

    # HUB MobileNetV3Large  # switch to keras
    if model_name.startswith('MobileNetV3Large'):
        """ HUB MobileNet V3 (all use input images sized 224x224)

        crop_224x224_plateau_RA_bs32
            loss: nan

        set rescaling_input=inputs, seem get out the nana but still stocking.
            [max val_acc]
              MobileNetV3Large-------------------------:
                           0.46556559205055237 Epoch@P5
        set rescaling_input=tf.keras.applications.mobilenet_v3.preprocess_input
            loss: nan
            
        [260 hub: fixed size 224x224,] 2022-02-08
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 8 9 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_MobileNetV3Large_hub_crop_plateau
        
        [keras] switch to build_efn_model() [change within 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py]

        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0,1
#        rescaling_input=scale_layer(inputs)
#        rescaling_input=inputs
#        rescaling_input=tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    
    
    # HUB NASNet: NASNetMobile NASNetLarge # 12,13
    if model_name.startswith('NASNetMobile'):
        """
        * * the important warnings seem is the reason of learn fail: * *
        CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
            warnings.warn('Custom mask layers require a config and must override '
        https://github.com/tensorflow/tensorflow/issues/52978
        * * It may be fixed in the upcoming TF2.8 in near future. * *
        """
        
        """
        The input images are expected to have color values in the range [0,1], following the common image input conventions. For this model, the size of the input images is fixed to height x width = 224 x 224 pixels.
        
            hard code of size!224 331! #12 NaN
            
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_NASNetMobile_hub_crop_plateau
        
        (260 hub) hard code of size!224 331! #12 NaN
            NASNet tf260 h5但json serial error.

        250 keras: 0.6989716291427612 Epoch@P0
            其他ep 0.6183 不變
            因為是硬改載入h5

        250 keras: 不手動載入h5 則loss: nan



        GOTO: rescale, resize layer僅在tf260之後有，改用260測試。
        
        260 hub:  * TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='inputs') fixed input 224!!
        
        260 hub + scale + resize: 1/255, 224x224
            Model: "NASNetMobile"
            _________________________________________________________________
            Layer (type)                 Output Shape              Param #
            =================================================================
            input_first (InputLayer)     [(None, 512, 512, 3)]     0
            _________________________________________________________________
            resizing (Resizing)          (None, 224, 224, 3)       0
            _________________________________________________________________
            rescaling (Rescaling)        (None, 224, 224, 3)       0
            _________________________________________________________________
            NASNetMobile (KerasLayer)    (None, 1056)              4269716
            _________________________________________________________________
            top_output (Dense)           (None, 5)                 5285
            =================================================================
            Total params: 4,275,001
            Trainable params: 4,238,263
            Non-trainable params: 36,738
            _________________________________________________________________
            
            loss : nan 無法訓練, take long initial ~5mnin
             
            換WCD試試
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop WCD RA 512 512 TrainSaveDir-2022-02-07_NASNetMobile_hub_crop_WCD
            also loss : nan 無法訓練, take long initial ~5mnin
            
            
            change ds 224x224 inputs
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop plateau RA 224 224 TrainSaveDir-2022-02-07_NASNetMobile_hub_crop_plateau
                Model: "NASNetMobile"
                _________________________________________________________________
                Layer (type)                 Output Shape              Param #
                =================================================================
                input_first (InputLayer)     [(None, 224, 224, 3)]     0
                _________________________________________________________________
                resizing (Resizing)          (None, 224, 224, 3)       0
                _________________________________________________________________
                rescaling (Rescaling)        (None, 224, 224, 3)       0
                _________________________________________________________________
                NASNetMobile (KerasLayer)    (None, 1056)              4269716
                _________________________________________________________________
                top_output (Dense)           (None, 5)                 5285
                =================================================================
                Total params: 4,275,001
                Trainable params: 4,238,263
                Non-trainable params: 36,738
                _________________________________________________________________
             also loss : nan 無法訓練, take long initial ~5mnin
                
                
             change ds 224x224 inputs, then remove resizing its no longer need.
                 Model: "NASNetMobile"
                _________________________________________________________________
                Layer (type)                 Output Shape              Param #
                =================================================================
                input_first (InputLayer)     [(None, 224, 224, 3)]     0
                _________________________________________________________________
                rescaling (Rescaling)        (None, 224, 224, 3)       0
                _________________________________________________________________
                NASNetMobile (KerasLayer)    (None, 1056)              4269716
                _________________________________________________________________
                top_output (Dense)           (None, 5)                 5285
                =================================================================
                Total params: 4,275,001
                Trainable params: 4,238,263
                Non-trainable params: 36,738
                _________________________________________________________________
              also loss : nan 無法訓練, take long initial ~5mnin
              
            260 + hub + ds resize224 + rescale:
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k resize plateau RA 224 224 TrainSaveDir-2022-02-07_NASNetMobile_hub_resize_plateau
                loss: nan
            
            260+hub + ds resize224: no rescale->rescaling_input = inputs
                loss: nan
            
            
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k resize WCD AA 224 224 TrainSaveDir-2022-02-07_NASNetMobile_hub_resize_WCD
                loss: nan
                
                
                
            [2022-02-16]
            [2022-02-23] 260 + HUB + SavedModel + (resize224+rescale01)
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop plateau RA 512 512 1
            nan
            [2022-02-23] 260 + HUB + SavedModel + (resize224+rescale01) + nasnet.preprocess_input
            nan
            no luck! skip to "qubvel"'s keras version!
           
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        #resize_layer_224 + scale [0,1]
        resize_input = resize_layer_224(inputs)
        rescaling_input = scale_layer(resize_input)
        #rescaling_input = tf.keras.applications.nasnet.preprocess_input(resize_input)
        
#        # Load model into KerasLayer
#        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
##        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True, arguments=dict(batch_norm_momentum=0.997))
#        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
#        base_model.build([None, 512, 512, 3])
#
##        # input with 0,1 and resize to 224x224
##        ##inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="input_first")
##        resize_layer = tf.keras.layers.Resizing(224, 224)
##        resize = resize_layer(inputs)
##        rescaling_input=scale_layer(resize)
#
#        # input with 0,1
#        rescaling_input = scale_layer(inputs)

#        # input no change
#        rescaling_input = inputs


    # HUB NASNet: NASNetLarge # 12,13
    if model_name.startswith('NASNetLarge'):
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        #resize_layer_224 + scale [0,1]
        resize_input = resize_layer_224(inputs)
        rescaling_input = scale_layer(resize_input)




    # HUB PNASNet "12 decoy" "Progressive Neural Architecture Search", from NAS.
    if model_name.startswith('PNASNet'):
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        
        
        
        
    # HUB EfficientNet V1 : EfficientNetB0 - B7 # 14 -> 21
    if model_name.startswith('EfficientNetB'):
        """
        EfficientNetB0    14
        EfficientNetB1    15
        EfficientNetB2    16
        EfficientNetB3    17
        EfficientNetB4    18
        EfficientNetB5    19
        EfficientNetB6    20
        EfficientNetB7    21


        We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use AutoML MNAS Mobile framework to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to EfficientNet-B7.


        For this module, the size of the input image is flexible, but it would be best to match the model training input, which is height x width = 224 x 224 pixels for this model. The input images are expected to have color values in the range [0,1], following the common image input conventions.
        
        EfficientNetB0_
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 14 15 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_EfficientNetB0_hub_crop_plateau
        [207.55354189872742] of epoch 1 to 0.76566
        [56.49730658531189] of epoch 2 to 0.82767
        [61.58566331863403] of epoch 3 to 0.83640
        [53.250446796417236] of epoch 4 to 0.85229
        
        EfficientNetB7_
        Total params: 64,110,485
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 21 22 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_EfficientNetB7_hub_crop_plateau
        [657.0217218399048] of epoch 1 to 0.84606
        [159.93874382972717] of epoch 2 from 0.84606
        [193.79346656799316] of epoch 3 from 0.84606 loss: nan
        ... loss: nan
        
        [2022-02-16]
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 21 22 imagenet1k crop plateau RA 512 512 1
        plateau 1e-5,
                from -inf to 0.00062 val_loss: nan seem too small lr?
                
        plateau 1e-3,
                from -inf to 0.00031    val_loss: nan
        """
        
        
        """
        2022-02-20
            Call trainable=True, when hub.klayer() and remove 1. base_model.trainable = True 2. training=False in finial build.

            *** We forget put the trainable=True when call hub.keraslayer, its different base_model.trainable=True!!!!! ***å
                from -inf to 0.00031 , seem not this issue~~
            
            
             $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 14 15 imagenet1k crop plateau RA 512 512 1
             from -inf to 0.25678
             Aborted (core dumped) Epoch 3/20
             
             
             Call hub.klayer(trainable=True) and WITH (1. base_model.trainable = True 2. training=False) in finial build.
                Aborted (core dumped) Epoch 3/20
                
            What if switch to k.apps?
                python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 14 15 imagenet1k crop plateau RA 512 512 1
                
                
        
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0,1
        rescaling_input = scale_layer(inputs)
        
    
    # HUB EfficientNet V2 : # 22 -> 28
    if model_name.startswith('EfficientNetV2'):
        """trained on imagenet-21k (Full ImageNet, Fall 2011 release).
        
        [2022-02-21 Update] EfficientNetV2 with tf260 + HUB work well. Somehow no input size warning.
        
        
        [2022-02-17 14:49:18] notify command "nv2111tf260kappa"
        * * Execute --> python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 15 31 imagenet1k crop plateau RA 512 512 5 * * [EfficientNetV2 is fine, EfficientNetV1 all switch to k.aps and retrain at 02-21.]
        
            $python3 Leaf_parse_test_acc_from_bench_dir.py 0 44 imagenet1k crop plateau RA 512 512
            ...
            22  EfficientNetV2B0                               87.3792  MB
            23  EfficientNetV2B1                               88.3141  MB
            24  EfficientNetV2B2                               88.0648  MB
            25  EfficientNetV2B3                               88.1895  MB
            26   EfficientNetV2S                               88.5323  MB
            27   EfficientNetV2M                               88.9997  MB
            28   EfficientNetV2L                               88.8127  MB
            29             VGG16                               87.4416  MB
            30             VGG19                               87.8467  MB
            ...
        
        
        [below test before 02-??]
        EfficientNetV2B0    22 with input size 224x224 Total params: 5,925,717
            For this model, the size of the input images is fixed to height x width = 224 x 224 pixels.
        EfficientNetV2B1    23 with input size 240x240, Total params: 6,937,529
            For this model, the size of the input images is fixed to height x width = 240 x 240 pixels.
        EfficientNetV2B2    24 with input size 260x260, Total params: 8,776,419
            For this model, the size of the input images is fixed to height x width = 260 x 260 pixels.
        EfficientNetV2B3    25 with input size 300x300, Total params: 12,938,307
            For this model, the size of the input images is fixed to height x width = 300 x 300 pixels.
                
            *[V2B0-3 seems to change the input size, so we use one of it with input 512s. not really,  it outcome different size of model!!]
        
        EfficientNetV2S    26  Total params: 20,337,765
            For this model, the size of the input images is fixed to height x width = 384 x 384 pixels.
        EfficientNetV2M    27 Total params: 53,156,793
            For this model, the size of the input images is fixed to height x width = 480 x 480 pixels.
        EfficientNetV2L    28 Total params: 117,753,253
            For this model, the size of the input images is fixed to height x width = 480 x 480 pixels.
        
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 22 23 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2B0_hub_crop_plateau
        [179.765882730484] of epoch 1 to 0.80212
        [53.02258658409119] of epoch 2 from 0.80212
        [73.8576922416687] of epoch 3 from 0.80212 loss: nan
        [53.339800119400024] of epoch 4 from 0.80212 loss: nan
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 23 24 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2B1_hub_crop_plateau
        [199.76409220695496] of epoch 1 to 0.00031 loss: nan
        loss: nan
            V2B1-2:
                [207.2085530757904] of epoch 1 to 0.00031
            V2B1-3:
                [202.06375694274902] of epoch 1 to 0.78872
                [54.44618558883667] of epoch 2 loss: nan
                
                
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 24 25 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2B2_hub_crop_plateau
        [196.12987852096558] of epoch 1 to 0.82798
        [55.208704710006714] of epoch 2 to 0.85291
            Number of REPLICAS: 7
            BATCH_SIZE: 4, MULTI_BATCH_SIZE: 28
                Epoch 2/20
                [2022-02-12 18:32:31.769122] Learning rate for epoch 2 is 9.999999747378752e-06
                535/535 [==============================] - 54s 100ms/step - loss: 0.5158 - accuracy: 0.8184 - val_loss: 0.4189 - val_accuracy: 0.8529
            Re-train again, bcs, gpu only 7 in training. Its very unstable situation on TWCC.
        
            [214.7973027229309] of epoch 1 to 0.81147
            [55.016533851623535] of epoch 2 to 0.84637
            [61.88282585144043] of epoch 3 from 0.84637 val_loss: nan
            [54.67004990577698] of epoch 4 from 0.84637 loss: nan
            loss: nan
            
            T2:
                [215.79860424995422] of epoch 1 to 0.82611
                loss: nan
            T3:
                [205.81049394607544] of epoch 1 to 0.82113
                [56.78717875480652] of epoch 2 to 0.84917
                loss: nan
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 25 26 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2B3_hub_crop_plateau
        [245.55664086341858] of epoch 1 to 0.00031 val_loss: nan
        [64.5748188495636] of epoch 2 loss: nan
        
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 26 27 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2S_hub_crop_plateau
        [296.83906173706055] of epoch 1 to 0.00000, val_loss: nan
        loss: nan
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 26 27 imagenet1k crop WCD RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2S_hub_crop_WCD
        loss: nan
        
        # Change input to EfficientNetV2S's 384x384 #
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 26 27 imagenet1k crop plateau RA 384 384 TrainSaveDir-2022-02-12_EfficientNetV2S_hub_crop_plateau
        [275.60606384277344] of epoch 1 to 0.83141
        [68.85362148284912] of epoch 2 from 0.83141 val_loss: nan
        loss: nan
        
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 27 28 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2M_hub_crop_plateau
        [444.48643684387207] of epoch 1 to 0.85977
        [122.38302206993103] of epoch 2 from 0.85977 val_loss: nan
        loss: nan
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 28 29 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-12_EfficientNetV2L_hub_crop_plateau
        [614.0859096050262] of epoch 1 to 0.86008
        [193.26175022125244] of epoch 2 val_loss: nan
        loss: nan
        
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0,1
        rescaling_input = scale_layer(inputs)
    


    # HUB VGG16 VGG19, 29  30
    if model_name.startswith('VGG16'):
        """ VGG only exist in keras.apps, not in the 260 hub. but 260 keras 'Not JSON Serializable:'. So SavedModel instead.
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with 0,1
        rescaling_input = scale_layer(inputs)
    



    # HUB ViT    ViT-B16
#     # ViT # tf.hub version [Waiting for twcc update CCS image for version 21.11]
#     if model_name.startswith('ViT'):
#         """Inputs to the model must:
#             1.be four dimensional Tensors of the shape (batch_size, height, width, num_channels). Note that the model expects images with channels_last property. num_channels must be 3.
#             2.be resized to 224x224 resolution.
#             3.have pixel values in the range [-1, 1].
#         """
#         """ValueError: Unknown SavedObject type: None
#         but work in wth tf2.6.0, tf2.7.0

#         """
#         if model_name.startswith('ViT_b8'):
#             handle="https://tfhub.dev/sayakpaul/vit_b8_fe/1"
#         if model_name.startswith('ViT_s16'):
#             handle="https://tfhub.dev/sayakpaul/vit_s16_fe/1"
#         num_classes=5
        
#         # ViT model as a layer
#         hub_layer = hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_b8_fe/1", trainable=True)
#         model = tf.keras.Sequential([
#                                         inputs,
#                                         hub_layer,
#                                         keras.layers.Dense(num_classes, activation="softmax"),
#                                         ])

    # HUB ViT , 31 32 33 34 35
    if model_name.startswith('ViT-'):
        """https://tfhub.dev/sayakpaul/collections/vision_transformer/1
        
        S16 https://tfhub.dev/sayakpaul/vit_s16_fe/1
        
        
        Inputs to the model must:
             1.be four dimensional Tensors of the shape (batch_size, height, width, num_channels). Note that the model expects images with channels_last property. num_channels must be 3.
             2.be resized to 224x224 resolution. [All hub ViTf use 224!!!!]
             3.have pixel values in the range [-1, 1].


        ViT-B8 Total params: 85,811,717
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 31 32 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-14_ViT-B8_hub_crop_plateau
        
            ValueError: Could not find matching function to call loaded from the SavedModel.
                * Tensor("inputs:0", shape=(None, 512, 512, 3), dtype=float32)
            Expected these arguments to match one of the following 1 option(s):
                * TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='inputs')
            
            
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 31 32 imagenet1k crop plateau RA 224 224 TrainSaveDir-2022-02-14_ViT-B8_hub_crop_plateau
            
            1. ValueError: Unable to create dataset (name already exists)
            2.
            Epoch 1/20
            [2022-02-14 14:45:52.471684] Learning rate for epoch 1 is 9.999999747378752e-06
            terminate called after throwing an instance of 'std::bad_alloc'
              what():  std::bad_alloc
              
              https://stackoverflow.com/questions/58647973/why-i-am-getting-terminate-called-after-throwing-an-instance-of-stdbad-alloc
              The error implies that your running out memory.

                special considerations:

                In some instances, it could be caused by memory fragmentation in that you actually have enough memory to service your task but because it's not contiguous, it can't be used.

                OR

                a process is allocated a large memory portion that some is left unoccupied and cannot be used by another process.

                Running tf.reset_default_graph() between training may help to free up memory in-case fragmentation is the real issue.
                The keras.clear_session() method is an alternative to tf.reset_default_graph() that may help free memory in-case of fragmentation.
                
              3. Segmentation fault (core dumped) [with scale_layer_2(inputs)]
              4. ValueError: Unable to create dataset (name already exists)
              
              5. * what if turn off "best_model_save" callback.
                    { it seems train well but can not save to h5, but SavedModel is ok. currently no callback for SavedModel save the best model.}
                  [313.9420425891876] of epoch 1 val_accuracy: 0.8068
                  [181.13868165016174] of epoch 2 val_accuracy: 0.8529 max for only 3 epochs
                  [196.69073128700256] of epoch 3 val_accuracy: 0.8411
                  
                  update: callback best_model_save can save by /xx/dir/ as the SavedModel**
              
              
        ViT-B32 to save testing time
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 33 34 imagenet1k crop plateau RA 224 224 TrainSaveDir-2022-02-14_ViT-B32_hub_crop_plateau
        
            1. * what if "best_model_save" callback with save_weights_only = True,. then model.load_weights(checkpoint_filepath).
            
                ValueError: Unable to create dataset (name already exists)
                [still can not save to h5 weight?]
            2. callback with save_weights_only = True, and filepath without h5,
                    filepath="./TrainSaveDir-2022-02-14_ViT-B32_hub_crop_plateau/b32/
                    will save a empty checkpoint file in /B32/
                    
            3. callback with save_weights_only = False, and filepath without h5,
                it is OK to save a SavedModel folder.
                [TODO] Test SavedModel reload and test_accuracy. [DONE it works.]
            
        """
        
        """[2022-02-21] redo check tf260+hub, 1.crop 224 , 2. crop 512 + resize224 (slightly better)
        
            1.crop 224:
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 33 34 imagenet1k crop plateau RA 224 224 1
            ViT-B32_0 :          0.855406641960144 Epoch@P15
            
            2. crop 512 + resize224:
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 33 34 imagenet1k crop plateau RA 512 512 1 (input size warming 224)
                512 + resize_layer_224 + scale [-1,1]
                T1:Aborted (core dumped)
                T2:0.8582 Epoch@P10
                
                
        """
        
        
        """ [TODO] base_model.load_weights() to bypass the fixed input shape?!
        
        Or weight=None, model.build(512,512,3) train from scratch but lr seem different?
        
        """
        
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        
      
#        # input with **[-1, 1]** but we just lazy use [0, 1]
##        rescaling_input = scale_layer(inputs)
#        # input with **[-1, 1]
#        rescaling_input = scale_layer_2(inputs)

        #resize_layer_224 + scale [-1,1]
        resize_input = resize_layer_224(inputs)
        rescaling_input = scale_layer_2(resize_input)
       
       
       
    # HUB MLP-Mixer, MixerB16 MixerL16 : 36 37
    if model_name.startswith('Mixer-'):
        """ 2022-02-22
        
            https://tfhub.dev/sayakpaul/collections/mlp-mixer/1
            
            ImageNet-1k fine-tuned:This model is an MLP-Mixer of type B-16 [1] pre-trained on the ImageNet-21k dataset [2] and fine-tuned on the ImageNet-1k dataset [2].
            Inputs to the model must:
                1. (batch_size, height, width, num_channels)
                2. be resized to 224x224 resolution.
                3. have pixel values in the range [-1, 1].
                
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 36 37 imagenet1k crop plateau RA 512 512 1
            [ok train]
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        #resize_layer_224 + scale [-1,1]
        resize_input = resize_layer_224(inputs)
        rescaling_input = scale_layer_2(resize_input)
    
    
    # EA, (External Attention Transformer)
    
    
    
    # HUB ConvMixer 38 39 40 [ConvMixer-1024/20 ConvMixer-768/32 ConvMixer-1536/20]
    if model_name.startswith('ConvMixer'):
        """ConvMixer is a simple model that uses only standard convolutions to achieve the mixing steps. Despite it's simplicity ConvMixer outperforms ViT and MLP-Mixer [3].
        
        The input images are expected to have color values in the range [0,1], following the common image input conventions. The expected size of the input images is height x width = 224 x 224 pixels by default in the defult channels last format. This outputs an array of size [-1, 1024] representing the pooled features.
        Notes
            Due to the signatures of the SavedModel, you should always follow the hub.KerasLayer layer with a tf.keras.layers.Reshape((1024,)) layer like shown in the above usage example. This allows you to place a Dense or any other layers for fine-tuning you might want to add.
            The outputs are already the pooled features, you would not want to add a pooling layer after this.
            
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 38 39 imagenet1k crop plateau RA 512 512 1
        
         * * * hub: https://tfhub.dev/rishit-dagli/convmixer-1024-20-fe/1
            Set ConvMixer-1024-20 with Reshape((1024,) and dense
            ERROR:absl:hub.KerasLayer is trainable but has zero trainable weights.
         or
            ValueError: Setting hub.KerasLayer.trainable = True is unsupported when calling a SavedModel signature.
            both trainable=True or base_model.trainable = True
            
         and remove signature and trainable=True
             
             ERROR:absl:hub.KerasLayer is trainable but has zero trainable weights.
             [2022-02-22] currently ConvMixer seem not workable for finetune in hub. Pull an issue in GitHub waiting the author's reply.
             
        """
        # Load model into KerasLayer
        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
#                                     signature="serving_default", output_key="output")

        #resize_layer_224 + scale [0,1]
        resize_input = resize_layer_224(inputs)
        rescaling_input = scale_layer(resize_input)
    
    
    
    # HUB BiT # 33
    if model_name.startswith('BiT'):
        """
        [2022-02-24]
        BiT is not a new network, it is using super-huge dataset to pre-train model. # BigTransfer : s-imagenet1k, m-imagenet21k, l-JTF-300M.
        
        https://tfhub.dev/google/collections/bit/1
        
        Big Transfer (BiT) is a recipe for pre-training image classification models on large supervised datasets and efficiently fine-tuning them on any given target task. The recipe achieves excellent performance on a wide variety of tasks, even when using very few labeled examples from the target dataset.

        A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly and N. Houlsby: Big Transfer (BiT): General Visual Representation Learning.
        
        In the paper, three families of models are presented: "BiT-S", pre-trained on ImageNet-1k (also known as ILSRCV-2012-CLS); "BiT-M", pre-trained on ImageNet-21k (also known as the "Full ImageNet, Fall 2011 release"); and "BiT-L", pre-trained on JFT-300M, a proprietary dataset. This collection contains the BiT-S and BiT-M families.

        Each family is composed of a ResNet-50 (R50x1), a ResNet-50 three times wider (R50x3), a ResNet-101 (R101x1), a ResNet-101 three times wider (R101x3), and our flagship architecture, a ResNet-152 four times wider (R152x4). Contrary to the original ResNet architecture, we used Group Normalization instead of Batch Normalization, and Weight Standardization of the convolution kernels.
        
        
        # former test #
        BiT hub need special go around.
        Note that it is important to initialize the new head to all zeros.
        
        The BiT example, from_logits=True and dense witout the activation is because the laster layer (dense) is without the softmax or sigmoid!!
        logits 層通常產生從 -infinity 到 +infinity 的值，softmax 層將其轉換為從 0 到 1 的值。 如果dense已經softmax把logits轉成0,1了，那loss時就要為from_logits=False了喔。
        
        # kernel_initializer='zeros' seems no effect!
        # using base_model=hub.KerasLayer(, trainable=True) is equal to base_model.trainable = True.
        """
        
        """ [2022-02-25]
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 38 43 imagenet1k crop plateau RA 512 512 1
            
            BiT-S-R50x1: e1 to 0.85104
            
            BiT-S-R101x3: e1 to 0.87223
            
            BiT-S-R152x4: (0) Resource exhausted:  Out of memory while trying to allocate 18178349440 bytes.
                BATCH_SIZE = 4
                [Do NOT work!! however reduce bs size may work for this case after all other model is trained.]
                # Set if memory growth should be enabled
                gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                BATCH_SIZE = 2, can train
                BiT-S-R152x4: BS2, e1 to 0.87005 [1810.997568845749] of epoch 1
                
        """
        # Load model into KerasLayer
        base_model = hub.KerasLayer(tf_hub_dict[model_name], name = model_name, trainable=True)
        # input with same 0-256
        rescaling_input = inputs
    

#    if not from_hub:
#        do we need bn, dropout, GAP on the top layer????



    #####################
    # HUB model compile #
    #####################
    
    
    # todo: remove hub's Xception, MBnet V2 V3. to keras.apps.
    if model_name in ["Xception", "", ""]:
        """# For some model still use keras.applications #"""
        print(f"* * * Get keras app model * * *")
        
        # Freeze the pretrained weights
        base_model.trainable = True #False #skip the TL so it should be change to True. and remove the free_model()
        print("base_model.trainable : ", base_model.trainable)
        print(f'Set other models')

        # move to Top
        # How to add training=False in base_model create
        #inputs = tf.keras.Input(shape=(120, 120, 3))
        #rescal = rescaling_input()(inputs)
        #b_m_output = base_model(inputs, training=False)

        b_m_output = base_model(rescaling_input, training=False)

        # Rebuild top
        gap2d = tf.keras.layers.GlobalAveragePooling2D()(b_m_output) #(base_model.output)
        #BNL = tf.keras.layers.BatchNormalization()(gap2d) #tood: remove#
        
        # for go around!!
        top_dropout_rate=0.4
        
        dropout = tf.keras.layers.Dropout(top_dropout_rate)(gap2d)#tood: remove# J add dropout, for flood 0.2 is ok. for leaf 0.4 is better. for foot 0.8 is fine.
        #outputs = tf.keras.layers.Dense(outputnum)(dropout)# remove activation for regression output (to default, the linear), , activation = 'relu' no help
        outputs = tf.keras.layers.Dense(outputnum, activation="softmax")(dropout)#todo: activation="softmax", default is "linear" activation: a(x) = x


        # Compile new model
        model = tf.keras.Model(inputs, outputs, name=model_name)

    
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), #0.0001 1e-4 #RMSprop , Adam, SGD Adadelta(learning_rate=0.001), if set lr_callback the learning_rate=0.001 will not effeced.
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['accuracy'])
                 
#    # no longer use, switch to k.apps
#    elif model_name in ["MobileNetV3Small", "MobileNetV3Large", ""]:
#        print(f'\n * * * hub: {tf_hub_dict[model_name]}')
#        base_model = hub.KerasLayer(tf_hub_dict[model_name], trainable=True, arguments=dict(batch_norm_momentum=0.997), name = model_name)
#
#        model=tf.keras.Sequential([
#            #tf.keras.Input(shape=(img_height, img_width, 3), name="input_first"),
#            tf.keras.layers.Rescaling(1./255),
#            base_model,
#            tf.keras.layers.Dense(outputnum, activation="softmax", name="top_output")
#            ])
#        model.build([None, 224, 224, 3])  # Batch input shape.
#        # Compile new model
#        model.compile(
#            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
#            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
#            metrics=['accuracy']
#            )

    elif model_name.startswith('ConvMixer'):
        """ConvMixer always follow the hub.KerasLayer layer with a tf.keras.layers.Reshape((1024,))
        """
        #convmixer_reshape = tf.keras.layers.Reshape((1024,))
        
        print(f'Set {model_name} with Reshape((1024,) and dense\n\n')
        base_model.trainable = True
        b_m_output = base_model(rescaling_input)
        
        #reshape_out = convmixer_reshape()()
#        reshape_output = tf.keras.layers.Reshape((1024,))(b_m_output)
        mix_reshape_layer = tf.keras.layers.Reshape((1024,))
        reshape_output = mix_reshape_layer(b_m_output)
        
        
        # Dense output
        outputs = tf.keras.layers.Dense(outputnum, activation="softmax", name="top_output")(reshape_output)
        
        # build new model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        # Compile new model
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
            )
    else:
        """# Otherwise use tf hub layers #"""
        print(f"* * * Get tf hub layers * * *")
        # Build TF HUB LAYERS #
        print(f'Set {model_name}')
        
        base_model.trainable = True #[TODO] use train=True in hublayer instead
        b_m_output = base_model(rescaling_input, training=False) #[TODO] no need for hub models
#        print(f'Remove base_model.trainable = True and base_model(rescaling_input, training=False')
#        b_m_output = base_model()(rescaling_input) # this is wrong call.
#        b_m_output = base_model(rescaling_input)
        
        # Dense output
        outputs = tf.keras.layers.Dense(outputnum, activation="softmax", name="top_output")(b_m_output)
        
        # build new model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        # Compile new model
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
            )
    
    return model, base_model
        
        
        
        
###################################
#                                 #
#                                 #
# Keras.applications build model  #
#                                 #
#                                 #
###################################
    
def build_efn_model(weight, model_name, outputnum, img_height, img_width, top_dropout_rate, drop_connect_rate):
    """ Formal keras.applications models."""
    
    
    # move to Top
    inputs = tf.keras.Input(shape=(img_height, img_width, 3)) #shape=(120, 120, 3), img_height, img_width shape=(img_height, img_width, 3)
    
    # Resize layers: for some model has fixed input size.
    resize_layer_224 = tf.keras.layers.Resizing(224, 224)
    resize_layer_331 = tf.keras.layers.Resizing(331, 331)
    
    
    # EfficientNetB@# #
#     # OK efn
#     if model_name.startswith('EfficientNetB'):# == "EfficientNetB0":
#         root_m_name = 'efficientnet'
#         fullnameofmodel = "tensorflow.keras.applications." + root_m_name #model_name #model_name.lower()
#         model = importlib.import_module(fullnameofmodel)
#         BaseCnn = getattr(model,model_name)
#         base_model = BaseCnn(include_top=False, weights="imagenet", input_shape=(120,120,3),drop_connect_rate=drop_connect_rate) #{'imagenet', None}
    # shorter version of OK efn
    if model_name.startswith('EfficientNetB0'):
        """For EfficientNet, input preprocessing is included as part of the model (as a Rescaling layer)."""
        
        """
        [2022-02-20] hub efnet usually nan after epoch1. so switch back to k.apps.
        
        EfficientNetB0
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 14 15 imagenet1k crop plateau RA 512 512 1
            [84.45566749572754] of epoch 6 from 0.86102 to 0.86600
            seems is ok to learn. so switch all efnet to k.apps.
            
        EfficientNetB1
            $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 15 16 imagenet1k crop plateau RA 512 512 1
            [update 2022-02-21]
            In tf260.k.apps, efnets also head into best v_acc or loss: nan at epoch 1 in the ratio of 0/5 ~ 2/5 rounds, but k.apps probability of event is less than tf260 hub. However, v_acc is still unstable e.g.
            
            * [keras.apps.efnet issue]
                1. warnings.warn('Custom mask layers require a config and must override ' (fixed after tf280)
                2. WARNING:absl:Importing a function (__inference_block6b_expand_activation_layer_call_and_return_conditional_losses_12731361) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.
                    (seem flowing after some cases of when happen loss nan not the all,)
                    https://github.com/tensorflow/similarity/issues/219
                    https://github.com/tensorflow/tensorflow/issues/40166
            
        """
        #BaseCnn = getattr(importlib.import_module("tensorflow.keras.applications.efficientnet"), model_name)
        #base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=weight)
        # NO extra rescale need, efn already include the scaling inside the model
        rescaling_input = inputs
        
    if model_name.startswith('EfficientNetB1'):
        base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights=weight)
        rescaling_input = inputs
    if model_name.startswith('EfficientNetB2'):
        base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights=weight)
        rescaling_input = inputs
    if model_name.startswith('EfficientNetB3'):
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights=weight)
        rescaling_input = inputs
    if model_name.startswith('EfficientNetB4'):
        base_model = tf.keras.applications.EfficientNetB4(include_top=False, weights=weight)
        rescaling_input = inputs
    if model_name.startswith('EfficientNetB5'):
        base_model = tf.keras.applications.EfficientNetB5(include_top=False, weights=weight)
        rescaling_input = inputs
    if model_name.startswith('EfficientNetB6'):
        base_model = tf.keras.applications.EfficientNetB6(include_top=False, weights=weight)
        rescaling_input = inputs
    if model_name.startswith('EfficientNetB7'):
        base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights=weight)
        rescaling_input = inputs
        
        
        
    # Xception #
    """When run in "tf" mode it actuallly expect the input to be uint8 between 0 and 255 and scales it to the range from -1.0 to 1.0.
    Check the docstring and the source code.""" #NOT TRUE
    """ For Xception, call tf.keras.applications.xception.preprocess_input on your inputs before passing them to the model.
    xception.preprocess_input will scale input pixels between -1 and 1."""
    if model_name.startswith('Xception'):
        BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.xception"), model_name)
        base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        rescaling_input = tf.keras.applications.xception.preprocess_input(inputs)

    # ResNet50 ResNet101 ResNet152 #
    if model_name.startswith('ResNet'):
        """For ResNet, call tf.keras.applications.resnet.preprocess_input on your inputs before passing them to the model.
        resnet.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel with
        respect to the ImageNet dataset, without scaling."""
        BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.resnet"), model_name)
        base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        rescaling_input = tf.keras.applications.resnet.preprocess_input(inputs)
        
    # InceptionV3 #
    if model_name.startswith('InceptionV3'):
        """For InceptionV3, call tf.keras.applications.inception_v3.preprocess_input on your inputs before passing them to the model.
        inception_v3.preprocess_input will scale input pixels between -1 and 1."""
        BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.inception_v3"), model_name)
        base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        rescaling_input = tf.keras.applications.inception_v3.preprocess_input(inputs)



    # MobileNet #
    if model_name.endswith('MobileNet'):
        """ For MobileNet, call tf.keras.applications.mobilenet.preprocess_input on your inputs before passing them to the model.
        mobilenet.preprocess_input will scale input pixels between -1 and 1."""
        """
        The weight of trained (224, 224) will be load for fine turn, bcs the mbnet preteing size (). But it not matter in FT task.
        
        CDR
        [max val_acc] ft_bench_imagenet_crop_512x512_CDR_RA_bs32
          MobileNet-------------------------:
                       0.858522891998291 Epoch@P19
        plateau
        [max val_acc]
          MobileNet-------------------------:
                       0.8501090407371521 Epoch@P17
                       
        plateau RA 512 crop:
        [max val_acc]
          MobileNet-------------------------:
               0.8566531538963318 Epoch@P13
        """
        #BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.mobilenet"), model_name)
        #base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        """Yep, I give up the buzz getattr call."""
        base_model = tf.keras.applications.MobileNet(include_top=False, weights=weight)
        rescaling_input = tf.keras.applications.mobilenet.preprocess_input(inputs)
        
    # MobileNetV2 #
    if model_name.startswith('MobileNetV2'):
        """ For MobileNetV2, call tf.keras.applications.mobilenet_v2.preprocess_input on your inputs before passing them to the model.
        mobilenet_v2.preprocess_input will scale input pixels between -1 and 1.
        
        
        keras.app only CDR can train without loss=nan, but val_accuracy: ~0.6183 after e2.
        [max val_acc]
          MobileNetV2-------------------------: 0.7360548377037048 Epoch@P0 actually at e1.
          
        plateau after two few times
        val_accuracy: 0.8158 e3, 0.6183 after e3.
        
        max of crop plateau keras hdf5
        0.8510439395904541 Epoch@P16
        
        """
        
        
        #BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.mobilenet_v2"), model_name)
        #base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        """Yep, I give up the buzz getattr call."""
        
        """
        [2022-02-15]
        [tf260+keras+ classifier_activation=None with N-round!]
        classifier_activation    A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True. Set classifier_activation=None to return the logits of the "top" layer. When loading pretrained weights, classifier_activation can only be None or "softmax".
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 6 7 imagenet1k crop plateau RA 512 512 1
        
            * * * Building keras.apps...
            WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
            
        [166.73478841781616] of epoch 1 to 0.73761
        [74.36339807510376] of epoch 2 to 0.74572
        [61.64659023284912] of epoch 3 from 0.74572
        
        [tf260+keras+with N-round!]
        T1
        [175.63299012184143] of epoch 1 to 0.72795
        [52.57041931152344] of epoch 2 from 0.72795 val_loss: nan
        T2
        [169.56960463523865] of epoch 1 o 0.65067 val_loss: nan
        T3
        [163.51381540298462] of epoch 1 to 0.74291
        [75.98137903213501] of epoch 6 to 0.84637
        [52.75170969963074] of epoch 20 from 0.86725
        
        [tf260+keras+with N-round! + remove preprocess_input]
         MobileNetV2_0 :              0.6444374918937683 Epoch@P1 can not learn!!
         
         
        """
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=weight)
        #base_model = tf.keras.applications.MobileNetV2(include_top=False, classifier_activation=None, weights=weight) # Ignored unless include_top=True. do not put it in.
        rescaling_input = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
 
 
 
    # MobileNetV3Small  # Total params: 1,535,093
    if model_name.startswith('MobileNetV3Small'):
        """MobileNetV3Small"""
        """
        [keras] switch to build_efn_model() [change within 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py]
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 7 8 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_MobileNetV3Small_hub_crop_plateau
        
            * * * Building keras.apps...
            WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
            
        #MobileNetV3Small crop 512 512 bs32 gpu*8
        [143.05812764167786] of epoch 1 to 0.68433
        [54.428372621536255] of epoch 2 to 0.74042
        [59.87986660003662] of epoch 3 to 0.77750
        ...
        [53.80861735343933] of epoch 18 to 0.84855 (best)
        [max val_acc]
          MobileNetV3Small-------------------------:
               0.8485509753227234 Epoch@P17 (for plotting curve start from 0.)
               
        [2022-02-15]
        [tf260+keras with N-round!]
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 7 8 imagenet1k crop plateau RA 512 512 1
        
            * * * Building keras.apps... Weights for input shape (224, 224) will be loaded as the default.
        [180.6282343864441] of epoch 1 to 0.70739
        [77.24401688575745] of epoch 2 to 0.75818
        [82.39694356918335] of epoch 3 0.78031
        MobileNetV3Small_0 :         0.8435649871826172 Epoch@P10
        
        
        [tf260+keras+ include_preprocessing=True with N-round!] seems better val_acc
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 7 8 imagenet1k crop plateau RA 512 512 1
        
                    * * * Building keras.apps... Weights for input shape (224, 224) will be loaded as the default.
        [160.50961661338806] of epoch 1 to 0.68090
        [76.38955640792847] of epoch 2 to 0.74135
        [81.97635841369629] of epoch 3 0.77563
        MobileNetV3Small_0 :         0.8600810170173645 Epoch@P19
        
        
        """
        #BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.MobileNetV3Small"), model_name)
        #base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        """tf250  no include_preprocessing for MobileNetV3, it implemented after tf260."""
        base_model = tf.keras.applications.MobileNetV3Small(
                        input_shape=None, alpha=1.0, minimalistic=False, include_top=False,
                        weights=weight, input_tensor=None, pooling=None,
                        dropout_rate=0.2,
                        include_preprocessing=True
                        )
        rescaling_input = inputs
        
    # MobileNetV3Large  #
    if model_name.startswith('MobileNetV3Large'):
        """For ModelNetV3, by default input preprocessing is included as a part of the model (as a Rescaling layer), and thus tf.keras.applications.mobilenet_v3.preprocess_input is actually a pass-through function. In this use case, ModelNetV3 models expect their inputs to be float tensors of pixels with values in the [0-255] range."""
        """can be disabled by setting include_preprocessing argument to False."""
        
        """
        [260 hub: fixed size 224x224,] 2022-02-08
        
        [keras] switch to build_efn_model() [change within 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py]
                
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 8 9 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_MobileNetV3Large_hub_crop_plateau
        
                    * * * Building keras.apps...
            WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.

        #MobileNetV3Large crop 512 512 bs32 gpu*8
        [162.6946840286255] of epoch 1 to 0.74104
        [53.809749126434326] of epoch 2 to 0.81427
        [63.57476997375488] of epoch 3 to 0.84107
        ...
        [53.2305428981781] of epoch 20 to 0.87348
        [max val_acc]
          MobileNetV3Large-------------------------:
               0.8734808564186096 Epoch@P19
               
               
        [2022-02-15]
        [tf260+keras+ include_preprocessing=True with N-round!] seems better val_acc not really
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 8 9 imagenet1k crop plateau RA 512 512 1
        [184.5608868598938] of epoch 1 to 0.76784
        [79.89947557449341] of epoch 2 to 0.82331
        [87.00471591949463] of epoch 3 to 0.84014
        0.8725459575653076 Epoch@P11
        """
       #BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.MobileNetV3Large"), model_name)
       #base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        base_model = tf.keras.applications.MobileNetV3Large(
                        input_shape=None, alpha=1.0, minimalistic=False, include_top=False,
                        weights='imagenet', input_tensor=None, pooling=None,
                        dropout_rate=0.2,
                        include_preprocessing=True
                        )
        rescaling_input = inputs
    
    
    

    # DenseNet121 DenseNet169 DenseNet201 # 9 10 11
    if model_name.startswith('DenseNet121'):
        """ 2022-02-09
        With tf260 keras:
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 9 10 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_DenseNet121_keras_crop_plateau
        
        WARNING:tensorflow:
        The following Variables were used a Lambda layer's call (tf.compat.v1.nn.fused_batch_norm_113), but
        are not present in its tracked objects:
          <tf.Variable 'conv5_block13_1_bn/gamma:0' shape=(128,) dtype=float32>
          <tf.Variable 'conv5_block13_1_bn/beta:0' shape=(128,) dtype=float32>
        It is possible that this is intended behavior, but it is more likely
        an omission. This is a strong indication that this layer should be
        formulated as a subclassed Layer rather than a Lambda layer.
        ...
        TypeError: Layer tf.nn.convolution was passed non-JSON-serializable arguments.
        """
        

        
        """
        With tf250 keras: 8gpu
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 9 10 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_DenseNet121_keras_crop_plateau
        
        [258.1814877986908] of epoch 1 to 0.82050
        [77.32694268226624] of epoch 2 to 0.84855
        [90.53892612457275] of epoch 3 to 0.85883
        
        [77.50968551635742] of epoch 7 to 0.87223
        
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 10 11 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_DenseNet169_keras_crop_plateau
        [358.13567328453064] of epoch 1 to 0.80399
        [100.38875389099121] of epoch 2 to 0.84917
        [122.18761730194092] of epoch 3 to 0.85323
        ...
        [102.82618021965027] of epoch 8 to 0.87597
        
        
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 11 12 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_DenseNet201_keras_crop_plateau
        [429.0091028213501] of epoch 1 to 0.82206
        [120.9476089477539] of epoch 2 to 0.84824
        [150.43617010116577] of epoch 3 to 0.86008
        ...
        [119.14348888397217] of epoch 16 from 0.87753
        """
        
        
        """ For DenseNet, call tf.keras.applications.densenet.preprocess_input on your inputs before passing them to the model."""
#        BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.densenet"), model_name)
#        base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
#
#        rescaling_input = tf.keras.applications.densenet.preprocess_input(inputs)
        
        
        """ [2022-02-15]  Do not use getattr, tf.k.apps.xx instead!
        With tf260 keras + SavedModel + tf.keras.applications:
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 9 10 imagenet1k crop plateau RA 512 512 1
        still "Lambda layer's call" issue!! Can not load the model parameter. the Total params: 5,125 is just dense not the whole model DenseNet121 = 8,062,504.
        
        $python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 9 10 imagenet1k crop plateau RA 512 512 1
        [301.9613585472107] of epoch 1 to 0.81209
        [131.21402215957642] of epoch 2 to 0.84294
        [135.336993932724] of epoch 3 to 0.85915
        [133.2428011894226] of epoch 7 to 0.87255 max
        """
        base_model = tf.keras.applications.densenet.DenseNet121(
                input_shape=None, include_top=False,
                weights=weight, input_tensor=None, pooling=None,
                )
        rescaling_input = tf.keras.applications.densenet.preprocess_input(inputs)
        
    # 10
    if model_name.startswith('DenseNet169'):
        base_model = tf.keras.applications.densenet.DenseNet169(
            input_shape=None, include_top=False,
            weights=weight, input_tensor=None, pooling=None,
            )
        rescaling_input = tf.keras.applications.densenet.preprocess_input(inputs)
    # 11
    if model_name.startswith('DenseNet201'):
        base_model = tf.keras.applications.densenet.DenseNet201(
                input_shape=None, include_top=False,
                weights=weight, input_tensor=None, pooling=None,
                )
        rescaling_input = tf.keras.applications.densenet.preprocess_input(inputs)
        
        
        
    # NASNet: NASNetMobile # 12
    if model_name.startswith('NASNetMobile'):
        """ Total params: 4,275,001
        tf 250 keras: 8GPU bs32
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-07_NASNetMobile_keras_crop_plateau
        
        250 keras: 0.6989716291427612 Epoch@P0
            其他ep 0.6183 不變
            因為是硬改手動載入h5

        250 keras: 不手動載入h5 則loss: nan

        250 keras + scale + resize: rescale, resize layer僅在tf260之後有，改用260測試。
        
        GOTO:
        260 hub + scale + resize:
        
        """
    
    
        """For NASNet, call tf.keras.applications.nasnet.preprocess_input on your inputs before passing them to the model."""
        """Optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
        for NASNetMobile It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (224, 224, 3)
        would be one valid value. For loading imagenet weights, input_shape should be (224, 224, 3)"""
        
        """otherwise the input shape has to be (331, 331, 3) for NASNetLarge. It should have exactly 3 inputs channels, and width and height should
        be no smaller than 32. E.g. (224, 224, 3) would be one valid value.  For loading imagenet weights, input_shape should be (331, 331, 3)"""
        
        """ NASNetMobile imagenet with 224: ted 10.x/5.x
            NASNetMobile None with 120: ted 12.x/4.x,  seem no different at fine tune phase. """
        
        """pre-set use inputs = tf.keras.Input([None, None, 3]) to fake run build mode to get the model weight.  Then, run
        on normal build with specific input size without imagenet-weight, and reload the weight by hand.
        Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/nasnet/nasnet_mobile_no_top.h5
        Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-large-no-top.h5
        https://github.com/keras-team/keras-applications/issues/78
        
        2022-02-09 recheck. this hand load h5 is not correct.
        兩個h5都使用過nan，但未測試過tf.keras.Input([None, None, 3])
        
        """
        """Very large model, NASNetLarge take 8xM parameters, take 800~300 sec for one epoch.
        Epoch 00015: val_accuracy did not improve from 0.87036
        [306.6111526489258] of epoch 15
        CPU times: user 7h 30min 14s, sys: 32min 2s, total: 8h 2min 16s
        Wall time: 1h 26min 32s
        
        NASNetMobile: but loss: nan after epcoh 1, need reduce the lr!!!!!!!!!!!!!!!!!!!!!!!!!!!!!![12/14]Fixed by seprating mobile/large to two functions.
        Epoch 00011: val_accuracy did not improve from 0.64444
        [124.91568398475647] of epoch 11
        CPU times: user 3h 19min 49s, sys: 3min 41s, total: 3h 23min 31s
        Wall time: 29min 27s
        """
        
        """[2022-02-16]
        260+keras+SavedModel: + weights=None base_model.load_weights take long initial time
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop plateau RA 512 512 1
        loss: nan
        
        260+keras+SavedModel: + weights=None without load_weights : == train from scratch
        loss: nan
        
         * comment scheduled_lr callback
        260+keras+SavedModel: + weights=None without load_weights : == train from scratch + learning_rate=0.01
         loss: nan
         
         260+keras+SavedModel: + weights=None without load_weights : == train from scratch + learning_rate=0.1
         loss: nan
         
         260+keras+SavedModel: + weights=None without load_weights : == train from scratch + learning_rate=0.000000001
          loss: nan
          
        * change plateau 0.00001 as same as WCD/CDR INIT_LR.
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop plateau RA 512 512 1
        
        
        [2022-02-16][2022-02-17 Update]
        * * the important warnings seems is the reason of learn fail: * *
        CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
              warnings.warn('Custom mask layers require a config and must override '
              
        https://github.com/tensorflow/tensorflow/issues/52978
         Google may fix it in the upcoming TF2.8 shortly.
        * * It may be fixed in the upcoming TF2.8 in near future. * *

              

        """
        
        """ [2022-02-27]
        260+keras+SavedModel + resize + nasnet.preprocess_input:
        
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 12 13 imagenet1k crop plateau "AA" 512 512 1
        
        
        """
        
        
#        # Pre download the model first for it weight later we need to reload it.
##         inputs = tf.keras.Input([None, None, 3])
#
#        rescaling_input = tf.keras.applications.nasnet.preprocess_input(inputs)
#
#
#        BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.nasnet"), model_name)
##         base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
##         base_model = BaseCnn(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3)) #{'imagenet', None} for set input to 120x120
##         base_model = BaseCnn(include_top=False, weights='imagenet') #{'imagenet', None} for set input to 120x120
#
#
#        # load weight by hand.
#        base_model = BaseCnn(include_top=False, weights=None, input_shape=(img_height, img_width, 3))
#
#        # forgote where got it@@
#        #base_model.load_weights('/home/u3148947/.keras/models/nasnet_mobile_no_top.h5') # If no load_weights, model.fit will take very long long time for initial when into the epoch 1.
#
#        #NASNet-mobile-no-top.h5 https://github.com/fchollet/deep-learning-models/releases/ #same as above, epoch 2 loss nan
#        base_model.load_weights('/home/u3148947/.keras/models/NASNet-mobile-no-top.h5') # If no load_weights, model.fit will take very long long time for initial when into the epoch 1.
#
#
##         inputs = tf.keras.layers.Resizing(224, 224) # tf >= 2.6.0, but currnet TWCC newest 21.08 is tf=2.5.0
##         i = tf.compat.v1.keras.layers.experimental.preprocessing.Resizing(224, 224)(inputs)
##         x = tf.cast(i, tf.float32)


#        base_model = tf.keras.applications.nasnet.NASNetMobile(
#            input_shape=(img_height, img_width, 3), include_top=False,
#            weights=None, input_tensor=None, pooling=None,
#            )
#        base_model.load_weights('/home/u3148947/.keras/models/nasnet_mobile_no_top.h5')
#        rescaling_input = tf.keras.applications.nasnet.preprocess_input(inputs)


        base_model=tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights=weight)
        resize_input = resize_layer_224(inputs)
        rescaling_input = tf.keras.applications.nasnet.preprocess_input(resize_input)
        


    # NASNet: NASNetLarge #
    if model_name.startswith('NASNetLarge'):
#        rescaling_input = tf.keras.applications.nasnet.preprocess_input(inputs)
#
#        BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.nasnet"), model_name)
#
#        # load weight by hand.
#        base_model = BaseCnn(include_top=False, weights=None, input_shape=(img_height, img_width, 3))
#        base_model.load_weights('/home/u3148947/.keras/models/nasnet_large_no_top.h5')
        base_model=tf.keras.applications.nasnet.NASNetLarge(include_top=False, weights=weight)
        resize_input = resize_layer_331(inputs)
        rescaling_input = tf.keras.applications.nasnet.preprocess_input(resize_input)



    # VGG 29  30, only exist in keras.apps
    # VGG16 # work with tf21.08
    """VGG16 not train even with None or Imagenet. pooling= is not the factor"""
    if model_name.startswith('VGG16'):
        """For VGG16, call tf.keras.applications.vgg16.preprocess_input on your inputs before passing them to the model.
        vgg16.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel
        with respect to the ImageNet dataset, without scaling."""
        
        """[2022-02-14]
        
        VGG16 Total params: 14,717,253
        
        260 keras: 'Not JSON Serializable:' when save model.
        
        
        
        250 keras:
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 29 30 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-14_VGG16_hub_crop_plateau
            
            [100.18129587173462] of epoch 1 to 0.75849
            [62.247276067733765] of epoch 2 to 0.83328
            [66.91538119316101] of epoch 3 to 0.85167
            [max val_acc] 0.8660018444061279 Epoch@P15
        

            
        VGG16 Total params: 14,717,253
        260 keras.apps SavedModel :
        python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 29 30 imagenet1k crop plateau RA 512 512 1
        loss: nan
        
        
        """
        #BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.vgg16"), model_name)
        #base_model = BaseCnn(include_top=False, weights='imagenet') #{'imagenet', None}
        
        base_model=tf.keras.applications.vgg16.VGG16(include_top=False, weights=weight)
        
        rescaling_input = tf.keras.applications.vgg16.preprocess_input(inputs)



    # VGG19 #
    if model_name.startswith('VGG19'):
        """For VGG19, call tf.keras.applications.vgg19.preprocess_input on your inputs before passing them to the model.
        vgg16.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel
        with respect to the ImageNet dataset, without scaling."""
        
        """ Do not use getattr in tf260, it is not workable.
        
        VGG19 Total params: 20,026,949
        250 keras:
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 30 31 imagenet1k crop plateau RA 512 512 TrainSaveDir-2022-02-14_VGG19_hub_crop_plateau
            
            [118.55681943893433] of epoch 1 to 0.74727
            [74.95128226280212] of epoch 2 to 0.83328
            [79.70657896995544] of epoch 3 to 0.83515
            [max val_acc] 0.8731691837310791 Epoch@P14
            
            
        260 keras.apps + SavedModel:
            python3 2022_Leaf_rewrite-for-model-benchmark-forConvert2Py.py 30 31 imagenet1k crop plateau RA 512 512 1
            
            
        
        """
        
        #BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications.vgg19"), model_name)
        #base_model = BaseCnn(include_top=False, weights=weight) #{'imagenet', None}
        
        base_model=tf.keras.applications.vgg16.VGG16(include_top=False, weights=weight)
                
        rescaling_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    
    
 
        
        
        

    # ViT # ViT-keras, but seems need more epoch to train.
    if model_name.startswith('ViT'):
        """For ViT (vit-keras) https://github.com/faustomorales/vit-keras
        
        https://www.kaggle.com/raufmomin/vision-transformer-vit-fine-tuning
        For from scratch implementation of ViT check out this notebook:
            https://www.kaggle.com/raufmomin/vision-transformer-vit-from-scratch

            Research Paper: https://arxiv.org/pdf/2010.11929.pdf
            Github (Official) Link: https://github.com/google-research/vision_transformer
            Github (Keras) Link: https://github.com/faustomorales/vit-keras

        There are models pre-trained on imagenet21k for the following architectures: ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32 and ViT-H/14.There are also the same models pre-trained on imagenet21k and fine-tuned on imagenet2012.
        
        pip install -U --quiet vit-keras # for imagenet21k pre-trained weight.
        pip install -U tensorflow-addons # for scratch
        
        default use 'sigmoid'
        """
        
        """base_model = vit.vit_b32
        Downloading data from https://github.com/faustomorales/vit-keras/releases/download/dl/ViT-B_32_imagenet21k+imagenet2012.npz
        
        base_model = vit.vit_b16
        Downloading data from https://github.com/faustomorales/vit-keras/releases/download/dl/ViT-B_16_imagenet21k+imagenet2012.npz
        
        
        https://github.com/faustomorales/vit-keras/releases/
         ViT-B_16_imagenet21k+imagenet2012.npz  331 MB
            ViT-B_16_imagenet21k.npz               394 MB
         ViT-B_32_imagenet21k+imagenet2012.npz  337 MB
            ViT-B_32_imagenet21k.npz               400 MB
         ViT-L_16_imagenet21k+imagenet2012.npz  1.14 GB
            ViT-L_16_imagenet21k.npz               1.22 GB
         ViT-L_32_imagenet21k+imagenet2012.npz  1.14 GB
            ViT-L_32_imagenet21k.npz               1.23 GB

        
        
        """
        """
        ValueError: Input 0 of layer global_average_pooling2d is incompatible with the layer: expected ndim=4, found ndim=2. Full shape received: (None, 768)
        seems need use Flatten() to capture laster feature output.
        """
        """AssertionError: image_size must be a multiple of patch_size
        600 / 16 = 37.5
        512 / 16 = 32, only works
        """
        
        """For ViT with TF.HUB
        https://tfhub.dev/sayakpaul/collections/vision_transformer/1
        https://github.com/google-research/vision_transformer
        
        Model    Top-1 Accuracy MiB
        B/8     85.948
        B/16    84.018          391 MiB
        B/32    79.436          398 MiB
        L/16    85.716          1243 MiB
        S/16    80.462          115 MiB
        """
        from vit_keras import vit
        """vit_keras
        AssertionError: image_size must be a multiple of patch_size
        SO, 600 not work for 16 or 32. Need 512!!!!
        """
        #import tensorflow_addons as tfa
        
        rescaling_input = inputs
        
        
#        if model_name == "ViT-B/8":
#            #vit_b16 vit_b32  vit_L16 vit_L32
#            base_model = vit.vit_b8(
#                image_size = img_width,
#                activation = 'softmax',
#                pretrained = True, #True,
#                include_top = True,
#                pretrained_top = False,
#                classes = 5)
                
        if model_name == "ViT-B16":
            #vit_b16 vit_b32  vit_L16 vit_L32
            base_model = vit.vit_b16(
                image_size = img_width,
                activation = 'softmax',
                pretrained = True, #True,
                include_top = True, # wait the verbose=2,output 1epoch per 1line.
                pretrained_top = False,
                classes = 5)
        if model_name == "ViT-B32":
            #vit_b16 vit_b32  vit_L16 vit_L32
            base_model = vit.vit_b32(
                image_size = img_width,
                activation = 'softmax',
                pretrained = True, #True,
                include_top = True,
                pretrained_top = False,
                classes = 5)
                
        if model_name == "ViT-L16":
            #vit_b16 vit_b32  vit_L16 vit_L32
            base_model = vit.vit_l16(
                image_size = img_width,
                activation = 'softmax',
                pretrained = True, #True,
                include_top = True,
                pretrained_top = False,
                classes = 5)
        if model_name == "ViT-L32":
            #vit_b16 vit_b32  vit_L16 vit_L32
            base_model = vit.vit_l32(
                image_size = img_width,
                activation = 'softmax',
                pretrained = True, #True,
                include_top = True,
                pretrained_top = False,
                classes = 5)



#     # template #
#     if model_name.startswith(''):
#         """For."""
#         BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications."), model_name)
#         base_model = BaseCnn(include_top=False, weights="imagenet", input_shape=(120,120,3)) #{'imagenet', None}
        
#         rescaling_input = tf.keras.applications.inception_v3.preprocess_input(inputs)


#     # template #
#     if model_name.startswith(''):
#         """For."""
#         BaseCnn = getattr(importlib.import_module("tensorflow.python.keras.applications."), model_name)
#         base_model = BaseCnn(include_top=False, weights="imagenet", input_shape=(120,120,3)) #{'imagenet', None}
        
#         rescaling_input = tf.keras.applications.inception_v3.preprocess_input(inputs)





    ############################
    # Keras.apps model compile #
    ############################
    
    # ViT # tf.hub version [Waiting for twcc update CCS image for version 21.11]
    if model_name.startswith('ViT'):
        """ViT was loaded above already."""
        print(f'Set {model_name}')
        
#        # IF set vit_keras model include_top = False
#        """WCD take ep19 with 0.722 val acc"""
#        b_m_output = base_model(rescaling_input)
#
#        # Rebuild top
#        gap2d = tf.keras.layers.Flatten()(b_m_output)
#        dropout = tf.keras.layers.Dropout(top_dropout_rate)(gap2d)
#        outputs = tf.keras.layers.Dense(outputnum, activation="softmax")(dropout)
#
#        # Compile new model
#        model = tf.keras.Model(inputs, outputs, name=model_name)
#
#        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
#                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
#                 metrics=['accuracy'])
                 
        # IF set vit_keras model include_top = True, val_acc 0.61~ 0.69 ep5.
        """ plateae take ep50p5, lr 0.01 ???"""
        vit_lr=0.01 #0.00001
        print(f'Set plateuaLR of {model_name}={vit_lr}')
        model=base_model
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=vit_lr), #0.0001 1e-4 #RMSprop , Adam, SGD Adadelta(learning_rate=0.001), if set lr_callback the learning_rate=0.001 will not effeced.
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['accuracy'])
#        pass
        

    elif model_name.startswith('vit_b8_fe'):
        model = tf.keras.Sequential(
            [
                rescaling_input,
                base_model,
                tf.keras.layers.Dense(outputnum, activation="softmax"),
            ]
        )
        
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), #0.0001 1e-4 #RMSprop , Adam, SGD Adadelta(learning_rate=0.001), if set lr_callback the learning_rate=0.001 will not effeced.
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
         metrics=['accuracy'])
    
    
    elif model_name.startswith('ConvMixer'):
        """TEST"""
        print(f'Set {model_name}')
        pass
        
    else:
        # Freeze the pretrained weights
        base_model.trainable = True #False #skip the TL so it should be change to True. and remove the free_model()
        print("base_model.trainable : ", base_model.trainable)
        print(f'Set other models')

        # move to Top
        # How to add training=False in base_model create
        #inputs = tf.keras.Input(shape=(120, 120, 3))
        #rescal = rescaling_input()(inputs)
        #b_m_output = base_model(inputs, training=False)

        b_m_output = base_model(rescaling_input, training=False)

        # Rebuild top
        gap2d = tf.keras.layers.GlobalAveragePooling2D()(b_m_output) #(base_model.output)
        #BNL = tf.keras.layers.BatchNormalization()(gap2d) #tood: remove#
        dropout = tf.keras.layers.Dropout(top_dropout_rate)(gap2d)#tood: remove# J add dropout, for flood 0.2 is ok. for leaf 0.4 is better. for foot 0.8 is fine.
        #outputs = tf.keras.layers.Dense(outputnum)(dropout)# remove activation for regression output (to default, the linear), , activation = 'relu' no help
        outputs = tf.keras.layers.Dense(outputnum, activation="softmax")(dropout)#todo: activation="softmax", default is "linear" activation: a(x) = x


        # Compile new model
        model = tf.keras.Model(inputs, outputs, name=model_name)

    
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),#0.00001 for [plateau] keep same as WCD and CDR's INIT_LR. #0.0001 1e-4 #RMSprop , Adam, SGD Adadelta(learning_rate=0.001), if set lr_callback the learning_rate=0.001 will not effeced.
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['accuracy'])
    
    return model, base_model
#
#
## [Models] ##
#
#

#
#
## [Models] ##
#
#

#
#
## [Models] ##
#
#


## Display sample of images #
#import math
#col_row = math.sqrt(int(BATCH_SIZE))
#
#plt.figure(figsize=(20, 20))
#for images, labels in train_ds_pre.take(1):
#    print('batch * multi:', len(labels))
#    for i in range(4): # for batch size to show 4 32
#        ax = plt.subplot(4, 4, i + 1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(f'labels:{labels[i]}, {CLASSES[labels[i]]}')
#        plt.axis("off")
        
    
# [WCD] StepWise warmup cosine decay learning rate [optimazer]
"""# Reference:
# https://colab.research.google.com/github/sayakpaul/ViT-jax2tf/blob/main/fine_tune.ipynb
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2
# https://colab.research.google.com/github/google-research/vision_transformer/blob/linen/vit_jax.ipynb"""
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

    # 2021-12-19
    # If use optimizers.schedules.LearningRateSchedule and to save full model, need to rewrite the get_config()
    # https://stackoverflow.com/questions/61557024/notimplementederror-learning-rate-schedule-must-override-get-config
    def get_config(self):
        config = {
            'learning_rate_base':self.learning_rate_base,
            'total_steps':self.total_steps,
            'warmup_learning_rate':self.warmup_learning_rate,
            'warmup_steps':self.warmup_steps,
        }
        return config


#num_train = tf.data.experimental.cardinality(train_ds_map_s)
#num_val = tf.data.experimental.cardinality(valid_ds_map_s)
#print(f"Number of training examples: {num_train}")
#print(f"Number of validation examples: {num_val}")


#EPOCHS = 20
#EPOCHS_WARM = 10
#TOTAL_STEPS = int((num_train / MULTI_BATCH_SIZE) * EPOCHS)
#WARMUP_STEPS = int((num_train / MULTI_BATCH_SIZE) * EPOCHS_WARM) #10 # warmup 2 epochs
#INIT_LR = 0.004 #4e-3
#WAMRUP_LR = 0.00001 #1e-5
#
#print(f'total step/EPOCHS: {TOTAL_STEPS}/{EPOCHS}, MULTI_BATCH_SIZE:{MULTI_BATCH_SIZE}')
#
#print(f'total step= (num_train / MULTI_BATCH_SIZE) * EPOCHS : {TOTAL_STEPS}=({num_train}/{MULTI_BATCH_SIZE})*{EPOCHS}')


#WCD = WarmUpCosine(
#    learning_rate_base=INIT_LR,
#    total_steps=TOTAL_STEPS,
#    warmup_learning_rate=WAMRUP_LR,
#    warmup_steps=WARMUP_STEPS,
#)
#
#lrs = [WCD(step) for step in range(TOTAL_STEPS)]
#plt.plot(lrs)
#plt.xlabel("Step", fontsize=14)
#plt.ylabel("LR", fontsize=14)
#plt.show()

#rng = [i for i in range(TOTAL_STEPS)]
#y = [WCD(step) for step in rng]
#sns.set(style='darkgrid')
#fig, ax = plt.subplots(figsize=(8, 4))
#plt.plot(rng, y)

#lrs = [WCD(step) for step in range(TOTAL_STEPS)]
#y = lrs
#x = [x for x in range(TOTAL_STEPS)]
#
#sns.set(style='white') # will affect the further plot include plt or sns.
#
#plt.plot(x,lrs)
#plt.xticks(x)
#
#plt.xlabel("Epoch", fontsize=14)
#plt.ylabel("Learning rate", fontsize=14)
#
#for a,b in zip(x, y):
#    #plt.text(a, b, str(b))
#    plt.scatter(a,b, color='black', alpha=0.2)
#    plt.annotate(f'{b:.8f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#
#plt.show()
#
#
#
#print('{} ~ {}'.format(min(y), max(y)))
#
#
#for i in y:
#    print(i.numpy())
#


# [CDR ]
"""warm Cosine Decay Restart"""

from matplotlib.ticker import FormatStrFormatter

#ep_num = EPOCHS

initial_learning_rate = INIT_LR
first_decay_steps = 5

CosineDecayCLRWarmUpLSW = tf.keras.experimental.CosineDecayRestarts(
          initial_learning_rate,
          first_decay_steps,
          t_mul=1.0,
          m_mul=1.0,
          alpha = initial_learning_rate,
          name="CCosineDecayRestarts")

#rng = [i for i in range(ep_num)]
#y = [CosineDecayCLRWarmUpLSW(x) for x in rng]
#sns.set(style='darkgrid')
#fig, ax = plt.subplots(figsize=(20, 6))
## plt.ylim(.0000000000000001, .01)# for too large loss
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.12f'))# for too small loss
#plt.plot(rng, y)
#plt.xticks(rng)
#
#for a,b in zip(rng, y):
#    #plt.text(a, b, str(b))
#    plt.scatter(a,b, color='black', alpha=0.2)
#    plt.annotate(f'{b:.8f}',xy=(a,b)) # offest text:, xytext=(10,10), textcoords='offset points'
#
#CDR = CosineDecayCLRWarmUpLSW



# 2021-11-25 #
#TODO: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback 改寫為callbacks.on_epoch_begin(epoch) 並加入datetime方便在訓練時觀察時間
class PrintLR(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         self.epoch=epoch
#         print(f"第 {epoch} 執行週期開始...")
#     def on_epoch_end(self, epoch, logs=None):
#         print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))


#     def on_epoch_begin(self, epoch, logs=None):
#         print(f'[{datetime.now()}] Learning rate for epoch {epoch + 1} is {model.optimizer.lr.numpy()}')
        
    # time of epoch
    def on_train_begin(self,  epoch, logs=None):
        self.times = []
    def on_epoch_begin(self,  epoch, logs=None):
        self.epoch_time_start = time.time()
        #print(f'[{datetime.now()}] Learning rate for epoch {epoch + 1} is {model_toe.optimizer.lr.numpy()}') #for epochwise
        print(f'[{datetime.now()}] Learning rate for epoch {epoch + 1} is {self.model.optimizer._decayed_lr(tf.float32).numpy()}') #for stepwise
        
#         s_time = time.time()
#         print("start")

    def on_epoch_end(self,  epoch, logs=None):
        wall_time = time.time() - self.epoch_time_start
        self.times.append(wall_time)
        #print(f'[{self.times}] of epoch {epoch + 1}')
        print(f'[{wall_time}] of epoch {epoch + 1}')





# Check trainable layers #
def count_model_trainOrNot_layers(model, printlayers=False):
    tt = 0
    nt = 0
    for layer in model.layers:
        if layer.trainable:
            tt +=1
            if printlayers:
                print(f'{layer.name}')
        else:
            nt +=1
    print('\n*********************************** Start fine tune ***********************************')
    print(f'tt: {tt}, nt:{nt}, total layers:{tt+nt}')
    print('*********************************** Start fine tune ***********************************\n')
    


    
###########################################
# Test DS accuracy in N rounds per models #
###########################################
""" How to run it:
roll_out_valid()
roll_out()
"""
label_true_all = []
label_pred_all = []

# rewrite to prediction with batch of ds, replace the list of file to speed up.
# todo: Checked right!:model_back.predict_on_batch [OK done 20200904]
def pred_on_batch(model_back, val_ds_pre, MULTI_BATCH_SIZE, test_samples):

    batch_n = 0
    acc_count= 0

    #for image_batch, label_batch in tqdm(val_ds_pre): #ds set to repeat forever
    for image_batch, label_batch in val_ds_pre: #ds set to repeat forever
        batch_n += 1
        pred_max = []
        pred = model_back.predict_on_batch(image_batch)
        
        label_batch_np = label_batch.numpy()
        label_true_all.extend(label_batch_np)
        #print('label_batch_np = ',label_batch_np)
        
        for i in range(MULTI_BATCH_SIZE):#BATCH_SIZE to MULTI_BATCH_SIZE if used Multi-GPU training
    #         print(i)
            try:
                score = tf.nn.softmax(pred[i])
                label_pred = np.argmax(score)
                pred_max.append(label_pred)
                
    #             print('label_batch_np[i] = ', label_batch_np[i])
                
                if label_batch_np[i] == label_pred:
                    acc_count += 1
            except IndexError:
                #print("End of batch")
                pass
        label_pred_all.extend(pred_max)

    print("acc_count =", acc_count, "  (if score == label[i] then count one.)")
    print("accuracy  = ", acc_count/test_samples, "%")
    print("Number of batch used = ",batch_n)
    
    return label_pred_all, label_true_all
    
    
# rewrite to prediction with batch of ds, replace the list of file to speed up.
# todo: Checked right!:model_back.predict_on_batch [OK done 20200904]
def pred_on_batch_org(model_back, val_ds_pre):

    batch_n = 0
    acc_count= 0

    for image_batch, label_batch in tqdm(val_ds_pre): #ds set to repeat forever
        batch_n += 1
        pred_max = []
        pred = model_back.predict_on_batch(image_batch)
        
        label_batch_np = label_batch.numpy()
        label_true_all.extend(label_batch_np)
        #print('label_batch_np = ',label_batch_np)
        
        for i in range(MULTI_BATCH_SIZE):#BATCH_SIZE to MULTI_BATCH_SIZE if used Multi-GPU training
    #         print(i)
            try:
                score = tf.nn.softmax(pred[i])
                label_pred = np.argmax(score)
                pred_max.append(label_pred)
                
    #             print('label_batch_np[i] = ', label_batch_np[i])
                
                if label_batch_np[i] == label_pred:
                    acc_count += 1
            except IndexError:
                #print("End of batch")
                pass
                
        label_pred_all.extend(pred_max)
        
        #print("pred =", pred_max)
    print("acc_count =", acc_count, "  (if score == label[i] then count one.)")
    print("accuracy  = ", acc_count/test_samples, "%")
    print("Number of batch used = ",batch_n)


# conf matrix 1
def plot_confusion_matrix_with_seaborn():
    # Plot confusion matrix with seaborn

    cm = tf.math.confusion_matrix(label_true_all, label_pred_all, num_classes=outputnum)
    print("abstract of cm: \n", cm)

    classes = CLASSES
    print("number of classes: ", len(CLASSES))

    # acc in %. Comment this line to be number of images.
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    print("abstract of cm in %: \n", cm)

    #set inner text scale
    # sns.set(font_scale=1.2)

    # Let label of xticks go to top
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    #set inner text scale, for inner of inner digits
    #sns.set(font_scale=1.2)

    sns.heatmap(
        cm, annot=True,
        fmt='.2f',
        cmap=plt.cm.Blues,
        vmin=0, vmax=1,
        xticklabels=classes,
        yticklabels=classes)
    plt.title('Confusion Matrix', fontsize='x-large')
#    plt.xlabel("Predicted class\n",fontsize=14, fontweight='bold')
#    plt.ylabel("True class",fontsize=14,fontweight='bold')
    plt.xlabel("Predicted class\n")
    plt.ylabel("True class")

    # Let y-label also matching matplotlib
    plt.yticks(rotation=0)
    # plt.title('Confusion Matrix', fontsize=20)
    
    dpi=100
    pgn=f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_confusion_matrix_seaborn.png'
    print(f'Save to {pgn} \n')
    plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)


def plot_confusion_matrix_with_seaborn_diag():
    # Plot confusion matrix with seaborn

    cm = tf.math.confusion_matrix(label_true_all, label_pred_all, num_classes=outputnum)
    print("abstract of cm: \n", cm)

    classes = CLASSES
    print("number of classes: ", len(CLASSES))

    # acc in %. Comment this line to be number of images.
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    print("abstract of cm in %: \n", cm)

    
    # diag of cm
    mask = np.ones_like(cm)
    mask[np.diag_indices_from(mask)] = False #True


    #set inner text scale
    # sns.set(font_scale=1.2)

    # Let label of xticks go to top
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    #set inner text scale, for inner of inner digits
    #sns.set(font_scale=1.2)

    # Outline frame of cm
    ax.axhline(y=0, color='k',linewidth=1)
    ax.axhline(y=cm.shape[1], color='k',linewidth=2)
    ax.axvline(x=0, color='k',linewidth=1)
    ax.axvline(x=cm.shape[0], color='k',linewidth=2)
    
    sns.heatmap(
        cm, annot=True,
        fmt='.2f',
        mask=mask,
        cmap=plt.cm.Blues,
        vmin=0, vmax=1,
        xticklabels=classes,
        yticklabels=classes)
    plt.title('Confusion Matrix', fontsize='x-large')
#    plt.xlabel("Predicted class\n",fontsize=14, fontweight='bold')
#    plt.ylabel("True class",fontsize=14,fontweight='bold')
    plt.xlabel("Predicted class\n")
    plt.ylabel("True class")

    # Let y-label also matching matplotlib
    plt.yticks(rotation=0)
    # plt.title('Confusion Matrix', fontsize=20)
    
    dpi=100
    pgn=f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_confusion_matrix_seaborn_diag.png'
    print(f'Save to {pgn} \n')
    plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
    
    
# conf matrix 2
def plot_confusion_matrix_with_pyplot():
    cm = tf.math.confusion_matrix(label_true_all, label_pred_all, num_classes=outputnum)
    print("abstract of cm: \n", cm)
    
    classes = CLASSES
    print("number of classes: ", len(CLASSES))
    
    # acc in %. Comment this line to be number of images.
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    print("abstract of cm in %: \n", cm)
       
    # Let label of xticks go to top
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) #plt.cm.Blues plt.cm.winter
    plt.title('Confusion Matrix', fontsize='x-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    print(tick_marks)
    plt.xticks(tick_marks, classes)#, rotation=-45)
    plt.yticks(tick_marks, classes)


    iters = [[i,j] for i in range(len(classes)) for j in range(len(classes))]
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    #plt.show()
    
    dpi=100
    pgn=f'./{log_dir_name}/{th}_{model_name}_{weight}_{crop}_{img_width}x{img_height}_{lr_name}_{augment}_bs{MULTI_BATCH_SIZE}_best_{monitor}_confusion_matrix_plot.png'
    print(f'Save to {pgn} \n')
    plt.savefig(pgn, bbox_inches = 'tight', dpi=dpi)
    

# Roll out for validation set
def roll_out_valid():
    """ Somehow, switch order of plot cm1 and cm2, the cm2 is correct plot.
    
    # pred
    pred_on_batch(best_model_reload, test_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    
    """
    # pred
    pred_on_batch(best_model_reload, valid_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    plot_confusion_matrix_with_seaborn_diag()
    
    
# Roll out final test data
def roll_out():
    """ Somehow, switch order of plot cm1 and cm2, the cm2 is correct plot.
    
    # pred
    pred_on_batch(best_model_reload, test_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    
    """
    # pred
    pred_on_batch(best_model_reload, test_ds_pre)

    # conf matrix 2
    plot_confusion_matrix_with_pyplot()

    # conf matrix 1
    plot_confusion_matrix_with_seaborn()
    plot_confusion_matrix_with_seaborn_diag()




def get_dir_size(path='.'):
    """bytes of directory
    print(get_dir_size('data/src'))
    
    Note: .stat().st_size output : Size in bytes of a plain file
    
    if 517MB when /1000 /1000 in (in decimal) (Cyberduck)
    
    model_size: 517671369 bytes, 493.6898889541626 MB (/1024 KB/1024 MB) (in binary) os.dir
    
    du -hs ./imagenet_crop_512x512_WCD_AA_bs32/ResNet101/0/ ->494M (in binary) Ubuntu
    
    
    1 Byte = 8 Bits
    1 Kilobyte (KB) = 1024 byte
    1 Megabyte (MB) = 1024 KB
    1 Gigabyte (GB) = 1024 MB
    1 Terabyte (TB) = 1024 GB
    """
    
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


    


