"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

"""
2020-07-27 For Water Segmentation App.

Usage:

time python3 flood_river.py train --dataset=/home/uu/data/flood_river/ --weights=imagenet

"""


"""
[2021-04-23] 補充access deep sensing論文中水位高度的第二種估算方法 mask rcnn water level
複製一份flood_river--> tire --> wheel_water_level

time python3 wheel_level.py train --dataset=/home/uu/data/wheel_level/ --weights=imagenet #也是不ㄒㄧㄥ
time python3 wheel_level.py train --dataset=/home/uu/data/wheel_level/ --weights=coco #error


 99/100 [============================>.] - ETA: 0s - loss: 0.1913 - rpn_class_loss: 0.0023 - rpn_bbox_loss: 0.0308 - mrcnn_class_loss: 0.0271 - mrcnn_bbox_loss: 0.0243 - mrcnn_mask_loss: 0.1068Traceback (most recent call last):
  File "wheel_level.py", line 398, in <module>
    train(model)
  File "wheel_level.py", line 229, in train
    layers='heads')
  File "/home/uu/twcc_gpfs/MaskRCNN/Mask_RCNN/mrcnn/model.py", line 2377, in train
    use_multiprocessing=True,
  File "/usr/local/lib/python3.6/dist-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/keras/engine/training.py", line 1732, in fit_generator
    initial_epoch=initial_epoch)
  File "/usr/local/lib/python3.6/dist-packages/keras/engine/training_generator.py", line 242, in fit_generator
    workers=0)
  File "/usr/local/lib/python3.6/dist-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/keras/engine/training.py", line 1791, in evaluate_generator
    verbose=verbose)
  File "/usr/local/lib/python3.6/dist-packages/keras/engine/training_generator.py", line 401, in evaluate_generator
    reset_metrics=False)
  File "/usr/local/lib/python3.6/dist-packages/keras/engine/training.py", line 1559, in test_on_batch
    outputs = self.test_function(ins)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/backend.py", line 3476, in __call__
    run_metadata=self.run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1472, in __call__
    run_metadata_ptr)
tensorflow.python.framework.errors_impl.InternalError: 2 root error(s) found.
  (0) Internal: Unable to launch cuBLAS gemm on stream 0x1753a8e0
         [[{{node cluster_26_1/xla_run}}]]
         [[cluster_28_1/merge_oidx_0/_5215]]
  (1) Internal: Unable to launch cuBLAS gemm on stream 0x1753a8e0
         [[{{node cluster_26_1/xla_run}}]]
0 successful operations.
0 derived errors ignored.


不論使用imagenet or coco，大約2~4 epoch之後即報錯，
改train with 2 image, 100step to 20 stpes, coco or imagenet --> 第一步即報錯
Old question, but may help others.
Try to close interactive sessions active in other processes (if IPython Notebook - just restart kernels). This helped me!

Additionally, I use this code to close local sessions in this kernel during experiments:

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()
但是關掉jupynb後，即跑兩epoch即報錯！



sftp://xdata1.twcc.ai/home/uu/twcc_gpfs/MaskRCNN/Mask_RCNN/logs/wheel_level20210423T1444/mask_rcnn_wheel_level_0003.h5
sftp://xdata1.twcc.ai/home/uu/twcc_gpfs/MaskRCNN/Mask_RCNN/samples/wheel_level/test_img/1.jpg


沒用
time python3 wheel_level.py splash --weights=/home/uu/twcc_gpfs/MaskRCNN/Mask_RCNN/logs/wheel_level20210423T1444/mask_rcnn_wheel_level_0003.h5 --image=/home/uu/twcc_gpfs/MaskRCNN/Mask_RCNN/samples/wheel_level/test_img/1.jpg

用
detect-wheel_level.ipynb
對test_img批次做分割
3 epoch效果不佳

"""

import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.4)


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
# ###########################################################


class TireConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "wheel_level"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # J: for TWCC 8*GPU, Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10 #100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



############################################################
#  Dataset
# ###########################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
#        self.add_class("flood_river", 1, "bridge")
#        self.add_class("flood_river", 2, "flood")
#        self.add_class("flood_river", 3, "riverbed")
        self.add_class("vehicle_tire", 1, "tire")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        #dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "vehicle_tire-export.json")))
        
        name_dict = {"tire":1}
        for item in annotations['assets']:
            image_id = annotations['assets'][item]['asset']['name']
            path = '/home/uu/data/wheel_level/' + image_id
            width = annotations['assets'][item]['asset']['size']['width']
            height = annotations['assets'][item]['asset']['size']['height']
                        
            polygons = []   
            name_id = []            
            for regions in annotations['assets'][item]['regions']:
                name_id.append(name_dict[regions['tags'][0]])
                all_points_x = []
                all_points_y = []
                for j in regions['points']:   
                    all_points_x.append(j['x'])
                    all_points_y.append(j['y'])
                polygons.append({'all_points_x':all_points_x,'all_points_y':all_points_y})

            self.add_image(
                "vehicle_tire",
                image_id=image_id,  # use file name as a unique image id
                path=path,
                width=width, 
                height=height,
                class_id=name_id,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "vehicle_tire":
            return super(self.__class__, self).load_mask(image_id)

        name_id = image_info["class_id"]
   
        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        class_ids = np.array(name_id, dtype=np.int32)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "vehicle_tire":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
   
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()
    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image))*255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print(r['rois'])

        for i, roi in enumerate(r['rois']):            
            skimage.io.imsave('roi_'+str(i)+'.jpg', image[roi[0]:roi[2],roi[1]:roi[3]])
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
# ###########################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TireConfig()
    else:
        class InferenceConfig(TireConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

""

