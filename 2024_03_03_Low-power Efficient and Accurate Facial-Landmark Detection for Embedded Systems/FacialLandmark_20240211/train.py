
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger


# albumentations
from functools import partial
# from albumentations import (
#     Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
#     Rotate
# )
import albumentations as A

from tqdm import tqdm

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

global image_h
global image_w
global num_landmarks


""" Hyperparameters """
image_h = 512
image_w = 512
num_landmarks = 51 #106



def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#def load_dataset(path):
#    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
#    train_y = sorted(glob(os.path.join(path, "train", "landmarks", "*.txt")))
#
#    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
#    valid_y = sorted(glob(os.path.join(path, "val", "landmarks", "*.txt")))
#
#    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
#    test_y = sorted(glob(os.path.join(path, "test", "landmarks", "*.txt")))
#
#    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


#def load_dataset(path):
#    train_x = sorted(glob(os.path.join(path, "300W", "images", "*.png")))
#    train_y = sorted(glob(os.path.join(path, "300W", "labels", "*.pts")))
#
#    valid_x = sorted(glob(os.path.join(path, "afw", "images", "*.jpg")))
#    valid_y = sorted(glob(os.path.join(path, "afw", "labels", "*.pts")))
#
#    test_x = sorted(glob(os.path.join(path, "afw", "images", "*.jpg")))
#    test_y = sorted(glob(os.path.join(path, "afw", "labels", "*.pts")))
#
#    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

face_dirs = ['IFPW',
'afw',
'helen',
'300W',
'ibug']
import itertools
# load all sub_dir into one train ds #
def load_dataset(path):
    np.random.seed(42)
    img_ls=[]
    pts_ls=[]
    
    def get_tupe_list():
        for face in face_dirs:
            tmp_img_ls = sorted(glob(os.path.join(path, face, "images", "*.*g")))
            img_ls.extend(tmp_img_ls)

            tmp_pts_ls = sorted(glob(os.path.join(path, face, "labels", "*.pts")))
            pts_ls.extend(tmp_pts_ls)
            
        print(f'number of image:{len(img_ls)}, number of pts:{len(pts_ls)}')
        
        print(f'#️⃣img ten:{img_ls[10]}')
        print(f'#️⃣pts ten:{pts_ls[10]}')
        
        return (img_ls, pts_ls)
    
    # speratre shuffle will not sysnc
#    np.random.shuffle(img_ls)
#    np.random.shuffle(pts_ls)
#    
#    print(f'img ten:{img_ls[10]}')
#    print(f'pts ten:{pts_ls[10]}')

    (img_ls, pts_ls) = get_tupe_list()
    train_xy = [list(i) for i in zip(img_ls, pts_ls)]
    
    print(f'sorted train_xy[10]: {train_xy[10]}')
    
    
    
    np.random.shuffle(train_xy)
    print(f'shuffled train_xy[10]: {train_xy[10]}')
    
    print(f'take x, y train_xy[10]: {train_xy[10][0]} {train_xy[10][1]}')
    
    split_size = int(len(train_xy) / 5)
    print(f'split_size: {split_size}')
    
    val_xy = train_xy[:split_size]
    train_xy = train_xy[split_size:]
    print(f'#️⃣val_xy:{len(val_xy)} \n#️⃣train_xy:{len(train_xy)}  ')
    

    #print(f'train_xy : {train_xy}')
    valid_x, valid_y = zip(*val_xy)
    print(f'val x,y : {valid_x[10]},{valid_y[10]}')
    
    train_x, train_y = zip(*train_xy)
    print(f'train x,y : {train_x[10]},{train_y[10]}')
    

    #return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    return (list(train_x), list(train_y)), (list(valid_x), list(valid_y)), (list(valid_x), list(valid_y))
    
#well land not the lank:)
def read_image_lankmarks(image_path, landmark_path):
    
    """ Check """
    #print(f'image_path: {image_path}')
    
    """ Image """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    image = cv2.resize(image, (image_w, image_h))
    #image = image/255.0
    #image = image.astype(np.float32) # TypeError: Image must be RGB image in uint8 format.

    """ Lankmarks """
    #data = open(landmark_path, "r").read()
    data = open(landmark_path, "r").read()[26:-2]
    landmarks = []
    
    #check
    #print(data)
#    print(f"Check pts too many values to uppack!")
#    print(f"✅ r {landmark_path}")

    #for line in data.strip().split("\n")[1:]: ##the example txt has a dump ""
#    count = 1
    for line in data.strip().split("\n"):
#        print(f"✅ line {count}: {line}")
        x, y = line.split(" ", 1)  #ValueError: too many values to unpack (expected 2) :force to 1 split!!
        #print(f"✅ {count}: {x} {y}")
#        count += 1
        
#        x = float(x)/w
#        y = float(y)/h

#        landmarks.append(x)
#        landmarks.append(y)


        x = float(x)/w * image_w #轉回0,1 轉回0,512，但以float，不是int給Album
        y = float(y)/h * image_h
        
        """ Check xy dtype"""
        print(f"x : {x}, y:{y}")
        
#        x = int(float(x)/w * image_w)
#        y = int(float(y)/h * image_h)
        
#        """ Let keypoints always less than <512 """
#        if x >= image_w:
#            x = int(image_w-1)
#        if y >= image_h:
#            y = int(image_h-1)
        
        
        
        #xy = (x, y)
        landmarks.append((x, y))


    landmarks = np.array(landmarks,  dtype=int) # dtype=np.float32) - Argument 'borderMode' is required to be an integer

    return image, landmarks


# Testing keypoints augment
prob=0.5
transforms = A.Compose([

            #New add: roate
            #A.Rotate(always_apply=False, p=prob, limit=(-3, 3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False), #>30 easy outer the image

            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=prob), # merge RandomBrightnessContrast and RandomContrast
            #A.RandomBrightness(limit=0.1, p=0.5),
            #A.RandomContrast(limit=0.2, p=0.5),
            
            A.ImageCompression(quality_lower=85, quality_upper=100, p=prob),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=prob),
            
            A.FancyPCA(alpha=0.1, always_apply=False, p=prob),
            A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=0, always_apply=False, p=prob),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=prob),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=prob),
    
            A.HorizontalFlip(p=prob),
            A.RandomResizedCrop(always_apply=False, height=image_w, width=image_h, scale=(0.95, 0.95), ratio=(1.0, 1.0), interpolation=0, p=prob),#xy become double need change dtype of label. # pp will outside the image.
#            A.IAAAffine(scale=0.9, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0, mode='reflect', always_apply=False, p=0.5),
            #A.Affine(scale=0.9, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, cval=0, mode='reflect', always_apply=False, p=0.5), # somehow,
#             A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5), interpolation=1, border_mode=2, value=(0, 0, 0), mask_value=None),
    #2021-02-26
#             A.IAAPerspective(scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5),#fallout image make train stop. NOT support keypoints!!!!!
            ]
            ,
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=True),  #currently not works for tf.ds yet.
            )
prob_dump=0.0
transforms_dump = A.Compose([

            #New add: roate
            #A.Rotate(always_apply=False, p=prob, limit=(-3, 3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False), #>30 easy outer the image

            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=prob_dump), # merge RandomBrightnessContrast and RandomContrast
            #A.RandomBrightness(limit=0.1, p=0.5),
            #A.RandomContrast(limit=0.2, p=0.5),
            
            A.ImageCompression(quality_lower=85, quality_upper=100, p=prob_dump),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=prob_dump),
            
            A.FancyPCA(alpha=0.1, always_apply=False, p=prob_dump),
            A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=0, always_apply=False, p=prob_dump),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=prob_dump),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=prob_dump),
    
            A.HorizontalFlip(p=prob_dump),
            A.RandomResizedCrop(always_apply=False, height=image_w, width=image_h, scale=(0.95, 0.95), ratio=(1.0, 1.0), interpolation=0, p=prob_dump),#xy become double need change dtype of label. # pp will outside the image.
#            A.IAAAffine(scale=0.9, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0, mode='reflect', always_apply=False, p=0.5),
            #A.Affine(scale=0.9, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, cval=0, mode='reflect', always_apply=False, p=0.5), # somehow,
#             A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5), interpolation=1, border_mode=2, value=(0, 0, 0), mask_value=None),
    #2021-02-26
#             A.IAAPerspective(scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5),#fallout image make train stop. NOT support keypoints!!!!!
            ]
            ,
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=True),  #currently not works for tf.ds yet.
            )



""" The sorce method do not tupe the xy to (x, y), but the Album need this form.
⏺️ landmarks: [0.35639685 0.38094288 0.3709075  0.34562573 0.40442926 0.34206435
 0.43936437 0.35029185 0.473686   0.36189276 0.5224809  0.36916402
 0.55873966 0.35745946 0.59468687 0.35445604 0.6316304  0.3649319
 0.64301056 0.3986515  0.50396395 0.39300892 0.50287277 0.42714468
 0.5030303  0.4660843  0.5035351  0.5057091  0.4622902  0.50932425
 0.4810329  0.5163749  0.50139666 0.52370554 0.5212005  0.51873904
 0.53573895 0.51152456 0.397747   0.38974404 0.41594183 0.37413055
 0.44061044 0.38059813 0.462033   0.39859337 0.43995765 0.4033456
 0.4140859  0.4024192  0.5385536  0.40543196 0.5600743  0.3864952
 0.58542216 0.3870027  0.6014736  0.40102947 0.5870127  0.40829158
 0.5612783  0.41012493 0.42888787 0.5575796  0.4519649  0.5587046
 0.48090333 0.55623823 0.4931655  0.56557906 0.51644766 0.56065
 0.5369601  0.56175035 0.56012815 0.56243616 0.5363229  0.57833934
 0.51524013 0.58580583 0.49226874 0.5863422  0.47647375 0.5844641
 0.45113432 0.5746232  0.43672726 0.5603641  0.47906238 0.566793
 0.49290186 0.57000816 0.5164001  0.5667943  0.55072325 0.5636691
 0.5164001  0.5667943  0.49290186 0.57000816 0.47906238 0.566793  ]
 
 -->
 
 [ (100, 200), (100, 200), (100, 200), (100, 200),...
 
 ]
 
 
 AND
 
 TypeError: Image must be RGB image in uint8 format.
 
 
 AND
 
     ValueError: Expected x for keypoint (512, 148, 0.0, 0.0) to be in the range [0.0, 512], got 512.
     原始xy就在邊框上512 148時，Album報錯！
     
     if x or y >512 = 511?
     
     OR
     
     讀取時無限不進位？？
     
    INVALID_ARGUMENT:  Incompatible shapes at component 1: expected [?,102] but got [1,100].
     helen/190546275_1.png 都是這一張在搞鬼，因為眼睛已經接近邊緣，經過裁剪後點就不見了！直接刪除這一張圖即可
     240001815_1.png
     
     或是減少裁減比例0.9 --> 0.95 還是有點被切掉
     445.png
     
 
 """

def vis_keypoints(image, keypoints, color=(0,255,0), diameter=2, pth="album_test/album_check.png"):
    im_k = image.copy()
    
    for (x,y) in keypoints:
        cv2.circle(im_k, (int(x), int(y)), diameter, color, -1)
        
    cv2.imwrite(pth, im_k)

def vis_keypoints_aug(image, keypoints, color=(124,255,124), diameter=2, pth="album_test/album_check.png"):
    im_k = image.copy()
    
    #for (x,y) in keypoints:
    x = keypoints[0]
    y = keypoints[1]
    for (x,y) in zip(x,y):
        cv2.circle(im_k, (int(x), int(y)), diameter, color, -1)
        
    cv2.imwrite(pth, im_k)
    
    
def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        f_pth = x # keep the file path
        #f_par = os.path.join(f_pth.split("/")[:-2])
        p_pth = f_pth.replace("images", "labels")
        #p_pth = f_pth.replace("jpg", "pts")
                
#        print(f"✅ p {y}")
        
        image, landmarks = read_image_lankmarks(x, y) #return a 512 image and 512 ranged pts for Album
        
        print(f"⏺️ [read_image_lankmarks] {f_pth} landmarks: {landmarks} 🈴 {len(landmarks)}")
        vis_keypoints(image, landmarks, pth=f"album_test/album_check_{name}.png")
        
        
        
        """ Album here """
        aug_data = transforms(image=image, keypoints=landmarks)
        #aug_data = transforms(image=image, keypoints=list(zip(*landmarks)))  # someone hit this
        image = aug_data["image"]
        landmarks  = aug_data["keypoints"]
        
        
        
        #INVALID_ARGUMENT:  0-th value returned by pyfunc_0 is uint8, but expects float
        #need move to after Album finished the work.
#        image = image/255.0
        image = image.astype(np.float32)
        #INVALID_ARGUMENT:  1-th value returned by pyfunc_0 is int64, but expects float
        #landmarks_512 = [(np.float32(i), np.float32(j)) for (i,j) in landmarks]
        
        """ convert Albun 512 ranged pts to [0,1] image_w(512->0,1)"""
        #org_img = cv2.imread(f_pth, cv2.IMREAD_COLOR)
        #h, w, _ = org_img.shape
        landmarks = [(np.float32(i/image_w), np.float32(j/image_h)) for (i,j) in landmarks] # landmarks 0,1
        
        """ IF aug_landmarks < 51 continue """
        if len(landmarks) < 51:
            print(f"❌ ❌  <102 {len(landmarks)} ")
            #continue
    
        print(f"⏺️ {f_pth} landmarks aug: {landmarks} 🈴 {len(landmarks)}")
        vis_keypoints(image, landmarks, pth=f"album_test/album_check_{name}_aug.png")
        
        
        
        #INVALID_ARGUMENT:  Incompatible shapes at component 1: expected [?,102] but got [1,51,2].
        #最後又要從[(x,y),(x,y),(x,y),...]==[batch, 51, 2] 打散回 [x y x y xy ...]==[batch, 102]
        landmarks_102=[]
        #for xy in landmarks:
        #    print(f'xy: {xy}')
        for (x,y) in landmarks:
            landmarks_102.append(x)
            landmarks_102.append(y)
            
        landmarks_102 = np.array(landmarks_102) #To np array
        
#        print(f"⏺️ {f_pth} landmarks aug to ,102]: {landmarks_102} 🈴 {len(landmarks_102)}")
        
        if len(landmarks_102) < 102:
            print(f"❌ ❌  <102 {len(landmarks)} ")
            print("")
            print("")
            print(f"mv {f_pth} ")
            print(f"mv {p_pth} ")
            print("")
            print("")
            
        
        """ Album here """
        
        
        """ Check ds-pts shape """
        print(f'🔎 Albu-code preprocess-out-pts shape landmarks: {len(landmarks)} {landmarks} {np.array(landmarks).shape} {type(landmarks)}') #AttributeError: 'list' object has no attribute 'shape'
        print(f'🔎 Albu-code preprocess-out-pts shape landmarks_102: {len(landmarks_102)} {landmarks_102} {np.array(landmarks_102).shape} {type(landmarks_102)}') #AttributeError: 'list' object has no attribute 'shape'
        exit()
        
        return image, landmarks_102

    image, landmarks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([image_h, image_w, 3])
    landmarks.set_shape([num_landmarks * 2])

    return image, landmarks




#                           #
#                           #
# For validation ds process #
#                           #
#                           #


def preprocess_val_ds(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        f_pth = x # keep the file path
        #f_par = os.path.join(f_pth.split("/")[:-2])
        p_pth = f_pth.replace("images", "labels")
        #p_pth = f_pth.replace("jpg", "pts")
                
#        print(f"✅ p {y}")
        
        image, landmarks = read_image_lankmarks(x, y) #return a 512 image and 512 ranged pts for Album
        
#        print(f"⏺️ {f_pth} landmarks: {landmarks} 🈴 {len(landmarks)}")
#        vis_keypoints(image, landmarks, pth=f"album_test/album_check_{name}.png")
        
        
        
        """ Album here """
        aug_data = transforms_dump(image=image, keypoints=landmarks)
        #aug_data = transforms(image=image, keypoints=list(zip(*landmarks)))  # someone hit this
        image = aug_data["image"]
        landmarks  = aug_data["keypoints"]
        
        
        
        #INVALID_ARGUMENT:  0-th value returned by pyfunc_0 is uint8, but expects float
        #need move to after Album finished the work.
        image = image/255.0
        image = image.astype(np.float32)
        #INVALID_ARGUMENT:  1-th value returned by pyfunc_0 is int64, but expects float
        #landmarks_512 = [(np.float32(i), np.float32(j)) for (i,j) in landmarks]
        
        """ convert Albun 512 ranged pts to [0,1] image_w(512->0,1)"""
        #org_img = cv2.imread(f_pth, cv2.IMREAD_COLOR)
        #h, w, _ = org_img.shape
        landmarks = [(np.float32(i/image_w), np.float32(j/image_h)) for (i,j) in landmarks] # landmarks 0,1
        
        
        """ IF aug_landmarks < 51 continue """
        if len(landmarks) < 51:
            print(f"❌ ❌  <102 {len(landmarks)} ")
            #continue
    
#        print(f"⏺️ {f_pth} landmarks aug: {landmarks} 🈴 {len(landmarks)}")
#        vis_keypoints(image, landmarks, pth=f"album_test/album_check_{name}_aug.png")
        
        
        
        #INVALID_ARGUMENT:  Incompatible shapes at component 1: expected [?,102] but got [1,51,2].
        #最後又要從[(x,y),(x,y),(x,y),...]==[batch, 51, 2] 打散回 [x y x y xy ...]==[batch, 102]
        landmarks_102=[]
        #for xy in landmarks:
        #    print(f'xy: {xy}')
        for (x,y) in landmarks:
            landmarks_102.append(x)
            landmarks_102.append(y)
            
        landmarks_102 = np.array(landmarks_102) #To np array
        
#        print(f"⏺️ {f_pth} landmarks aug to ,102]: {landmarks_102} 🈴 {len(landmarks_102)}")
        
        if len(landmarks_102) < 102:
            print(f"❌ ❌  <102 {len(landmarks)} ")
            print("")
            print("")
            print(f"mv {f_pth} ")
            print(f"mv {p_pth} ")
            print("")
            print("")
            
        
        """ Album here """
        
        
#        """ Check ds-pts shape """
#        print(f'🔎 [val] Albu-code preprocess-out-pts shape landmarks: {len(landmarks)} {landmarks} {np.array(landmarks).shape} {type(landmarks)}') #AttributeError: 'list' object has no attribute 'shape'
#        print(f'🔎 [val] Albu-code preprocess-out-pts shape landmarks_102: {len(landmarks_102)} {landmarks_102} {np.array(landmarks_102).shape} {type(landmarks_102)}') #AttributeError: 'list' object has no attribute 'shape'
#        exit()
        
        return image, landmarks_102

    image, landmarks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([image_h, image_w, 3])
    landmarks.set_shape([num_landmarks * 2])

    return image, landmarks



#                           #
#                           #
# For validation ds process #
#                           #
#                           #




#def preprocess_for_check_outdoor_pts(x, y):
##    def f(x, y):
##        #print(f'x : {x}') # if not with map(), the x is list [..] of image path
##    
##        #x = x.decode()
##        #y = y.decode()
#        
#    num_of_x = len(x)
#    print(f'num_of_x : {num_of_x}')
#    
#    for x, y in tqdm(zip(x,y), total=num_of_x):
#
#        """ Extract the name """
#        name = x.split("/")[-1].split(".")[0]
#        f_pth = x # keep the file path
#        #f_par = os.path.join(f_pth.split("/")[:-2])
#        p_pth = f_pth.replace("images", "labels")
#        #p_pth = f_pth.replace("jpg", "pts")
#                
##        print(f"✅ p {x} {y}")
#        
#        image, landmarks = read_image_lankmarks(x, y)
#        
##        print(f"⏺️ {f_pth} landmarks: {landmarks} 🈴 {len(landmarks)}")
##        vis_keypoints(image, landmarks, pth=f"album_test/album_check_{name}.png")
#        
#        
#        
#        """ Album here """
#        aug_data = transforms(image=image, keypoints=landmarks)
#        #aug_data = transforms(image=image, keypoints=list(zip(*landmarks)))  # someone hit this
#        image = aug_data["image"]
#        landmarks  = aug_data["keypoints"]
#        
#        
#        
#        #INVALID_ARGUMENT:  0-th value returned by pyfunc_0 is uint8, but expects float
#        #need move to after Album finished the work.
#        #image = image/255.0
#        image = image.astype(np.float32)
#        #INVALID_ARGUMENT:  1-th value returned by pyfunc_0 is int64, but expects float
#        landmarks = [(np.float32(i), np.float32(j)) for (i,j) in landmarks]
#        
#        """ IF aug_landmarks < 51 continue """
#        if len(landmarks) < 51:
#            print(f"❌ ❌  <102 {len(landmarks)} ")
#            #continue
#    
##        print(f"⏺️ {f_pth} landmarks aug: {landmarks} 🈴 {len(landmarks)}")
##        vis_keypoints(image, landmarks, pth=f"album_test/album_check_{name}_aug.png")
#        
#        
#        
#        #INVALID_ARGUMENT:  Incompatible shapes at component 1: expected [?,102] but got [1,51,2].
#        #最後又要從[(x,y),(x,y),(x,y),...]==[batch, 51, 2] 打散回 [x y x y xy ...]==[batch, 102]
#        landmarks_102=[]
#        #for xy in landmarks:
#        #    print(f'xy: {xy}')
#        for (x,y) in landmarks:
#            landmarks_102.append(x)
#            landmarks_102.append(y)
##        print(f"⏺️ {f_pth} landmarks aug to ,102]: {landmarks_102} 🈴 {len(landmarks_102)}")
#        
#        if len(landmarks_102) < 102:
#            print(f"❌ ❌  <102 {len(landmarks)} ")
#            print("")
#            print("")
#            print(f"mv {f_pth} ")
#            print(f"mv {p_pth} ")
#            print("")
#            print("")
#            
#        
#        """ Album here """
#            
#            
##            return image, landmarks_102
#            
##    f(x, y)
#    
##    image, landmarks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
##    image.set_shape([image_h, image_w, 3])
##    landmarks.set_shape([num_landmarks * 2])
##
##    return image, landmarks


def tf_dataset(x, y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    
#    # small train for fast test #
#    ds = ds.take(100)
#    print(f'samll train_ds: {ds.cardinality()}')
#    #exit()
    
    ds = ds.shuffle(buffer_size=ds.cardinality(), reshuffle_each_iteration=True).map(preprocess)
    ds = ds.batch(batch).prefetch(buffer_size=AUTOTUNE)
    ds = ds.cache()
    return ds

def tf_dataset_val(x, y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    #ds = tf.data.Dataset.from_tensor_slices((x, y)).map(preprocess_val_ds)
    ds = ds.shuffle(buffer_size=len(ds), reshuffle_each_iteration=False).map(preprocess_val_ds)
    ds = ds.batch(batch).prefetch(buffer_size=AUTOTUNE)
    ds = ds.cache()
    return ds
    
def build_model(input_shape, num_landmarks):
    """ (pts,)"""
    inputs = L.Input(input_shape)

    #backbone = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=1.0)
    #EfficientNetV2B0 : default 0,255
    backbone = EfficientNetV2B0(include_top=False, weights="imagenet", input_tensor=inputs)

    backbone.trainable = True

    x = backbone.output
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.2)(x)
    outputs = L.Dense(num_landmarks*2, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model


def build_model_SeparableConv2D(input_shape, num_landmarks):
    """ [?, 1, 1, pts]"""
    #inputs = L.Input(input_shape)

    #backbone = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=1.0)
    #EfficientNetV2B0 : default 0,255
    backbone = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=input_shape,) #input_tensor=inputs) # 3/15 replace by input_shape=

    backbone.trainable = True
    
#    x = backbone.output
#    x = L.GlobalAveragePooling2D()(x)
#    x = L.Dropout(0.2)(x)
#    outputs = L.Dense(num_landmarks*2, activation="sigmoid")(x)
#
#    model = tf.keras.models.Model(inputs, outputs)
#    return model

    inputs = layers.Input(input_shape)
    x = keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = backbone(x)
    x = L.Dropout(0.3)(x)
    x = L.SeparableConv2D(
            num_landmarks*2, kernel_size=5, strides=1, activation="relu"
    )(x)
    outputs = L.SeparableConv2D(
            num_landmarks*2, kernel_size=3, strides=1, activation="sigmoid"
    )(x)

    return keras.Model(inputs, outputs, name="keypoint_detector")
    

    

def CosineDecayCLRWarmUpLSW_3_warmup(epoch):
    
    #step_size = 25 # currently best for foot pp
    max_lr = 0.001#0.005 #1e-3 #1e-2 # currently best for foot pp
    base_lr = 1e-6 # 1e-6 1e-7

    # warm up
    lr_init_ep = base_lr #1e-3 # 1e-3 = 0.001
    lr_ramp_ep = 5 # 50 # set 5 for fast test
    lr_sus_ep  = 5
    #lr_decay   = 0.8


    #initial_learning_rate = 1e-3 # 1e-2 = 0.02
    first_decay_steps = 10


    lr_decayed_fn = (
      tf.keras.experimental.CosineDecayRestarts(
          max_lr,#initial_learning_rate,
          first_decay_steps,
          t_mul=1,
          m_mul=1,
          alpha = 0.0000001,
          name="CCosineDecayRestarts"))
    
#     return lr_decayed_fn(epoch)
    
    
    # warm up
    if epoch < lr_ramp_ep:
        return (max_lr - base_lr) / lr_ramp_ep * epoch + base_lr
    else:
        return lr_decayed_fn(epoch-lr_ramp_ep)
        
    #return lr


def plot_lr(epoch=100, lr=0.01, pth="./"):
    rng = [i for i in range(epoch)]
    y = [CosineDecayCLRWarmUpLSW_3_warmup(x) for x in rng]
    # print(y)

    # sns.set(style='darkgrid') # Remove this seting sns的風格無法控制 使用matplotlib本身的功能#
    fig, ax = plt.subplots(figsize=(20, 6))
    # plt.ylim(.0000000000000001, .01)# for too large loss
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))# for too small loss
    ax.grid() # Enabling grid lines
    # set the limits
    # ax.set_xlim([0, 200])
    plt.plot(rng, y)
    # plt.xticks(np.arange(0, 205, step=5))
    plt.annotate(f'y0: {y[0]:.10f}', xy=(0.05, 0.001), xytext=(0, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate(f'y-1: {y[-1]:.10f}', xy=(0.73, 0), xytext=(0, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.savefig(f'./{pth}_LR.png')
   
    
##                           #
## Save val image for check  #
##                           #
#
#
#def show_predictions(dataset=None, num=1, model=model):
#    if dataset:
#        for image, mask in dataset.take(num):
#            pred_mask = model.predict(image)
#            display_kares_array([image[0], mask[0], create_mask(pred_mask)])
#    else:
#        display_kares_array([sample_image, sample_mask,
#                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
#                 
#class DisplayCallback(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs=None):
#        clear_output(wait=True)
#        show_predictions(dataset=val_xy)
#        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
#        
##                           #
## Save val image for check  #
##                           #


if __name__ == "__main__":
    
    # ------- SPQR ------- #
    is_multi_GPU_training = "YES" #"NO"
    
    # tf MirroredStrategy seting
    strategy = tf.distribute.MirroredStrategy()
    REPLICAS = strategy.num_replicas_in_sync
    print('\nNumber of REPLICAS: {}\n'.format(REPLICAS))
    
    BATCH_SIZE = 16 #32
    MULTI_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
    print('BATCH_SIZE: {}, MULTI_BATCH_SIZE: {}'.format(BATCH_SIZE, MULTI_BATCH_SIZE))
    
    # tf.data autotune
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_reshuffle_each_iteration = True
    ds_shuffle_buffer_size = MULTI_BATCH_SIZE #64 # len(train_ds)
    print("# ------- SPQR ------- #\n")
    


    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    input_shape = (image_h, image_w, 3) #maybe gray scale is better for FLMD
    batch_size = MULTI_BATCH_SIZE #32
    lr = 1e-2
    num_epochs = 100
    
    """ SavedModel """
    SavedModel = "efv2b0_alb.5_0101_g4_CDR_BCE_512_lr2"

    """ Paths """
    #    dataset_path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/LaPa"
    dataset_path = "../data/ivslab_facial_train/"
    model_path = os.path.join("files", SavedModel)
    csv_path = os.path.join("files", SavedModel, "data.csv")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")
    
#    exit()

    """ Check ds out of image points """
    #preprocess_for_check_outdoor_pts(train_x, train_y)
    #exit()
    
    #preprocess_for_check_outdoor_pts(valid_x, valid_y)
    #exit()
    
    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset_val(valid_x, valid_y, batch=batch_size) #Donot made augmentation on valid ds

#    """ Check ds out of image points """
#    for element in tqdm(train_ds, total=train_ds.cardinality().numpy()):
#        #print(element[1][0])
#        #pass
#        continue
#    #exit()
#    
#    for element in tqdm(valid_ds, total=valid_ds.cardinality().numpy()):
#        continue
#    exit()
    
    """ Model """
    if is_multi_GPU_training == "YES":
        with strategy.scope():
            model = build_model(input_shape, num_landmarks)
            
            model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr))
            #metrics=[tf.keras.metrics.MeanSquaredError()] #same as lose, keep it.
            #model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr))

    else:
        model = build_model(input_shape, num_landmarks)
        
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr))
        #metrics=[tf.keras.metrics.MeanSquaredError()] #same as lose, keep it.
        #model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr))
        
        
    """ check model output shape """
    model.summary()
    #; exit()
    

    """ LR callback"""
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(CosineDecayCLRWarmUpLSW_3_warmup)
         
    plot_lr(num_epochs, lr, pth=SavedModel)
    #exit()
    
    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        #ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        lr_schedule,
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=False)
    ]

    print(f"image_h:{image_h}, image_w:{image_w}, batch_size:{batch_size}, MULTI_BATCH_SIZE:{MULTI_BATCH_SIZE}, num_landmarks:{num_landmarks}, num_epochs:{num_epochs}, lr:{lr}, loss:MSE, SavedModel:{SavedModel}")
    
    model.fit(train_ds,
        validation_data=valid_ds,
        epochs=num_epochs,
        callbacks=callbacks
    )




### ...
