###################################
#
#
# 2024-03-11
#
# predict the TEST image without pts.
#
#
###################################

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from train import create_dir, load_dataset, image_h, image_w, num_landmarks

global image_h
global image_w
global num_landmarks

def plot_lankmarks(image, landmarks):
    h, w, _ = image.shape
    radius = int(h * 0.005)

    for i in range(0, len(landmarks), 2):
        x = int(landmarks[i] * w)
        y = int(landmarks[i+1] * h)

        image = cv2.circle(image, (x, y), radius, (255, 0, 0), -1)

    return image

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Hyperparameters """
#    image_h = 512
#    image_w = 512
#    num_landmarks = 51 #106
    print(f"check {image_h} x {image_w}")

    """ SavedModel """
    SavedModel = "efv2b0_alb.5_0101_g4_CDR_mse_224_sc2d_0-255_11102_0315"
    
    
    """ Paths """
    #    dataset_path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/LaPa"
    dataset_path = "../data/test_1/ivslab_facial_test_private_qualification"
    model_path = os.path.join("files", SavedModel)
    #csv_path = os.path.join("files", SavedModel, "data.csv")

#    """ Loading the dataset """
#    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
#    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
#    print("")

    """ Load the model """
    model = tf.keras.models.load_model(model_path)
    # model.summary()


    """ Get test image list """
    test_img_list = sorted(glob(os.path.join(dataset_path, "*.png")))
    


    """ Prediction test image only """
    for x in tqdm(test_img_list, total=len(test_img_list)):
    #for x, y in tqdm(zip(valid_x[:10], valid_y[:10]), total=len(train_x)):
    #for x, y in tqdm(zip(train_x, train_y), total=len(train_x)):
    #for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        #print(f"org_size: h:{h}, w:{w}")
        image_x = image
        image = cv2.resize(image, (image_w, image_h))
#        image = image/255.0 ## (512, 512, 3) ##這次album image not convert back to !!!
        image = np.expand_dims(image, axis=0) ## (1, 512, 512, 3)
        image = image.astype(np.float32)

#        """ Landmarks """
#        #data = open(y, "r").read()
#        data = open(y, "r").read()[26:-2]
#        landmarks = []
#        #for line in data.strip().split("\n")[1:]:
#        for line in data.strip().split("\n"):
#            x, y = line.split(" ")
#            x = float(x)/image_x.shape[1]
#            y = float(y)/image_x.shape[0]
#
#            landmarks.append(x)
#            landmarks.append(y)
#
#        landmarks = np.array(landmarks, dtype=np.float32)

        """ Prediction """
        pred = model.predict(image, verbose=0)[0]
        pred = pred.astype(np.float32)

#        print(f"{pred} {np.array(pred).shape} {type(pred)}")
        #exit()
        
        pred = pred.reshape(-1, 102)[0]
#        print(f"{pred} {np.array(pred).shape} {type(pred)}")
#        exit()
        
        """ Save pts as txt files """
        """
        version: 1
        n_points: 51
        {
        ...51 lines
        }
        """
        
        f = open(f"test_data/{name}.txt", 'w')
        f.write(f'version: 1\n')
        f.write(f'n_points: 51\n')
        f.write('{\n')
        
        #print(f"pred :{pred} {pred.shape}")
        for i in range(0, len(pred), 2):
            #x = int(pred[i] * w)
            #y = int(pred[i+1] * h)
            x = pred[i] * w
            y = pred[i+1] * h
            
            #print(f"org_size_xy: {x} {y}")
            
            # write 52 xy to txt
            f.write(f'{x:.3f} {y:.3f}\n')
            
        f.write('}')
        f.write('\n')
        f.close()
        

#        """ Saving the results """
#        #gt_landmarks = plot_lankmarks(image_x.copy(), landmarks)
#        pred_landmarks = plot_lankmarks(image_x.copy(), pred)
#        #line = np.ones((image_x.shape[0], 10, 3)) * 255
#
#        #cat_images = np.concatenate([gt_landmarks, line, pred_landmarks], axis=1)
#        cv2.imwrite(f"results/{name}.png", pred_landmarks)
#
#
#
#        exit()


    ## ...
