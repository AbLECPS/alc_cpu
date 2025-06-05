# import libraries defined in this project
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import tensorflow.keras.metrics
from sklearn import preprocessing
from tensorflow.keras.optimizers import Adam
#from callbacks.trainingmonitor import TrainingMonitor
#from callbacks.epochcheckpoint import EpochCheckpoint
import tensorflow.keras.backend as K 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.utils
from sklearn.model_selection import train_test_split
from os import path 
from imutils import paths
import cv2

import tensorflow.keras.backend as K
import tensorflow as tf
import sys
import os
from sklearn.metrics import confusion_matrix

from scipy.signal import find_peaks 
import warnings
import pickle
import yaml
import time
from alc_utils.common import load_python_module
#import compute_am


alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)


def fix_folder_path(folder_path):
    if (not folder_path):
        return None
    pos = folder_path.find('jupyter')
    if (pos == -1):
        return folder_path
    folder_path = folder_path[pos:]
    if (alc_working_dir_name):
        ret = os.path.join(alc_working_dir_name, folder_path)
        return ret
    return None


# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def compute_iou(y_pred, y_true):
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()

     current = confusion_matrix(y_true, y_pred,labels=[0,2])
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return IoU[1]

def mean_IoU(y_true, y_pred):
    s = K.shape(y_true)
    # reshape such that w and h dim are multiplied together
    y_true_reshaped = K.reshape( y_true, tf.stack( [-1, s[1]*s[2], s[-1]] ) )
    y_pred_reshaped = K.reshape( y_pred, tf.stack( [-1, s[1]*s[2], s[-1]] ) )
    # correctly classified
    clf_pred = K.one_hot( K.argmax(y_pred_reshaped), num_classes = s[-1])
    equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped
    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(y_true_reshaped,axis=1) + K.sum(y_pred_reshaped,axis=1)
    iou = intersection / (union_per_class - intersection)
    iou_mask = tf.is_finite(iou)
    iou_masked = tf.boolean_mask(iou,iou_mask)
    return K.mean( iou_masked )

def map_to_label(x):   
    if(x[1]>200):
        return 1
    elif(x[2]>200):
        return 2
    else:
        return 0
        
def convert(im):
    for i in range(99):
        for j in range(511):
            print (str(i) + ',' +str(j))
            print (im[i][j][0])
    return im



def label_to_image(x1):
    x = int(round(x1))
    if x > 0:
        return 255
    else:
        return 0


def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image




def compute_accuracy(pred,image):
    tp =0
    tn =0
    fp =0
    fn=0
    acc = 0
    
    pipe_pos_pred = get_pipe_pos_from_semseg(pred)
    pipe_pos_gt = get_pipe_pos_from_semseg(image)
    if pipe_pos_gt == None :
        if pipe_pos_pred == None :
            tn =1
        else:
            fp = 1
    if pipe_pos_pred == None :
        if pipe_pos_gt != None :
            fn =1
        
    if pipe_pos_gt != None and pipe_pos_pred != None :
        acc =  1 - abs(pipe_pos_gt - pipe_pos_pred)
        tp = 1
    
    #print(acc)
    return tp, tn, fp, fn, acc
            
        

def get_pipe_pos_from_semseg(image):
    
    image_np = image[:,:,2]
    #image_np = np.where(image_np<100,0,image_np) 
    
    # looking for peaks in the image:
    pos_array = []
    [height, width] = np.shape(image_np)
    for i in range(height):
        peaks, _ =  find_peaks(image_np[i,:], height=50, width=2)
        #print(peaks.all())
        #print (np.nanmean(peaks.all()))
        if np.isnan(peaks.all()):
            continue
        if np.isreal(peaks.all()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                val = np.nanmean(peaks) / width
                if (not np.isnan(val)):
                    pos_array.append(val)
    #printp(pos_array)
    if len(pos_array) > 10:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pos = np.nanmean(pos_array)
            var = np.nanvar(pos_array)
            return pos
    
    return None
    
def run(lec_path, data_path, out_filename):
    lec_path = fix_folder_path(lec_path)
    data_path = fix_folder_path(data_path)
    
    lec_module_file = os.path.join(os.path.dirname(lec_path), 'LECModel.py')
    if os.path.exists(lec_module_file):
        model_module = load_python_module(lec_module_file)
    else:
        print(' lec module file not found in '+ lec_module_file)
        raise Exception ('lec module not found in '+lec_module_file)

    revmfunc = np.vectorize(label_to_image)
    mfunc = np.vectorize(map_to_label)
    training_images = []
    training_masks = []
    val_images = []
    val_masks = []
    masksp = []
    content = list(paths.list_images(os.path.join(data_path,"gt")))
    for imagePath in sorted(content):
        masksp.append(imagePath)
    imagesp = []
    content = list(paths.list_images(os.path.join(data_path,"scan")))
    for imagePath in sorted(content):
        imagesp.append(imagePath)

    images = []
    masks = []
    gt_images = []
    for imagePath in imagesp:
        image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        image = image.reshape((image.shape[0],image.shape[1],1))
        image = (np.float32(image)) / 255
        images.append(image)
        

    for imagePath in masksp:
        image = cv2.imread(imagePath,cv2.IMREAD_COLOR)
        image = image.reshape((image.shape[0],image.shape[1],3))
        gt_images.append(image)
        mask = np.zeros((image.shape[0],image.shape[1]))
        mask[image[:,:,2]>200] = 2
        masks.append(mask)
        
        
    X = np.asarray(images)
    y = np.asarray(masks)
    dict={}
    #with open('params.yml') as f:
    #    dict = yaml.load(f,Loader=yaml.SafeLoader)
    model = model_module.get_model(**dict)
    model.summary()
    model.load_weights(lec_path)
    start = time.process_time()
    preds = model.predict(X,batch_size=2)
    end = time.process_time()
    compute_time = (end-start)/(X.shape[0])
    #am_result, nominal_index, high_index = compute_am.run(am_test_path,X)

    count = 0 
    sums = 0
    accs = 0
    results = []
    count_tp=count_tn=count_fp=count_fn = 0
    for i in range(X.shape[0]):
        
        pred0 = (preds[i]*255.0).astype(np.uint8)
        #pred0 = np.where(pred0<100,0,pred0)
        pred1 =  revmfunc(pred0)
        pred1 = pred1.copy()
        pred1 = np.float32(pred1)
        tp,tn,fp,fn,accuracy = compute_accuracy(pred1,gt_images[i])

        if (tp == 1):
            predmask = np.zeros((pred1.shape[0],pred1.shape[1]))
            predmask[pred1[:,:,2]>50] = 2
            if (np.count_nonzero(y[i] == 2)==0):
                continue
            #iou = compute_iou(pred2.argmax(axis=-1),y[i].argmax(axis=-1))
            #iou = compute_iou(pred0.argmax(axis=-1),y[i].argmax(axis=-1))
            iou = compute_iou(predmask, y[i])
            results.append([iou,tp,tn,fp,fn,accuracy, iou])
            sums+= iou
            accs += accuracy
            count+=1
        count_tp +=tp
        count_tn +=tn
        count_fp += fp
        count_fn += fn

    if (count):   
        mean_iou = sums*1.0/count
        mean_acc = accs*1.0/count
    else:
        mean_iou = 0
        mean_acc = 0
    tp_rate  = count_tp#*1.0/(X.shape[0])
    tn_rate  = count_tn#*1.0/(X.shape[0])
    fp_rate  = count_fp#*1.0/(X.shape[0])
    fn_rate  = count_fn#*1.0/(X.shape[0])
    if (tp_rate + fp_rate)==0:
        precision = 0
    else:
        precision = tp_rate*1.0/(tp_rate +fp_rate)

    if (tp_rate + fn_rate)==0:
        recall  = 0
    else:
        recall    = tp_rate*1.0/(tp_rate +fn_rate)

    pickle.dump(results, open(out_filename,'wb'))
    #print("MeanIoU:",mean_iou)

    return  mean_iou, mean_acc, tp_rate, tn_rate, fp_rate, fn_rate, precision, recall, compute_time, results#, am_result, nominal_index, high_index


# if (len(sys.argv) not in [4,5]):
#     print("Usage: run_test.py <lec_path> <data_folder> <out_file(.pkl)>")

# lec_path= sys.argv[1]
# data_path= sys.argv[2]
# out_filename =sys.argv[3]

