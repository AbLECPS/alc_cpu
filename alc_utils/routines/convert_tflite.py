from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import tensorflow.keras.metrics
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K 
import argparse
import numpy as np
import tensorflow.keras.utils
import cv2
import tensorflow.keras.backend as K
import tensorflow as tf
import sys
import os
import LECModel

def run(input_path=''):
    model = LECModel.get_model()
    fname = os.path.join("model_weights.h5")
    model.load_weights(fname)
    if (not input_path):
        input_path= os.path.join(os.getenv('ALC_WORKING_DIR'), 'jupyter', 'admin_BlueROV', 'LEC2_data', 'data', 'sss_rplidar10','scan')#
        if (not os.path.exists(input_path)):
            print('input data path for quantization not specified. default path does not exist. default-path = '+ input_path)
            sys.exit(1)

    imagesp = []
    content = (os.listdir(input_path))
    for imagePath in sorted(content):
        imagesp.append(imagePath)
        
    images = []
    for imagePath in imagesp:
        image = cv2.imread(input_path + '/'+ imagePath,cv2.IMREAD_GRAYSCALE)
        image = image.reshape((image.shape[0],image.shape[1],1))
        image = (np.float32(image)) / 255
        images.append(image)
        
        
    print(len(images))
    X = np.asarray(images)


    model.input.set_shape((1,) + model.input.shape[1:])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] #4 options - but only one option available as others are experimental or deprecated.

    def representative_data_gen():
        for xi in X:
          sample_tensor = tf.reshape(tf.convert_to_tensor(xi, dtype=tf.float32), model.input.shape)
          yield [sample_tensor]
          
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    with open('lec2_quant.tflite', 'wb') as f:
      f.write(tflite_model)

    os.system("edgetpu_compiler lec2_quant.tflite")
