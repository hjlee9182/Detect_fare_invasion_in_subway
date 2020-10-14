import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense,Activation,GlobalAveragePooling2D

def addTopModelMobileNet(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)

    return top_model


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='../../data/video/crawl/IMG_4641.mp4')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()
    
    mobilenet = MobileNet(include_top=False , input_shape=(224,224,3))
    fc = addTopModelMobileNet(mobilenet,3)
    mobile = Model(inputs=mobilenet.input,outputs=fc)
    mobile.load_weights('weight.hdf5')

    #logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    print('hi')
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    num = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        #image = cv2.resize(image,(w,h))
        print('hi')
        humans = e.inference(image,resize_to_default=(w>0 and h>0 ), upsample_size=4.0)
        print('hi')
        if not args.showBG:
            image = np.zeros(image.shape)
        image,a,b = TfPoseEstimator.draw_humans(image, humans,mobile, imgcopy=False)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(f'../../data/inference/{num}.jpg',image)
        print(num)
        num+=1
        
        #cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        #if cv2.waitKey(1) == 27:
        #    break

    #cv2.destroyAllWindows()
    print('finish!!')
logger.debug('finished+')
