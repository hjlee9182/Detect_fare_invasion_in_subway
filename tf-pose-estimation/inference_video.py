import argparse
import logging
import time
import os
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense,Activation,GlobalAveragePooling2D


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def addTopModelMobileNet(bottom_model,num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)

    return top_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='../../data/video/jump/IMG_4636.mp4')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--num',type=int,default = 3)
    parser.add_argument('--weight',type=str,default= 'weight.hdf5')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    mobilenet = MobileNet(include_top=False, input_shape=(224,224,3))
    output_num = args.num
    fc = addTopModelMobileNet(mobilenet,output_num)
    mobile = Model(inputs=mobilenet.input,outputs=fc)
    mobile.load_weights('./weight/'+args.weight)
   
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video")
    fps_time = 0
    num = 0
    #video = cv2.VideoWriter('../../data/first.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,(w,h))
    while cap.isOpened():
        ret_val,image = cap.read()
        
        if image is None:
            #video.release()
            print('hi')
            break

        humans = e.inference(image,resize_to_default=(w>0 and h>0), upsample_size=4.0)

        image,a,b = TfPoseEstimator.draw_humans(image,humans,mobile,output_num,imgcopy=False)

        cv2.putText(image,"FPS: %f" % (1.0 / (time.time() - fps_time)), (10,10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        
        #video.write(image)
        cv2.imwrite(f'../../data/inference/{num}.jpg',image)
        print(num)
        num+=1

        fps_time = time.time()
   #    break
    #cv2.destroyAllWindows()
    print('finish!!')
    #video.release()
logger.debug('finished+')
