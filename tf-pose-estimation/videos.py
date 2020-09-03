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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='../../data/video/crawl/IMG_4641.mp4')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    li = ['../../data/video/crawl','../../data/video/jump','../../data/video/walk'] 
    #li = ['../../data/video/crawl']
    num = 0
    for path in li:
        save_path = '../../data/video/test/'
        motion = path.split('/')[-1]
        save_path +=motion+'/'
        for i in os.listdir(path):
            
            #os.makedirs(save_path+i)
            #cap = cv2.VideoCapture(path+'/'+i)
            cap = cv2.VideoCapture(args.video)
            print(args.video)
            if cap.isOpened() is False:
                print("Error opening video stream or file")
            while cap.isOpened():
                ret_val, image = cap.read()
                #image = cv2.resize(image,(w,h))
                humans = e.inference(image,resize_to_default=(w>0 and h>0 ), upsample_size=4.0)
                if humans==False:
                    print('finish')
                    break
                if not args.showBG:
                    image = np.zeros(image.shape)
                 
                image,a,normal = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                if a==0:
                    print('no human')
                    continue
                #cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(f'{save_path+i}/{num}.jpg',cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                num+=1
                print('hi')
                #for img in normal:
                #    cv2.imwrite(f'../../data/test/{num}.jpg',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                #    num+=1
                #cv2.imwrite(f'../../data/test/{num}.jpg',image)
                #num+=1
                #cv2.imshow('tf-pose-estimation result', image)
                fps_time = time.time()
            print('one video finish') 
                #if cv2.waitKey(1) == 27:
        #    break
    #cv2.destroyAllWindows()
    #print('finish!!')
logger.debug('finished+')
