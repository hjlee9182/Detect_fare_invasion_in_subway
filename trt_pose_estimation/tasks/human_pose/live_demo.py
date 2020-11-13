# -*- coding: utf-8 -*-
import json
import cv2
import torch
import torch2trt
import torchvision.transforms as transforms
import PIL.Image
import time
import numpy as np
import argparse
import threading
from io import BytesIO
from alert import notifier
from os.path import join
import trt_pose.coco
import trt_pose.models
import matplotlib.pyplot as plt
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='0', help='Input Test Video')
parser.add_argument('--output_dir', type=str, default='', help='Input output directory')

args = parser.parse_args()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
WIDTH = 224
HEIGHT = 224


topology = trt_pose.coco.coco_category_to_topology(human_pose)


num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()


OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print("trt model loaded!!!")

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def get_bbox(features):
    feature_x = []
    feature_y = []
    for i in range(0,len(features),2):
        feature_x.append(features[i])
        feature_y.append(features[i+1])
    xmin = min(feature_x) 
    ymin = min(feature_y) 
    xmax = max(feature_x) 
    ymax = max(feature_y) 

    x_std = int(np.array(feature_x).std())
    y_std = int(np.array(feature_y).std() * 0.8)
    
    xmin = max(0, xmin - x_std)
    xmax = min(1080, xmax + x_std)
    ymin = max(0, ymin - y_std)
    ymax = min(1920, ymax + y_std)

    return int(xmin), int(ymin), int(xmax), int(ymax)    
    
def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def detect_alert(key_points, image):
    global prev_std
    global prev_stat
    global DETECT
    alert = False
    std = 0
    if key_points == {}:
        return prev_stat
    for key_point in key_points.values():
        x, y = key_point[0], key_point[1]
        std = np.array(key_point).var()
        
        if prev_std - std > 1000 or prev_stat:
            alert = True
            cv2.putText(image, 'Alert!!!!',
                    (max(0, x-50), max(0, y-100)),cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 3)
            xmin, ymin, xmax, ymax = get_bbox(key_point)
            crop_image = image[ymin:ymax, xmin:xmax, :]
            cv2.imwrite('output/crop.jpg', crop_image)
            DETECT = True
        else:
            prev_stat = False
    prev_std = std
    prev_stat = alert

def execute(image):
    data = preprocess(image)
    start = time.time()
    cmap, paf = model_trt(data)
    end = time.time()
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    key_points = draw_objects(image, counts, objects, peaks)

    detect_alert(key_points, image)
    image = cv2.resize(image, (480,640), interpolation=cv2.INTER_LINEAR)
    if DETECT == 3:
        t.start()
    #if abnormal:
    #    cv2.putText(image, 'Abnormal!!',
    #                (15, 50),cv2.FONT_HERSHEY_SIMPLEX,
    #                0.7, (255, 0, 0), 2)
#    return bgr8_to_jpeg(image[:, ::-1, :])
    return image

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)
WINDOW_NAME = 'Video Gesture Recognition'
cap = cv2.VideoCapture(args.video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# env variables
full_screen = False
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 480)
cv2.moveWindow(WINDOW_NAME, 250, 200)
cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)
#video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
#video_fps       = cap.get(cv2.CAP_PROP_FPS)
#out = cv2.VideoWriter(join(args.output_dir,'output_video.mp4'), video_FourCC, video_fps, (480,640))
prev_std = 0
prev_stat = False
i_frame = 0
prev = time.time()
DETECT = False
t = threading.Thread(target=notifier)
f_txt = open(args.video_path[:-4]+'.txt', 'w')
while True:
    res, img = cap.read()
    i_frame +=1 
    if not res:
      print("Video Finished!!")
      break
    print("##### Frame :",i_frame)
    if i_frame % 3 == 0:
        continue
    image = execute(img)
    now = time.time()
    cv2.putText(image, '{:.2f} Vid/s'.format(1 / (now-prev)),
                    (15, 30),cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)
    prev = now
      #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    f_txt.write(str(DETECT)+'\n')
    #cv2.imshow(WINDOW_NAME,image)
    #out.write(image)
    #cv2.imwrite('output/'+str(i_frame)+'.jpg', image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key ==27:
        break

cap.release()
#out.release()
cv2.destroyAllWindows()

    
