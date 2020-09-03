import cv2
import os
import shutil

#train_test_split ratio
ratio = 0.7

#absolute path
img_dir = '/home/jmpark/g_project/test/datasets/lee/real'

number_of_img = len(next(os.walk(img_dir))[2])

number_of_train = int( ratio*number_of_img)
number_of_valid = number_of_img-number_of_train

train_data_dir = '/home/jmpark/g_project/test/Detect_fare_invasion_in_subway/Datasets/train_data'
valid_data_dir = '/home/jmpark/g_project/test/Detect_fare_invasion_in_subway/Datasets/valid_data'

if not os.path.exists(train_data_dir):
	os.mkdir(train_data_dir)

if not os.path.exists(valid_data_dir):
	os.mkdir(valid_data_dir)

#move train set
for i in range(0,number_of_train):
	src = '/'+next(os.walk(img_dir))[2][i]
	shutil.copy(img_dir+src, train_data_dir)
#move valid set
for i in range(number_of_train, number_of_img):
	src = '/'+next(os.walk(img_dir))[2][i]
	shutil.copy(img_dir+src, valid_data_dir)

print('# of whole img data : ', number_of_img)
print('# of train img data : ', len(next(os.walk(train_data_dir))[2]))
print('# of valid img data : ', len(next(os.walk(valid_data_dir))[2]))
