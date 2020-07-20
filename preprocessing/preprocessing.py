import cv2
import os

path = input()
folder_name = path.split('/')[-1]
print(folder_name)
change_name = 'renum_'+folder_name

li = os.listdir(path)
for index,image in enumerate(li):
        print(image)
        img = cv2.imread(path+'/'+image,cv2.IMREAD_COLOR)
        image2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(change_name+'/'+str(index)+'.jpg',image2)
