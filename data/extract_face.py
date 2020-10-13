
import insightface
import urllib
import urllib.request
from random import randint
import cv2
from imutils import paths
import time
import imutils
import random
import numpy as np
import argparse
import imutils
import numpy as np
from skimage import io,transform
# import skimage

model = insightface.model_zoo.get_model('retinaface_r50_v1')
model.prepare(ctx_id = -1, nms=0.4)
imagePaths = list(paths.list_images("./Partial_Mask"))
p=0
for imagePath in imagePaths:
    p+=1
    print("number ",p)
    name=imagePath.split('/')[-1]
    # if "c6_" in name:
    #     continue
    # if "c5_" in name:
    #     continue
    # else:
    print("image :  ",imagePath)
    # image = io.imread(imagePath)
    image=cv2.imread(imagePath)
    image=cv2.resize(image,(224,224))
    # w,h=image.shape[:2]
    # image = transform.resize(image, (224, 224))

        # for i in range(len(faces)):
    try:
        faces, landmark = model.detect(image, threshold=0.5, scale=1.0)
        box = faces[-1].astype(np.int)
        # color = (0,0,255)
            # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        (x, y, w, h)=(box[0], box[1],(box[2]-box[0]), (box[3]-box[1]))
                # (x, y, w, h)=bbox[0][:-1]
                # print(type(x))
        try:
            sub_img=image[y:y+h+5,x-3:x+w+3]
        except:
            print("error")
            sub_img=image[y:y+h,x:x+w]
        cv2.imwrite("./p/"+str(randint(0,10000))+name+".jpg",sub_img)
    except:
        print("error-2")
        continue










#     # cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
#     cv2.imshow("imahw",image)
# cv2.waitKey(0)
#     # print(bbox)
#     # print(x)