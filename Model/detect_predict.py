# -*- coding: utf-8 -*-
"""

@author: Amal Mathew
Created on Thu Sept 20

"""

from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import argparse
import insightface
import cv2
import numpy as np
from imutils import paths
import time
import imutils
import random
import numpy as np
import argparse
import imutils
import config
import cv2
import os
import sys

# initialising 
dir_labels=()
dir_predict=()
num_class=0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
	help="path to input image")
args = vars(ap.parse_args())


checkpoint ='../checkpoints/inception/'
best_model= '../checkpoints/inception/best_model.h5'
path_to_logs ='../logs/inception/'

start_epoch=0




device_name = tf.test.gpu_device_name()
print("[INFO] Processing with ",device_name)
if device_name!=None and device_name.split(":")[-2] =="GPU":
	device_name='/device:GPU:0'
else:
	device_name='/device:CPU:0'
with tf.device(device_name):
	#findings the labels
	for file in os.listdir(config.train_path) :
		temp_tuple=(file,'null')
		dir_labels=dir_labels+temp_tuple
		dir_labels=dir_labels[:-1]
		num_class=num_class+1

	print('[INFO]::Loading pretrained Model....')
	model=load_model(best_model)

	if args["image"]!=None:

		image=args['image']
		name=image.split('/')[-1]
		detector = insightface.model_zoo.get_model('retinaface_r50_v1')
		detector.prepare(ctx_id = -1, nms=0.4) #ctx=1 if gpu

		print("[INFO] Loading Image...")
		try :
			image = cv2.imread(image)
			orig = image.copy()
			xx,yy=orig.shape[:2]
		except AttributeError :
			print("[INFO] Attribute error")
			sys.exit()
		faces, landmark = detector.detect(image, threshold=0.5, scale=1.0)
		for i in range(len(faces)):

			start = time.time()
			box = faces[i].astype(np.int)
			# color = (0,0,255)
			# cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
			(x, y, w, h)=(box[0], box[1],(box[2]-box[0]), (box[3]-box[1]))
		    # print(type(x))
			sub_image=image[y:y+h-3,x+3:x+w-3]
    		# cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
    		# cv2.imshow("imahw",image)
			# pre-process the image for classification
			sub_image = cv2.resize(sub_image, (224,224))
			sub_image = img_to_array(sub_image)
			sub_image = np.expand_dims(sub_image, axis=0)


			proba = model.predict(sub_image)[0]
			print("proba",proba)
			idxs = np.argsort(proba)[::-1][:1]
			print("idxs",idxs)
			label= dir_labels[idxs[0]]
			lb= label
			print("dir_la",dir_labels)
			print("label",label)
			probas= float(proba[idxs])

			labels = "{}: {:.2f}%".format(label, probas * 100)
			labels = "{}".format(label)

			for (label, p) in zip(dir_labels, proba):
				print("{}: {:.2f}%".format(label, float(p * 100)))

				# draw the label on the image
			output = imutils.resize(orig, width=400)
			if lb=="Mask":
				print("green")
				color=(0,255,0)
			if lb=="No_Mask":
				print("red")
				color=(0,0,255)
			else:
				color=(0,125,125)
			cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
			cv2.putText(image, labels, (x,y-10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)

			duration=time.time()-start
			print("Duration:",duration)
		# image=cv2.resize(image,(520,520))
		cv2.imshow("Output",image)
		cv2.waitKey(0)




