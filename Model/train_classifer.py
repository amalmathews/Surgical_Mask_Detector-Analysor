
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2,InceptionResNetV2
from tensorflow.keras.models import load_model
from epochCheckpoint import EpochCheckpoint
from batchGenerator import BatchGenerator
import config
from keras import backend as K
import random
# from lrfinder import LRFinder
# from lrfinder2 import lr_finder
from keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import AveragePooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str,
                default="../data/train/",
                help="path to input dataset")

ap.add_argument("-m", "--train",
                type=str,
	            help="path to *specific* model checkpoint to load")

ap.add_argument("-l", "--load",
                type=str,
	            help="path to load checkpoint")

ap.add_argument("-e", "--start-epoch",
                type=int, default=0,
	            help="epoch to restart training at")

ap.add_argument("-n", "--epochs",
                type=int,
                default=5,
	            help="no of epochs")

# ap.add_argument("-p", "--plot",
#                 type=str,
#                 default="./Plots/plot.png",
#                 help="path to output loss/accuracy plot")

args = vars(ap.parse_args())
if args['load']:
    load_path=args['load'].split('/')[-2]
# path_to_logs ='../logs/'
# checkpoint= '../checkpoints/'
if (args["train"] or load_path) =='inception':
    path_to_logs ='../logs/inception/'
elif (args["train"] or load_path) =='mobilenet':
    path_to_logs ='../logs/mobilenet/'


if (args["train"] or load_path) =='inception':
    checkpoints ='../checkpoints/inception/'
elif (args["train"] or load_path) =='mobilenet':
    checkpoints ='../checkpoints/mobilenet/'

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 20
BS = 32
dir_labels = ()
num_class = 0
device_name = tf.test.gpu_device_name()
print("[INFO] Processing with GPU ",device_name)
with tf.device('/device:GPU:0'):
	for file in os.listdir(config.train_path):
		temp_tuple = (file, 'null')
		dir_labels = dir_labels + temp_tuple
		# print(dir_labels)
		dir_labels = dir_labels[:-1]
		# print(dir_labels)
		num_class = num_class + 1
	print('\nThe total number of classes are',num_class)

	print("[INFO] loading images...")
	imagePaths = list(paths.list_images(args["dataset"]))
	random.shuffle(imagePaths)
	data = []
	labels = []

	# loop over the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]
		for i in range(num_class):
			if label == dir_labels[i]:
				label=i

		labels.append(label)

	# convert the data and labels to NumPy arrays
	# data = np.array(data, dtype="float32")
	# labels = np.array(labels)
	# perform one-hot encoding on the labels


	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(imagePaths,labels, test_size=0.3)


	trainY = to_categorical(trainY, num_classes=num_class)
	testY = to_categorical(testY, num_classes=num_class)
	# lb = LabelBinarizer()
	# trainY = lb.fit_transform(trainY)
	# trainY = to_categorical(trainY)
	#
	# testY = lb.fit_transform(testY)
	# testY = to_categorical(testY)

	training_batch_generator = BatchGenerator(trainX, trainY, config.batch_size)
	validation_batch_generator = BatchGenerator(testX,testY , config.batch_size)





	# load the MobileNetV2 network, ensuring the head FC layer sets are
	# left off
	# baseModel = MobileNetV2(weights="imagenet", include_top=False,
	# 	input_tensor=Input(shape=(224, 224, 3)))
	#
	#
	# top_model = baseModel.output
	# top_model = AveragePooling2D(pool_size=(7, 7))(top_model)
	# top_model = Flatten(name="flatten")(top_model)
	# top_model = Dense(128, activation="relu")(top_model)
	# top_model = Dropout(0.5)(top_model)
	# top_model = Dense(3, activation="softmax")(top_model)
	#
	# model = Model(inputs=baseModel.input, outputs=top_model)
	#
	# for layer in baseModel.layers:
	# 	layer.trainable = False



	if args['train'] == 'mobilenet':
		input_tensor = Input(shape=(224, 224, 3))

		base_model =MobileNetV2(weights='imagenet', include_top=False ,input_tensor=input_tensor)
		# base_model = MobileNet(include_top=False, weights=None, input_shape=input_tensor, alpha=1.0, depth_multiplier=1)
		# base_model.load_weights('./mobilenet_1_0_224_tf.h5') # give the path for downloaded weights
		top_model = base_model.output


		top_model = GlobalAveragePooling2D()(top_model)
		top_model = Flatten(name="flatten")(top_model)
		top_model = Dropout(0.5)(top_model)
		top_model = Dense(4096, activation="relu")(top_model)
		top_model = Dropout(0.5)(top_model)
		top_model = Dense(4096, activation="relu")(top_model)
		top_model = Dropout(0.5)(top_model)
		top_model = Dense(128, activation="relu")(top_model)
		top_model = Dropout(0.5)(top_model)
		top_model = Dense(128, activation="relu")(top_model)
		top_model = Dense(3, activation="softmax")(top_model)

		model = Model(inputs=base_model.input, outputs=top_model)

		for layer in base_model.layers:
					layer.trainable = True

		print("[INFO] compiling model...")
		opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / args['epochs'])
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


	elif args['train'] == 'inception':
		print('[INFO]::Trainig with inceptionNet...')
		input_tensor = Input(shape=(224, 224, 3))

		base_model =InceptionResNetV2(weights='imagenet', include_top=False ,input_tensor=input_tensor)
		top_model = base_model.output

		# top_model = Flatten(name="flatten")(top_model)
		top_model = GlobalAveragePooling2D()(top_model)
		top_model = Flatten(name="flatten")(top_model)
		top_model = Dropout(0.5)(top_model)
		top_model = Dense(256, activation="relu")(top_model)
		top_model = Dense(3, activation="softmax")(top_model)

		# top_model = Dense(num_class, activation="softmax")(top_model)

		model = Model(inputs=base_model.input, outputs=top_model)

		for layer in base_model.layers:
			layer.trainable = True

		opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / args['epochs'])
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

	else:
		print('[INFO]::Loading pretrained Model....')
		model=load_model(args["load"])
		K.get_value(model.optimizer.lr)
		K.set_value(model.optimizer.lr, 0.001)
	tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=path_to_logs,
															 histogram_freq=0,
															 write_graph=True,
															 write_images=False)
	# lr_finder = LRFinder(min_lr=0.01, max_lr=0.0001)
	# lrf = lr_finder(model,begin_lr=1e-8, end_lr=1e0, num_epochs=20)
	# lr_rate = LearningRateScheduler(lrf.lr_schedule)
	# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01,
	#                           patience=2, min_lr=0.009)

	callbacks = [EpochCheckpoint(checkpoints, every=1,startAt=0),tbCallBack]

	# train the head of the network
	print("[INFO] training head...")
	H = model.fit(training_batch_generator,
				  validation_data=validation_batch_generator,
				  steps_per_epoch=len(trainX) // config.batch_size,
				  epochs=args['epochs'],
				  callbacks=callbacks)














# # make predictions on the testing set
# print("[INFO] evaluating network...")
# predIdxs = model.predict(testX, batch_size=BS)
# predIdxs = np.argmax(predIdxs, axis=1)
# print(classification_report(testY.argmax(axis=1),
# 							predIdxs,
# 							target_names=lb.classes_))
# 
# # serialize the model to disk
# print("[INFO] saving mask detector model...")
# model.save(config.model_path, save_format="h5")

# # plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])