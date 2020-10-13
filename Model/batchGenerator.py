import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import keras


class BatchGenerator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        data=[]
        for file_name in batch_x:
            try:
                # print(file_name)
                img = cv2.imread(file_name)
                img = cv2.resize(img, (224, 224))
                img = img_to_array(img)
                data.append(img)
            except:
                print(file_name)
                print("continuing")
                continue


        data=np.array(data)
        y=np.array(batch_y)
        return data , y


        # return np.array([
        #     resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
        #     for file_name in batch_x]) / 255.0, np.array(batch_y)
