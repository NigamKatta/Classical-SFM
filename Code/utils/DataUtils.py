import os
import cv2
import csv
import numpy as np

class DataUtils:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        
    def load_images(self):
        fileList = os.listdir(self.dataPath)
        image_name = []
        images = {}
        for file in fileList:
            if('.png' in file):
                name = file.split(".")[0]
                image_name.append(name)
                images[name] = cv2.imread(os.path.join(self.dataPath, file))
        return images, image_name

    def load_intrinsic(self):
        K = []
        with open(self.dataPath + 'calibration.txt') as file:
            reader = csv.reader(file, delimiter = ' ')
            for row in reader:
                row_K = [float(row[i]) for i in range(3)]
                K.append(row_K)
        return np.array(K)

        