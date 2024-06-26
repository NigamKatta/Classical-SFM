import cv2
import numpy as np
import csv

class FeatureUtils:
    def __init__(self):
        self.n_features = []
        self.matching_files = ['matching1.txt', 'matching2.txt', 'matching3.txt', 'matching4.txt']
        self.match_file_idex = [1, 2, 3, 4]
        self.matches = {}
        self.total_matches = 0

    def read_matching_files(self, dataPath):
        for  image_i_idx, match_file in zip(self.match_file_idex, self.matching_files):
            with open(dataPath + match_file, 'r') as file:
                reader = csv.reader(file, delimiter=' ')
                for row_idx, row in enumerate(reader):
                    if(row_idx == 0):
                        self.n_features.append(int(row[1]))
                        continue
                    n_matches = int(row[0])
                    self.total_matches += n_matches - 1
                    n = 1
                    image_i_u = float(row[4])
                    image_i_v = float(row[5])
                    j_idx = 6
                    while(n < n_matches):
                        image_j_idx = int(row[j_idx])
                        image_j_u = float(row[j_idx + 1])
                        image_j_v = float(row[j_idx + 2])

                        if self.matches.get((image_i_idx, image_j_idx)) is not None:
                            self.matches[(image_i_idx, image_j_idx)].append([(image_i_u, image_i_v), (image_j_u, image_j_v)])
                        else:
                            self.matches[(image_i_idx, image_j_idx)] = [[(image_i_u, image_i_v), (image_j_u, image_j_v)]]
                        n  += 1
                        j_idx += 3
        return self.matches           