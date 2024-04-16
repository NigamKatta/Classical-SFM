import cv2
import argparse
import utils.DataUtils as DataUtils
import utils.FeatureUtils as FeatureUtils
from epipolarGeometryMatrices import epipolar_geometry_matrices
import numpy as np
from linearTriangulation import *
import matplotlib.pyplot as plt

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='C:\\Users\\Nigam Katta\\OneDrive - Georgia Institute of Technology\\Desktop\\Gatech\\git\\SFM\\Data\\')
    args = parser.parse_args()

    data_utils = DataUtils(args.dataPath)
    imgs, image_names = data_utils.load_images()
    
    # Reading Camera Intrinsic parameters
    K = data_utils.load_intrinsic()
    print("Intrinsic:", K)

    featureUtils = FeatureUtils()
    matched_features = featureUtils.read_matching_files(args.dataPath)
    #print(len(matched_features))

    # Fundamental matrix estimation
    epipolarMat = epipolar_geometry_matrices(args, matched_features, data_utils)

    # images pairs
    pairs = [(1, 2)]

    for image_pair in pairs:
        inliner = epipolarMat.perform_ransac_F(image_pair)
    
    # Estimating the E matrix
    E = epipolarMat.estimate_essential()
    print("E", E)

    F = epipolarMat.get_F()
    print("F", F)

    # Estimating the R and t matrix of that corresponding frames. 
    # We will have four solution from which we need to select the best suitable one by cheirality condition
    R, T = epipolarMat.estimate_RTset(E)
    print(R, T)

    # Estimating the 3 position of the feature points using Triangulation
    R0 = np.eye(3)
    T0 = np.zeros((3,1))
    fig, ax = plt.subplots()
    max_points = 0
    for Ri,Ci in zip(R,T):
        point_3D = triangulation(R0, T0, Ri, Ci, inliner[image_pair], K)
        X = point_3D.T
        print(X.shape)  
        ax.scatter(X[0], X[1])
        ax.set_xlabel("x")
        ax.set_ylabel("z")

        #Checking cheirality condition to select appropriate R and T
        valid_points = check_cheirality(Ci, Ri, point_3D)

        if(valid_points > max_points):
            max_point = valid_points
            C_final = Ci
            R_final = Ri
            X_final = X
    plt.show()



    