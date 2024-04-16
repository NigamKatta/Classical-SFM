import utils.DataUtils as DataUtils
import numpy as np
import random

class epipolar_geometry_matrices:

    def __init__(self, args, matches, data_utils):
        self.epsilon = 0.7
        self.num_iters = 1000
        self.num_inliners = 0
        self.iter_inliers = {}
        self.final_inliers = {}
        self.data_utils = data_utils
        random.seed(40)

        self.F = np.zeros((3, 3))
        self.E = np.zeros((3, 3))
        self.matched_feature = matches

    def estimate_fundamental(self, eight_points):
        A = []
        for point in eight_points:
            a, b = point[1]
            c, d = point[0]
            row = [a * c, c * b, c, d * a, d * b, d, a, b, 1]
            A.append(row)
        A = np.array(A)

        # We chose the last eigen value corresponding eigne vector as it will show very less variance.
        # while in PCA we chose the top as they show large variance involved in the dataset.
        U, S, V = np.linalg.svd(A, full_matrices=True)
        F = V[:][8]
        F = F.reshape((3, 3))

        # Adding rank two constraint
        U_F, S_F, V_F = np.linalg.svd(F, full_matrices=True)
        S_F[-1] = 0
        S_F = np.diag(S_F)
        F = np.dot(U_F, np.dot(S_F, V_F))
        return F

    def estimate_essential(self):
        F = self.F
        K = self.data_utils.load_intrinsic()

        # We know the relation between E and F as 
        # E = K' F K, where K is the camera intrinsic
        E_intermediate = np.matmul(np.matmul(K.T, F), K)
        
        # In order to satify the rank 2 constraint, we forcfully zero out the third eigen value in S
        # and reconstruct the E matrix.
        U, S, V = np.linalg.svd(E_intermediate, full_matrices=True)
        S = [1, 1, 0]
        self.E = np.dot(U, np.dot(np.diag(S), V))
        return self.E
    
    def perform_ransac_F(self, image_pair):
        
        self.num_inliners = 0
        for i in range(self.num_iters):
            self.iter_inliers.clear()
            count_inliers = 0
            point_pairs_8 = random.sample(self.matched_feature[image_pair], 8)
            F = self.estimate_fundamental(point_pairs_8)
            point_pairs = self.matched_feature[image_pair]

            for point_pair in point_pairs:
                point1 = point_pair[0]
                point2 = point_pair[1]
                
                # Appending 1 to the points to see them in homogeneous coordinates
                point1 = np.expand_dims(np.array([point1[0], point1[1], 1]), axis=1)
                point2 = np.expand_dims(np.array([point2[0], point2[1], 1]), axis=1)
                
                # We perform X.t F X = 0
                # Epipolar Constrain: We know the F.X will give the epipolar line in the second image.
                # As the epipolar lines will touch the X', and we know that X' vector will be perpendicular to the 
                # Epipolar line, when we do X.t F X we get product as 0 as they are perpendicular
                prod = abs(np.matmul(np.matmul(point2.T, F), point1))

                if prod < self.epsilon:
                    count_inliers += 1
                    self.update_inliers(image_pair, point_pair)

            if (self.num_inliners < count_inliers):
                self.num_inliners = count_inliers
                self.final_inliers[image_pair] = self.iter_inliers[image_pair]
                self.F = F

        return self.final_inliers
    
    def get_F(self):
        return self.F
    
    def get_E(self):
        return self.E
    
    def update_inliers(self, img_idxs, point_pair):
        image_i_idx, image_j_idx = img_idxs
        image_i_u, image_i_v = point_pair[0]
        image_j_u, image_j_v = point_pair[1]
        if self.iter_inliers.get((image_i_idx, image_j_idx)) is not None:
            self.iter_inliers[(image_i_idx, image_j_idx)].append([(image_i_u, image_i_v), (image_j_u, image_j_v)])
        else:
            self.iter_inliers[(image_i_idx, image_j_idx)] = [[(image_i_u, image_i_v), (image_j_u, image_j_v)]]

    def estimate_RTset(self, E):
        U, S, V = np.linalg.svd(E)

        # We multiple the W with 
        # here W is a skew-symetric matrix
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

        # The possible rotations would be
        R1 = np.dot(U, np.dot(W, V))
        R2 = np.dot(U, np.dot(W, V))
        R3 = np.dot(U, np.dot(W.T, V))
        R4 = np.dot(U, np.dot(W.T, V))

        # The translation matrix
        T1 = U[:, 2]
        T2 = -U[:, 2]
        T3 = U[:, 2]
        T4 = -U[:, 2]

        R = [R1, R2, R3, R4]
        T = [T1, T2, T3, T4]

        for i in range(len(R)):
            if(np.linalg.det(R[i]) < 0):
                R[i] = - R[i]
                T[i] = -T[i]
        R, T = self.chk_det(R, T)
        return R, T
    
    def chk_det(self, Rs,Cs):
        Rn,Cn = [],[]
        assert(len(Rs) == len(Cs))
        print("Number of R and T: ", len(Rs))
        for R,C in zip(Rs,Cs):
            if(np.round(np.linalg.det(R))==-1):
                Rn.append(-R)
                Cn.append(-C)
            else:
                Rn.append(R)
                Cn.append(C)
        return Rn,Cn