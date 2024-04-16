import numpy as np
from IPython import embed


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def triangulation(R1,C1,R2,C2,x,K):
    
    # Homogeneous Coordinates
    x1 = np.array([p[0] for p in x])
    x2 = np.array([p[1] for p in x])

    # P = K ([I | T] * R)
    # Projection matrix of two set of features are calculated
    C1 = C1.reshape((3,1))
    T1 = -R1.dot(C1) 
    P1 = K@np.hstack((R1,T1))
    
    C2 = C2.reshape((3,1))
    T2 = -R2.dot(C2) 
    P2 = K@np.hstack((R2,T2))
    
    Xn = []

    # looping over all points 
    for u,v in zip(x1,x2):
        uh = np.append(u,1)
        vh = np.append(v,1)

        A1 = skew(uh)@P1
        A2 = skew(vh)@P2
        A = np.vstack((A1,A2))

        U,S,V = np.linalg.svd(A)
        X = V[np.argmin(S),:]
        X = X/X[3]
        Xn.append(X[0:3])
    Xn = np.array(Xn)

    return Xn


#  condition1 (R3[X - T] > 0) 
#  condition2 Z > 0 i.e; X[2] > 0
#  Then the point considered as correct point in aligned with R and T
def check_cheirality(T, R, points):
    R3 = R[:, 2]
    R3 = R3.reshape((3, 1))
    C = T.reshape((3,1))
    count_points = 0
    #print(points[0])
    for X in points:
        X = X.reshape((3,1))
        check1 = X[2]
        check2 = R3.T @ (X - C)
        if(check1 > 0 and check2 > 0):
            count_points += 1   

    return count_points