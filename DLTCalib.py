import numpy as np 

def DLT(listOf3Dpoints, listOf2DPoints):
    # listOf3Dpoints : list of all the 3D world points, at least 6 poinsts is needed in this list
    # listOf2DPoint : list of all the 2D image points, at least 6 poinsts is needed in this list
    
    
    # x represents the 2D image points (u,v)
    # P: the Calibration Matrix, contais the 11 parameters with the homogeneous coords.
    # X: represents the 3D world point (x,y,z)
    
    # this function returs calibration matrix.
    
    matrix_Form_Result_list = []
    unknowns_Params = []
    
    
    # matrix transformation of the mathematics equations 
    """
         - matrix_Form_Result_list : for each point we have 2 equations, so for 6 points we need 
         12 rows matrix, this matrix has all the poinst coordinates(2D & 3D)
         
         - unknowns_Params :The calibration Matrix parameters will be in one vector (11x1)
         
     
    """
    # each point will have 
    for i in range(listOf3Dpoints.shape[0]):
        matrix_Form_Result_list.append( [-listOf3Dpoints[i,0],
                                  -listOf3Dpoints[i,1],
                                  -listOf3Dpoints[i,2],
                                  -1, 0, 0, 0, 0, 
                                   listOf2DPoints[i,0]*listOf3Dpoints[i,0],
                                   listOf2DPoints[i,0]*listOf3Dpoints[i,1],
                                   listOf2DPoints[i,0]*listOf3Dpoints[i,2],
                                   listOf2DPoints[i,0]] )
                                   
                                                          
        matrix_Form_Result_list.append( [0,0,0,0,-listOf3Dpoints[i,0],
                                  -listOf3Dpoints[i,1],
                                  -listOf3Dpoints[i,2],
                                    -1, 
                                   listOf2DPoints[i,1]*listOf3Dpoints[i,0],
                                   listOf2DPoints[i,1]*listOf3Dpoints[i,1],
                                   listOf2DPoints[i,1]*listOf3Dpoints[i,2],
                                                        listOf2DPoints[i,1]])                                                  
                                                          
    
      # turn the list to Arrray 
    matrix_Form_Result_Array = np.asarray(matrix_Form_Result_list) # 32x12 matrix, 8 points
                                                         
         
    """         
     we take the svd for the result matrix and it will result U,E,V matrices and the last column
     V is the answer(unknowns_Params) of the calibration matrix
   
    """
    #M = np.dot(matrix_Form_Result_Array , matrix_Form_Result_Array.T)
    U,E,V = np.linalg.svd(matrix_Form_Result_Array) 
    #q,r= np.linalg.qr(V)
    
    
    #unknowns_Params = np.transpose(V)
    #return V
    return V[-1, :] / V[-1, -1]    
    

def Normalize(dim, points):
     """
     Parameters
     ----------
     dim : integer
        diminsion of points.
     points : 3D or 2D points 
    
     Returns
     -------
     inverMatrix : Transformation Matrix, we use in denormalizing.
     normalizedPoints : normalized points

     """
     # points : points to be normalized
     # mu : the mean for each column (all x's and all y's and all z's)
     # sd : standard deviation
     normalizedPoints = np.copy(points)
     mu, sd = np.mean(points,0), np.std(points) 
     # translate all the values around the mean, and multiply by the standard
     for i in range(normalizedPoints.shape[1]):
         for j in range(normalizedPoints.shape[0]):
             #points[j,i] = (points[j,i] - mu[i])  / sd
             normalizedPoints[j,i] = (normalizedPoints[j,i] * sd) + mu[i]
     
     # Get the Transformation Matrix to use for denoramalization later         
     if dim == 2:
        transMatrix = np.array([[sd, 0, mu[0]], [0, sd, mu[1]], [0, 0, 1]])
     else:
        transMatrix = np.array([[sd, 0, 0, mu[0]], [0, sd, 0, mu[1]], [0, 0, sd, mu[2]], [0, 0, 0, 1]]) 
   
     inverMatrix = np.linalg.inv(transMatrix)
     
     return  inverMatrix , normalizedPoints # return normalized points 
    
def test():
    #3D points 
    #xyz = np.array([[0,0,0,1,1], [0,12.3,0,1,1], [14.5,12.3,0,1], [14.5,0,0,1], [0,0,14.5,1], [0,12.3,14.5,1], [14.5,12.3,14.5,1], [14.5,0,14.5,1]])
    #2D corresponding points 
    #uv1 = np.array([[1302,1147,1],[1110,976,1],[1411,863,1],[1618,1012,1],[1324,812,1],[1127,658,1],[1433,564,1],[1645,704,1]])
   
    #t = np.array([[-0.1, -0.1, 1,1],[-0.1, -0.033333333333333, 1,1],[-0.1, 0.033333333333333, 1,1],[-0.1, 0.1, 1,1],[-0.033333333333333, -0.1, 1,1],[-0.033333333333333, -0.033333333333333, 1,1],[-0.033333333333333, 0.033333333333333, 1,1],[-0.033333333333333, 0.1, 1,1],[0.033333333333333, -0.1, 1,1],[0.033333333333333, -0.033333333333333, 1,1],[0.033333333333333, 0.033333333333333, 1,1],[0.033333333333333, 0.1, 1,1],[0.1, -0.1, 1,1],[0.1, -0.033333333333333, 1,1],[0.1, 0.033333333333333, 1,1],[0.1, 0.1, 1,1]])
    #s = np.array([[490,362,1], [490,462,1], [490,562,1], [490,662,1], [590,362,1], [590,426,1],[590,562,1],[590,662,1],[690,362,1],[690,462,1],[690,562,1],[690,662,1],[790,362,1],[790,462,1],[790,562,1],[790,662,1]]) 

    cube3Dpoint= np.array([[-0.100000000000000,-0.100000000000000,0.900000000000000,1],
                        [-0.100000000000000 ,0.100000000000000,0.900000000000000,1],
                        [0.100000000000000, 0.100000000000000 ,0.900000000000000,1],
                        [0.100000000000000 , -0.100000000000000 ,0.900000000000000,1],
                        [-0.100000000000000, -0.100000000000000, 1.10000000000000,1],
                        [-0.100000000000000, 0.100000000000000, 1.10000000000000,1],
                        [0.100000000000000, 0.100000000000000 , 1.10000000000000,1],
                        [0.100000000000000 ,-0.100000000000000, 1.10000000000000,1]])

    cube2Dpoint= np.array([[473.333333333333 ,345.333333333333,1],
    [473.333333333333 ,678.666666666667,1],
    [806.666666666667 ,678.666666666667,1],[806.666666666667, 345.333333333333,1],[503.636363636364 ,375.636363636364,1],[503.636363636364 ,648.363636363636,1],[776.363636363636 ,648.363636363636,1],[776.363636363636 ,375.636363636364,1]])
    
    # Normalize the data
    transMatrix3D, resultNormPoints3D = Normalize(3,cube3Dpoint)
    transMatrix2D, resultNormPoints2D = Normalize(2,cube2Dpoint)
    CalibParms = DLT(resultNormPoints3D,resultNormPoints2D)

    
    #Calibration matrix
    
    H = CalibParms.reshape(3,4) # return the calibration matrix 
    
    # Denormalize the points data using the transformation matrix returned from Normalize function 
    H = np.dot( np.dot( np.linalg.pinv(transMatrix2D), H ), transMatrix3D )
    H = H / H[-1, -1]
    #L = np.ndarray.flatten(H)
    
    q,r = np.linalg.qr(H) # QR Decomposition 
    #ss = np.dot(q,q.T)
  
   
    return H # Calibration Matrix 
    




Calib = test()    
print('Calibration Matrix:', Calib)
     
 
     
     