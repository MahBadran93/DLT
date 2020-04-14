import numpy as np 
import  cv2 

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
        matrix_Form_Result_list.append( [listOf3Dpoints[i,0],
                                  listOf3Dpoints[i,1],
                                  listOf3Dpoints[i,2],
                                  1, 0, 0, 0, 0, 
                                   -listOf2DPoints[i,0]*listOf3Dpoints[i,0],
                                   -listOf2DPoints[i,0]*listOf3Dpoints[i,1],
                                   -listOf2DPoints[i,0]*1,
                                   -listOf2DPoints[i,0]] )
                                   
                                                          
        matrix_Form_Result_list.append( [0,0,0,0,listOf3Dpoints[i,0],
                                  listOf3Dpoints[i,1],
                                  listOf3Dpoints[i,2],
                                    1, 
                                   -listOf2DPoints[i,1]*listOf3Dpoints[i,0],
                                   -listOf2DPoints[i,1]*listOf3Dpoints[i,1],
                                   -listOf2DPoints[i,1]*1,
                                                        -listOf2DPoints[i,1]])                                                  
                                                          
    
      # turn the list to Arrray 
        matrix_Form_Result_Array = np.asarray(matrix_Form_Result_list) # 16x12 matrix, 8 points  
                                                          
                                                         
         
    """         
     we take the svd for the result matrix and it will result U,E,V matrices and the las column
     V transposed is the answer(unknowns_Params)
   
    """
    U,E,V = np.linalg.svd(matrix_Form_Result_Array) 
    unknowns_Params = np.transpose(V)
    return unknowns_Params[V.shape[0]-1,:] / unknowns_Params[-1,-1]
    
    

def Normalize(points):
    # points : points to be normalized
     # mu : take the mean for each column (all x's and all y's and all z's)
     # sd : standard deviation
     mu, sd = np.mean(points,0), np.std(points) 
     # translate all the values around the mean, and then devide by the standard
     # deviation to change the scale 
     for i in range(points.shape[1]):
         for j in range(points.shape[0]):
             points[j,i] = (points[j,i] - mu[i])  / sd
         
     return points # return normalized points 
    
def test():
    #3D points 
    xyz = np.array([[0,0,0,1], [0,12.3,0,1], [14.5,12.3,0,1], [14.5,0,0,1], [0,0,14.5,1], [0,12.3,14.5,1], [14.5,12.3,14.5,1], [14.5,0,14.5,1]])
    #2D corresponding points 
    uv1 = np.array([[1302,1147,1],[1110,976,1],[1411,863,1],[1618,1012,1],[1324,812,1],[1127,658,1],[1433,564,1],[1645,704,1]])
   
    t = np.array([[-0.1, -0.1, 1],[-0.1, -0.033333333333333, 1],[-0.1, 0.033333333333333, 1],[-0.1, 0.1, 1],[-0.033333333333333, -0.1, 1],[-0.033333333333333, -0.033333333333333, 1],[-0.033333333333333, 0.033333333333333, 1],[-0.033333333333333, 0.1, 1],[0.033333333333333, -0.1, 1],[0.033333333333333, -0.033333333333333, 1],[0.033333333333333, 0.033333333333333, 1],[0.033333333333333, 0.1, 1],[0.1, -0.1, 1],[0.1, -0.033333333333333, 1],[0.1, 0.033333333333333, 1],[0.1, 0.1, 1]])
    s = np.array([[490,362], [490,462], [490,562], [490,662], [590,362], [590,426],[590,562],[590,662],[690,362],[690,462],[690,562],[690,662],[790,362],[790,462],[790,562],[790,662]]) 

    # Normalize the data
    normX = Normalize(t)
    normUV = Normalize(s)
    
    CalibParms = DLT(normX,normUV)
    #Calibration matrix
    H = CalibParms.reshape(3,4)

    return H
    




Calib = test()    
     
 
     
     