import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def DGI_3D(image, entry_point):
    """
    This function takes a 4D array of a neuron and calculates its directional growth indext (DGI) from the entry point based on the the formula:

    DGI = length(V)/length(sum(|Vi|))

    It returns a datafram containing the values of the orientation vector for each time point.
    
    Parameter:
    entry_point:    Array with 4 values that correspond to the entry point of the dendrite into the neuropil. Values correspond to the 'TZYX' coordinates of the point.
    
    neuron:         4D array containing the dendrite to analyze. It should only contain parts of the dendrite.
    """
    pixel_co = np.argwhere(image)
    # shifting all points to make entry point the center of (0,0,0)
    norm_pixel_values = pixel_co - entry_point
    norm_pixel_values[:,1] *= -1 ##### to reverse the Y axis numbering upward 
    vec_length = np.linalg.norm(norm_pixel_values, axis=1) # sum all points_vectors (maximum length)
    dbp_index = np.argwhere(vec_length == 0)
    norm_pixel_values = np.delete(norm_pixel_values, (dbp_index), axis=0)
    vec_length = np.delete(vec_length, dbp_index)
    ori_vec = norm_pixel_values.sum(axis=0) #calculate (Z,Y,X) of vector sum
    ori_vec_length = np.linalg.norm(ori_vec) #Calculate the length of vector sum
    Dgi = ori_vec_length/vec_length.sum() #calculate DGI which is maximum_length/length_vect_sum 
    av_vect = norm_pixel_values.mean(axis=0)
    av_vect_length = np.linalg.norm(av_vect)   
    return Dgi

def rotate(origin, point, angle, direction = 'counterclockwise'):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    if direction == 'counterclockwise':
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    elif direction == 'clockwise':
        qx = ox + math.cos(angle) * (px - ox) + math.sin(angle) * (py - oy)
        qy = oy - math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def angle_between(p1,p2):
    """
    angle beteen 2 points
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1-ang2 % (2 * np.pi)))

#everything as one...
def metric_dump(image,entry_point,plot=False):
    """

    """
    coords = (np.argwhere(image) - entry_point).T
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    y_v1, x_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    y_v2, x_v2 = evecs[:, sort_indices[1]]
    # # calculate the angle of rotation of the eigenvectors relative to the original coordinate space and get the rotation matrix
    theta = np.arctan((x_v1)/(y_v1))  
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    inv_rot_mat = np.linalg.inv(rotation_mat)

    # # rotate the original coordinates
    rotated_coords = rotation_mat * coords

    # # plot the transformed blob
    y_transformed, x_transformed = rotated_coords.A

    # # we want the minimum and maximum values along each axis
    x_max = np.max(x_transformed)
    x_min = np.min(x_transformed)
    y_max = np.max(y_transformed)
    y_min = np.min(y_transformed)

    # # the fraction of pixels in each direction
    frac_x_pos = len(x_transformed[x_transformed>0])/len(x_transformed)
    frac_x_neg = 1 - frac_x_pos
    frac_y_pos = len(y_transformed[y_transformed>0])/len(y_transformed)
    frac_y_neg = 1 - frac_y_pos

    # # The difference
    diff_x = abs(frac_x_pos - frac_x_neg)
    diff_y = abs(frac_y_pos - frac_y_neg)

    # # then the difference times the fraction in that direction
    x_pos = diff_x * frac_x_pos
    x_neg = diff_x * frac_x_neg
    y_pos = diff_y * frac_y_pos
    y_neg = diff_y * frac_y_neg

    # # and finally, multiply by the scalar value from the first point. 
    final_x_pos = x_max * x_pos
    final_x_neg = x_min * x_neg
    final_y_pos = y_max * y_pos
    final_y_neg = y_min * y_neg

    # ## sort out the vectors - we have the points, and the origin is [0,0], so rotate the second point by -theta degrees
    x_pos_final = rotate(origin = [0,0], point = [final_x_pos,0], angle = theta, direction = 'clockwise')
    x_neg_final = rotate(origin = [0,0], point = [final_x_neg,0], angle = theta, direction = 'clockwise')
    y_pos_final = rotate(origin = [0,0], point = [0,final_y_pos], angle = theta, direction = 'clockwise')
    y_neg_final = rotate(origin = [0,0], point = [0,final_y_neg], angle = theta, direction = 'clockwise')

    # # get angle of each vector
    x_pos_angle = angle_between([0,0],x_pos_final)
    x_neg_angle = angle_between([0,0], x_neg_final)
    y_pos_angle = angle_between([0,0], y_pos_final)
    y_neg_angle = angle_between([0,0], y_neg_final)

    df = pd.DataFrame.from_dict({'axis':['x_positive','x_negative','y_positive','y_negative'],
                                'Fraction_weight':[x_pos,x_neg,y_pos,y_neg],
                                'Pixel_scale':[x_max,x_min,y_max,y_min],
                                'Angle':[x_pos_angle,x_neg_angle,y_pos_angle,y_neg_angle],
                                'xy': [x_pos_final,x_neg_final,y_pos_final,y_neg_final]})

    asymmetry = (diff_x + diff_y)/2

    if plot:
        x, y = coords[0], coords[1]
        cent_x = sum(rotated_coords.A.T[:,0]) / len(rotated_coords.A.T)
        cent_y = (sum(rotated_coords.A.T[:,1]) / len(rotated_coords.A.T))
        scale = 20
        plt.plot(x, y, 'k.',alpha=0.2)
        plt.plot([x_v1*-scale*2, x_v1*scale*2],
                [y_v1*-scale*2, y_v1*scale*2], color='red')
        plt.plot([x_v2*-scale, x_v2*scale],
                [y_v2*-scale, y_v2*scale], color='blue')
        # plt.plot(x, y, 'k.')
        plt.axis('equal')
        plt.gca().invert_yaxis()  # Match the image system with origin at top left
        plt.show()
        # x_transformed, y_transformed = transformed_mat.A
        plt.plot(x_transformed, y_transformed, 'g.',alpha=0.1)
        plt.scatter(cent_x,cent_y,c='k')
        plt.axis('equal')
        plt.gca().invert_yaxis()  # Match the image system with origin at top left
        plt.show()

    return asymmetry, df, rotated_coords.A.T