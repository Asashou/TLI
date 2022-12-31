# from importlib.abc import Traversable
from cProfile import label
from re import L
from timeit import default_timer as timer
from turtle import shape
from tqdm import tqdm
import numpy as np
import utils.datautils as datautils
import os
import matplotlib.pyplot as plt
from textwrap import wrap
import csv
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
from read_roi import read_roi_zip as co_zip
import cv2
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math


def cal_lifetimes(neuron, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
    """
    This function takes a 4D array as an input and calculates the lifetimes of the pixels over time. 
    The array should only contain the pixels that are part of the dendrite.
    It first binarizes the image then multiplies the previous to the current volume to see if the pixel survived.
    Afterwards it adds the volume of the binary image of the current index to the last volume, thereby increasing
    the count of each pixel that is still "alive".
    
    Parameter:
    ------------------
    neuron: 4D array of dendrite
    
    Returns:
    -------------------
    
    neuron_lifetimes: 4D array of the same shape as the input array but with pixel values as their lifetimes in every stack.
    """
    
    neuron_lifetimes = np.empty(neuron.shape)
    neuron_binary = neuron.copy()
    
    neuron_binary[neuron_binary > 0] = 1
    neuron_lifetimes[0] = neuron_binary[0]
    
    for i in tqdm(range(1,neuron_binary.shape[0])):
        current_lifetimes = (neuron_binary[i]*neuron_lifetimes[i-1]) + neuron_binary[i]
        neuron_lifetimes[i] = current_lifetimes

    neuron_lifetimes = neuron_lifetimes.astype('uint8')
    
    if save == True:
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            save_name = str(save_path+'stable_image.tif')
        else:
            save_name = str(save_path+'stable_'+save_file)
        if '.tif' not in save_name:
            save_name +='.tif'
        datautils.save_image(save_name, neuron_lifetimes, xy_pixel=xy_pixel, z_pixel=z_pixel)

    return neuron_lifetimes


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
    entry_point = np.array(entry_point)
    # shifting all points to make entry point the center of (0,0,0)
    norm_pixel_values = pixel_co - entry_point
    norm_pixel_values[:,1] *= -1 ##### to reverse the Y axis numbering upward 
    vec_length = np.linalg.norm(norm_pixel_values, axis=1) # sum all points_vectors (maximum length)
    dbp_index = np.argwhere(vec_length == 0)
    norm_pixel_values = np.delete(norm_pixel_values, (dbp_index), axis=0)
    vec_length = np.delete(vec_length, dbp_index)
    ori_vec = norm_pixel_values.sum(axis=0) #calculate (Z,Y,X) of vector sum
    # p1 = np.array(ori_vec[1:])-entry_point[1:]
    ori_vec_angle = np.arctan2(*np.array(ori_vec[1:])) #calculate angle of ori_vex
    ori_vec_length = np.linalg.norm(ori_vec) #Calculate the length of vector sum
    Dgi = ori_vec_length/vec_length.sum() #calculate DGI which is maximum_length/length_vect_sum 
    av_vect = norm_pixel_values.mean(axis=0)
    av_vect_length = np.linalg.norm(av_vect)   
    return Dgi, ori_vec_angle

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
    img_PC = np.argwhere(image)
    img_PC = img_PC - entry_point
    img_PC[:,0] *= -1
    coords = img_PC.T
    coords = np.vstack((coords[1],coords[0]))

    # find the covariance matrix:
    cov_mat = np.cov(coords)

    # get eigen vectors and values
    evals, evecs = np.linalg.eig(cov_mat)

    # get the order of the eigen values and sort the eigenvectors
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    # calculate the angle of rotation of the eigenvectors relative to the original coordinate space and get the rotation matrix
    # theta = abs(np.arctan((x_v1)/(y_v1)))
    theta = np.arctan((x_v1)/(y_v1))
    # print(theta)
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    inv_rot_mat = np.linalg.inv(rotation_mat)

    # rotate the original coordinates
    rotated_coords = rotation_mat * coords

    # plot the transformed blob
    x_transformed, y_transformed = rotated_coords.A
    trans_cen = np.array((y_transformed.mean(), x_transformed.mean()))
    PC1_cen_angle = np.arctan2(*trans_cen) 

    # we want the minimum and maximum values along each axis
    x_max = np.max(x_transformed)
    x_min = np.min(x_transformed)
    y_max = np.max(y_transformed)
    y_min = np.min(y_transformed)

    # the fraction of pixels in each direction
    frac_x_pos = len(x_transformed[x_transformed>0])/len(x_transformed)
    frac_x_neg = 1 - frac_x_pos
    frac_y_pos = len(y_transformed[y_transformed>0])/len(y_transformed)
    frac_y_neg = 1 - frac_y_pos

    # The difference
    diff_x = abs(frac_x_pos - frac_x_neg)
    diff_y = abs(frac_y_pos - frac_y_neg)

    # then the difference times the fraction in that direction
    x_pos = diff_x * frac_x_pos
    x_neg = diff_x * frac_x_neg
    y_pos = diff_y * frac_y_pos
    y_neg = diff_y * frac_y_neg

    # and finally, multiply by the scalar value from the first point. 
    final_x_pos = x_max * x_pos
    final_x_neg = x_min * x_neg
    final_y_pos = y_max * y_pos
    final_y_neg = y_min * y_neg

    ## sort out the vectors - we have the points, and the origin is [0,0], so rotate the second point by -theta degrees

    x_pos_final = rotate(origin = [0,0], point = [final_x_pos,0], angle = theta, direction = 'clockwise')
    x_neg_final = rotate(origin = [0,0], point = [final_x_neg,0], angle = theta, direction = 'clockwise')
    y_pos_final = rotate(origin = [0,0], point = [0,final_y_pos], angle = theta, direction = 'clockwise')
    y_neg_final = rotate(origin = [0,0], point = [0,final_y_neg], angle = theta, direction = 'clockwise')

    # get angle of each vector
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
    x_asymmetry = min(frac_x_neg,frac_x_pos)*min(x_max,abs(x_min))/(max(frac_x_neg,frac_x_pos)*max(x_max,abs(x_min)))
    y_asymmetry = min(frac_y_neg,frac_y_pos)*min(y_max,abs(y_min))/(max(frac_y_neg,frac_y_pos)*max(y_max,abs(y_min)))
    x_scale = x_pos+abs(x_neg)
    y_scale = y_pos+abs(y_neg)
    if x_scale > y_scale:
        PC1_asymmetry = x_asymmetry
        PC2_asymmetry = y_asymmetry
    else:
        PC1_asymmetry = y_asymmetry
        PC2_asymmetry = x_asymmetry
    
    asymmetries = [asymmetry, x_asymmetry, y_asymmetry, PC1_asymmetry, PC2_asymmetry]

    if plot:
        x, y = coords[0], coords[1]
        cent_x = x_transformed.mean()
        cent_y = y_transformed.mean()
        scale = 20
        plt.scatter(x, y, marker='.',c='k',alpha=0.2)
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
        plt.text(cent_x+5,cent_y+5,'centroid')
        plt.axis('equal')
        plt.gca().invert_yaxis()  # Match the image system with origin at top left
        plt.show()

    return asymmetries, df, rotated_coords.A.T, evals



def transform_point(y=200, x=80, 
                    x0=0, y0=0, #center of ellipse1 
                    a0=50, b0=100, #major and minor of ellispe1
                    x1=0, y1=0, #center of ellipse2 (ref)
                    a1=20, b1=40, # major and minor of ellipse2
                    X_length=150, #X_dim of final array
                    Y_length=250): #Y_dim of final array
    """
    This function transform the  y,x coordinates of a point in ellipse1 
    to y_f,x_f coordinate of the corresponding point in ellipse2
    INPUT: y,x coordinates of the point in ellipse1 
           y0,x0,a0,b0 center and dimaters of ellipse1 
           y1,x1,a1,b1 center and dimaters of ellipse2
    Returns y_f, x_f coordinates of the corresponding point in ellipse2
    NOTE: the point has to be inside the ellipse1 and not on the boarder
    """

    #definning start point for calculating angle
    S_radian = np.radians(0)
    Sy0, Sx0 = int(y0+b0*np.sin(S_radian)), int(x0+a0*np.cos(S_radian))
    
    #getting angle (in radians) between start and point-of-interest
    vect, ref = (y-y0,x-x0), (Sy0-y0,Sx0-x0)
    unit_vector_1 = vect/np.linalg.norm(vect)
    unit_vector_2 = ref / np.linalg.norm(ref)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    p_radian = np.arccos(dot_product)
    p_deg = np.degrees(p_radian)
    #finding the cross point in ellipse1 based on the angle, center and diameters
    if p_deg == 90.0:
        Px0 = x0
        if y > y0:
            Py0 = y0 + b0
        else:
            Py0 = y0 - b0
    else:
        try:
            for i in [0]:
                if vect[-1]<0:
                    Px0 = int(x0-a0*b0/(np.sqrt(b0**2+a0**2*np.tan(p_radian)**2))) 
                else:
                    Px0 = int(x0+a0*b0/(np.sqrt(b0**2+a0**2*np.tan(p_radian)**2))) 
                if vect[0]<0:
                    Py0 = y0-int(np.sqrt((np.tan(p_radian)*(Px0-x0))**2))
                else:
                    Py0 = y0+int(np.sqrt((np.tan(p_radian)*(Px0-x0))**2))
        except:
            for i in [0]:
                print('failed at step Px0 because radian', p_radian, y,x)
                Px0 = 0
                Py0 = 0
    # calculating ratio of point length to cross length
    cross = (Py0-y0,Px0-x0)
    vect_length = np.linalg.norm(vect)
    cross_length = np.linalg.norm(cross)
    ratio = vect_length/cross_length 

    #finding the cross point in ellipse2 based on the angle, center and diameters
    if p_deg == 90.0:
        Px1 = x1
        if y > y1:
            Py1 = y1 + b1
        else:
            Py1 = y1 - b1
    else:
        try:
            for i in [0]:
                if vect[-1]<0:
                    Px1 = x1-int(a1*b1/(np.sqrt(b1**2+a1**2*np.tan(p_radian)**2))) 
                else:
                    Px1 = x1+int(a1*b1/(np.sqrt(b1**2+a1**2*np.tan(p_radian)**2))) 
                if vect[0]<0:
                    Py1 = y1-int(np.tan(p_radian)*(Px1-x1))
                else:
                    Py1 = y1+int(np.tan(p_radian)*(Px1-x1))
        except:
            for i in [0]:
                print("this point didn't work", p_radian, x, y)
                Px1 = 0
                Py1 = 0
    
    # calculating cross point to ellipse2 and point_length
    cross2 = (Py1-y1,Px1-x1)
    cross2_length = np.linalg.norm(cross2)
    vect2_length = ratio*cross2_length

    # x_f, x_f coordinated of the transformed point from ellipse1 to ellipse2
    try:
        for i in [0]:
            if Py1 < y1:
                y_f = y1-int(vect2_length*np.sqrt(np.sin(p_radian)**2))
            else:
                y_f = y1+int(vect2_length*np.sqrt(np.sin(p_radian)**2))
            if Px1 < x1:
                x_f = x1-int(vect2_length*np.sqrt(np.cos(p_radian)**2))
            else:
                x_f = x1+int(vect2_length*np.sqrt(np.cos(p_radian)**2))
    except:
        for i in [0]:
            print("this point wasn't transformed", x, y)
            y_f, x_f = 0, 0
    return y_f, x_f


def transform_point2(theta=0, y=200, x=80, 
                    x0=0, y0=0, #center of ellipse1 
                    a0=50, b0=100, #major and minor of ellispe1
                    x1=0, y1=0, #center of ellipse2 (ref)
                    a1=20, b1=40, # major and minor of ellipse2
                    X_length=150, #X_dim of final array
                    Y_length=250): #Y_dim of final array

                    """
                    This function transform the  y,x coordinates of a point in ellipse1 
                    to y_f,x_f coordinate of the corresponding point in ellipse2
                    INPUT: y,x coordinates of the point in ellipse1 
                        y0,x0,a0,b0 center and dimaters of ellipse1 
                        y1,x1,a1,b1 center and dimaters of ellipse2
                    Returns y_f, x_f coordinates of the corresponding point in ellipse2
                    NOTE: the point has to be inside the ellipse1 and not on the boarder
                    """
                    # find angle and distance of the point (y,x) from the center of ellipse1
                    p1 = (y-y0,x-x0)
                    p1_theta = np.arctan2(*p1)+theta
                    p1_dist = np.linalg.norm(p1)
                    
                    # find the coordinates and distance of edge point on ellipse1 that has the same angle/p1_theta
                    edge1_x = (x/abs(x)) * (a0*b0/np.sqrt((a0**2+b0**2*np.tan(p1_theta)**2))+x0)
                    edge1_y = (y/abs(y)) * (np.sqrt(1- (edge1_x/b0)**2)*a0+y0)
                    edge1_dist = np.linalg.norm((edge1_y-y0,edge1_x-x0))

                    # find the coordinates and distance of edge point on ellipse2 that has the same angle/p1_theta
                    edge2_x = (x/abs(x)) * (a1*b1/np.sqrt((a1**2+b1**2*np.tan(p1_theta)**2))+x1)
                    edge2_y = (y/abs(y)) * (np.sqrt(1- (edge2_x/b1)**2)*a1+y1)
                    # edge2_x = a1*b1/np.sqrt((a1**2+b1**2+np.tan(p1_theta)**2))+x1
                    # edge2_y = edge2_x*np.tan(p1_theta)
                    edge2_dist = np.linalg.norm((edge2_y-y1,edge2_x-x1))

                    # find ratio of p1_dist to edge1_dist
                    dist_ratio = p1_dist/edge1_dist
                    print(dist_ratio, p1_dist, edge1_dist)

                    # find corresponding distance of the point in ellipse2
                    dist = edge2_dist * dist_ratio

                    # find coordinates of the point in ellipse2
                    x_f = dist*np.cos(p1_theta)+x1
                    y_f = dist*np.sin(p1_theta)+y1

                    return y_f, x_f, dist_ratio