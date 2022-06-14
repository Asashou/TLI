# from importlib.abc import Traversable
from cProfile import label
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

def stable_N(neuron, stab_limit=4, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
    """
    this is a simple function that gets an array where the values are the pixels' lifetimes,
    and set anything below the stable_limit to 0 and all other values to 1
    """
    stable_neuron = neuron.copy()
    stable_neuron[stable_neuron<stab_limit] = 0
    stable_neuron[stable_neuron>0] = 1
    if save == True:
        stable_neuron = stable_neuron.astype('uint16')
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            save_name = str(save_path+'stable_image.tif')
        else:
            save_name = str(save_path+'stable_'+save_file)
        if '.tif' not in save_name:
            save_name +='.tif'
        datautils.save_image(save_name, stable_neuron, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return stable_neuron

def N_volume(neuron, stable=np.zeros((1)), normalize=False, start_t=36, plot=True, save=True, save_path='', save_file=''):
    neuron[neuron != 0] = 1
    stable[stable != 0] = 1
    
    # definning timepoints
    T_length = np.arange(len(neuron))
    timepoints = [start_t+(i*0.25) for i in T_length] 

    output_volumes = pd.DataFrame({'timepoints':timepoints,
                                    'vol_all':neuron.sum(axis=(1,2,3)), 
                                    'vol_stable':stable.sum(axis=(1,2,3))})
    
    if save_path != '' and save_path[-1] != '/':
        save_path += '/'

    if plot:
        fig_name = save_path+save_file+'_neu_vol.pdf'
        #ploting the results
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(timepoints, output_volumes.vol_all, label='all pixels')
        plt.plot(timepoints, output_volumes.vol_stable, label='all pixels')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Neuron Volume Over Time')
        plt.ylabel("\n".join(wrap('No of pixels [a.u.]',30)))
        plt.xlabel("Hours After Puparium Formation [h]")
        plt.savefig(fig_name, bbox_inches='tight')

    if save == True:
        if save_file == '':
            save_file = "transient_Pxs.csv"
        csv_file = save_path+save_file
        output_volumes.to_csv(csv_file, sep=';')
    return output_volumes

def vect_alpha(vect, ref=(0,1,0)):
    """
    This function takes a point cloud of (N,D) and calculates the angles between these vectors and the ref vector (tuple)
    all vectors are assumed to start from center
    return angles calculated as numpy (N,1)
    """
    # print(vect, ref)
    if len(vect.shape)>1:
        vect = vect.ravel()
    # vec_norm = np.linalg.norm(vect, axis=1)
    # unit_vector_1 = vect/vec_norm[:,None]
    unit_vector_1 = vect / np.linalg.norm(vect)
    unit_vector_2 = ref / np.linalg.norm(ref)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    # dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    vect_deg = np.degrees(angle)
    if vect[-1]<0:
        vect_deg = 360 - vect_deg
    # print(vect_deg)
    return vect_deg

def calculate_DGI(entry_point, neuron, subtype='a', start_t=36, save=True, save_path='', save_file=''):
    """
    This function takes a 4D array of a neuron and calculates its directional growth indext (DGI) from the entry point based on the the formula:

    DGI = length(V)/length(sum(|Vi|))

    It returns a datafram containing the values of the orientation vector for each time point.
    
    Parameter:
    entry_point:    Array with 4 values that correspond to the entry point of the dendrite into the neuropil. Values correspond to the 'TZYX' coordinates of the point.
    
    neuron:         4D array containing the dendrite to analyze. It should only contain parts of the dendrite.
    """
    subtype = subtype.upper()
    pixel_co = np.argwhere(neuron>0)
    # shifting all points to make entry point the center of (0,0,0,0)
    norm_pixel_values = np.zeros(pixel_co.shape)
    norm_pixel_values = pixel_co - entry_point
    norm_pixel_values[:,2] *= -1 ##### to reverse the Y axis numbering upward 
    output = []
    for i in tqdm(range(int(max(norm_pixel_values[:,0])+1))):
        age = i*0.25+start_t
        timepoint = norm_pixel_values[norm_pixel_values[:,0]== i]
        timepoint = np.delete(timepoint,0,1) # deleting the time compoenet/axis?
        # timepoint = np.delete(timepoint,0,1) # deleting the Z compoenet/axis, WHY?
        y_spread = timepoint[:,1].max() - timepoint[:,1].min()
        x_spread = timepoint[:,2].max() - timepoint[:,2].min()
        vec_length = np.linalg.norm(timepoint, axis=1) # sum all points_vectors (maximum length)
        dbp_index = np.argwhere(vec_length == 0)
        timepoint = np.delete(timepoint, (dbp_index), axis=0)
        vec_length = np.delete(vec_length, dbp_index)
        ori_vec = timepoint.sum(axis=0) #calculate (Z,Y,X) of vector sum
        ori_vec_length = np.linalg.norm(ori_vec) #Calculate the length of vector sum
        Dgi = ori_vec_length/vec_length.sum() #calculate DGI which is maximum_length/length_vect_sum 
        av_vect = timepoint.mean(axis=0)
        av_vect_length = np.linalg.norm(av_vect)
        ref_ax = {'A':(0,0,1), 'B':(0,0,-1), 'C':(0,-1,0), 'D':(0,1,0)}
        ori_vec_deg = vect_alpha(ori_vec[:,], ref=ref_ax)
        # ori_vec_deg = vect_alpha(ori_vec[:,None].T)[0][0]
        norm_px_deg = []
        for px in timepoint:
            norm_px_deg.append(vect_alpha(px) - ori_vec_deg)
        deg_variance = np.var(norm_px_deg)
        PC_std = timepoint.std()
        # norm_pixel_deg = vect_alpha(timepoint) - ori_vec_deg
        # deg_variance = norm_pixel_deg.var()
        output.append([age,ori_vec,vec_length.sum(), 
                        av_vect, av_vect_length, 
                        ori_vec_deg,deg_variance,
                        PC_std, Dgi,
                        y_spread, x_spread])
    DGIs_columns = ['timepoints', 'ori_vec', 'Max_Vec_length', 
                    'av_vect', 'av_vect_length', 
                    'ori_vec_deg', 'deg_variance', 'PC_std', 'DGI', 'y_spread', 'x_spread']
    DGIs = pd.DataFrame(output, columns=DGIs_columns)

    if save == True:
        if save_file == '':
            save_file = "DGIs.csv"
        csv_file = save_path+save_file
        DGIs.to_csv(csv_file, sep=';')
    
    return DGIs


def transform_point(y=200, x=80, 
                    x0=80, y0=120, #center of ellipse1 
                    a0=50, b0=100, #major and minor of ellispe1
                    x1=30, y1=45, #center of ellipse2 (ref)
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

def roi_img(img_nD, ROI_2D):
    """
    draw a 2D_mask from Fiji roi, broadcast it to a 3D_image and return the masked 3D_image
    ROI_2D: is a fiji ROI
    return the 3D_mask array and the masked nD_img
    """
    if ROI_2D['type'] == 'oval':
        x0 = ROI_2D['left']+int(ROI_2D['width']/2); a = int(ROI_2D['width']/2)  # x center, half width                                       
        y0 = ROI_2D['top']+int(ROI_2D['height']/2); b = int(ROI_2D['height']/2)  # y center, half height                                      
        X = np.linspace(0, img_nD.shape[-1],img_nD.shape[-1])  # x values of interest
        Y = np.linspace(0, img_nD.shape[-2],img_nD.shape[-2])[:,None]  # y values of interest, as a "column" array
        mask_roi = ((X-x0)/a)**2 + ((Y-y0)/b)**2 <= 1  # True for points inside the ellipse
        mask_roi = mask_roi.astype(int)
    elif ROI_2D['type'] == 'freehand':
        rois = np.array(list(zip(ROI_2D['x'],ROI_2D['y'])))
        rois = rois.astype(int)
        mask_roi = np.zeros([img_nD.shape[-2],img_nD.shape[-1]])
        mask_roi = cv2.fillPoly(mask_roi, pts =[rois], color=(255,255,255))
        mask_roi[mask_roi>0] = 1
    Roi_3D = np.broadcast_to(mask_roi, img_nD.shape)
    masked_nD = Roi_3D * img_nD
    return masked_nD, Roi_3D

def col_occupancy(neuron, cols_zip, 
                  norm_cols, normalize_cols=False, 
                  nor_fact=1, start_t=36, plot=True, 
                  save=True, save_path='', 
                  save_file=''):
    # definning timepoints
    cols_occ = {'timepoints': [start_t+(i*0.25) for i in range(0,neuron.shape[0])]}
    # if normalize_cols:
    #     norm_output = np.zeros_like(neuron[0,0])
    for key, val in tqdm(cols_zip.items()):
        nue_sub, col_3D = roi_img(neuron, val)
        col_size = col_3D.sum()
        cols_occ[key] =  nue_sub.sum(axis=(1,2,3))/col_size
################# START REPLACED
        # if val['type'] == 'oval':
        #     x0 = val['left']+int(val['width']/2); a = int(val['width']/2)  # x center, half width                                       
        #     y0 = val['top']+int(val['height']/2); b = int(val['height']/2)  # y center, half height                                      
        #     X = np.linspace(0, neuron.shape[-1],neuron.shape[-1])  # x values of interest
        #     Y = np.linspace(0, neuron.shape[-2],neuron.shape[-2])[:,None]  # y values of interest, as a "column" array
        #     column = ((X-x0)/a)**2 + ((Y-y0)/b)**2 <= 1  # True for points inside the ellipse
        #     column = column.astype(int)
        # elif val['type'] == 'freehand':
        #     rois = np.array(list(zip(val['x'],val['y'])))
        #     rois = rois.astype(int)
        #     column = np.zeros([neuron.shape[-2],neuron.shape[-1]])
        #     column = cv2.fillPoly(column, pts =[rois], color=(255,255,255))
        #     column[column>0] = 1
        # col_filter = np.broadcast_to(column, neuron.shape)
        # col_size = col_filter.sum()
        # nue_sub = col_filter * neuron # pixels occupied by neuron in the column
        # cols_occ[key] =  nue_sub.sum(axis=(1,2,3))/col_size
################# END REPLACED
        # if normalize_cols:
        #     lifetimes = cal_lifetimes(nue_sub, save=False,
        #                               xy_pixel=0.076, z_pixel=0.4)
        #     nue_sub_last = lifetimes[-1]
        #     nue_sub_last[nue_sub_last<4] = 0
        #     nue_sub_last = nue_sub_last.max(axis=0)
        #     neu_PC = np.argwhere(nue_sub_last)
        #     neu_PC = np.array([neu_PC[:,0],neu_PC[:,1],nue_sub_last[neu_PC[:,0],neu_PC[:,1]]]).T
        #     x1 = norm_cols[key]['left']+int(norm_cols[key]['width']/2); 
        #     a1 = int(norm_cols[key]['width']/2)  # x center, half width                                       
        #     y1 = norm_cols[key]['top']+int(val['height']/2)
        #     b1 = int(norm_cols[key]['height']/2)  # y center, half height                                      
        #     for point in tqdm(neu_PC, desc='normalizing columns'):
        #         y_f, x_f = transform_point(y=point[0], x=point[1], 
        #                                     x0=x0, y0=y0, #center of ellipse1 
        #                                     a0=a, b0=b, #major and minor of ellispe1
        #                                     x1=x1, y1=y1, #center of ellipse2 (ref)
        #                                     a1=a1, b1=b1)
        #         norm_output[y_f, x_f] = point[-1]
    cols_occ = pd.DataFrame(cols_occ)
    # if normalize_cols:
    #     tif.imwrite(save_path+save_file+'_normalized_cols.tif', norm_output)
    if save_path != '' and save_path[-1] != '/':
        save_path += '/'
    if plot:
        fig_name = save_path+save_file+'_col_occupancy.pdf'
        #ploting the results
        plt.figure(figsize=(8, 6), dpi=80)
        col_names = list(cols_occ.loc[:, cols_occ.columns != 'timepoints'].columns)
        # for col in col_names:
        #     print(col)
        #     plt.plot(cols_occ.timepoints, cols_occ[col], label=col)
        cols_occ.plot(x='timepoints', y=col_names, kind='line')
        # plt.plot(cols_occ.timepoints, cols_occ.loc[:, cols_occ.columns != 'timepoints'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Column Occupancy')
        plt.ylabel("\n".join(wrap('Column Occupancy [a.u.]',30)))
        plt.xlabel("Hours After Puparium Formation [h]")
        plt.savefig(fig_name, bbox_inches='tight')
    if save == True:
        if save_file == '':
            save_file = "col_occupancy.csv"
        csv_file = save_path+save_file
        if '.csv' not in csv_file:
            csv_file +='.csv'
        cols_occ.to_csv(csv_file, sep=';')
    return cols_occ