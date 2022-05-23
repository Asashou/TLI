# from importlib.abc import Traversable
from timeit import default_timer as timer
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

def vect_alpha(vect, ref):
    """
    This function takes a point cloud of (N,D) and calculates the angles between these vectors and the ref vector (tuple)
    all vectors are assumed to start from center
    return angles calculated as numpy (N,1)
    """
    vec_norm = np.linalg.norm(vect, axis=1)
    unit_vector_1 = vect/vec_norm[:,None]
    unit_vector_2 = ref / np.linalg.norm(ref)
    dot_product = np.dot(unit_vector_1, unit_vector_2[:,None])
    angle = np.arccos(dot_product)
    vect_deg = np.degrees(angle)
    return vect_deg


def calculate_DGI(entry_point, neuron, start_t=36, save=True, save_path='', save_file=''):
    """
    This function takes a 4D array of a neuron and calculates its directional growth indext (DGI) from the entry point based on the the formula:

    DGI = length(V)/length(sum(|Vi|))

    It returns a datafram containing the values of the orientation vector for each time point.
    
    Parameter:
    entry_point:    Array with 4 values that correspond to the entry point of the dendrite into the neuropil. Values correspond to the 'TZYX' coordinates of the point.
    
    neuron:         4D array containing the dendrite to analyze. It should only contain parts of the dendrite.
    """
    pixel_co = np.argwhere(neuron>0)
    # shifting all points to make entry point the center of (0,0,0,0)
    norm_pixel_values = np.zeros(pixel_co.shape)
    norm_pixel_values = pixel_co - entry_point
    # norm_pixel_values[:,0] = pixel_co[:,0]
    # norm_pixel_values[:,1] = pixel_co[:,1] - entry_point[1]
    # norm_pixel_values[:,2] = pixel_co[:,2] - entry_point[2]
    # norm_pixel_values[:,3] = pixel_co[:,3] - entry_point[3]
    norm_pixel_values[:,2] *= -1 ##### to reverse the Y axis numbering upward 
    DGIs_columns = ['timepoints', 'ori_vec', 'Max_Vec_length', 'ori_vec_deg', 'deg_variance', 'DGI']
    DGIs = pd.DataFrame(columns=DGIs_columns)
    for i in tqdm(range(int(max(norm_pixel_values[:,0])))):
        age = i*0.25+start_t
        timepoint = norm_pixel_values[int(np.argwhere(norm_pixel_values[:,0]==i)[0]):int(np.argwhere(norm_pixel_values[:,0]==i)[-1]),:]
        timepoint = np.delete(timepoint,0,1) # deleting the time compoenet/axis?
        # timepoint = np.delete(timepoint,0,1) # deleting the Z compoenet/axis, WHY?
        vec_length = np.linalg.norm(timepoint, axis=1) # maximun length of all vectors
        dbp_index = np.argwhere(vec_length == 0)
        timepoint = np.delete(timepoint, (dbp_index), axis=0)
        vec_length = np.delete(vec_length, dbp_index)
        ori_vec = timepoint.sum(axis=0) #calculate (Z,Y,X) of vector sum
        ori_vec_length = np.linalg.norm(ori_vec) #Calculate the length of vector sum
        Dgi = np.divide(ori_vec_length, vec_length.sum()) #calculate DGI which is maximum_length/length_vect_sum 
        ref_vect = (0,1,0)
        print('before')
        start_time = timer()
        ori_vec_deg = vect_alpha(ori_vec[:,None].T, ref_vect)[0]
        norm_pixel_deg = vect_alpha(timepoint, ref_vect) - ori_vec_deg
        deg_variance = norm_pixel_deg.var()
        print('after', timer() - start_time)
        Info = np.zeros([1,6])
        Info[0,0] = age
        Info[0,1] = ori_vec
        Info[0,2] = vec_length.sum()
        Info[0,3] = ori_vec_deg
        Info[0,4] = deg_variance
        Info[0,5] = Dgi
        DGIs = DGIs.append(pd.DataFrame(Info, columns=DGIs_columns))

    if save == True:
        if save_file == '':
            save_file = "DGIs.csv"
        csv_file = save_path+save_file
        DGIs.to_csv(csv_file, sep=';')
    
    return DGIs

def col_occupancy(neuron, cols_zip, nor_fact=1, start_t=36, plot=True, save=True, save_path='', save_file=''):
    # T_length = np.arange(len(cols_hist[list(cols_hist.keys())[0]]))
    # definning timepoints
    cols_occ = {'timepoints': [start_t+(i*0.25) for i in range(0,neuron.shape[0])]}
    for key, val in tqdm(cols_zip.items()):
        if val['type'] == 'oval':
            x0 = val['left']+int(val['width']/2); a = int(val['width']/2)  # x center, half width                                       
            y0 = val['top']+int(val['height']/2); b = int(val['height']/2)  # y center, half height                                      
            x = np.linspace(0, neuron.shape[-2],neuron.shape[-1])  # x values of interest
            y = np.linspace(0, neuron.shape[-2],neuron.shape[-1])[:,None]  # y values of interest, as a "column" array
            column = ((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1  # True for points inside the ellipse
            column = column.astype(int)
        elif val['type'] == 'freehand':
            rois = np.array(list(zip(val['x'],val['y'])))
            rois = rois.astype(int)
            column = np.zeros([neuron.shape[-2],neuron.shape[-1]])
            column = cv2.fillPoly(column, pts =[rois], color=(255,255,255))
            column[column>0] = 1
        col_filter = np.broadcast_to(column, neuron.shape)
        col_size = col_filter.sum()
        nue_sub = col_filter * neuron # pixels occupied by neuron in the column
        cols_occ[key] =  nue_sub.sum(axis=(1,2,3))/col_size
    cols_occ = pd.DataFrame(cols_occ)
    if save_path != '' and save_path[-1] != '/':
        save_path += '/'
    if plot:
        fig_name = save_path+save_file+'_col_occupancy.pdf'
        #ploting the results
        plt.figure(figsize=(8, 6), dpi=80)
        col_names = list(cols_occ.loc[:, cols_occ.columns != 'timepoints'].columns)
        cols_occ.plot(x='my_timestampe', y=col_names, kind='line')
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


    # for col in tqdm(cols, desc='calculating Col occupancy', leave=False):
    #     ind += 1
    #     cols_hist['col_'+str(ind)] = []
    #     filter = np.broadcast_to(col, neuron.shape)
    #     filter = filter.copy()
    #     filter -= filter.min()
    #     filter = filter/filter.max()
    #     col_size = filter.sum()
    #     nue_sub = filter * neuron # pixels occupied by neuron in the column
    #     for t in tqdm(nue_sub, leave=False):
    #         cols_hist['col_'+str(ind)].append(t.sum()/col_size)
    # #convert the results to dataframe
    # occupancy = pd.DataFrame(cols_hist)  
    # # definning timepoints
    # T_length = np.arange(len(cols_hist[list(cols_hist.keys())[0]]))
    # occupancy['timepoints'] = [start_t+(i*0.25) for i in range(0,len(occupancy.index))] 
    # if save_path != '' and save_path[-1] != '/':
    #     save_path += '/'
    # if plot:
    #     fig_name = save_path+save_file+'_col_occupancy.pdf'
    #     #ploting the results
    #     plt.figure(figsize=(8, 6), dpi=80)
    #     cols = list(cols_hist.keys())
    #     for col in cols:
    #         plt.plot(occupancy.timepoints, occupancy[col], label=col)
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #     plt.title('Column Occupancy')
    #     plt.ylabel("\n".join(wrap('Column Occupancy [a.u.]',30)))
    #     plt.xlabel("Hours After Puparium Formation [h]")
    #     plt.savefig(fig_name, bbox_inches='tight')
    
    # if save == True:
    #     if save_file == '':
    #         save_file = "col_occupancy.csv"
    #     csv_file = save_path+save_file
    #     if '.csv' not in csv_file:
    #         csv_file +='.csv'
    #     occupancy.to_csv(csv_file, sep=';')
    # return occupancy