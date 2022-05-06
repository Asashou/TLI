from importlib.abc import Traversable
from tqdm import tqdm
import numpy as np
import utils.datautils as datautils
import os
import matplotlib.pyplot as plt
from textwrap import wrap
import csv
import pandas as pd

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
    norm_pixel_values = np.zeros(pixel_co.shape)
    norm_pixel_values[:,0] = pixel_co[:,0]
    norm_pixel_values[:,1] = pixel_co[:,1] - entry_point[1]
    norm_pixel_values[:,2] = pixel_co[:,2] - entry_point[2]
    norm_pixel_values[:,3] = pixel_co[:,3] - entry_point[3]
    norm_pixel_values[:,2] *= -1

    DGIs = pd.DataFrame(columns=['timepoints','Ori_Vec_Y','Ori_Vec_X', 'Ori_Vec_Length', 'DGI'])
    for i in range(int(max(norm_pixel_values[:,0]))):
        age = i*0.25+start_t
        timepoint = norm_pixel_values[int(np.argwhere(norm_pixel_values[:,0]==i)[0]):int(np.argwhere(norm_pixel_values[:,0]==i)[-1]),:]
        timepoint = np.delete(timepoint,0,1)
        timepoint = np.delete(timepoint,0,1)
        vec_length = np.linalg.norm(timepoint, axis=1)
        dbp_index = np.argwhere(vec_length == 0)
        timepoint = np.delete(timepoint, (dbp_index), axis=0)
        vec_length = np.delete(vec_length, dbp_index)
        ori_vec = timepoint.sum(axis=0)
        ori_vec_length = np.linalg.norm(ori_vec)
        DGI = np.divide(ori_vec_length, vec_length.sum())
        Info = np.zeros([1,5])
        Info[0,0] = age
        Info[0,1] = ori_vec[0]
        Info[0,2] = ori_vec[1]
        Info[0,3] = ori_vec_length
        Info[0,4] = DGI
        DGIs = DGIs.append(pd.DataFrame(Info,
                                columns=['timepoints','Ori_Vec_Y','Ori_Vec_X','Ori_Vec_Length','DGI']
                                ))
    if save == True:
        if save_file == '':
            save_file = "DGIs.csv"
        csv_file = save_path+save_file
        DGIs.to_csv(csv_file, sep=';')
    
    return DGIs

def col_occupancy(neuron, cols, nor_fact=1, start_t=36, plot=True, save=True, save_path='', save_file=''):
    cols_hist = {}
    ind = 0
    for col in tqdm(cols, desc='calculating Col occupancy', leave=False):
        ind += 1
        cols_hist['col_'+str(ind)] = []
        filter = np.broadcast_to(col, neuron.shape)
        filter = filter.copy()
        filter -= filter.min()
        filter = filter/filter.max()
        col_size = filter.sum()
        nue_sub = filter * neuron # pixels occupied by neuron in the column
        for t in tqdm(nue_sub, leave=False):
            cols_hist['col_'+str(ind)].append(t.sum()/col_size)
    #convert the results to dataframe
    occupancy = pd.DataFrame(cols_hist)  
    # definning timepoints
    T_length = np.arange(len(cols_hist[list(cols_hist.keys())[0]]))
    occupancy['timepoints'] = [start_t+(i*0.25) for i in range(0,len(occupancy.index))] 
    if save_path != '' and save_path[-1] != '/':
        save_path += '/'
    if plot:
        fig_name = save_path+save_file+'_col_occupancy.pdf'
        #ploting the results
        plt.figure(figsize=(8, 6), dpi=80)
        cols = list(cols_hist.keys())
        for col in cols:
            plt.plot(occupancy.timepoints, occupancy[col], label=col)
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
        occupancy.to_csv(csv_file, sep=';')
    return occupancy