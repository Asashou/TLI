from importlib.abc import Traversable
from tqdm import tqdm
import numpy as np
import utils.datautils as datautils
import os
import matplotlib.pyplot as plt
from textwrap import wrap
import csv
import pandas as pd

# def stable_branch(img, stab_limit=4, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
#     stable_img = img.copy()
#     # deleting unstabilized pixels: ones that don't remain at least an hour
#     for t in tqdm(np.arange(img[stab_limit-1:].shape[0]), desc='filtering_px', leave=False):
#         for z in np.arange(img.shape[1]):
#             for y in np.arange(img.shape[2]):
#                 for x in np.arange(img.shape[3]): 
#                     if img[t:t+stab_limit,z,y,x].sum() < stab_limit:
#                         stable_img[t+stab_limit-1,z,y,x] = 0

#     # make the first 4 timepoints the same (you can ignore first hour of analysis)
#     stable_img[0] = stable_img[1] = stable_img[2] = stable_img[1]
#     if save == True:
#         if save_path != '' and save_path[-1] != '/':
#             save_path += '/'
#         if save_file == '':
#             save_name = str(save_path+'stable_image.tif')
#         else:
#             save_name = str(save_path+'stable_'+save_file)
#         if '.tif' not in save_name:
#             save_name +='.tif'
#         datautils.save_image(save_name, stable_img, xy_pixel=xy_pixel, z_pixel=z_pixel)
#     return stable_img

# def px_lifetimes(image_4D):
#     lifetimes = []
#     px_ind = _4D_to_PC(image_4D[0])
#     for i in px_ind:
#         z,y,x = i[0], i[1], i[2]
#         px = image_4D[:,z,y,x]
#         if px.sum() > 0:
#             iter = np.where(px == 1)[0]
#             life = 1
#             if len(iter) == 1:
#                 lifetimes.append([iter[0],z,y,x,1])
#             else:
#                 for i, t in enumerate(iter[1:]):
#                     if t - iter[i] == 1:
#                         life += 1
#                     elif t - iter[i] > 1:
#                         lifetimes.append([iter[i],z,y,x,life])
#                         life = 1
#                 # if life > 1:
#                 lifetimes.append([iter[-1],z,y,x,life])
#     lifetimes = np.array(lifetimes)
#     return lifetimes

# def stable_px(image_4D, st_limit = 4, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
#     lifetimes = px_lifetimes(image_4D)
#     st_px = lifetimes[lifetimes[:,-1]>st_limit-1]
#     stable_img = np.empty_like(image_4D)
#     for px in st_px:
#         z, y, x = px[1], px[2], px[3]
#         for i in np.arange(px[-1]-st_limit,-1,-1):
#             t = px[0] - i
#             stable_img[t,z,y,x] = 1
#     if save == True:
#         if save_path != '' and save_path[-1] != '/':
#             save_path += '/'
#         if save_file == '':
#             save_name = str(save_path+'stable_image.tif')
#         else:
#             save_name = str(save_path+'stable_'+save_file)
#         if '.tif' not in save_name:
#             save_name +='.tif'
#         datautils.save_image(save_name, stable_img, xy_pixel=xy_pixel, z_pixel=z_pixel)
#     return stable_img

# def _4D_to_PC(img):
#     limit = img.min() -1
#     px = np.argwhere(img>limit)
#     return px

# def _PC_to_4D(PC, img_shape):
#     img = np.zeros(img_shape)
#     for px in PC:
#         t,z,y,x = px[0],px[1],px[2],px[3]
#         img[t,z,y,x] = 1
#     return img

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

    all_sizes = []
    for t in tqdm(np.arange(neuron.shape[0]), desc='calculating neuron volume'):
        all_sizes.append(neuron[t].sum())

    if stable.sum() != 0:
            stable_sizes = []
            for t in tqdm(np.arange(stable.shape[0]), desc='calculating stable volume'):
                stable_sizes.append(stable[t].sum())
    else:
        stable_sizes = np.zeros(stable.shape[0])
    
    #normalization (optinal)
    if normalize:
        norm_factor = all_sizes[-1]
        all_sizes = [val/norm_factor for val in all_sizes]
        stable_sizes = [val/norm_factor for val in stable_sizes]
    
    # definning timepoints
    T_length = np.arange(len(all_sizes))
    timepoints = [start_t+(i*0.25) for i in T_length] 

    # making a dataframe of the results
    output_volumes = pd.DataFrame({'timepoints':timepoints,'vol_all':all_sizes, 'vol_stable':stable_sizes})
    
    if save_path != '' and save_path[-1] != '/':
        save_path += '/'

    if plot:
        fig_name = save_path+save_file+'_neu_vol.pdf'
        #ploting the results
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(timepoints, output_volumes.vol_all, label='all pixels')
        if sum(stable_sizes) != 0:
            plt.plot(timepoints, output_volumes.vol_stable, label='stable pixels')
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
        # if '.csv' not in csv_file:
        #     csv_file +='.csv'
        # with open(csv_file, 'w', newline='') as csvfile:
        #     fieldnames = ['timepoint', 'Total Neuron volume', 'stable Neuron volume']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     for t, val in enumerate(all_sizes):
        #         writer.writerow({'timepoint' : timepoints[t], 'Total Neuron volume' : val, 'stable Neuron volume': stable_sizes[t]})
        # csvfile.close()
    return output_volumes

def trans_px_levels(lifetimes, stab_limit=4, start_t=36, plot=True, save=True, save_path='', save_file=''):
    """"
    takes 4D_image, where the values are the lifetimes of these pixels in the 4D_image
    """
    lifetimes = lifetimes.astype('float32')

    trans_Vol = []
    trans_per = []
    # trans_Vol = [0 for i in range(stab_limit-2)]
    # trans_per = [0 for i in range(stab_limit-2)]
    for stack in tqdm(lifetimes):
        all_px = len(stack[stack>0])
        transient = len(stack[(stack < stab_limit) & (stack > 0)])
        # stack[stack>0] = 1
        trans_Vol.append((all_px - transient))
        trans_per.append(((all_px - transient)/all_px))
    
    # definning timepoints
    T_length = np.arange(len(lifetimes))
    timepoints = [start_t+(i*0.25) for i in T_length] 

    if save_path != '' and save_path[-1] != '/':
        save_path += '/'

    # convert results to dataframe
    output_transient = pd.DataFrame({'timepoints':timepoints,'transient_px_N':trans_Vol, 'transient_px_per':trans_per})

    if plot:
        fig_name = save_path+save_file+'_transient.pdf'
        #ploting the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,10))
        ax1.plot(timepoints, output_transient.transient_px_N)
        ax1.set_title('Number of Transient Pixels')
        ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Number of Transient Pixels')
        ax2.plot(timepoints, output_transient.transient_px_per)
        ax2.set_title('Percentage of Transient Pixels')
        ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Percentage of Transient Pixels')
        plt.savefig(fig_name, bbox_inches='tight')
    if save == True:
        if save_file == '':
            save_file = "transient_Pxs.csv"
        csv_file = save_path+save_file
        output_transient.to_csv(csv_file, sep=';')
    
    return output_transient

# def trans_px_levels(neuron, stable, start_t=36, plot=True, save=True, save_path='', save_file=''):
#     """
#     calculate number and percentage of transient pixels between stable neuron and all_neuron
#     """
#     neuron[neuron != 0] = 1
#     stable[stable != 0] = 1
#     neuron = neuron.astype('float32')
#     stable = stable.astype('float32')

#     transient = []
#     trans_per = []
#     for t in tqdm(np.arange(stable.shape[0]), desc='calculating tansient'):
#         transient.append((neuron[t].sum()-stable[t].sum()))
#         trans_per.append((neuron[t].sum()-stable[t].sum())/neuron[t].sum())

#     # definning timepoints
#     T_length = np.arange(len(transient))
#     timepoints = [start_t+(i*0.25) for i in T_length] 

#     if save_path != '' and save_path[-1] != '/':
#         save_path += '/'

#     output_transient = pd.DataFrame({'timepoints':timepoints,'transient_px_N':transient, 'transient_px_%':trans_per})

#     if plot:
#         fig_name = save_path+save_file+'_transient.pdf'
#         #ploting the results
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,10))
#         ax1.plot(timepoints, transient)
#         ax1.set_title('Number of Transient Pixels')
#         ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Number of Transient Pixels')
#         ax2.plot(timepoints, trans_per)
#         ax2.set_title('Percentage of Transient Pixels')
#         ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Percentage of Transient Pixels')
#         plt.savefig(fig_name, bbox_inches='tight')
    
#     if save == True:
#         if save_file == '':
#             save_file = "transient_Pxs.csv"
#         csv_file = save_path+save_file
#         output_transient.to_csv(csv_file, sep=';')
#         # if '.csv' not in csv_file:
#         #     csv_file +='.csv'
#         # with open(csv_file, 'w', newline='') as csvfile:
#         #     fieldnames = ['timepoint', 'No. of transient', 'percentage']
#         #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         #     writer.writeheader()
#         #     for t, val in enumerate(transient):
#         #         writer.writerow({'timepoint' : timepoints[t], 'No. of transient' : val, 'percentage': trans_per[t]})
#         # csvfile.close()
    
#     return output_transient


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

    DGIs = pd.DataFrame(columns=['timepoints','Ori Vec_Y','Ori Vec_X', 'Ori Vec Length', 'DGI'])
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
                                columns=['timepoints','Ori Vec_Y','Ori Vec_X','Ori Vec Length','DGI']
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

    #normalization step
    for col, ocup in cols_hist.items():
        cols_hist[col] = [val/nor_fact for val in ocup]
    
    # definning timepoints
    T_length = np.arange(len(cols_hist[list(cols_hist.keys())[0]]))
    timepoints = [start_t+(i*0.25) for i in T_length] 

    if save_path != '' and save_path[-1] != '/':
        save_path += '/'

    if plot:
        fig_name = save_path+save_file+'_col_occupancy.pdf'
        #ploting the results
        plt.figure(figsize=(8, 6), dpi=80)
        for col, val in cols_hist.items():
            plt.plot(timepoints, val, label=col)
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
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['timepoint']
            for col in cols_hist.keys():
                fieldnames.append(col)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, t in enumerate(timepoints):
                fill_line = {'timepoint':t}
                for col, val in cols_hist.items():
                    fill_line[col] = val[i]
                writer.writerow(fill_line)
        csvfile.close()
    
    return cols_hist
