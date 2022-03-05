from tqdm import tqdm
import numpy as np
import utils.datautils as datautils
import os
import matplotlib.pyplot as plt
from textwrap import wrap
import csv

def stable_branch(img, stab_limit=4, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
    stable_img = img.copy()
    # deleting unstabilized pixels: ones that don't remain at least an hour
    for t in tqdm(np.arange(img[stab_limit-1:].shape[0]), desc='filtering_px', leave=False):
        for z in np.arange(img.shape[1]):
            for y in np.arange(img.shape[2]):
                for x in np.arange(img.shape[3]): 
                    if img[t:t+stab_limit,z,y,x].sum() < stab_limit:
                        stable_img[t+stab_limit-1,z,y,x] = 0

    # make the first 4 timepoints the same (you can ignore first hour of analysis)
    stable_img[0] = stable_img[1] = stable_img[2] = stable_img[1]
    if save == True:
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            save_name = str(save_path+'stable_image.tif')
        else:
            save_name = str(save_path+'stable_'+save_file)
        if '.tif' not in save_name:
            save_name +='.tif'
        datautils.save_image(save_name, stable_img, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return stable_img

def col_occupancy(neuron, cols, nor_fact=1, start_t=36, plot=True, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
    cols_hist = {}
    ind = 0
    for col in tqdm(cols, desc='calculating Col occupancy', leave=False):
        ind += 1
        cols_hist['col_'+str(ind)] = []
        filter = np.broadcast_to(col, neuron.shape)
        col_size = filter.sum()
        nue_sub = filter * neuron # pixels occupied by neuron in the column
        for t in tqdm(nue_sub, leave=False):
            cols_hist['col_'+str(ind)].append(t.sum()/col_size)

    #normalization step
    for col, ocup in cols_hist.items():
        cols_hist[col] = ocup/nor_fact
    
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
                fill_line = [t]
                for val in cols_hist.values():
                    fill_line.append(val[i])
                writer.writerow(fill_line)
        csvfile.close()
    
    return cols_hist
    

