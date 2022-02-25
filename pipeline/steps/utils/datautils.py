import ants
from skimage import io, filters
from tqdm import tqdm
import numpy as np
import os
from timeit import default_timer as timer
import tifffile as tif
import cv2 as cv
from sklearn import metrics
import csv
from skimage.filters import median

def save_image(name, image, xy_pixel=0.0764616, z_pixel=0.4):
    """save provided image by name with provided xy_pixel, and z_pixel resolution as metadata"""
    if len(image.shape) == 3:
        dim = 'ZYX'
    elif len(image.shape) == 4:
        dim = 'TZYX'
    if image.dtype != 'uint16': ###this part to be omitted later
        print('image type is not uint16')
        image = image.astype('uint16')
    tif.imwrite(name, image, imagej=True, dtype=image.dtype, resolution=(1./xy_pixel, 1./xy_pixel),
                metadata={'spacing': z_pixel, 'unit': 'um', 'finterval': 1/10,'axes': dim})
    return

def get_file_names(path, group_by='', order=True, nested_files=False, criteria='tif'):
    """
    returns a list of all files' names in the given directory and its sub-folders
    the list can be filtered based on the 'group_by' str provided
    the files_list is sorted in reverse if the order is set to True. 
    The first element of the list is used later as ref
    """
    if os.path.isfile(path):
        file_list = [path]
    else:
        file_list = []
        if nested_files == False:
            file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        else:
            for path, subdirs, files in os.walk(path):
                for name in files:
                    file_list.append(os.path.join(path, name))
        file_list = [file for file in file_list if group_by in file]
        file_list = [file for file in file_list if criteria in file]
        file_list.sort(reverse=order)    
    return file_list

def split_convert(image, ch_names):
    """deinterleave the image into dictionary of two channels"""
    # for i in tqdm(range(1), desc = 'split_convert'):
    start_time = timer()
    image_ch = {}
    for ind, ch in enumerate(ch_names):
        image_ch[ch] = image[ind::len(ch_names)]
    if len(ch_names) > 1:
        image_ch[ch_names[-1]] = median(image_ch[ch_names[-1]])
    # for ch, img in image_ch.items():
    #     image_ch[ch] = img_limits(img, limit=0)
    print('split_convert runtime', timer()-start_time)
    return image_ch

def files_to_4D(files_list, ch_names=[''], 
                save=True, save_path='', save_file='', 
                xy_pixel=1, z_pixel=1, ddtype='uint16'):
    """
    read files_list, load the individual 3D_img tifffiles, 
    and convert them into a dict of 4D-arrays of the identified ch
    has the option of saving is as 8uint image
    """
    start_time = timer()
    image_4D = {ch:[] for ch in ch_names}
    files_list.sort()
    for file in tqdm(files_list, desc = 'compiling_files'):
        image = tif.imread(file)
        image = split_convert(image, ch_names=ch_names)
        for ch in ch_names:
            image_4D[ch].append(image[ch])
    z_dim = min([len(img) for img in image_4D[ch_names[0]]])
    print(image_4D.keys(), type(image_4D[ch_names[-1]]), len(image_4D[ch_names[-1]]))
    for ch in ch_names:
        print('compiling the', ch, 'channel')
        image_4D[ch] = [stack[0:z_dim] for stack in image_4D[ch]]
        image_4D[ch] = np.array(image_4D[ch])  
    if save == True:
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            name1 = os.path.basename(files_list[0])
            name2 = os.path.basename(files_list[1])
            for s in name1:
                if s in name2:
                    save_file += s
                else:
                    break
        for ch, img in image_4D.items():
            save_name = save_path+'4D_'+ch+'_'+save_file
            if os.path.splitext(save_name)[-1] not in ['.tif','.tiff']:
                save_name += '.tif'
            save_image(save_name, img, xy_pixel=xy_pixel, z_pixel=z_pixel)
    print('files_to_4D runtime', timer()-start_time)
    return image_4D

def read_files(path, group_by ,compile=True, ch_names=[''], 
                save=True, save_path='', save_file='', 
                xy_pixel=1, z_pixel=1, ddtype='uint16'):
    files_list = get_file_names(path=path, group_by=group_by, order=True, nested_files=False, criteria='tif')
    if compile:
        image_4D = files_to_4D(files_list, ch_names=ch_names, 
                            save=save, save_path=save_path, save_file=save_file, 
                            xy_pixel=xy_pixel, z_pixel=z_pixel, ddtype=ddtype)
    else:
        temp = ch_names.copy()
        temp.sort()
        image_4D = {ch:tif.imread(files_list[ind]) for ind, ch in enumerate(temp)} 
    return image_4D


def check_similarity(ref, image):
    try:
        for i in [1]:
            image = image.numpy()
            ref = ref.numpy()
    except:
        pass
    check = sum(metrics.pairwise.cosine_similarity(image.ravel().reshape(1,-1), 
                           ref.ravel().reshape(1,-1)))[0]
    # print('check_similarity of image to ref is', check)
    return check


def similarity_4D(image_4D, save=True, save_path='', save_file=''):
    similairties = [1]
    for t in tqdm(np.arange(len(image_4D[1:])), desc='cosine_sim for timepoint'):
        similairties.append(check_similarity(image_4D[t], image_4D[t+1]))
    if save == True:
        if save_file == '':
            save_file = "similarity_check.csv"
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        checks_file = save_path+save_file
        if '.csv' not in checks_file:
            checks_file +='.csv'
        with open(checks_file, 'w', newline='') as csvfile:
            print('worked')
            fieldnames = ['timepoint', 'cosine_similarity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ind, check in enumerate(similairties):
                writer.writerow({'timepoint' : ind+1, 'cosine_similarity' : check})
        csvfile.close()
    return similairties




