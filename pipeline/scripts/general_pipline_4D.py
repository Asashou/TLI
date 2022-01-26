# We import all our dependencies.
import argparse
import os
from pickle import FALSE
import cv2 as cv
import numpy as np 
import tifffile as tif
from detect_delimiter import detect
from n2v.models import N2V
import ants
from skimage import filters
from skimage.registration import phase_cross_correlation as corr
from scipy import ndimage
from scipy.ndimage import gaussian_filter as gf
import csv
from skimage import io
import skimage.transform as tr
import psutil
import gc
from PIL import Image
from scipy import ndimage, spatial, stats
import matplotlib.pyplot as plt
from sklearn import metrics
from skimage.filters import gaussian, threshold_otsu
from scipy import ndimage
from tqdm import tqdm
# import h5py

# functions
def mem_use():
    print('memory usage')
    print('cpu_percent', psutil.cpu_percent())
    print(dict(psutil.virtual_memory()._asdict()))
    print('percentage of used RAM', psutil.virtual_memory().percent)
    print('percentage of available memory', psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

def str2bool(v):
    """this function convert str to corresponding boolean value"""
    options = ("yes", "true", "t", 'y', 'no','false', 'n','f')
    if v.lower() in options:
        return str(v).lower() in ("yes", "true", "t", "1")
    else:
        return v

def txt2dict(path):
    print('getting info from', path)
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == '#':
            lines.remove(line)
    for ind, line in enumerate(lines):
        if '#' in line:
            lines[ind] = line[0:line.index('#')]
        elif '\n' in line:
            lines[ind] = line.replace('\n','')
    for line in lines:
        if line == '':
            lines.remove(line)
    delimiter = detect(lines[0])
    print(len(lines),'lines found in txt_file with', delimiter, 'as the delimiter')
    try:
        for i in [0]:
            lines = [item.strip().rsplit(delimiter, 2) for item in lines]
            input_txt = {item[0].strip(): item[1].strip() for item in lines}
    except:
        print('failed to read txt_file')
    for key, val in input_txt.items():
        if ',' in val:
            try:
                input_txt[key] = tuple(map(int, val.split(',')))
            except:
                try:
                    input_txt[key] = [item.strip() for item in val.split(',')]
                except:
                    pass
        else:
            try:
                input_txt[key] = float(val)
            except:
                input_txt[key] = str2bool(val)    
    ### adding some default parameters if missing in info_txt
    if 'sigma' not in input_txt.keys():
        input_txt['sigma'] = 0
    if 'steps' not in input_txt.keys():
        input_txt['steps'] = ['all']    
    if 'reg_subset' not in input_txt.keys():
        input_txt['reg_subset'] = [0,0]
    if 'metric' not in input_txt.keys():
        input_txt['metric'] = 'mattes'
    if 'check_ch' not in input_txt.keys():
        input_txt['check_ch'] = input_txt['ch_names'][0]
    if 'double_register' not in input_txt.keys():
        input_txt['double_register'] = False
    #### reasign un-recognized parameters
    if type(input_txt['ch_names']) != list:
        input_txt['ch_names'] = [input_txt['ch_names']]
    if type(input_txt['drift_corr']) != list:
        input_txt['drift_corr'] = [input_txt['drift_corr']]
    if type(input_txt['steps']) == str:
        input_txt['steps'] = [input_txt['steps'].lower()]
    elif type(input_txt['steps']) == tuple:
        input_txt['steps'] = [s.lower() for s in input_txt['steps']]
    if 'all' in input_txt['steps']:
        input_txt['steps'] = ['compile','preshift', 'trim','postshift', 'ants', 'n2v', 'clahe', 'mask']
    if 'check_ch' not in input_txt['ch_names']:
        print('channel defined for similarity_check not recognized, so ch_0 used')
        input_txt['check_ch'] = input_txt['ch_names'][0]
    print(input_txt)
    return input_txt

def get_file_names(path, group_by='', order=True, nested_files = False):
    """returns a list of all files' names in the given directory and its sub-folders
    the list can be filtered based on the 'group_by' str provided
    the files_list is sorted in reverse if the order is set to True. 
    The first element of the list is used later as ref"""
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
        file_list.sort(reverse=order)    
    return file_list

def img_limits(img, limit=2000, ddtype=np.uint16):
    max_limits = {np.uint8: 255, np.uint16: 65530}
    print('image old limits', img.min(), img.max())
    img = img - img.min()
    if limit == 0:
        limit = img.max()
    if limit > max_limits[ddtype]:
        limit = max_limits[ddtype]
        print('the limit provided is larger than alocated dtype. limit reassigned as appropriate', limit)
    img = img/img.max()
    img = img*limit
    img = img.astype(ddtype)
    print('image new limits and type', img.min(), img.max(), img.dtype, 'with limit', limit)
    return img

def split_convert(image, ch_names):
    """deinterleave the image into dictionary of two channels"""
    image_ch = {}
    for ind, ch in enumerate(ch_names):
        image_ch[ch] = image[ind::len(ch_names)]
    # if len(ch_names) > 1:
    #     image_ch[ch_names[-1]] = filters.median(image_ch[ch_names[-1]])
    for ch, img in image_ch.items():
        image_ch[ch] = img_limits(img, limit=0)
    return image_ch

def files_to_4D(files_list, ch_names=[''], 
                save=True, save_path='', save_file='', 
                xy_pixel=1, z_pixel=1, ddtype=np.uint8, dim='TZYX'):
    """
    read files_list, load the individual 3D_img tifffiles, 
    and convert them into a dict of 4D-arrays of the identified ch
    has the option of saving is as 8uint image
    """
    image_4D = {ch:[] for ch in ch_names}
    files_list.sort()
    for file in files_list:
        image = io.imread(file)
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
        if save_path[-1] != '/':
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
            for tim, stack in enumerate(img):
                img[tim] = img_limits(stack, limit=0, ddtype=ddtype)
            save_image(save_name, img, xy_pixel=xy_pixel, z_pixel=z_pixel, dim='TZYX')
    print(image_4D.keys(), type(image_4D[ch_names[-1]]), len(image_4D[ch_names[-1]]))
    return image_4D

def img_subset(img, subset):
    print('subsetting the image')
    try:
        subset_img = img[subset[0]:subset[1],subset[2]:subset[3],subset[4]:subset[5]]
    except:
        print('failed to subset image')
    return subset_img

def rot_flip(img, flip, angle=0):
    vert = ['vertical', 'vertically', 2, -1]
    hort = ['horizontally', 'horizontal', 1]
    if flip in vert:
        flipped = img[:,:,::-1]
        print('flipped image vertically')
    elif flip in hort:
        flipped = img[:,::-1,:]
        print('flipped image horizontally')
    else:
        flipped = img.copy()
    
    if len(flipped.shape) == 2:
        flipped = ndimage.rotate(flipped, angle, reshape=False)
        print('rotated image by', angle)
    else:
        for ind, sli in enumerate(flipped):
            flipped[ind] = ndimage.rotate(sli, angle, reshape=False)
        print('rotated image by', angle)
    return flipped

def save_image(name, image, xy_pixel=0.0764616, z_pixel=0.4, dim='TZYX'):
    """save provided image by name with provided xy_pixel, and z_pixel resolution as metadata"""
    tif.imwrite(name, image, imagej=True, resolution=(1./xy_pixel, 1./xy_pixel),
                metadata={'spacing': z_pixel, 'unit': 'um', 'finterval': 1/10,'axes': dim})

def mask_image(volume, return_mask = False ,sig = 2):
    """
    Create a binary mask from a 2 or 3-dimensional np.array.
    Method normalizes the image, converts it to greyscale, then applies gaussian bluring (kernel width set to 2 by default, can be changed with sig parameter).
    This is followed by thresholding the image using the isodata method and returning a binary mask. 
    Parameters
    ----------
    image           np.array
                    np.array of an image (2 or 3D)
    return_mask     bool
                    If False (default), the mask is subtracted from the original image. If True, a boolian array is returned, of the shape of the original image, as a mask. 
    sig             Int
                    kernel width for gaussian smoothing. set to 2 by default.
    Returns
    -------
    mask            np.array
                    Returns a binary np.array of equal shape to the original image, labeling the masked area.
    """
    image = volume.copy()
    # if input image is 2D...
    image = image.astype('float32')
    # normalize to the range 0-1
    image -= image.min()
    image /= image.max()
    # blur and grayscale before thresholding
    blur = gaussian(image, sigma=sig)
    # perform adaptive thresholding
    t = threshold_otsu(blur.ravel())
    mask = blur > t
    # convert to bool
    mask = np.array(mask, dtype=bool)
    if return_mask == False:
        image[mask==False] = 0
        return image
    else:
        return mask

def mask_subset(image, xy_pixel, z_pixel, sig=2, file='', save=True, save_path='', save_file=''):
    if file != '':
        image = Image.open(file)
        image = np.array(image)
    file_name = os.path.basename(file)
    mask = image.copy()
    for i, img in enumerate(image):
        try:
            mask[i] = mask_image(img, return_mask=False ,sig=sig)
        except:
            mask[i] = mask[i]
    img_limits(mask, limit=2000, ddtype=np.uint16)
    if save == True:
        if save_file == '':
            save_name = save_path+'mask_'+file_name
        else:
            save_name = save_path+'mask_'+save_file
        if '.tif' not in save_name:
            save_name += '.tif'
        img_save = img_limits(mask, limit=2000)
        save_image(save_name, img_save, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return mask

def antspy_regi(fixed, moving, drift_corr, metric='mattes',
                reg_iterations=(40,20,0), 
                aff_iterations=(2100,1200,1200,10), 
                aff_shrink_factors=(6,4,2,1), 
                aff_smoothing_sigmas=(3,2,1,0),
                grad_step=0.2, flow_sigma=3, total_sigma=0,
                aff_sampling=32, syn_sampling=32):

    """claculate drift of image from ref using Antspy with provided drift_corr"""
    try:
        fixed= ants.from_numpy(np.float32(fixed))
    except:
        pass
    try:
        moving= ants.from_numpy(np.float32(moving))
    except:
        pass
    
    shift = ants.registration(fixed, moving, type_of_transform=drift_corr, 
                              aff_metric=metric, syn_metric=metric,
                              reg_iterations=(reg_iterations[0],reg_iterations[1],reg_iterations[2]), 
                              aff_iterations=(aff_iterations[0],aff_iterations[1],aff_iterations[2],aff_iterations[3]), 
                              aff_shrink_factors=(aff_shrink_factors[0],aff_shrink_factors[1],aff_shrink_factors[2],aff_shrink_factors[3]), 
                              aff_smoothing_sigmas=(aff_smoothing_sigmas[0],aff_smoothing_sigmas[1],aff_smoothing_sigmas[2],aff_smoothing_sigmas[3]),
                              grad_step=grad_step, flow_sigma=flow_sigma, total_sigma=total_sigma,
                              aff_sampling=aff_sampling, syn_sampling=syn_sampling)
    print(shift['fwdtransforms'])
    return shift

def check_similarity(ref, image):
    check = sum(metrics.pairwise.cosine_similarity(image.ravel().reshape(1,-1), 
                           ref.ravel().reshape(1,-1)))[0]
    # print('check_similarity of image to ref is', check)
    return check

def antspy_drift(fixed, moving, shift, check=True):
    if check == True:
        try:
            fixed= fixed.numpy()
        except:
            pass
        try:
            moving= moving.numpy()
        except:
            pass    
        check_ref = fixed.copy()
        pre_check = check_similarity(check_ref, moving)
    try:
        fixed= ants.from_numpy(np.float32(fixed))
    except:
        pass
    try:
        moving= ants.from_numpy(np.float32(moving))
    except:
        pass
    """shifts image based on ref and provided shift"""
    vol_shifted = ants.apply_transforms(fixed, moving, transformlist=shift).numpy()
    if check == True:
        post_check = check_similarity(check_ref, vol_shifted)
        print('similarity_check', pre_check, 'improved to', post_check)
        if (pre_check - post_check) > 0.1:
            vol_shifted = moving.numpy()
            print('similarity_check was smaller after shift, so shift was ignored:', pre_check, '>>', post_check)
    return vol_shifted

def apply_ants_channels(ref, image, drift_corr,  xy_pixel, 
                        z_pixel, ch_names, ref_ch=-1,
                        metric='mattes',
                        reg_iterations=(40,20,0), 
                        aff_iterations=(2100,1200,1200,10), 
                        aff_shrink_factors=(6,4,2,1), 
                        aff_smoothing_sigmas=(3,2,1,0),
                        grad_step=0.2, flow_sigma=3, total_sigma=0,
                        aff_sampling=32, syn_sampling=3,  
                        check_ch='',                       
                        save=True, save_path='',save_file=''):
    """calculate and apply shift on both channels of image based on ref, which is dictionary of two channels.
    if save is True, save shifted channels individually with provided info"""
    for ch, value in ref.items():
        try:
            ref[ch]= ants.from_numpy(np.float32(value))
        except:
            pass
    for ch, value in image.items():
        try:
            image[ch]= ants.from_numpy(np.float32(value))
        except:
            pass
    shift = antspy_regi(ref[ch_names[ref_ch]], image[ch_names[ref_ch]], drift_corr, metric,
                        reg_iterations=reg_iterations, 
                        aff_iterations=aff_iterations, 
                        aff_shrink_factors=aff_shrink_factors, 
                        aff_smoothing_sigmas=aff_smoothing_sigmas,
                        grad_step=grad_step, flow_sigma=flow_sigma, 
                        total_sigma=total_sigma,
                        aff_sampling=aff_sampling, 
                        syn_sampling=syn_sampling)
    shifted = image.copy()
    for ch, img in shifted.items():
        shifted[ch]= antspy_drift(ref[ch],img,shift=shift['fwdtransforms'],check=False)
    if check_ch in image.keys():
        check_ref = ref[check_ch].numpy()
        pre_check = check_similarity(check_ref, image[check_ch].numpy())
        post_check = check_similarity(check_ref, shifted[check_ch])
        print('similarity_check', pre_check, 'improved to', post_check)
        image = shifted.copy()
        if (pre_check - post_check) > 0.1:
            for ch, img in shifted.items():
                image[ch] = img.numpy() 
            print('similarity_check was smaller after shift, so shift was ignored:', pre_check, '>>', post_check)
    else:
        print(check_ch, 'not a recognized ch in image')
        image = shifted.copy()
    if save == True:
        for ch, img in image.items():
            img_save = img_limits(image[ch])
            save_name = str(save_path+drift_corr+'_'+ch+'_'+save_file)
            if '.tif' not in save_name:
                save_name += '.tif'
            save_image(save_name, img_save, xy_pixel=xy_pixel, z_pixel=z_pixel)       
    return image, shift

def apply_ants_4D(image, drift_corr,  xy_pixel, 
                  z_pixel, ch_names=[1], ref_t=0,
                  ref_ch=-1, metric='mattes',
                  reg_iterations=(40,20,0), 
                  aff_iterations=(2100,1200,1200,10), 
                  aff_shrink_factors=(6,4,2,1), 
                  aff_smoothing_sigmas=(3,2,1,0),
                  grad_step=0.2, flow_sigma=3, total_sigma=0,
                  aff_sampling=32, syn_sampling=3,  
                  check_ch='',                       
                  save=True, save_path='',save_file=''):
    """"""
    if isinstance(image, dict):
        image = {ch_names:image}
    for ch in ch_names:
            image[ch] = ants.from_numpy(np.float32(image[ch]))
    if ref_t== -1:
        ref_t= len(image[ch_names[-1]])-1
    fixed = {image[ch][ref_t].copy() for ch in ch_names}
    scope = np.arange(0,ref_t)
    scope = np.concatenate((scope, np.arange(ref_t,len(image[ch_names[-1]]))))
    print('ants seq for 4D regi',scope)
    shifts = {}
    for i in scope:
        moving = {image[ch][i].copy() for ch in ch_names}
        shifted, shifts[i] = apply_ants_channels(fixed, moving, drift_corr=drift_corr,  xy_pixel=xy_pixel, 
                        z_pixel=z_pixel, ch_names=ch_names, ref_ch=ref_ch,
                        metric=metric,
                        reg_iterations=reg_iterations, 
                        aff_iterations=aff_iterations, 
                        aff_shrink_factors=aff_shrink_factors, 
                        aff_smoothing_sigmas=aff_smoothing_sigmas,
                        grad_step=grad_step, flow_sigma=flow_sigma, total_sigma=total_sigma,
                        aff_sampling=aff_sampling, syn_sampling=syn_sampling,  
                        check_ch=check_ch,                       
                        save=save, save_path=save_path,save_file=save_file)
        for ch in ch_names:
            image[ch][i] = shifted[ch] 
    if save == True:
        for ch, img in image.items():
            img_save = img_limits(img)
            save_name = str(save_path+drift_corr+'_'+ch+'_'+save_file)
            if '.tif' not in save_name:
                save_name += '.tif'
            save_image(save_name, img_save, xy_pixel=xy_pixel, z_pixel=z_pixel)       
    return image, shifts
def phase_corr(fixed, moving, sigma):
    if fixed.shape > moving.shape:
        print('fixed image is larger than moving', fixed.shape, moving.shape)
        fixed = fixed[tuple(map(slice, moving.shape))]
        print('fixed image resized to', fixed.shape)
    elif fixed.shape < moving.shape:
        print('fixed image is smaller than moving', fixed.shape, moving.shape)
        moving = moving[tuple(map(slice, fixed.shape))]
        print('moving image resized to', moving.shape)
    fixed = gf(fixed, sigma=sigma)
    moving = gf(moving, sigma=sigma)
    print('applying phase correlation')
    try:
        for i in [0]:
            shift, error, diffphase = corr(fixed, moving)
    except:
        for i in [0]:
            shift, error, diffphase = np.zeros(len(moving)), 0, 0
            print("couldn't perform PhaseCorr, so shift was casted as zeros")
    return shift

def N2V_predict(model_name, model_path, xy_pixel=1, z_pixel=1, image=0, file='', save=True, save_path='', save_file=''):
    """apply N2V prediction on image based on provided model
    if save is True, save predicted image with provided info"""
    if file != '':
        image = Image.open(file)
        image = np.array(image)
    file_name = os.path.basename(file)
    model = N2V(config=None, name=model_name, basedir=model_path)
    predict = model.predict(image, axes='ZYX', n_tiles=None)
    if save == True:
        if save_file == '':
            save_name = str(save_path+'N2V_'+file_name)
        else:
            save_name = str(save_path+'N2V_'+save_file)
        if '.tif' not in save_name:
            save_name +='.tif'
        img_save = img_limits(predict)
        save_image(save_name, img_save, xy_pixel=xy_pixel, z_pixel=z_pixel, dim='TZYX')
    return predict

def apply_clahe(kernel_size, xy_pixel=1, z_pixel=1, image=0, file='', clipLimit=1, save=True, save_path='', save_file=''):
    """apply Clahe on image based on provided kernel_size and clipLimit
    if save is True, save predicted image with provided info"""
    if file != '':
        image = imread(file)
    if image.min()<0:
        image = (image - image.min())
    image = image.astype(np.uint16)
    print(image.dtype)
    file_name = os.path.basename(file)
    image_clahe= np.empty(image.shape)
    clahe_mask = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=kernel_size)
    for ind, slice in enumerate(image):
        image_clahe[ind] = clahe_mask.apply(slice)
        image_clahe[ind] = cv.threshold(image_clahe[ind], 
                            thresh=np.percentile(image_clahe[ind], 95), 
                            maxval=image_clahe[ind].max(), 
                            type= cv.THRESH_TOZERO)[1]
    if save == True:
        if save_file == '':
            save_name = save_path+'clahe_'+file_name
        else:
            save_name = save_path+'clahe_'+save_file
        if '.tif' not in save_name:
            save_name += '.tif'
        img_save = img_limits(image_clahe, limit=0)
        save_image(save_name, img_save, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return image_clahe

######################

def main():
    parser = argparse.ArgumentParser(description='read info.txt file and perform preprocessing pipline on prvided path',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('txt_path', help='provide path to info.txt file')
    args = parser.parse_args()
    
    ##### this part is for reading variables' values and info.txt
    input_txt = txt2dict(args.txt_path)
    
    ####### compiling the single 3D tif files to 4D_image, or reading 4D_image(s)
    file_4D = input_txt['group'].split('_')[0]+'.tif'
    if os.path.isdir(input_txt['path_to_data']):
        files_list = get_file_names(input_txt['path_to_data'], 
                                    group_by=input_txt['group'], 
                                    order=input_txt['reference_last'])
        print('the first 5 files (including ref) are', files_list[0:5])
        if 'compile' in input_txt['steps']:
            # loading raw skimage files into 4D array and saving raw 4D
            print('compiling 3D image_files into dict of 4D_images of specified channels')
            image_4D = files_to_4D(files_list, ch_names=input_txt['ch_names'], save=True, 
                                save_path=input_txt['save_path'], 
                                save_file='raw_'+file_4D, 
                                xy_pixel=input_txt['xy_pixel'], 
                                z_pixel=input_txt['z_pixel'], 
                                ddtype=np.uint8)
        else:
           temp = input_txt['ch_names'].copy()
           temp.sort()
           image_4D = {ch:io.imread(files_list[ind]) for ind, ch in enumerate(temp)} 
    elif os.path.isfile(input_txt['path_to_data']):
        image_4D = {input_txt['ch_names'][0]:io.imread(input_txt['path_to_data'])}
        files_list = [i for i in np.arange(len(image_4D))]
        file_4D = os.path.basename(input_txt['path_to_data'])
        file_4D = file_4D.split('_')[0]+'.tif'    
    print(type(image_4D), image_4D.keys(), type(image_4D[input_txt['ch_names'][-1]]), len(image_4D[input_txt['ch_names'][-1]]))
    # print(image_4D.items(), image_4D[input_txt['ch_names'][-1]].shape)

    ####### initial registration of images using phase_correlation on red channel, last channel
    if 'preshift' in input_txt['steps']:
        print('applying preshift')
        pre_shifts = {}
        ref_im = image_4D[input_txt['ch_names'][-1]]
        current_shift = [0 for i in ref_im[0].shape]
        for ind, stack in enumerate(ref_im):
            pre_shifts[files_list[ind]] = phase_corr(ref_im[0], stack, input_txt['sigma']) 
            current_shift = [sum(x) for x in zip(current_shift, pre_shifts[files_list[ind]])] 
            print('current pre_shift', current_shift)
            for ch, img in image_4D.items(): 
                image_4D[ch][ind] = ndimage.shift(img[ind], current_shift) 
        if input_txt['save_pre_shift'] == True:
            for ch, img in image_4D.items():
                name = input_txt['save_path']+'PhaseCorr_'+ch+'_'+file_4D
                save_image(name, img, xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],dim='TZYX')

        shift_file = input_txt['save_path']+"PhaseCorr_shifts.csv"
        with open(shift_file, 'w', newline='') as csvfile:
            fieldnames = ['timepoint', 'phase_shift']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for timepoint, shift in pre_shifts.items():
                writer.writerow({'timepoint' : timepoint, 'phase_shift' : shift})
        csvfile.close()

    ###### optional deletion of last quater of slices in Z_dim of each 3D image
    #### this is to reduce the run time for Ants a little bit
    if 'trim' in input_txt['steps']:
        trim = int((3*image_4D[input_txt['ch_names'][-1]].shape[1])/4)
        print('trimming all images in Z dim to 0:', trim)
        for ch, img in image_4D.items():
            image_4D[ch] = img[:,0:trim]

    ###### applying Ants registration based on the last (red) channel
    if 'ants' in input_txt['steps']:
        parameters = {'grad_step':0.2, 'flow_sigma':3, 'total_sigma':0,
                      'aff_sampling':32, 'aff_random_sampling_rate':0.2, 
                      'syn_sampling':32, 'reg_iterations':(40,20,0), 
                      'aff_iterations':(2100,1200,1200,10), 
                      'aff_shrink_factors':(6,4,2,1), 
                      'aff_smoothing_sigmas':(3,2,1,0)}
        for para in parameters.keys():
            try:
                parameters[para] = input_txt[para]
            except:
                pass
        if 'ants_ref_st' not in input_txt.keys():
            input_txt['ants_ref_st'] = 0
        ref_t = input_txt['ants_ref_st']
        if isinstance(ref_t, int) == False or ref_t < 0 or ref_t > len(image_4D[input_txt['ch_names'][-1]]):
            ref_t = 0
        ants_shifts = {}
        for i, drift_t in enumerate(input_txt['drift_corr']):
            ants_step = str(i+1)+'_'+drift_t
            try:
                metric_t = input_txt['metric'][i]
            except:
                for i in [0]:
                    print('optimization metric not recognized. mattes used instead')
                    metric_t = 'mattes'            
            image_4D, ants_shifts[ants_step] = apply_ants_4D(image_4D, 
                                                            drift_corr=drift_t,  
                                                            xy_pixel=input_txt['xy_pixel'], 
                                                            z_pixel=input_txt['z_pixel'], 
                                                            ch_names=input_txt['ch_names'], 
                                                            ref_t=ref_t,
                                                            ref_ch=-1, 
                                                            metric=metric_t,
                                                            reg_iterations=parameters['reg_iterations'], 
                                                            aff_iterations=parameters['aff_iterations'], 
                                                            aff_shrink_factors=parameters['aff_shrink_factors'], 
                                                            aff_smoothing_sigmas=parameters['aff_smoothing_sigmas'],
                                                            grad_step=parameters['grad_step'], 
                                                            flow_sigma=parameters['flow_sigma'], 
                                                            total_sigma=parameters['total_sigma'],
                                                            aff_sampling=parameters['aff_sampling'], 
                                                            syn_sampling=parameters['syn_sampling'], 
                                                            check_ch=input_txt['ch_names'][0],                       
                                                            save=True, 
                                                            save_path=input_txt['save_path'],
                                                            save_file=str(i+1)+'_'+file_4D)
            print('finished ants run with', drift_t)
        ###### saving shifts mats as csv
        shift_file = input_txt['save_path']+"ants_shifts.csv"
        with open(shift_file, 'w', newline='') as csvfile:
            fieldnames = ['reg_step', 'timepoint', 'ants_shift']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for reg_step, shifts in ants_shifts.items():
                for timepoint, ind_shift in shifts.items():
                    writer.writerow({'reg_step': reg_step, 'timepoint' : timepoint+1, 'ants_shift' : ind_shift})
        csvfile.close()  
        ###### doing final similarity check after antspy, and saving values
        similairties = {}
        for t, img in enumerate(image_4D[input_txt['ch_names'][0]][1:]):
            img_t = image_4D[input_txt['ch_names'][0]][t]
            similairties[t+1] = check_similarity(img_t, img)
        checks_file = input_txt['save_path']+"similarity_check.csv"
        with open(checks_file, 'w', newline='') as csvfile:
            fieldnames = ['reg_step', 'file', 'similarity_check']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for reg_step, checks in similarity_check.items():
                for file, similarity_check in checks.items():
                    writer.writerow({'reg_step': reg_step, 'file' : file, 'similarity_check' : similarity_check})
        csvfile.close()

    if 'n2v' in input_txt['steps']:
        for ind, stack in enumerate(image_4D[input_txt['ch_names'][-1]]):
            image_4D[input_txt['ch_names'][0]][ind] = N2V_predict(image=stack,
                                                                    model_name=input_txt['model_name'], 
                                                                    model_path=input_txt['model_path'], 
                                                                    save=False, save_path=input_txt['save_path'],
                                                                    xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
                                                                    save_file=input_txt['ch_names'][0]+'_'+file_4D)
        name = input_txt['save_path']+'N2V_GFP_'+file_4D
        save_image(name, image_4D[input_txt['ch_names'][0]][ind], 
                    xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'], dim='TZYX')
    if 'clahe' in input_txt['steps']:
        for ind, stack in enumerate(image_4D[input_txt['ch_names'][-1]]):
            image_4D[input_txt['ch_names'][0]][ind] = apply_clahe(kernel_size=input_txt['kernel_size'], 
                                                                    image=img, clipLimit=input_txt['clipLimit'], 
                                                                    xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
                                                                    save=False, save_path=input_txt['save_path'], 
                                                                    save_file=input_txt['ch_names'][0]+'_'+file_4D)
        name = input_txt['save_path']+'Clahe_GFP_'+file_4D
        save_image(name, image_4D[input_txt['ch_names'][0]][ind], 
                    xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'], dim='TZYX')
    
    ###### masking 
    # if 'mask' in input_txt['steps']:


    # if 'ants' in input_txt['steps']:
    #     parameters = {'grad_step':0.2, 'flow_sigma':3, 'total_sigma':0,
    #                   'aff_sampling':32, 'aff_random_sampling_rate':0.2, 
    #                   'syn_sampling':32, 'reg_iterations':(40,20,0), 
    #                   'aff_iterations':(2100,1200,1200,10), 
    #                   'aff_shrink_factors':(6,4,2,1), 
    #                   'aff_smoothing_sigmas':(3,2,1,0)}
    #     for para in parameters.keys():
    #         try:
    #             parameters[para] = input_txt[para]
    #         except:
    #             pass
    #     ants_ref_no = str(input_txt['ants_ref_no'])
    #     try:
    #         for i in [0]:
    #             file = [file for file in files_list if ants_ref_no in file][0]
    #             print(file)
    #             ref = io.imread(file)
    #             print(os.path.basename(file), 'is used as ref for Antspy')
    #             start = files_list.index(file)
    #     except:
    #         for i in [0]:
    #             print("couldn't find the ref file specified in info.txt")
    #             ref = io.imread(files_list[0])
    #             print(os.path.basename(files_list[0]), 'is used instead as ref for Antspy')
    #             start = 0
    #     if start > 0:
    #         scope1 = np.arange(start, -1, -1)
    #         scope2 = np.arange(start, len(files_list), 1)
    #         scope = np.concatenate((scope1, scope2))
    #     else:
    #         scope = np.arange(0,len(files_list),1)
    #     print('registration sequence:', scope) 
    #     ref = split_convert(ref, input_txt['ch_names'])
    #     if 'preshift' in input_txt['steps']:
    #         start_ref = ref.copy()
    #         pre_shifts = {}
    #     if 'postshift' in input_txt['steps']:
    #         post_shifts = {}
    #     ants_shift = {str(i+1)+'_'+drift_t:{} for i, drift_t in enumerate(input_txt['drift_corr'])}
    #     similarity_check = {str(i+1)+'_'+drift_t:{} for i, drift_t in enumerate(input_txt['drift_corr'])}
    #     similarity_check['0_unregi'] = {}
    #     if sum(input_txt['reg_subset']) != 0:
    #         print('image subset indecies', input_txt['reg_subset'])
    #         for ch, img in ref.items():
    #             ref[ch] = img_subset(img, input_txt['reg_subset'])
    # else:
    #     scope = np.arange(0,len(files_list),1)

    # for round, ind in enumerate(scope):
    #     file_path = files_list[ind]
    #     file = os.path.basename(file_path)
    #     print(ind,'working on ',file)
    #     image = io.imread(file_path)
    #     image = split_convert(image, input_txt['ch_names'])

    #     ## rotating and flipping the image
    #     for ch, img in image.items():
    #         image[ch] = rot_flip(img, input_txt['flip'], input_txt['angle'])

    #     if 'ants' in input_txt['steps']:
    #         if 'preshift' in input_txt['steps']:
    #             if ind == start:
    #                 pre_ref = start_ref.copy()
    #                 pre_shifts[file] = [0 for i in pre_ref[input_txt['ch_names'][-1]].shape]
    #                 current_shift = pre_shifts[file]
    #             else:
    #                 pre_shifts[file] = phase_corr(pre_ref[input_txt['ch_names'][-1]], 
    #                                             image[input_txt['ch_names'][-1]], input_txt['sigma'])
    #                 current_shift = [sum(x) for x in zip(current_shift, pre_shifts[file])]
    #                 pre_ref = image.copy()
    #                 for ch in image.keys():
    #                     image[ch] = ndimage.shift(image[ch], current_shift)
    #             print('current pre_shift', current_shift)
    #             if input_txt['save_pre_shift'] == True:
    #                 final = np.concatenate((np.empty_like(image[input_txt['ch_names'][0]]), 
    #                                         np.empty_like(image[input_txt['ch_names'][0]])))
    #                 final[0::2]= image[input_txt['ch_names'][0]]
    #                 final[1::2]= image[input_txt['ch_names'][-1]]
    #                 name = input_txt['save_path']+'PhaseCorr_'+file
    #                 save_image(name, final, xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'])
            
    #         if sum(input_txt['reg_subset']) != 0:
    #             for ch, img in image.items():
    #                 image[ch] = img_subset(img, input_txt['reg_subset'])

    #         for i, drift_t in enumerate(input_txt['drift_corr']): 
    #             ants_step = str(i+1)+'_'+drift_t
    #             if ind == start:
    #                 for ch, img in image.items():
    #                     save_name = input_txt['save_path']+drift_t+'_'+ch+'_'+file
    #                     save_image(save_name, img, 
    #                                xy_pixel=input_txt['xy_pixel'], 
    #                                z_pixel=input_txt['z_pixel'])
    #                 similarity_ref = image[input_txt['check_ch']]
    #                 similarity_check[ants_step] = check_similarity(similarity_ref,similarity_ref)
    #                 similarity_check['0_unregi'][file] = check_similarity(similarity_ref,similarity_ref)
    #                 print(file, 'was saved without applying ants on itself')
    #             else:
    #                 if i == 0:
    #                     similarity_check['0_unregi'][file] = check_similarity(similarity_ref,image[input_txt['check_ch']])
    #                 print('applying antspy with method',drift_t,'on file',file)
    #                 try:
    #                     metric_t = input_txt['metric'][i]
    #                 except:
    #                     for i in [0]:
    #                         print('optimization metric not recognized. mattes used instead')
    #                         metric_t = 'mattes'
    #                 unshifted_check = check_similarity(similarity_ref,image[input_txt['check_ch']])
    #                 shifted_img, ants_shift[ants_step][file] = apply_ants_channels(ref=ref, image=image, drift_corr=drift_t, 
    #                                                                             ch_names=input_txt['ch_names'],
    #                                                                             metric=metric_t, ref_ch=-1,
    #                                                                             reg_iterations=parameters['reg_iterations'], 
    #                                                                             aff_iterations=parameters['aff_iterations'], 
    #                                                                             aff_shrink_factors=parameters['aff_shrink_factors'], 
    #                                                                             aff_smoothing_sigmas=parameters['aff_smoothing_sigmas'],
    #                                                                             grad_step=parameters['grad_step'], 
    #                                                                             flow_sigma=parameters['flow_sigma'], 
    #                                                                             total_sigma=parameters['total_sigma'],
    #                                                                             aff_sampling=parameters['aff_sampling'], 
    #                                                                             syn_sampling=parameters['syn_sampling'], 
    #                                                                             xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
    #                                                                             check_ch=input_txt['check_ch'],
    #                                                                             save=False, save_path=input_txt['save_path'],
    #                                                                             save_file=str(i+1)+file)
    #                 similarity_check[ants_step] = check_similarity(similarity_ref,shifted_img[input_txt['check_ch']])
    #                 diff_1 = similarity_check['0_unregi'][file] - similarity_check[ants_step]
    #                 diff_2 = unshifted_check - similarity_check[ants_step] 
    #                 if diff_1 < 0.05 and diff_2 < 0.05:
    #                     print('similarity check improved from', unshifted_check, 'to', similarity_check[ants_step])
    #                     image = shifted_img.copy()
    #                 else:
    #                     print('similarity chack was worse after', drift_t, unshifted_check, '>>',similarity_check[ants_step])
    #                     print('this antpy transformation was ignored')  
    #         similarity_ref = image[input_txt['check_ch']]

    #         if len(input_txt['ch_names']) > 1:
    #             if input_txt['double_register'] == True:
    #                 similarity_check['final'] = {}
    #                 reg_rd = 0
    #                 for i, drift_t in enumerate(input_txt['drift_corr']):
    #                     if drift_t in ['Rigid','Similarity','Affine']:
    #                         if ind == start:
    #                             similarity_ref2 = image[input_txt['check_ch']]
    #                         else:
    #                             print('applying', drift_t, 'with first channel as ref')
    #                             unshifted_check = check_similarity(similarity_ref2,image[input_txt['check_ch']])
    #                             shifted_img, shift_2 = apply_ants_channels(ref=ref, image=image, 
    #                                                                                             drift_corr=drift_t, 
    #                                                                                             ch_names=input_txt['ch_names'],
    #                                                                                             metric=metric_t, ref_ch=0,
    #                                                                                             reg_iterations=parameters['reg_iterations'], 
    #                                                                                             aff_iterations=parameters['aff_iterations'], 
    #                                                                                             aff_shrink_factors=parameters['aff_shrink_factors'], 
    #                                                                                             aff_smoothing_sigmas=parameters['aff_smoothing_sigmas'],
    #                                                                                             grad_step=parameters['grad_step'], 
    #                                                                                             flow_sigma=parameters['flow_sigma'], 
    #                                                                                             total_sigma=parameters['total_sigma'],
    #                                                                                             aff_sampling=parameters['aff_sampling'], 
    #                                                                                             syn_sampling=parameters['syn_sampling'], 
    #                                                                                             xy_pixel=input_txt['xy_pixel'], 
    #                                                                                             z_pixel=input_txt['z_pixel'],
    #                                                                                             check_ch=input_txt['check_ch'],
    #                                                                                             save=False, save_path=input_txt['save_path'],
    #                                                                                             save_file=str(i+1)+file)
    #                             shifted_check = check_similarity(similarity_ref2,shifted_img[input_txt['check_ch']])
    #                             diff_3 = unshifted_check - shifted_check 
    #                             if diff_3 < 0.05:
    #                                 print('similarity check in 2nd ants_round improved from', unshifted_check, 'to', shifted_check)
    #                                 image = shifted_img.copy()
    #                                 reg_rd += 1
    #                             else:
    #                                 print('similarity chack in 2nd ants_round was worse after', drift_t, unshifted_check, '>>',shifted_check)
    #                                 print('this antpy transformation was ignored')  
    #                             similarity_ref2 = image[input_txt['check_ch']] 
    #                             similarity_check['final'][file] =  shifted_check
    #                 # if reg_rd > 0:
    #                 #     print('saving image after 2nd round of Antspy')
    #                 #     for ch, img in image.items():
    #                 #         save_name = input_txt['save_path']+'finalAnts_'+ch+'_'+file
    #                 #         save_image(save_name, img, 
    #                 #                 xy_pixel=input_txt['xy_pixel'], 
    #                 #                 z_pixel=input_txt['z_pixel'])                            
    #         ############ chnaging ref to shifted image every X runs/files based on reset_ref
    #         if ind % input_txt['ref_reset'] == 0:
    #             print(ind, input_txt['ref_reset'], ind % input_txt['ref_reset'])
    #             print('changing the ref image')
    #             ref = image.copy()

    #         if 'postshift' in input_txt['steps']:
    #             if round == 0:
    #                 for ch, img in image.items():
    #                     save_name = input_txt['save_path']+'PhaseCorr2_'+ch+'_'+file
    #                     save_image(save_name, img, 
    #                             xy_pixel=input_txt['xy_pixel'], 
    #                             z_pixel=input_txt['z_pixel'])
    #                 print(file, 'was saved without post_shift')
    #                 post_ref = start_ref.copy()
    #                 post_shifts[file] = [0 for i in post_ref[input_txt['ch_names'][0]].shape]
    #                 current_shift_2 = post_shifts[file]
    #                 print('current post_shift has been resetted', current_shift_2)
    #             elif ind == start:
    #                 print('second round on ref image is ignored')
    #                 post_ref = start_ref.copy()
    #                 post_shifts[file] = [0 for i in post_ref[input_txt['ch_names'][0]].shape]
    #                 current_shift_2 = post_shifts[file]
    #                 print('current post_shift has been resetted', current_shift_2)
    #             elif similarity_check[ants_step]<0.94:
    #                 post_shifts[file] = phase_corr(post_ref[input_txt['ch_names'][0]], 
    #                                             image[input_txt['ch_names'][0]], input_txt['sigma'])
    #                 current_shift_2 = [sum(x) for x in zip(current_shift_2, post_shifts[file])]
    #                 print(post_shifts[file], current_shift_2)
    #                 post_ref = image.copy()
    #                 for ch in image.keys():
    #                     image[ch] = ndimage.shift(image[ch], current_shift_2) 
    #                 print('current post_shift', current_shift_2)
    #             else:
    #                 print('post_shift was not applied because similarity metric is high enough', similarity_check[ants_step])  
    #             for ch, img in image.items():
    #                 image[ch] -= image[ch].min()
    #                 image[ch] = image[ch].astype(np.uint16)
    #                 save_name = input_txt['save_path']+'PhaseCorr2_'+ch+'_'+file  
    #                 save_image(save_name, image[ch], 
    #                             xy_pixel=input_txt['xy_pixel'], 
    #                             z_pixel=input_txt['z_pixel']) 

    #     if 'ants' not in input_txt['steps'] and len(input_txt['ch_names'])>1:
    #         name = input_txt['save_path']+input_txt['ch_names'][-1]+'_'+file
    #         save_image(name, image[input_txt['ch_names'][-1]], 
    #                    xy_pixel=input_txt['xy_pixel'], 
    #                    z_pixel=input_txt['z_pixel'])

    #     img = image[input_txt['ch_names'][0]]
    #     if 'n2v' in input_txt['steps']:
    #         print('applying n2v on', file)
    #         img = N2V_predict(image=img,
    #                           model_name=input_txt['model_name'], 
    #                           model_path=input_txt['model_path'], 
    #                           save=True, save_path=input_txt['save_path'],
    #                           xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
    #                           save_file=input_txt['ch_names'][0]+'_'+file)
    #     if 'clahe' in input_txt['steps']:
    #         print('applying clahe on', file)
    #         img = apply_clahe(kernel_size=input_txt['kernel_size'], 
    #                           image=img, clipLimit=input_txt['clipLimit'], 
    #                           xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
    #                           save=True, save_path=input_txt['save_path'], 
    #                           save_file=input_txt['ch_names'][0]+'_'+file)

    #     if 'mask' in input_txt['steps']:
    #         img = mask_subset(img, sig=10,
    #                             save=True, save_path=input_txt['save_path'],
    #                             xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
    #                             save_file=input_txt['ch_names'][0]+'_'+file)        

    #     del image, img
    #     gc.collect()
    #     print('memory usage after gc.collect')
    #     mem_use()        

    # # saving preshift and shift matrices
    # if 'ants' in input_txt['steps']: 
    #     shift_file = input_txt['save_path']+"ants_shifts.csv"
    #     with open(shift_file, 'w', newline='') as csvfile:
    #         fieldnames = ['reg_step', 'file', 'ants_shift']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for reg_step, shifts in ants_shift.items():
    #             for file, ants_shift in shifts.items():
    #                 writer.writerow({'reg_step': reg_step, 'file' : file, 'ants_shift' : ants_shift})
    #     csvfile.close()

    #     checks_file = input_txt['save_path']+"similarity_check.csv"
    #     with open(checks_file, 'w', newline='') as csvfile:
    #         fieldnames = ['reg_step', 'file', 'similarity_check']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for reg_step, checks in similarity_check.items():
    #             for file, similarity_check in checks.items():
    #                 writer.writerow({'reg_step': reg_step, 'file' : file, 'similarity_check' : similarity_check})
    #     csvfile.close()

    # if 'preshift' in input_txt['steps']:  
    #     shift_file = input_txt['save_path']+"PhaseCorr_shifts.csv"
    #     with open(shift_file, 'w', newline='') as csvfile:
    #         fieldnames = ['file', 'phase_shift']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for file, shift in pre_shifts.items():
    #             writer.writerow({'file' : file, 'phase_shift' : shift})
    #     csvfile.close()

    # if 'postshift' in input_txt['steps']:  
    #     shift_file = input_txt['save_path']+"PhaseCorr2_shifts.csv"
    #     with open(shift_file, 'w', newline='') as csvfile:
    #         fieldnames = ['file', 'phase2_shift']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for file, shift in post_shifts.items():
    #             writer.writerow({'file' : file, 'phase2_shift' : shift})
    #     csvfile.close() 
    
if __name__ == '__main__':
    main()