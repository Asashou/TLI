from timeit import default_timer as timer
import os
import cv2 as cv
import numpy as np 
import tifffile as tif
from n2v.models import N2V
from tqdm import tqdm
from skimage.filters import gaussian, threshold_otsu

import utils.datautils as datautils


def N2V_predict(model_name, model_path, xy_pixel=1, z_pixel=1, image=0, file='', save=True, save_path='', save_file=''):
    """apply N2V prediction on image based on provided model
    if save is True, save predicted image with provided info"""
    if file != '':
        image = tif.imread(file)
    file_name = os.path.basename(file)
    model = N2V(config=None, name=model_name, basedir=model_path)
    predict = model.predict(image, axes='ZYX', n_tiles=None)
    # if predict.min() != 0:
    #     predict = datautils.img_limits(predict, limit=0)
    if save == True:
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            save_name = str(save_path+'N2V_'+file_name)
        else:
            save_name = str(save_path+'N2V_'+save_file)
        if '.tif' not in save_name:
            save_name +='.tif'
        datautils.save_image(save_name, predict, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return predict

def N2V_4D(image_4D, model_name, model_path, xy_pixel=1, z_pixel=1, save=True, save_path='', save_file=''):
    for st in tqdm(range(len(image_4D)), desc='applying N2V'): 
        image_4D[st] = N2V_predict(image=image_4D[st],
                                    model_name=model_name, 
                                    model_path=model_path, 
                                    save=False)
    if save == True:
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            save_name = str(save_path+'N2V_4D.tif')
        else:
            save_name = str(save_path+'N2V_'+save_file)
        if '.tif' not in save_name:
            save_name +='.tif'
        datautils.save_image(save_name, image_4D, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return image_4D

def apply_clahe(kernel_size, xy_pixel=1, z_pixel=1, image=0, file='', clipLimit=1, save=True, save_path='', save_file=''):
    """apply Clahe on image based on provided kernel_size and clipLimit
    if save is True, save predicted image with provided info"""
    if file != '':
        image = tif.imread(file)
    if image.min()<0:
        image = (image - image.min())
    image = image.astype('uint16')
    print(image.dtype)
    file_name = os.path.basename(file)
    image_clahe= np.empty(image.shape)
    clahe_mask = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=kernel_size)
    for ind, slice in enumerate(image):
        if image_clahe.min() != 0:
            image_clahe = datautils.img_limits(image_clahe, limit=0)
        image_clahe[ind] = clahe_mask.apply(slice)
        image_clahe[ind] = cv.threshold(image_clahe[ind], 
                            thresh=np.percentile(image_clahe[ind], 95), 
                            maxval=image_clahe[ind].max(), 
                            type= cv.THRESH_TOZERO)[1]
    # if image_clahe.min() != 0:
    #     image_clahe = datautils.img_limits(image_clahe, limit=0)
    if save == True:
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            save_name = save_path+'clahe_'+file_name
        else:
            save_name = save_path+'clahe_'+save_file
        if '.tif' not in save_name:
            save_name += '.tif'
        datautils.save_image(save_name, image_clahe, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return image_clahe

def clahe_4D(image_4D, kernel_size, clipLimit=1, xy_pixel=1, z_pixel=1, save=True, save_path='', save_file=''):
    for ind, stack in enumerate(image_4D):
        image_4D[ind] = apply_clahe(image=stack,
                                    kernel_size=kernel_size, 
                                    clipLimit=clipLimit, 
                                    save=False)
    if save == True:
        if save_path != '' and save_path[-1] != '/':
            save_path += '/'
        if save_file == '':
            save_name = str(save_path+'clahe_4D.tif')
        else:
            save_name = str(save_path+'clahe_'+save_file)
        if '.tif' not in save_name:
            save_name +='.tif'
        datautils.save_image(save_name, image_4D, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return image_4D

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
    for i in tqdm(range(1), desc = '3D_mask'):
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

def mask_4D(image, xy_pixel=1, z_pixel=1, sig=2, save=True, save_path='', save_file=''):
    start_time = timer()
    mask = image.copy()
    mask_image = image.copy()
    for i, img in enumerate(image):
        print('calculating mask for stack#', i)
        try:
            mask[i] = mask_image(img, return_mask=True ,sig=sig)
            mask_image[i] = mask_image(img, return_mask=False ,sig=sig)
        except:
            mask[i] = mask[i]
            mask_image[i] = mask_image[i]
    if save == True:
        if save_file == '':
            save_name = save_path+'masked_image.tif'
            mask_name = save_path+'mask.tif'
        else:
            mask_name = save_path+'mask_'+save_file
            save_name = save_path+'image_mask_'+save_file
        if '.tif' not in save_name:
            save_name += '.tif'
        datautils.save_image(mask_name, mask, xy_pixel=xy_pixel, z_pixel=z_pixel)
        datautils.save_image(save_name, mask_image, xy_pixel=xy_pixel, z_pixel=z_pixel)
    print('mask_4D runtime', timer()-start_time)
    return mask, mask_image