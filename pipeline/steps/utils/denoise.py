from timeit import default_timer as timer
import os
import cv2 as cv
import numpy as np 
import tifffile as tif
from n2v.models import N2V
from tqdm import tqdm
# from skimage.filters import gaussian, threshold_otsu
import operator

import utils.datautils as datautils


def N2V_predict(model_name, model_path, n_tile=(2,4,4), xy_pixel=1, z_pixel=1, image=0, file='', save=True, save_path='', save_file=''):
    """apply N2V prediction on image based on provided model
    if save is True, save predicted image with provided info"""
    if file != '':
        image = tif.imread(file)
    file_name = os.path.basename(file)
    model = N2V(config=None, name=model_name, basedir=model_path)
    predict = model.predict(image, axes='ZYX', n_tiles=n_tile)
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

def N2V_4D(image_4D, model_name, model_path, n_tile=(2,4,4), xy_pixel=1, z_pixel=1, save=True, save_path='', save_file=''):
    model = N2V(config=None, name=model_name, basedir=model_path)
    for st in tqdm(range(len(image_4D)), desc='applying N2V', leave=False):
        image_4D[st] = model.predict(image_4D[st], axes='ZYX', n_tiles=n_tile) 
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
        file_name = os.path.basename(file)
    image -= image.min()
    image = image.astype('uint16')
    image_clahe= np.empty(image.shape)
    clahe_mask = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=kernel_size)
    for ind, slice in enumerate(image):
        image_clahe[ind] = clahe_mask.apply(slice)
        image_clahe[ind] = cv.threshold(image_clahe[ind], 
                            thresh=np.percentile(image_clahe[ind], 95), 
                            maxval=image_clahe[ind].max(), 
                            type= cv.THRESH_TOZERO)[1]
    image_clahe = datautils.img_limits(image_clahe, limit=image.max(), ddtype='int16')
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
    for st in tqdm(range(len(image_4D)), desc='applying clahe', leave=False):
        image_4D[st] = apply_clahe(image=image_4D[st],
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

def segment_3D(image, neu_no=10, max_neu_no=30, min_size=5000, xy_pixel=1, z_pixel=1, save=True, save_path='', save_file=''):
    s = ndimage.generate_binary_structure(len(image.shape), len(image.shape))
    labeled_array, num_labels = ndimage.label(image, structure=s)
    labels = np.unique(labeled_array)
    labels = labels[labels!=0]
    neu_sizes = {}
    for l in labels:
        neu_sizes[l] = (labeled_array == l).sum()/(labeled_array == l).max()
        # print((labeled_array == l).sum(), neu_sizes[l])
    avg_size = np.mean(list(neu_sizes.values()))
    # print('average, min and max segments sizes', avg_size, np.min(list(neu_sizes.values())), np.max(list(neu_sizes.values())))
    if min_size != 0:
        for ind, l in enumerate(labels):
            if neu_sizes[l] < min_size:
                # print(neu_sizes[l])
                labels[ind] = 0
        labels = labels[labels!=0]
    if neu_no != 0 and num_labels > neu_no:
        for ind, l in enumerate(labels):
            if neu_sizes[l] < avg_size:
                # print(neu_sizes[l])
                labels[ind] = 0
        labels = labels[labels!=0]
        # print('segments after first filtering', len(labels))
    if max_neu_no != 0 and len(labels) > max_neu_no:
        sorted_sizes = sorted(neu_sizes.items(), key=operator.itemgetter(1), reverse=True)
        sorted_sizes = sorted_sizes[0:max_neu_no]
        labels = [[l][0][0] for l in sorted_sizes]
        # print('# segments after second filtering', len(labels))
    # print('segments after first filtering', len(labels))
    neurons = {}
    for ind, l in enumerate(labels):
        labels[ind] = ind+1
        neuron = labeled_array.copy()
        neuron[neuron != l] = 0
        neuron[neuron == l] = ind+1
        neuron = neuron.astype('uint16')
        # print('values and size of neuron:', neuron.min(), neuron.max(), neuron.sum()/(ind+1))
        if neuron.sum() != 0 and neuron.sum() < np.prod(np.array(neuron.shape)):
            neurons[ind+1] = neuron
        else:
            print('this segment was removed because its empty')
        if save == True:
            if save_file == '':
                save_name = str(save_path+str(ind)+'_neuron.tif')
            else:
                save_name = str(save_path+'neuron_'+str(ind)+'_'+save_file)
            if '.tif' not in save_name:
                save_name +='.tif'
            datautils.save_image(save_name, neuron, xy_pixel=xy_pixel, z_pixel=z_pixel)             
    return neurons

def segment_4D(image_4D, neu_no=5, 
                max_neu_no=5, min_size=5000,
                xy_pixel=1, z_pixel=1,
                filter=True,
                save=True, save_path='', save_file=''):
    start_time = timer()
    final_neurons = segment_3D(image_4D[0], neu_no=neu_no, min_size=min_size, 
                                max_neu_no=max_neu_no, save=True, save_path=save_path)
    final_neurons = {l:[arr] for l, arr in final_neurons.items()}
    # print('identified neurons in first timepoint', final_neurons.keys())
    for img_3D in tqdm(image_4D[1:], desc='matching segments', leave=False):
        current_neurons = segment_3D(img_3D, neu_no=neu_no, min_size=min_size, 
                                    max_neu_no=max_neu_no, save=False, save_path=save_path)
        # print('identified neurons in first timepoint', ind+1, final_neurons.keys())
        for l, neu_list in final_neurons.items():
            neu = neu_list[-1]
            # neu[neu != 0] = 1
            neu_size = neu.sum() / neu.max()
            # print('size of segment %i in previous timepoint' %neu_size)
            diff = np.prod(np.array(neu.shape))
            ID = 0
            for t, neu_1 in current_neurons.items():
                # neu_1[neu_1 != 0] = 1
                neu1_size = neu_1.sum() / neu_1.max()
                print('size of segment %i in current timepoint' %neu1_size)
                # size_dif = abs(neu1_size - neu_size)
                # if size_dif/neu_size > 0.5:
                #     print('segment %i in current timepoint will be based due to big size diff' %t)
                # else:
                cur_diff = abs((neu/neu.max() - neu_1/neu_1.max())).sum()
                print('previous and current diffs for segment', t, diff,cur_diff)
                if cur_diff != 0:
                    if cur_diff < diff:
                        diff = cur_diff
                        ID = t
            if ID != 0:
                try:
                    final_neurons[l].append(current_neurons[ID])
                    # print('segment', ID, 'in timepoint', ind+2,'was assigned to segment', l, 'in final image')
                except:
                    pass
                    # print("no similar neuron was found at timepoint", ind+2)
        # print('finished segmenting timepoint', ind+2)
        current_neurons = None
    for l, neu_4D in final_neurons.items():
        neu_4D = np.array(neu_4D)
        if save == True:
            if save_file == '':
                save_name = str(save_path+str(l)+'_seg.tif')
            else:
                save_name = str(save_path+'seg_'+str(l)+'_'+save_file)
            if '.tif' not in save_name:
                save_name +='.tif'
            datautils.save_image(save_name, neu_4D, xy_pixel=xy_pixel, z_pixel=z_pixel)
    if filter == True:
        for l, mask_4D in final_neurons.items():
            neuron = image_4D.copy()
            neuron[mask_4D==0] = 0
            if save == True:
                if save_file == '':
                    save_name = str(save_path+str(l)+'_neuron.tif')
                else:
                    save_name = str(save_path+'neuron_'+str(l)+'_'+save_file)
                if '.tif' not in save_name:
                    save_name +='.tif'
                datautils.save_image(save_name, neuron, xy_pixel=xy_pixel, z_pixel=z_pixel)
            final_neurons[l] = neuron
            neuron = None
    print('segmentation runtime', timer()-start_time)
    return final_neurons