from timeit import default_timer as timer
import numpy as np 
from tqdm import tqdm
from skimage.filters import gaussian, threshold_otsu
import operator
from scipy import ndimage
import utils.datautils as datautils

def segment_3D(image, neu_no=10, max_neu_no=30, min_size=5000, xy_pixel=1, z_pixel=1, save=True, save_path='', save_file=''):
    s = ndimage.generate_binary_structure(len(image.shape), len(image.shape))
    labeled_array, num_labels = ndimage.label(image, structure=s)
    labels = np.unique(labeled_array)
    labels = labels[labels!=0]
    neu_sizes = {}
    for l in labels:
        neu_sizes[l] = (labeled_array == l).sum()/(labeled_array == l).max()
    avg_size = np.mean(list(neu_sizes.values()))
    if min_size != 0:
        for ind, l in enumerate(labels):
            if neu_sizes[l] < min_size:
                labels[ind] = 0
        labels = labels[labels!=0]
    if neu_no != 0 and num_labels > neu_no:
        for ind, l in enumerate(labels):
            if neu_sizes[l] < avg_size:
                labels[ind] = 0
        labels = labels[labels!=0]
    if max_neu_no != 0 and len(labels) > max_neu_no:
        sorted_sizes = sorted(neu_sizes.items(), key=operator.itemgetter(1), reverse=True)
        sorted_sizes = sorted_sizes[0:max_neu_no]
        labels = [[l][0][0] for l in sorted_sizes]
    neurons = {}
    for ind, l in enumerate(labels):
        labels[ind] = ind+1
        neuron = labeled_array.copy()
        neuron[neuron != l] = 0
        neuron[neuron == l] = ind+1
        neuron = neuron.astype('uint16')
        if neuron.sum() != 0 and neuron.sum() < np.prod(np.array(neuron.shape)):
            neurons[ind+1] = neuron
        else:
            pass
            # print('this segment was removed because its empty')
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
    for img_3D in tqdm(image_4D[1:], desc='matching segments', leave=False):
        current_neurons = segment_3D(img_3D, neu_no=neu_no, min_size=min_size, 
                                    max_neu_no=max_neu_no, save=False, save_path=save_path)
        for l, neu_list in final_neurons.items():
            neu = neu_list[-1]
            diff = np.prod(np.array(neu.shape))
            ID = 0
            for t, neu_1 in current_neurons.items():
                cur_diff = abs((neu/neu.max() - neu_1/neu_1.max())).sum() #compare difference of fit
                if cur_diff != 0:
                    if cur_diff < diff:
                        diff = cur_diff
                        ID = t
            if ID != 0:
                try:
                    final_neurons[l].append(current_neurons[ID])
                except:
                    pass
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