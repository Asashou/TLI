from timeit import default_timer as timer
import numpy as np 
from tqdm import tqdm
from skimage.filters import gaussian, threshold_otsu
import operator
from scipy import ndimage
import utils.datautils as datautils

def segment_neuron(image_4D, neu_no=1, xy_pixel=1, z_pixel=1, save=True, save_path='', save_file=''):
    s = ndimage.generate_binary_structure(len(image_4D.shape), len(image_4D.shape))
    labeled_array, num_labels = ndimage.label(image_4D, structure=s)
    labels = np.unique(labeled_array)
    labels = labels[labels!=0]
    if neu_no != 0:
        neuron = image_4D.copy()
        neuron[labeled_array!=neu_no] = 0
    else:
        neuron = labeled_array.copy()
    if save == True:
        if save_file == '':
            save_name = save_path+'neuron.tif'
        else:
            save_name = save_path+'seg_'+save_file
        if '.tif' not in save_name:
            save_name += '.tif'
        datautils.save_image(save_name, neuron, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return neuron