# We import all our dependencies.
import argparse
import os
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
    print('image old limits', img.min(), img.max())
    img = img - img.min()
    if limit != 0:
        img = img/img.max()
        img = img*limit
    img = img.astype(ddtype)
    print('image new limits and type', img.min(), img.max(), img.dtype)
    return img

def split_convert(image, ch_names):
    """deinterleave the image into dictionary of two channels"""
    image_ch = {}
    for ind, ch in enumerate(ch_names):
        image_ch[ch] = image[ind::len(ch_names)]
    if len(ch_names) > 1:
        image_ch[ch_names[-1]] = filters.median(image_ch[ch_names[-1]])
    for ch, img in image_ch.items():
        image_ch[ch] = img_limits(img, limit=0)
    return image_ch

def save_image(name, image, xy_pixel=0.0764616, z_pixel=0.4):
    """save provided image by name with provided xy_pixel, and z_pixel resolution as metadata"""
    tif.imwrite(name, image, imagej=True, resolution=(1./xy_pixel, 1./xy_pixel),
                metadata={'spacing': z_pixel, 'unit': 'um', 'finterval': 1/10,'axes': 'ZYX'})

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
                              aff_shrink_factors=aff_shrink_factors, 
                              aff_smoothing_sigmas=aff_smoothing_sigmas,
                              grad_step=grad_step, flow_sigma=flow_sigma, total_sigma=total_sigma,
                              aff_sampling=aff_sampling, syn_sampling=syn_sampling)
    print(shift)
    return shift

def antspy_drift(fixed, moving, shift):
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
    # vol_shifted = img_limits(vol_shifted)
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
                        save=True, save_path='',save_file=''):
    """calculate and apply shift on both channels of image based on ref, which is dictionary of two channels.
    if save is True, save shifted channels individually with provided info"""
    for ch, value in ref.items():
        try:
            ref[ch]= ants.from_numpy(np.float32(value))
        except:
            pass
    for ch, value in image.items():
        image[ch]= ants.from_numpy(np.float32(value))
    shift = antspy_regi(ref[ch_names[ref_ch]], image[ch_names[ref_ch]], drift_corr, metric,
                        reg_iterations=reg_iterations, 
                        aff_iterations=aff_iterations, 
                        aff_shrink_factors=aff_shrink_factors, 
                        aff_smoothing_sigmas=aff_smoothing_sigmas,
                        grad_step=grad_step, flow_sigma=flow_sigma, 
                        total_sigma=total_sigma,
                        aff_sampling=aff_sampling, 
                        syn_sampling=syn_sampling)
    for ch, img in image.items():
        image[ch] = antspy_drift(ref[ch],img,shift=shift['fwdtransforms'])
        if save == True:
            img_save = img_limits(image[ch])
            save_name = str(save_path+drift_corr+'_'+ch+'_'+save_file)
            if '.tif' not in save_name:
                save_name += '.tif'
            save_image(save_name, img_save, xy_pixel=xy_pixel, z_pixel=z_pixel)       
    return image, shift

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
    print('applying pre-shift with phase correlation')
    try:
        for i in [0]:
            shift, error, diffphase = corr(fixed, moving)
    except:
        for i in [0]:
            shift, error, diffphase = np.zeros(len(moving)), 0, 0
            print("couldn't perform PhaseCorr, so shift was casted as zeros")
    return shift

def N2V_predict(model_name, model_path, xy_pixel, z_pixel, image=0, file='', save=True, save_path='', save_file=''):
    """apply N2V prediction on image based on provided model
    if save is True, save predicted image with provided info"""
    if file != '':
        image = tif.imread(file)
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
        save_image(save_name, img_save, xy_pixel=xy_pixel, z_pixel=z_pixel)
    return predict

def apply_clahe(kernel_size, xy_pixel, z_pixel, image=0, file='', clipLimit=1, save=True, save_path='', save_file=''):
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
    with open(args.txt_path) as f:
        lines = f.readlines()
    delimiter = detect(lines[0])
    lines = [item.strip().rsplit(delimiter, 2) for item in lines]
    input_txt = {item[0].strip(): item[1].strip() for item in lines}
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
    # globals().update(input_txt)
    print('getting info from', args.txt_path)

    if type(input_txt['ch_names']) != list:
        input_txt['ch_names'] = [input_txt['ch_names']]
    
    if type(input_txt['drift_corr']) != list:
        input_txt['drift_corr'] = [input_txt['drift_corr']]
    
    if 'sigma' not in input_txt.keys():
        input_txt['sigma'] = 0

    if 'steps' not in input_txt.keys():
        input_txt['steps'] = ['all']
    if type(input_txt['steps']) == str:
        input_txt['steps'] = [input_txt['steps'].lower()]
    elif type(input_txt['steps']) == tuple:
        input_txt['steps'] = [s.lower() for s in input_txt['steps']]
    if 'all' in input_txt['steps']:
        input_txt['steps'] = ['preshift', 'postshift', 'ants', 'n2v', 'clahe']
    
    if 'metric' not in input_txt:
        input_txt['metric'] = 'mattes'

    print(input_txt)
    mem_use()

    #######
    files_list = get_file_names(input_txt['path_to_data'], 
                                group_by=input_txt['group'], 
                                order=input_txt['reference_last'])
    print('the first 5 files (including ref) are', files_list[0:5])
    
    if 'ants' in input_txt['steps']:
        parameters = {'grad_step':0.2, 'flow_sigma':3, 'total_sigma':0,
                      'aff_sampling':32, 'aff_random_sampling_rate':0.2, 
                      'syn_sampling':32, 'reg_iterations':(40, 20, 0), 
                      'aff_iterations':(2100, 1200, 1200, 10), 
                      'aff_shrink_factors':(6, 4, 2, 1), 
                      'aff_smoothing_sigmas':(3, 2, 1, 0)}
        for para in parameters.keys():
            try:
                parameters[para] = input_txt[para]
            except:
                pass
        
        ants_ref_no = str(input_txt['ants_ref_no'])
        print(ants_ref_no)
        try:
            for i in [0]:
                file = [file for file in files_list if ants_ref_no in file][0]
                print(file)
                ref = io.imread(file)
                print(os.path.basename(file), 'is used as ref for Antspy')
                start = files_list.index(file)
        except:
            for i in [0]:
                print("couldn't find the ref file specified in info.txt")
                ref = io.imread(files_list[0])
                print(os.path.basename(files_list[0]), 'is used instead as ref for Antspy')
                start = 0
        if start > 0:
            scope1 = np.arange(start, -1, -1)
            scope2 = np.arange(start, len(files_list), 1)
            scope = np.concatenate((scope1, scope2))
        else:
            scope = np.arange(0,len(files_list),1)
        print('registration sequence:', scope) 
        ref = split_convert(ref, input_txt['ch_names'])
        ants_shift = {i:{} for i in range(len(input_txt['drift_corr']))}
        if 'preshift' in input_txt['steps']:
            pre_shifts = {}
        if 'postshift' in input_txt['steps']:
            post_shifts = {}

    else:
        scope = np.arange(0,len(files_list),1)

    for t, ind in enumerate(scope):
        file = files_list[ind]
        save_file = os.path.basename(file)
        print(ind,'working on ',save_file)
        image = io.imread(file)
        image = split_convert(image, input_txt['ch_names'])
        if 'ants' in input_txt['steps']:
            if 'preshift' in input_txt['steps']:
                if ind == start:
                    pre_ref = ref.copy()
                    current_shift = [0 for i in pre_ref[input_txt['ch_names'][-1]].shape]
                else:
                    pre_shifts[ind] = phase_corr(pre_ref[input_txt['ch_names'][-1]], 
                                                image[input_txt['ch_names'][-1]], input_txt['sigma'])
                    current_shift = [sum(x) for x in zip(current_shift, pre_shifts[ind])]
                    pre_ref = image.copy()
                    for ch in image.keys():
                        image[ch] = ndimage.shift(image[ch], current_shift)
                print('current pre_shift', current_shift)
                if input_txt['save_pre_shift'] == True:
                    final = np.concatenate((np.empty_like(image[input_txt['ch_names'][0]]), 
                                            np.empty_like(image[input_txt['ch_names'][0]])))
                    final[0::2]= image[input_txt['ch_names'][0]]
                    final[1::2]= image[input_txt['ch_names'][-1]]
                    name = input_txt['save_path']+'PhaseCorr_'+save_file
                    save_image(name, final, xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'])

            for i, drift_t in enumerate(input_txt['drift_corr']): 
                if ind == start:
                    for ch, img in image.items():
                        save_name = input_txt['save_path']+drift_t+'_'+ch+'_'+save_file
                        save_image(save_name, img, 
                                   xy_pixel=input_txt['xy_pixel'], 
                                   z_pixel=input_txt['z_pixel'])
                    print(save_file, 'was saved without applying ants on itself')
                else:
                    print('applying antspy with method',drift_t,'on file',save_file)
                    try:
                        metric_t = input_txt['metric'][i]
                    except:
                        metric_t = 'mattes'
                    image, ants_shift[i][ind] = apply_ants_channels(ref=ref, image=image, drift_corr=drift_t, 
                                                                    ch_names=input_txt['ch_names'],
                                                                    metric=metric_t, ref_ch=-1,
                                                                    reg_iterations=(reg_iterations[0],reg_iterations[1],reg_iterations[2]), 
                                                                    aff_iterations=(aff_iterations[0],aff_iterations[1],aff_iterations[2],aff_iterations[3]), 
                                                                    aff_shrink_factors=parameters['aff_shrink_factors'], 
                                                                    aff_smoothing_sigmas=parameters['aff_smoothing_sigmas'],
                                                                    grad_step=parameters['grad_step'], 
                                                                    flow_sigma=parameters['flow_sigma'], 
                                                                    total_sigma=parameters['total_sigma'],
                                                                    aff_sampling=parameters['aff_sampling'], 
                                                                    syn_sampling=parameters['syn_sampling'], 
                                                                    xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
                                                                    save=True, save_path=input_txt['save_path'],
                                                                    save_file=save_file)
                # ########### chnaging ref to shifted image every X runs/files based on reset_ref
                if ind % input_txt['ref_reset'] == 0:
                    print('changing the ref image')
                    ref = image.copy()
            if 'postshift' in input_txt['steps']:
                save_file_p = 'rGFP_'+save_file
                if ind == start:
                    for ch, img in image.items():
                        save_name = input_txt['save_path']+'Rigid_'+ch+'_'+save_file_p
                        save_image(save_name, img, 
                                   xy_pixel=input_txt['xy_pixel'], 
                                   z_pixel=input_txt['z_pixel'])
                    print(save_file, 'was saved without applying ants on itself')
                else:
                    print('applying antspy with method','Rigid','on green_ch of file',save_file)
                    metric_t = 'meansquares'
                    image, post_shifts[ind] = apply_ants_channels(ref=ref, image=image, drift_corr='Rigid', 
                                                                    ch_names=input_txt['ch_names'],
                                                                    metric=metric_t, ref_ch=0,
                                                                    reg_iterations=parameters['reg_iterations'], 
                                                                    aff_iterations=parameters['aff_iterations'], 
                                                                    aff_shrink_factors=parameters['aff_shrink_factors'], 
                                                                    aff_smoothing_sigmas=parameters['aff_smoothing_sigmas'],
                                                                    grad_step=parameters['grad_step'], 
                                                                    flow_sigma=parameters['flow_sigma'], 
                                                                    total_sigma=parameters['total_sigma'],
                                                                    aff_sampling=parameters['aff_sampling'], 
                                                                    syn_sampling=parameters['syn_sampling'], 
                                                                    xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
                                                                    save=True, save_path=input_txt['save_path'],
                                                                    save_file=save_file_p)                
        if 'ants' not in input_txt['steps'] and len(input_txt['ch_names'])>1:
            name = input_txt['save_path']+input_txt['ch_names'][-1]+'_'+save_file
            save_image(name, image[input_txt['ch_names'][-1]], 
                       xy_pixel=input_txt['xy_pixel'], 
                       z_pixel=input_txt['z_pixel'])
        img = image[input_txt['ch_names'][0]]
        if 'n2v' in input_txt['steps']:
            # if ind == start and i>0:
            #     continue
            print('applying n2v on', save_file)
            img = N2V_predict(image=img,
                              model_name=input_txt['model_name'], 
                              model_path=input_txt['model_path'], 
                              save=True, save_path=input_txt['save_path'],
                              xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
                              save_file=input_txt['ch_names'][0]+'_'+save_file)
        if 'clahe' in input_txt['steps']:
            # if ind == start and i>0:
            #     continue
            print('applying clahe on', save_file)
            img = apply_clahe(kernel_size=input_txt['kernel_size'], 
                              image=img, clipLimit=input_txt['clipLimit'], 
                              xy_pixel=input_txt['xy_pixel'], z_pixel=input_txt['z_pixel'],
                              save=True, save_path=input_txt['save_path'], 
                              save_file=input_txt['ch_names'][0]+'_'+save_file)
        print('finished applying pipline for ', save_file)
        print('memory usage before gc.collect')
        mem_use()
        del image, img
        gc.collect()
        print('memory usage after gc.collect')
        mem_use()        



    # saving preshift and shift matrices
    if 'ants' in input_txt['steps']: 
        shift_file = open(input_txt['save_path']+"ants_shifts.csv", "w")
        writer = csv.writer(shift_file)
        for key, value in ants_shift.items():
            try:
                writer.writerow([key, value])
            except:
                pass    
        print('saved ants_shifts')
        shift_file.close()
    if 'preshift' in input_txt['steps']:                
        shift_file = open(input_txt['save_path']+"PhaseCorr_shifts.csv", "w")
        writer = csv.writer(shift_file)
        for key, value in pre_shifts.items():
            writer.writerow([key, value])
        shift_file.close()
    mem_use()
    
if __name__ == '__main__':
    main()