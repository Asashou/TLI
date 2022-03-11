import ants
from tqdm import tqdm
import numpy as np
import utils.datautils as datautils
import os
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation as corr
import csv
from scipy import ndimage

def antspy_drift_corr(img_4D_r, img_4D_g, ch_names, save_path, save_name, ref_t=0, drift_corr='Rigid', metric='CC'):
    """
    This function takes the folder containing all the files to each channel as an input and drift corrects them.
    It uses the files from ch1 to do the drift correction and applies the same correction to ch2.
    By default antspy uses the SyN drift correction algorithm.
    
    path_to_data_ch1: Path to files for ch1 
    path_to_data_ch2: Path to files for ch2
    savepath: Path to folder where the final files should be saved in
    name_ch1: Name for the final file for ch1
    name_ch2: Name for the final file for ch2
    name_shifts: Name for the file containing the shifts for each volume
    """

    shifts = []

    scope = np.arange(ref_t,-1,-1)
    scope = np.concatenate((scope, np.arange(ref_t,len(img_4D_r))))

    for i in tqdm(scope, desc = 'applying_antspy'):
        if i == ref_t:
            ch1_name = save_path+ch_names[0]+'_'+drift_corr+'_'+save_name+'_'+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch1_name, img_4D_g[ref_t], xy_pixel=0.0764616, z_pixel=0.4)
            ch2_name = save_path+ch_names[1]+'_'+drift_corr+'_'+save_name+'_'+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch2_name, img_4D_r[ref_t], xy_pixel=0.0764616, z_pixel=0.4)

            last = ants.from_numpy(np.float32(img_4D_r[ref_t]))

        else:        
            moving_ch1 = ants.from_numpy(np.float32(img_4D_g[i]))
            moving_ch2 = ants.from_numpy(np.float32(img_4D_r[i]))
            
            shift = ants.registration(fixed=last, moving=moving_ch2, type_of_transform=drift_corr, syn_metric=metric)
            
            vol_shifted_ch1 = ants.apply_transforms(fixed=last, moving=moving_ch1, transformlist=shift['fwdtransforms'])
            vol_shifted_ch2 = shift['warpedmovout']
            last = shift['warpedmovout'].copy()

            vol_shifted_ch1 = np.int16(vol_shifted_ch1.numpy())
            vol_shifted_ch2 = np.int16(vol_shifted_ch2.numpy())

            ch1_name = save_path+ch_names[0]+'_'+drift_corr+'_'+save_name+'_'+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch1_name, vol_shifted_ch1, xy_pixel=0.0764616, z_pixel=0.4)
            ch2_name = save_path+ch_names[1]+'_'+drift_corr+'_'+save_name+'_'+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch2_name, vol_shifted_ch2, xy_pixel=0.0764616, z_pixel=0.4)

            shifts.append(shift['fwdtransforms'])

            for el in shift['fwdtransforms']:
                os.remove(el)
            for el in shift['invtransforms']:
                try:
                    os.remove(el)
                except:
                    pass
                    
            del vol_shifted_ch1, vol_shifted_ch2, shift, moving_ch1, moving_ch2
    shifts_name = save_path+drift_corr+'_'+save_name+'.csv'
    shifts_name = shifts_name.replace('.tif','')
    np.savetxt(X=shifts, fname=shifts_name, delimiter=', ', fmt='% s')

    return

def phase_corr(fixed, moving, sigma):
    # setting the length of fixed and moving images to be same (smaller length)
    if fixed.shape > moving.shape:
        fixed = fixed[tuple(map(slice, moving.shape))]
    elif fixed.shape < moving.shape:
        moving = moving[tuple(map(slice, fixed.shape))]
    # applying gaussian blur into fixed and moving images
    fixed = gaussian(fixed, sigma=sigma)
    moving = gaussian(moving, sigma=sigma)
    # calculating phase_correlation between fixed and moving images
    shift, error, diffphase = corr(fixed, moving)
    return shift

def phase_corr_4D(image, sigma, xy_pixel=1, 
                  z_pixel=1, ch_names=[1], 
                  ref_ch=-1,                      
                  save=True, save_path='',
                  save_file='', save_shifts=True):
    if isinstance(image, dict) == False:
        image = {ch_names[0]:image}
    pre_shifts = {}
    if len(ch_names) == 1:
        ref_ch = ch_names[0]
    else:
        try:
            ref_ch = ch_names[ref_ch]
        except:
            ref_ch = ch_names[-1]
    ref_im = image[ref_ch]
    current_shift = [0 for i in ref_im[0].shape]
    for ind in tqdm(np.arange(len(ref_im[1:])), desc='applying phase_corr'):
        pre_shifts[ind+1] = phase_corr(ref_im[ind], ref_im[ind+1], sigma) 
        current_shift = [sum(x) for x in zip(current_shift, pre_shifts[ind+1])] 
        for ch, img in image.items(): 
            image[ch][ind] = ndimage.shift(img[ind], current_shift) 
    if save == True:
        for ch, img in image.items():
            save_name = str(save_path+'PhaseCorr_'+ch+'_'+save_file)
            if '.tif' not in save_name:
                save_name += '.tif'
            datautils.save_image(save_name, img, xy_pixel=xy_pixel, z_pixel=z_pixel)   
        shift_file = save_path+"PhaseCorr_shifts.csv"
        with open(shift_file, 'w', newline='') as csvfile:
            fieldnames = ['timepoint', 'phase_shift']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for timepoint, shift in pre_shifts.items():
                writer.writerow({'timepoint' : timepoint+1, 'phase_shift' : shift})
        csvfile.close()
    return image, pre_shifts