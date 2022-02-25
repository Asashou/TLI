import ants
from tqdm import tqdm
import numpy as np
import utils.datautils as datautils

def antspy_drift_corr(img_4D_r, img_4D_g, ch_names, save_path, save_name, name_shifts, ref_t=0, drift_corr='Rigid'):
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

    for i in tqdm(scope):
        if i == ref_t:
            ch1_name = save_path+ch_names[0]+'_'+save_name+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch1_name, img_4D_g[ref_t], xy_pixel=0.0764616, z_pixel=0.4)
            ch2_name = save_path+ch_names[1]+'_'+save_name+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch2_name, img_4D_r[ref_t], xy_pixel=0.0764616, z_pixel=0.4)

            last = ants.from_numpy(np.float32(img_4D_r[ref_t]))

        else:        
            moving_ch1 = ants.from_numpy(np.float32(img_4D_g[i]))
            moving_ch2 = ants.from_numpy(np.float32(img_4D_r[i]))
            
            shift = ants.registration(fixed=last, moving=moving_ch2, type_of_transform=drift_corr)
            
            vol_shifted_ch1 = ants.apply_transforms(fixed=last, moving=moving_ch1, transformlist=shift['fwdtransforms'])
            vol_shifted_ch2 = shift['warpedmovout']
            last = shift['warpedmovout'].copy()

            vol_shifted_ch1 = np.int16(vol_shifted_ch1.numpy())
            vol_shifted_ch2 = np.int16(vol_shifted_ch2.numpy())

            ch1_name = save_path+ch_names[0]+'_'+save_name+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch1_name, vol_shifted_ch1, xy_pixel=0.0764616, z_pixel=0.4)
            ch2_name = save_path+ch_names[1]+'_'+save_name+str(f"{i+1:03d}")+'.tif'
            datautils.save_image(ch2_name, vol_shifted_ch2, xy_pixel=0.0764616, z_pixel=0.4)

            shifts.append(shift['fwdtransforms'])
                    
            del vol_shifted_ch1, vol_shifted_ch2, shift, moving_ch1, moving_ch2

    np.savetxt(X=shifts, fname=save_path+name_shifts, delimiter=', ', fmt='% s')
    return