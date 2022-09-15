from skimage import morphology, io
import numpy as np
from scipy import ndimage
import tqdm
import tifffile
import cv2
from os.path import join as pjoin
from dipy.viz import regtools
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric
from dipy.align.imaffine import transform_centers_of_mass




def dendritic_space_transform(neuron_hexagon, neuron_center, save_path, data_path, file_name):
    
    """
    This function will calculate the transform matrix from the hexagon lattice of the dendrite to a standardized hexagon.
    It takes in the neuron_hexagon and neuron_center coordinates as a numpy array.
    It returns the dipy transformation object for the aligning the centers and for the transformation.
    It saves the images of the transform.
    The image space is supposed to be 400 x 400 pixels.

    Bugs: 
    It might be possible that the shape generation does not lead to a filled object which can lead to no succesful transform.

    Parameters:
    neuron_hexagon: should contain the six 2D coordinates of the column centers of the surrounding columns in image space
    neuron_center: should contain a single 2D coordinate for the central column in image space
    save_path: path to save the diffeomorphic grid deformation (for visualisation)
    data_path: path to a single channgel time lapse imaging file
    file_name: file_name of the transformed file
    
    
    
    Some variables:
    filled_to_filled: center of mass transform for two filled and adjusted hexagons that are used for the transformation
    raw_transform: moving the raw image to the center of mass of the regular hexagon
    mapping: transform object for diffeomorphic mapping between regular hexagon and neuron hexagon
    """
    neuron_hexagon = np.array(neuron_hexagon, np.int32)
    neuron_center = np.array(neuron_center, np.int32)
    neuron_hexagon_raw = neuron_hexagon - neuron_center

    neuron_hexagon = np.int32(np.round(neuron_hexagon_raw * (200/np.abs(neuron_hexagon_raw[:,1]).max()))) + np.array([200,200])
    neuron_hexagon_raw = neuron_hexagon_raw + neuron_center
    
    image = np.zeros((400,400))

    neuron_hexagon = neuron_hexagon.reshape((-1, 1, 2))
    isClosed = True
    thickness = 1

    neuron_hexagon_shape = cv2.polylines(image, [neuron_hexagon],
                          isClosed, thickness)

    neuron_hex_filled = np.array(ndimage.binary_fill_holes(neuron_hexagon_shape), dtype=np.float32)
    io.imshow(neuron_hex_filled)
    
    image = np.zeros((400,400))

    reg_hex = np.array([[200,0],
                    [26.8,100],
                    [26.8,300],
                    [200,400],
                    [373.2,300],
                    [373.2,100]
                   ], np.int32)

    reg_hex = reg_hex.reshape((-1, 1, 2))

    reg_hex_im = cv2.polylines(image, [reg_hex],
                          isClosed, thickness)
    reg_hex_im = morphology.dilation(reg_hex_im)

    subtract = reg_hex_im - morphology.erosion(reg_hex_im)


    reg_hex_filled = np.array(ndimage.binary_fill_holes(reg_hex_im), dtype=np.float32)-subtract
    
    image = np.zeros((400,400))
    
    
    neuron_hexagon_raw = neuron_hexagon_raw.reshape((-1, 1, 2))

    raw_im = cv2.polylines(image, [neuron_hexagon_raw],
                      isClosed, thickness)

    raw_im_filled = np.array(ndimage.binary_fill_holes(raw_im), dtype=np.float32)
    io.imshow(raw_im_filled)
                                       
    filled_to_filled = transform_centers_of_mass(static=reg_hex_filled,
                                static_grid2world=np.eye(3),
                                moving=neuron_hex_filled,
                                moving_grid2world=np.eye(3)
                               )
    neuron_hex_filled = filled_to_filled.transform(neuron_hex_filled)
                                       
    regtools.overlay_images(reg_hex_filled, neuron_hex_filled)
        
    raw_transform = transform_centers_of_mass(static=reg_hex_filled,
                                          static_grid2world=np.eye(3),
                                          moving=raw_im_filled,
                                          moving_grid2world=np.eye(3)
                                             )
    
    regtools.overlay_images(reg_hex_filled, raw_im_filled)
    
    dim = reg_hex_filled.ndim
    metric = SSDMetric(dim)
    level_iters = [200, 100, 50, 25]

    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50, ss_sigma_factor=2)
    mapping = sdr.optimize(reg_hex_filled, neuron_hex_filled)
    
    regtools.plot_2d_diffeomorphic_map(mapping, 10, fname=save_path + 'diffeomorphic_map.png', dpi=1000)
    
    data = tifffile.imread(data_path)
    
    stack_transformed = np.zeros(data.shape, np.int16)
    
    for timepoint in tqdm.tqdm(range(0,data.shape[0])):
        for vol_slice in range(0,data.shape[1]):
            stack_transformed[timepoint,vol_slice] = mapping.transform(
                raw_transform.transform(data[timepoint, vol_slice])
            )

    tifffile.imwrite(file=save_path + file_name, data=stack_transformed, **{'imagej':'TZYX', 'bigtiff':True})

    print('Done!')
    
    return stack_transformed
