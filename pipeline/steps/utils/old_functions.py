# def stable_branch(img, stab_limit=4, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
#     stable_img = img.copy()
#     # deleting unstabilized pixels: ones that don't remain at least an hour
#     for t in tqdm(np.arange(img[stab_limit-1:].shape[0]), desc='filtering_px', leave=False):
#         for z in np.arange(img.shape[1]):
#             for y in np.arange(img.shape[2]):
#                 for x in np.arange(img.shape[3]): 
#                     if img[t:t+stab_limit,z,y,x].sum() < stab_limit:
#                         stable_img[t+stab_limit-1,z,y,x] = 0

#     # make the first 4 timepoints the same (you can ignore first hour of analysis)
#     stable_img[0] = stable_img[1] = stable_img[2] = stable_img[1]
#     if save == True:
#         if save_path != '' and save_path[-1] != '/':
#             save_path += '/'
#         if save_file == '':
#             save_name = str(save_path+'stable_image.tif')
#         else:
#             save_name = str(save_path+'stable_'+save_file)
#         if '.tif' not in save_name:
#             save_name +='.tif'
#         datautils.save_image(save_name, stable_img, xy_pixel=xy_pixel, z_pixel=z_pixel)
#     return stable_img

# def px_lifetimes(image_4D):
#     lifetimes = []
#     px_ind = _4D_to_PC(image_4D[0])
#     for i in px_ind:
#         z,y,x = i[0], i[1], i[2]
#         px = image_4D[:,z,y,x]
#         if px.sum() > 0:
#             iter = np.where(px == 1)[0]
#             life = 1
#             if len(iter) == 1:
#                 lifetimes.append([iter[0],z,y,x,1])
#             else:
#                 for i, t in enumerate(iter[1:]):
#                     if t - iter[i] == 1:
#                         life += 1
#                     elif t - iter[i] > 1:
#                         lifetimes.append([iter[i],z,y,x,life])
#                         life = 1
#                 # if life > 1:
#                 lifetimes.append([iter[-1],z,y,x,life])
#     lifetimes = np.array(lifetimes)
#     return lifetimes

# def stable_px(image_4D, st_limit = 4, save=True, save_path='', save_file='', xy_pixel=1, z_pixel=1):
#     lifetimes = px_lifetimes(image_4D)
#     st_px = lifetimes[lifetimes[:,-1]>st_limit-1]
#     stable_img = np.empty_like(image_4D)
#     for px in st_px:
#         z, y, x = px[1], px[2], px[3]
#         for i in np.arange(px[-1]-st_limit,-1,-1):
#             t = px[0] - i
#             stable_img[t,z,y,x] = 1
#     if save == True:
#         if save_path != '' and save_path[-1] != '/':
#             save_path += '/'
#         if save_file == '':
#             save_name = str(save_path+'stable_image.tif')
#         else:
#             save_name = str(save_path+'stable_'+save_file)
#         if '.tif' not in save_name:
#             save_name +='.tif'
#         datautils.save_image(save_name, stable_img, xy_pixel=xy_pixel, z_pixel=z_pixel)
#     return stable_img

# def _4D_to_PC(img):
#     limit = img.min() -1
#     px = np.argwhere(img>limit)
#     return px

# def _PC_to_4D(PC, img_shape):
#     img = np.zeros(img_shape)
#     for px in PC:
#         t,z,y,x = px[0],px[1],px[2],px[3]
#         img[t,z,y,x] = 1
#     return img

# def trans_px_levels(lifetimes, stab_limit=4, start_t=36, plot=True, save=True, save_path='', save_file=''):
#     """"
#     takes 4D_image, where the values are the lifetimes of these pixels in the 4D_image
#     """
#     lifetimes = lifetimes.astype('float32')

#     trans_Vol = []
#     trans_per = []
#     # trans_Vol = [0 for i in range(stab_limit-2)]
#     # trans_per = [0 for i in range(stab_limit-2)]
#     for stack in tqdm(lifetimes):
#         all_px = len(stack[stack>0])
#         transient = len(stack[(stack < stab_limit) & (stack > 0)])
#         # stack[stack>0] = 1
#         trans_Vol.append((all_px - transient))
#         trans_per.append(((all_px - transient)/all_px))
    
#     # definning timepoints
#     T_length = np.arange(len(lifetimes))
#     timepoints = [start_t+(i*0.25) for i in T_length] 

#     if save_path != '' and save_path[-1] != '/':
#         save_path += '/'

#     # convert results to dataframe
#     output_transient = pd.DataFrame({'timepoints':timepoints,'transient_px_N':trans_Vol, 'transient_px_per':trans_per})

#     if plot:
#         fig_name = save_path+save_file+'_transient.pdf'
#         #ploting the results
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,10))
#         ax1.plot(timepoints, output_transient.transient_px_N)
#         ax1.set_title('Number of Transient Pixels')
#         ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Number of Transient Pixels')
#         ax2.plot(timepoints, output_transient.transient_px_per)
#         ax2.set_title('Percentage of Transient Pixels')
#         ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Percentage of Transient Pixels')
#         plt.savefig(fig_name, bbox_inches='tight')
#     if save == True:
#         if save_file == '':
#             save_file = "transient_Pxs.csv"
#         csv_file = save_path+save_file
#         output_transient.to_csv(csv_file, sep=';')
    
#     return output_transient

# def trans_px_levels2(neuron, stable, start_t=36, plot=True, save=True, save_path='', save_file=''):
#     """
#     calculate number and percentage of transient pixels between stable neuron and all_neuron
#     """
#     neuron[neuron != 0] = 1
#     stable[stable != 0] = 1
#     neuron = neuron.astype('float32')
#     stable = stable.astype('float32')

#     transient = []
#     trans_per = []
#     # for t in tqdm(np.arange(stable.shape[0]), desc='calculating tansient'):
#     for t in np.arange(stable.shape[0]):
#         transient.append((neuron[t].sum()-stable[t].sum()))
#         trans_per.append((neuron[t].sum()-stable[t].sum())/neuron[t].sum())

#     # definning timepoints
#     T_length = np.arange(len(transient))
#     timepoints = [start_t+(i*0.25) for i in T_length] 

#     if save_path != '' and save_path[-1] != '/':
#         save_path += '/'

#     output_transient = pd.DataFrame({'timepoints':timepoints,'transient_px_N':transient, 'transient_px_%':trans_per})

#     if plot:
#         fig_name = save_path+save_file+'_transient.pdf'
#         #ploting the results
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,10))
#         ax1.plot(timepoints, transient)
#         ax1.set_title('Number of Transient Pixels')
#         ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Number of Transient Pixels')
#         ax2.plot(timepoints, trans_per)
#         ax2.set_title('Percentage of Transient Pixels')
#         ax1.set(xlabel="Hours After Puparium Formation [h]", ylabel='Percentage of Transient Pixels')
#         plt.savefig(fig_name, bbox_inches='tight')
    
#     if save == True:
#         if save_file == '':
#             save_file = "transient_Pxs.csv"
#         csv_file = save_path+save_file
#         output_transient.to_csv(csv_file, sep=';')
#         # if '.csv' not in csv_file:
#         #     csv_file +='.csv'
#         # with open(csv_file, 'w', newline='') as csvfile:
#         #     fieldnames = ['timepoint', 'No. of transient', 'percentage']
#         #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         #     writer.writeheader()
#         #     for t, val in enumerate(transient):
#         #         writer.writerow({'timepoint' : timepoints[t], 'No. of transient' : val, 'percentage': trans_per[t]})
#         # csvfile.close()
    
#     return output_transient


# def segment_3D(image, neu_no=10, max_neu_no=30, min_size=5000, xy_pixel=1, z_pixel=1, save=True, save_path='', save_file=''):
#     s = ndimage.generate_binary_structure(len(image.shape), len(image.shape))
#     labeled_array, num_labels = ndimage.label(image, structure=s)
#     labels = np.unique(labeled_array)
#     labels = labels[labels!=0]
#     neu_sizes = {}
#     for l in labels:
#         neu_sizes[l] = (labeled_array == l).sum()/(labeled_array == l).max()
#     avg_size = np.mean(list(neu_sizes.values()))
#     if min_size != 0:
#         for ind, l in enumerate(labels):
#             if neu_sizes[l] < min_size:
#                 labels[ind] = 0
#         labels = labels[labels!=0]
#     if neu_no != 0 and num_labels > neu_no:
#         for ind, l in enumerate(labels):
#             if neu_sizes[l] < avg_size:
#                 labels[ind] = 0
#         labels = labels[labels!=0]
#     if max_neu_no != 0 and len(labels) > max_neu_no:
#         sorted_sizes = sorted(neu_sizes.items(), key=operator.itemgetter(1), reverse=True)
#         sorted_sizes = sorted_sizes[0:max_neu_no]
#         labels = [[l][0][0] for l in sorted_sizes]
#     neurons = {}
#     for ind, l in enumerate(labels):
#         labels[ind] = ind+1
#         neuron = labeled_array.copy()
#         neuron[neuron != l] = 0
#         neuron[neuron == l] = ind+1
#         neuron = neuron.astype('uint16')
#         if neuron.sum() != 0 and neuron.sum() < np.prod(np.array(neuron.shape)):
#             neurons[ind+1] = neuron
#         else:
#             pass
#             # print('this segment was removed because its empty')
#         if save == True:
#             if save_file == '':
#                 save_name = str(save_path+str(ind)+'_neuron.tif')
#             else:
#                 save_name = str(save_path+'neuron_'+str(ind)+'_'+save_file)
#             if '.tif' not in save_name:
#                 save_name +='.tif'
#             datautils.save_image(save_name, neuron, xy_pixel=xy_pixel, z_pixel=z_pixel)             
#     return neurons

# def segment_4D(image_4D, neu_no=5, 
#                 max_neu_no=5, min_size=5000,
#                 xy_pixel=1, z_pixel=1,
#                 filter=True,
#                 save=True, save_path='', save_file=''):
#     # start_time = timer()
#     final_neurons = segment_3D(image_4D[0], neu_no=neu_no, min_size=min_size, 
#                                 max_neu_no=max_neu_no, save=False, save_path=save_path)
#     final_neurons = {l:[arr] for l, arr in final_neurons.items()}
#     for img_3D in tqdm(image_4D[1:], desc='matching segments', leave=False):
#         current_neurons = segment_3D(img_3D, neu_no=neu_no, min_size=min_size, 
#                                     max_neu_no=max_neu_no, save=False, save_path=save_path)
#         for l, neu_list in final_neurons.items():
#             neu = neu_list[-1]
#             diff = np.prod(np.array(neu.shape))
#             ID = 0
#             for t, neu_1 in current_neurons.items():
#                 cur_diff = abs((neu/neu.max() - neu_1/neu_1.max())).sum() #compare difference of fit
#                 if cur_diff != 0:
#                     if cur_diff < diff:
#                         diff = cur_diff
#                         ID = t
#             if ID != 0:
#                 try:
#                     final_neurons[l].append(current_neurons[ID])
#                 except:
#                     pass
#         current_neurons = None
#     for l, neu_4D in final_neurons.items():
#         neu_4D = np.array(neu_4D)
#         if save == True:
#             if save_file == '':
#                 save_name = str(save_path+str(l)+'_seg.tif')
#             else:
#                 # save_name = str(save_path+'seg_'+str(l)+'_'+save_file)
#                 save_name = str(save_path+save_file+'_mask')
#             if '.tif' not in save_name:
#                 save_name +='.tif'
#             datautils.save_image(save_name, neu_4D, xy_pixel=xy_pixel, z_pixel=z_pixel)
#     # if filter == True:
#     #     for l, mask_4D in final_neurons.items():
#     #         neuron = image_4D.copy()
#     #         neuron[mask_4D==0] = 0
#     #         if save == True:
#     #             if save_file == '':
#     #                 save_name = str(save_path+str(l)+'_neuron.tif')
#     #             else:
#     #                 save_name = str(save_path+'neuron_'+str(l)+'_'+save_file)
#     #             if '.tif' not in save_name:
#     #                 save_name +='.tif'
#     #             datautils.save_image(save_name, neuron, xy_pixel=xy_pixel, z_pixel=z_pixel)
#     #         final_neurons[l] = neuron
#     #         neuron = None
#     # print('segmentation runtime', timer()-start_time)
#     return final_neurons