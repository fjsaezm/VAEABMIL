from tqdm import tqdm
import tifffile
import zarr
import h5py
import cv2
import os
import numpy as np
import torch
import random

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_att_hist(ax, f_pred, y_true, T_true, bag_idx, legend=True):
    """
    Args:
        s_pred: (num_inst,) attention score of the instances
        y_true: (num_inst,) label of the instances
        T_true: (num_bags,) label of the bags
        bag_idx: (num_inst,) maps each instance to its bag    
    """

    print('Plotting attention values distribution')

    pos_bags_idx = np.where(T_true == 1)[0] # positive bags

    idx_keep = np.isin(bag_idx, pos_bags_idx)

    y_true = y_true[idx_keep]
    f_pred = f_pred[idx_keep]
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    pos_inst = f_pred[pos_idx]
    neg_inst = f_pred[neg_idx]
    
    counts_pos, bins_pos = np.histogram(pos_inst, bins=20, density=False)
    counts_pos = counts_pos / counts_pos.sum()
    ax.hist(bins_pos[:-1], bins_pos, weights=counts_pos, label='Positive instances', edgecolor='black', alpha=0.7, color='tab:red')
    
    counts_neg, bins_neg = np.histogram(neg_inst, bins=20, density=False)
    counts_neg = counts_neg / counts_neg.sum()
    ax.hist(bins_neg[:-1], bins_neg, weights=counts_neg, label='Negative instances', edgecolor='black', alpha=0.7, color='tab:green')
    
    ax.set_ylim(0, 1)

    if legend:
        ax.legend(frameon=False)

    return ax

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def read_scan_slices(data_dir, inst_paths, size=512, resize_size=10):

    slices_list = []
    pbar = tqdm(total=len(inst_paths))
    for inst_path in inst_paths:
        inst_name = inst_path.split('/')[-1]
        pbar.update(1)
        img = np.load(data_dir+inst_name)
        # img = cv2.imread(f'{DATA_DIR}/images/' + patch_name + '.png')
        if resize_size != size:
            img = cv2.resize(img, (resize_size, resize_size))
        slices_list.append(img)
    return slices_list

def read_wsi_patches(data_dir, coords_path, bag_name, size=512, resize_size=10, ext='.tif'):
    wsi_path = os.path.join(data_dir, bag_name + ext)
    inst_coords = np.array(h5py.File(os.path.join(coords_path, bag_name + '.h5'), 'r')['coords'])
    bag_len = len(inst_coords)
    patches_list = []
    row_list = []
    column_list = []
    pbar = tqdm(range(bag_len), total=bag_len)
    for i in pbar:
        coord = inst_coords[i]
        x, y = coord
        
        store = tifffile.imread(wsi_path, aszarr=True)
        z = zarr.open(store, mode='r')
        wsi = z[0]
        patch = wsi[y:y+size, x:x+size]
        patch = cv2.resize(patch, (resize_size, resize_size))
        patches_list.append(patch)

        row = int(y / size)
        column = int(x / size)
        row_list.append(row)
        column_list.append(column)

    row_array = np.array(row_list)
    column_array = np.array(column_list)

    # normalize rows

    row_array = row_array - row_array.min()
    column_array = column_array - column_array.min()

    return patches_list, row_array, column_array

def pad_canvas(canvas_wsi, row_array, column_array, num_pad=5, resize_size=10):
    num_pad = 5

    # pad the image
    canvas_wsi = np.pad(canvas_wsi, ((num_pad*resize_size, num_pad*resize_size), (num_pad*resize_size, num_pad*resize_size), (0, 0)), mode='constant', constant_values=255)

    # update row and column arrays
    row_array = row_array + num_pad
    column_array = column_array + num_pad

    return canvas_wsi, row_array, column_array