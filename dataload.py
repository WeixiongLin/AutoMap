
from __future__ import print_function
import cv2
import os
import torch
import numpy as np
import pandas as pd
import argparse
import math
import warnings
import random
from skimage import color
import Augmentor
from typing import Optional, List, Callable, Any
from glob import glob
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from skimage.io import imread, imsave
import sklearn
import skimage
from sklearn.model_selection import train_test_split


# 把数据集划分为 TrainSet 和 ValidSet
def dataset_split(train_csv_dir):
    train_data_list =  pd.read_csv(train_csv_dir).sample(frac=1)
    train_size = int(0.9 * len(train_data_list))
    train_data_info = train_data_list[0:train_size]
    valid_data_info = train_data_list[train_size:]
    print('train size :%d, valid size: %d' %(train_size,len(train_data_list)-train_size))
    return train_data_info, valid_data_info

def dataset_split_half(train_csv_dir):
    train_data_list =  pd.read_csv(train_csv_dir).sample(frac=1)
    train_size = int(0.45 * len(train_data_list))
    train_data_info = train_data_list[0:train_size]
    valid_data_info = train_data_list[train_size:int(0.5 * len(train_data_list))]
    print('train size :%d, valid size: %d' %(len(train_data_info),len(valid_data_info)))
    return train_data_info, valid_data_info


class Brats2018_test(Dataset):
    def __init__(self, test_dataset_dir):
        test_data_list =  pd.read_csv(test_dataset_dir)
        self.datafile_path = np.asarray(test_data_list.iloc[:,0])
        datafile_img=[]
        for i in range(len(self.datafile_path)):
            datafile_img.append((os.path.join(self.datafile_path[i], os.path.basename(self.datafile_path[i])+'_flair.nii.gz'),
                                os.path.join(self.datafile_path[i], os.path.basename(self.datafile_path[i])+'_seg.nii.gz')))
        self.datafile_img = datafile_img
        self.data_len = len(self.datafile_img)
    
    def __getitem__(self,index):
        img_path, mask_path = self.datafile_img[index]
        itk_img = sitk.ReadImage(img_path) 
        img_array = sitk.GetArrayFromImage(itk_img)
        img_array = img_array.transpose(2, 1, 0)
        itk_mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(itk_mask)
        mask_array = mask_array.transpose(2, 1, 0)
        img_array=self.normlize(img_array) 
        mask_array [mask_array > 0]=1
        
        return img_array, mask_array #(torch.tensor(volume.copy(), dtype=torch.float),
               # torch.tensor(seg_volume.copy(), dtype=torch.float))
    
    def normlize(self, x):
        #normalized by clipping them to the 0.5 and 99.5 percentiles
        #(x-mean)/std
        x_min =x.min()+(x.max()-x.min())*0.05
        x_max =x.max()-(x.max()-x.min())*0.05
        x[x>x_max]=x_max
        x[x<x_min]=x_min
        x=(x - x_min) / (x_max- x_min)
        x = (x-x.mean())/x.std()#sklearn.preprocessing.scale(x)
        return x

    def __len__(self):
        return len(self.datafile_img)


class Brats2018_train(Dataset):
    def __init__(self, train_csv,crop_size, train=True):
        #self.data_info = pd.read_csv(train_csv)
        self.datafile_path = np.asarray(train_csv.iloc[:,0])
        datafile_img = []
        #datafile_mask = []
        if train:
            for i in range(len(self.datafile_path)):
                datafile_img.append((os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_flair.nii.gz'),
                                    os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_seg.nii.gz')))
                """
                datafile_img.append((os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_t1.nii.gz'),
                                    os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_seg.nii.gz')))
                datafile_img.append((os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_t1ce.nii.gz'),
                                    os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_seg.nii.gz')))
                datafile_img.append((os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_t2.nii.gz'),
                                    os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_seg.nii.gz')))
                """
        else:
            for i in range(len(self.datafile_path)):
                datafile_img.append((os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_flair.nii.gz'),
                                    os.path.join(self.datafile_path[i],os.path.basename(self.datafile_path[i])+'_seg.nii.gz')))
        self.datafile_img = datafile_img
        self.data_len = len(self.datafile_img)
        self.crop_size = crop_size
        self.train = train
    
    def __getitem__(self,index):
        img_path, mask_path = self.datafile_img[index]
        itk_img = sitk.ReadImage(img_path) 
        img_array = sitk.GetArrayFromImage(itk_img)
        img_array = img_array.transpose(2, 1, 0)
        itk_mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(itk_mask)
        mask_array = mask_array.transpose(2, 1, 0)
        mask_array [mask_array > 0]=1
        img_array=self.normlize(img_array) 
        volume=img_array[np.newaxis,:,:,:]
        seg_volume=mask_array[np.newaxis,:,:,:]
       
        volume, seg_volume = self.aug_sample(volume, seg_volume)
       
        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float))
    
    def normlize01(self, x):
        #normalized by clipping them to the 0.5 and 99.5 percentiles
        #normalize to 01
        x_min =x.min()+(x.max()-x.min())*0.05
        x_max =x.max()-(x.max()-x.min())*0.05
        x[x>x_max]=x_max
        x[x<x_min]=x_min
        return (x - x_min) / (x_max- x_min)
    
    def normlize(self, x):
        #normalized by clipping them to the 0.5 and 99.5 percentiles
        #(x-mean)/std
        x_min =x.min()+(x.max()-x.min())*0.05
        x_max =x.max()-(x.max()-x.min())*0.05
        x[x>x_max]=x_max
        x[x<x_min]=x_min
        x=(x - x_min) / (x_max- x_min)
        x= (x-x.mean())/x.std()
        return x


    def aug_sample(self,x,y):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]
        """
        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
            if random.random() < 0.5:
                x = skimage.util.random_noise(x,mode='gaussian',seed=None,clip=True)
            
        else:
            x, y = self.center_crop(x, y)

        return x, y
    
    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        return crop_volume, crop_seg

    
    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    
    def __len__(self):
        return len(self.datafile_img)


def get_test_patches(x,crop_size):
    height, width, depth = x.shape
    h_crop = math.ceil(height/crop_size[0])
    w_crop = math.ceil(width/crop_size[1])
    d_crop = math.ceil(depth/crop_size[2])
    patch_number = h_crop*w_crop*d_crop
    patches = np.zeros((patch_number,crop_size[0],crop_size[1],crop_size[2]))
    patch_index = 0
    for i in range(h_crop):
        h_s = i*crop_size[0]
        h_f = (i+1)*crop_size[0]
        if  h_f>height:
            h_s = height-crop_size[0]
            h_f = height
        for j in range(w_crop):
            w_s = j*crop_size[1]
            w_f = (j+1)*crop_size[1]
            if w_f > width:
                w_s = width-crop_size[1]
                w_f = width
            for d in range(d_crop):
                d_s = d*crop_size[2]
                d_f = (d+1)*crop_size[2]
                if d_f > depth:
                    d_s = depth-crop_size[2]
                    d_f = depth
                patches[patch_index,0:h_f-h_s,0:w_f-w_s,0:d_f-d_s] = x[h_s:h_f,w_s:w_f,d_s:d_f]
                patch_index+=1
    return patches


def re_test_patches(x,patches,crop_size):
    re_x = np.zeros(np.shape(x))
    patch_number = len(patches)
    height, width, depth = x.shape
    h_crop = math.ceil(height/crop_size[0])
    w_crop = math.ceil(width/crop_size[1])
    d_crop = math.ceil(depth/crop_size[2])
    patch_index = 0
    patches = np.array(patches)
    #print(patches.shape)
    for i in range(h_crop):
        h_s = i*crop_size[0]
        h_f = (i+1)*crop_size[0]
        if  h_f>height:
            h_s = height-crop_size[0]
            h_f = height
        for j in range(w_crop):
            w_s = j*crop_size[1]
            w_f = (j+1)*crop_size[1]
            if w_f > width:
                w_s = width-crop_size[1]
                w_f = width
            for d in range(d_crop):
                d_s = d*crop_size[2]
                d_f = (d+1)*crop_size[2]
                if d_f > depth:
                    d_s = depth-crop_size[2]
                    d_f = depth
                re_x[h_s:h_f,w_s:w_f,d_s:d_f] = patches[patch_index,0:h_f-h_s,0:w_f-w_s,0:d_f-d_s]
                patch_index+=1
    return re_x
