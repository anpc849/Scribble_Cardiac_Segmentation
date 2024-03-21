import os
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from torchvision import transforms, utils
import itertools
from scipy import ndimage
from PIL import Image
from albumentations import *
from albumentations.pytorch import *
from scipy.ndimage.interpolation import zoom

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform_both=None,
        transform_image=None,
        output_size=(256, 256),
    ):
        self._base_dir = base_dir  #default: data/ACDC/  0: background   1: RV (right ventricle)   2: Myo (myocardium)  3: LV (left ventricle)   4: UA (unannotated pixels)
        self.sample_list = []
        self.split = split
        self.transform_both = transform_both
        self.transform_image = transform_image
        self.output_size = output_size

        if self.split == "train":                   
            with open(self._base_dir + "/train_patients.list", "r") as f:
                for line in f:
                    # Strip parentheses and split by commas
                    indices = line.strip()[1:-1].split(',')
                    # Convert strings to integers and store as a tuple
                    indices = tuple(map(int, indices))
                    self.sample_list.append(indices)

        elif self.split == "val":
            with open(self._base_dir + "/val_patients.list", "r") as f:
                for line in f:
                    # Strip parentheses and split by commas
                    indices = line.strip()[1:-1].split(',')
                    # Convert strings to integers and store as a tuple
                    indices = tuple(map(int, indices))
                    self.sample_list.append(indices)

        elif self.split == "test":
            with open(self._base_dir + "/test_patients.list", "r") as f:
                for line in f:
                    # Strip parentheses and split by commas
                    indices = line.strip()[1:-1].split(',')
                    # Convert strings to integers and store as a tuple
                    indices = tuple(map(int, indices))
                    self.sample_list.append(indices)

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx] # case is a tuple (index_patient, index_frame, index_slice)
        h5f = h5py.File(self._base_dir + "ACDC_training_slices/patient{:03d}_frame{:02d}_slice_{}.h5".format(case[0], case[1], case[2]), "r")
        image = h5f["image"][:]
        gt = h5f["label"][:]
        label = h5f['scribble'][:]
        
        image = image.astype(np.float32)
        gt = gt.astype(np.uint8)
        label = label.astype(np.uint8)

        ## normalize (the datasset haved normalized)
        #image =  image / np.max(image)

        ## resize
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        ## augment dataset
        if self.transform_both is not None:
            augmented_sample = self.transform_both(image=image, mask=label)
            image = augmented_sample['image']
            label = augmented_sample['mask']
        if self.transform_image is not None:
            augmented_sample = self.transform_image(image=image)
            image = augmented_sample['image']

        # clip to 0 to 1
        image = np.clip(image, 0, 1)

        #zero-centre
        image = (image-0.5)/0.5

        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        label = torch.from_numpy(label)
        gt = torch.from_numpy(gt)

        sample = {'image': image, 'label': label, 'gt': gt, 'idx': idx}

        return sample