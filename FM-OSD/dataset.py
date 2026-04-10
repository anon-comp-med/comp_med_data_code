"""
Dataset for training and testing
"""

import torch.utils.data as data
from albumentations import Compose, KeypointParams, Affine
import cv2
import pandas as pd
import os
import torch.utils.data as data
import PIL.Image as Image
import os
import torch
import numpy as np
from torchvision.transforms import transforms
import pandas as pd
import random


random_seed = 2026
random.seed(random_seed)

# Creates an augmented dataset
class DataAugmentGen(data.Dataset):

    def __init__(self, img_name, img_dir, anno_pth, cfg_dataset):

        self.img_name = img_name 
        self.img_dir = img_dir
        self.anno_pth = anno_pth

        self.cfg_dataset = cfg_dataset

        self.aug = Compose([
            Affine(
                translate_percent=0.02,
                scale=(0.99, 1.01),
                rotate=(-5, 5),
                p=1.0,
            )
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=False))

        # Read in the CSV
        df = pd.read_csv(self.anno_pth, header=None)
        if self.cfg_dataset.TYPE == "hand":
            self.img_name = int(self.img_name)
        
        # Read in the landmarks
        result_row = df[df.iloc[:, 0] == self.img_name]          
        result = result_row.iloc[0, 1: 2 * self.cfg_dataset.KEY_POINTS + 1].to_numpy()                           
        self.kps = result.reshape(-1, 2)


    def __len__(self):
        return 1
    
    def kps_in_bounds(self, kps, height, width):
        for x, y in kps:
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        return True
    
    # Returns the augmented image and GT locations
    def __getitem__(self, index):
        
        pth_img = os.path.join(self.img_dir, str(self.img_name) + self.cfg_dataset.IMAGE_EXT)
        
        image = cv2.imread(pth_img) 

        augment = self.aug(image=image, keypoints = self.kps)

        h, w = augment["image"].shape[:2]
        
        valid = True
        if not self.kps_in_bounds(augment["keypoints"], h, w):
            return image, self.kps, False

        return augment["image"], augment["keypoints"], valid


# Loads downsampled image for global stage training
class TrainDatasetNew(data.Dataset):
    # image_dir is the path to the augmented template images
    def __init__(self, image_dir, csv_path, load_size, cfg_dataset, img_name_txt = None):

        self.img_name_txt = img_name_txt
        if img_name_txt is None:
            self.img_names = os.listdir(image_dir)
            self.img_full_paths = [os.path.join(image_dir, name) for name in self.img_names]

            # Remove file extension
            self.img_names = [name.split(".")[0] for name in self.img_names]
        else:
            with open(img_name_txt, 'r') as file:
                self.img_names = [line.strip() for line in file]
            
            self.img_full_paths = [os.path.join(image_dir, name + cfg_dataset.IMAGE_EXT) for name in self.img_names]
                    
        self.load_size = load_size
        self.cfg_dataset = cfg_dataset

        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        self.prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Read in GT from CSV, stored in (x, y) format
        self.df = pd.read_csv(csv_path, header=None)

        self.resize = transforms.Resize((self.load_size, self.load_size), interpolation=transforms.InterpolationMode.LANCZOS)


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, index):
        
        # Read in image (and store original size)
        img_path = self.img_full_paths[index]
        img = Image.open(img_path).convert('RGB') 
        width, height = img.size        

        # Get landmarks
        img_name = self.img_names[index]
        if self.cfg_dataset.TYPE == "hand" and self.img_name_txt is not None:
            img_name = int(img_name)
        result_row = self.df[self.df.iloc[:, 0] == img_name]
        result = result_row.iloc[0, 1: 2 * self.cfg_dataset.KEY_POINTS + 1].to_numpy(dtype=np.float32)                           
        kps = result.reshape(-1, 2)
        kps = kps[:, ::-1].copy()   
        kps = torch.Tensor(kps).long()  

        # Downsample and normalize
        image = self.resize(img)
        image = self.prep(image)

        # Downsample landmarks also
        image_y = image.shape[-2]
        image_x = image.shape[-1]
        kps_down = torch.zeros_like(kps)
        kps_down[:,0] = kps[:,0] / height * image_y
        kps_down[:,1] = kps[:,1] / width * image_x
        kps_down = torch.floor(kps_down).long()

        # kps in (y, x) format
        return image, kps, kps_down, img_path, torch.tensor([height, width])


class InferDataset(data.Dataset):
    # img_name_txt is path to text file with file names to use
    def __init__(self, image_dir, csv_path, cfg_dataset, img_name_txt, num_templates=None):

        self.cfg_dataset = cfg_dataset
        
        # Read in txt file of names
        with open(img_name_txt, 'r') as file:
            self.img_names = [line.strip() for line in file]

        # Randomly choose a subset of templates
        if num_templates is not None:
            self.img_names = random.sample(self.img_names, num_templates)

        # Full image paths
        self.img_full_paths = [os.path.join(image_dir, name + cfg_dataset.IMAGE_EXT) for name in self.img_names]

        # Load csv
        self.df = pd.read_csv(csv_path, header=None, dtype={0: str})
        self._anno_map = {str(row[0]): row[1: 2 * self.cfg_dataset.KEY_POINTS + 1].to_numpy(dtype=np.int64) for _, row in self.df.iterrows()}      


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        img_path = self.img_full_paths[index]
        img_name = self.img_names[index]
        
        # Read in landmarks (make sure correct order for img names and img paths)
        kps = self._anno_map[img_name].reshape(-1, 2)
        kps = kps[:, ::-1].copy()   
        kps = torch.from_numpy(kps) 

        # Read in image and get the size
        with Image.open(img_path) as img:
            width, height = img.size

        # Return land, full_path, and original image size
        return kps, img_path, torch.tensor([height, width])

