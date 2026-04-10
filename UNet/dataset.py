"""
Dataloader for UNet
"""

from torch.utils.data import Dataset
import albumentations as A
import shutil
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import sys

class LandmarkDataset(Dataset):
    
    def __init__(self, annotated_img_dir, annotation_path, txt_file_path, cfg_dataset, 
                 is_train = False, is_val = False, is_test = False, perform_augmentation=False, force=False):        

        self.annotated_img_dir = annotated_img_dir
        self.annotation_path = annotation_path
        self.txt_file_path = txt_file_path
        self.cfg_dataset = cfg_dataset
        self.perform_augmentation = perform_augmentation
        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test
        self.downsampled_image_width = cfg_dataset.CACHED_IMAGE_SIZE[0]
        self.downsampled_image_height = cfg_dataset.CACHED_IMAGE_SIZE[1]
        self.force = force

        if self.is_train:
            self.mode = "train"
        elif self.is_val:
            self.mode = "val"
        elif self.is_test:
            self.mode = "test"

        # Resize (preserving aspect ratio)
        self.preprocessing = A.Compose([
            A.LongestMaxSize(max_size=max(self.downsampled_image_width, self.downsampled_image_height))
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
               
        self.augmentation = A.Compose([
            A.PadIfNeeded(min_height=512, min_width=512, position='bottom_left'),
            A.Affine(translate_percent={"x": (-0.08, 0.08), "y": (-0.04, 0.04)}, scale=(0.9, 1.1), rotate=(-15, 15), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=False, p=1.0),  
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.0, p=1.0),  
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),            
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.val_augmentation = A.Compose([
            A.PadIfNeeded(min_height=512, min_width=512, position='bottom_left')
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.test_augmentation = A.Compose([
            A.PadIfNeeded(min_height=512, min_width=512, position='bottom_left')
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))        


        self.db = self.cache()


    def make_cache_folder(self):

        def make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        self.cache_data_dir = "cache/" + self.cfg_dataset.TYPE + "/split/"        
        self.cache_path = self.cache_data_dir + self.mode + "_" + os.path.splitext(os.path.basename(self.txt_file_path))[0] + "/" # Will hold padded images
        self.cache_old_path = self.cache_data_dir + self.mode + "_" + os.path.splitext(os.path.basename(self.txt_file_path))[0]  + self.cfg_dataset.IMAGE_EXT + "/"  # Will hold original images

        if os.path.exists(self.cache_path):
            
            if self.force:
                shutil.rmtree(self.cache_path, ignore_errors=True)
                shutil.rmtree(self.cache_old_path, ignore_errors=True)
            else:
                return (self.cache_old_path, True)

        make_dir(self.cache_data_dir)        
        make_dir(self.cache_path)
        make_dir(self.cache_old_path)

        imgs = []
        with open(self.txt_file_path, 'r') as file:
            imgs = [line.strip() for line in file]

        # Form the full file paths to the images
        form_path = lambda img : self.annotated_img_dir + img + self.cfg_dataset.IMAGE_EXT
        train_full_paths = list(map(form_path, imgs))

        # Move the images into the cache path
        for file_path in tqdm(train_full_paths, desc=f"Creating {self.mode} Split"):
            shutil.copy(file_path, self.cache_old_path)

        return (self.cache_old_path, False)


    # Pre-process and save the image in cache
    # Return the augmented keypoints and shape
    def get_landmarks_img(self, img_name, image_path, exists):
                   
        # Read keypoints from CSV
        df = pd.read_csv(self.annotation_path, header=None)

        if self.cfg_dataset.TYPE == "hand":
            img_name = int(img_name)

        result_row = df[df.iloc[:, 0] == img_name]
        result = result_row.iloc[0, 1: 2 * self.cfg_dataset.KEY_POINTS + 1].to_numpy()                           
        kps = result.reshape(-1, 2)

        # Pre-process        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
        shape = image.shape
        augment = self.preprocessing(image=image, keypoints=kps)
        kps_resized = augment["keypoints"]
        image_resized = augment["image"]

        filename_wo_ext = os.path.splitext(os.path.basename(image_path))[0]
        path = os.path.join(self.cache_path, filename_wo_ext + ".png")

        if not exists:            
            im = Image.fromarray(image_resized)
            im.save(path, format='png')          

        return kps_resized, shape, path


    def cache(self):

        # exists is true if the cache directory already exists
        (data_path, exists) = self.make_cache_folder()
        data_path = Path(data_path)

        # Relative image paths
        imgs = sorted([f.as_posix() for f in data_path.iterdir() if f.is_file()])

        # Only image names (no extension)
        imgs_names = sorted([f.stem for f in data_path.iterdir() if f.is_file()])


        db = []

        for i, img in tqdm(enumerate(imgs_names), total=len(imgs_names), desc=f"Processing {self.mode} Data"):

            # Read the CSV and get the thing as a numpy array
            kps_np_array, shape, path = self.get_landmarks_img(img, imgs[i], exists)

            original_image_height, original_image_width = shape
            scale_factor = max(original_image_width / self.downsampled_image_width,
                                original_image_height / self.downsampled_image_height)
                        
            # Update db
            db.append({
                "cached_img_path" : path,
                "cached_landmarks" : kps_np_array,
                "cached_meta" : {
                    "scale_factor": scale_factor,
                    #"pixel_size" : self.cfg_dataset.PIXEL_SIZE,
                    "mre_per_pixel": 1,
                    "cached_name": img # Add name for unsup
                },                 
            })

        return db     



    def __len__(self):
        return len(self.db)  


    def __getitem__(self, idx):

        data = self.db[idx]
        img_path = data["cached_img_path"]
        landmarks = data["cached_landmarks"]
        meta = data["cached_meta"]


        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        if self.perform_augmentation or self.is_train:
            aug = self.augmentation
        elif self.is_val:
            aug = self.val_augmentation
        else:
            aug = self.test_augmentation      
        

        # Repeat augmentation if landmarks out of bounds
        bounds = np.array([self.downsampled_image_height, self.downsampled_image_width])
        max_retries = 10000
        success = False

        for i in range(max_retries):
            # Note elastic deformations will remove landmarks out of bounds
            augmented = aug(image=image, keypoints=landmarks)
            image_aug = augmented["image"]
            landmarks_aug = augmented["keypoints"]

            if (landmarks_aug >= 0).all() and (landmarks_aug < bounds).all() \
                    and len(landmarks_aug) == self.cfg_dataset.KEY_POINTS:
                success = True
                break
        
        if not success:
            print("ERROR with Augmentation in __getitem__")
            sys.exit()


        all_channels = np.zeros([self.cfg_dataset.KEY_POINTS, image_aug.shape[0], image_aug.shape[1]])   

        for i in range(self.cfg_dataset.KEY_POINTS):                                     
            try:
                x, y = landmarks_aug[i].astype(int)
                all_channels[i, y, x] = 1.0
            except IndexError:
                print("ERROR WITH CHANNELS")
                print(landmarks_aug.shape)
                print(image_aug.shape)
                sys.exit()


        if self.cfg_dataset.TYPE == "hand":
            meta["mre_per_pixel"] = self.cfg_dataset.DISTANCE / np.linalg.norm(landmarks_aug[0] - landmarks_aug[4])

        if self.cfg_dataset.TYPE == "head":
            meta["mre_per_pixel"] = self.cfg_dataset.PIXEL_SIZE[0]
        
        if self.cfg_dataset.TYPE == "pelvis":

            img_path = data["cached_img_path"]
            name = os.path.splitext(os.path.basename(img_path))[0]

            if name == "FT1096-V5_36_38_months":    # Only for this file is the pixel size different
                meta["mre_per_pixel"] = 0.168
            else:           
                meta["mre_per_pixel"] = self.cfg_dataset.PIXEL_SIZE[0]

        image_aug = np.expand_dims(image_aug, axis = 0)

        return image_aug, all_channels, meta