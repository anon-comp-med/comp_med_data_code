"""
Generate augmented data to use for training 
"""

import argparse
import torch
import numpy as np

import os
from tqdm import tqdm

import save_utils
import dataset

import csv
import cv2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cfg',
                help='The path to the configuration file for the experiment',
                required=False,
                type=str)

    parser.add_argument('--train',
                help='The path to a text file containing the image names to use for aug',
                type=str,
                required=False,
                default='')
    
    parser.add_argument('--images',
                    help='The path to a directory containing all the images (in jpg)',
                    type=str,
                    required=False,
                    default='')

    parser.add_argument('--annotations',
                        help='The path to the csv file containing annotations',
                        type=str,
                        required=False,
                        default='')
    
    parser.add_argument('--save_dir',
                        help='Path to the folder to store augmented images',
                        type=str,
                        required=False,
                        default='')
    
    parser.add_argument('--save_anno',
                    help='Path to the file to store augmented ground truths',
                    type=str,
                    required=False,
                    default='')

    parser.add_argument('--max_iter', 
                        help="Number of augmented images per training image",
                        default=500, 
                        type=int)

    args = parser.parse_args()

    cfg = save_utils.prepare_config(args.cfg)

    os.makedirs(args.save_dir, exist_ok=True)
    save_anno_dir = os.path.dirname(os.path.abspath(args.save_anno)) or "."
    os.makedirs(save_anno_dir, exist_ok=True)

    # Get all image names
    imgs = []
    with open(args.train, 'r') as file:
        imgs = [line.strip() for line in file]

    img_to_land = {}
    
    for img_name in imgs:
        # Loader for image
        one_shot_loader = dataset.DataAugmentGen(img_name, args.images, args.annotations, cfg.DATASET)

        # Create max_iter training examples
        for iter_num in tqdm(range(args.max_iter), desc="Processing"):

            # Returns the augmented image (full res), and landmark locations
            valid = False
            num_iter = 0
            while not valid:
                image_np, landmarks, valid = one_shot_loader.__getitem__(0)
                if not valid:
                    num_iter += 1
                if num_iter >= 50:
                    valid = True # Will just use un-augmented image
                    print("Using Unaug")

            # Save image
            new_img_name = f"{img_name}_{str(iter_num)}"
            image_path = os.path.join(args.save_dir, f"{new_img_name}.png")
            success = cv2.imwrite(image_path, image_np)

            if not success:
                print("Error saving augmented image")
            
            flat_kps = np.asarray(landmarks, dtype=int).reshape(-1).tolist()
            img_to_land[new_img_name] = flat_kps


    # Save CSV with augmented kps
    with open(args.save_anno, "w", newline="") as f:
        writer = csv.writer(f)

        for key, values in img_to_land.items():
            writer.writerow([str(key)] + values)



