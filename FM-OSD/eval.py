"""
Generates summary statistics
"""

import numpy as np
import csv
import os
import json

import torch

import pandas as pd
from pathlib import Path
import sys

from tqdm import tqdm

def radial(pt1, pt2, factor=[1, 1]):
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5

class Evaluater(object):
    def __init__(self, pred, gt, cfg_dataset, eval_radius=[], eres = [], meta = [], csv_path = None):
        self.pred = np.array(pred)
        self.gt = np.array(gt)
        self.RE_list = list()
        self.num_landmark = self.gt.shape[1]

        self.recall_radius = eval_radius  # 2mm etc
        self.recall_rate = list()

        self.total_list = dict()

        self.cfg_dataset = cfg_dataset

        self.eres = eres

        self.pixel_to_mm = 1

        self.meta = meta

        self.csv_path = csv_path


    def calculate(self): 
        diff = self.pred - self.gt # (B (or T),N,2)
        diff = np.power(diff, 2).sum(axis = -1) # (B, N)
        diff = np.sqrt(diff)  

        self.pixel_to_mm = 1
        if self.cfg_dataset.TYPE == "hand":
            # Use distance for spacing
            self.pixel_to_mm = self.cfg_dataset.DISTANCE / np.linalg.norm(self.gt[:, 0, :] - self.gt[:, 4, :], axis=-1)
            self.pixel_to_mm = self.pixel_to_mm[:, None]
        elif self.cfg_dataset.TYPE == "head" or self.cfg_dataset.TYPE == "pelvis":
            self.pixel_to_mm = self.cfg_dataset.PIXEL_SIZE[0]
        else:
            print("Error, no dataset type")
            sys.exit()   

        diff = diff * self.pixel_to_mm
        self.RE_list = diff

        print(diff.shape)
        
        return None
    

    
    def get_dp(self, logger, radial_errors):
        radius = 2.0

        def get_vals(mask):
            if not np.any(mask):
                return np.zeros((1,radial_errors.shape[1]), dtype=float)
            return radial_errors[mask]
        
        def group_rate(mask):
            K = radial_errors.shape[1]
            if not np.any(mask):
                return np.zeros(K, dtype=float)

            return (radial_errors[mask] < radius).sum(axis=0) * 100 / radial_errors[mask].shape[0]

        radius = 2.0

        male_mask = np.array([m["gender"] == "M" for m in self.meta])
        female_mask = np.array([m["gender"] == "F" for m in self.meta])

        old_male_mask = np.array([m["gender"] == "M" and m["age"] == "Old" for m in self.meta])
        old_female_mask = np.array([m["gender"] == "F" and m["age"] == "Old" for m in self.meta])
        young_male_mask = np.array([m["gender"] == "M" and m["age"] == "Young" for m in self.meta])
        young_female_mask = np.array([m["gender"] == "F" and m["age"] == "Young" for m in self.meta])

        male = group_rate(male_mask)
        female = group_rate(female_mask)
        old_male = group_rate(old_male_mask)
        old_female = group_rate(old_female_mask)
        young_male = group_rate(young_male_mask)
        young_female = group_rate(young_female_mask)

        logger.info("--------------------------------------------------------------------------")

        # Make a CSV with the MREs, then gender, then age
        if self.csv_path != None:
            with open(self.csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                           
                for i in tqdm(range(radial_errors.shape[0])):  
                    radial_error_row = radial_errors[i, :].tolist()  
                    
                    gender = self.meta[i]["gender"]
                    age = self.meta[i]["age"]
                    
                    # Write the combined row to the CSV
                    writer.writerow(radial_error_row + [gender, age])
            logger.info("Finished Writing to CSV")

        logger.info("--------------------------------------------------------------------------")

        # Per KP DP
        dp_gender = np.abs(male - female)
        logger.info("Gender DP")
        logger.info(' '.join(map(str, dp_gender)))

        arrays = [old_male, old_female, young_male, young_female]
        stacked = np.stack(arrays)        
        max_diff = stacked.max(axis=0) - stacked.min(axis=0)
        logger.info("Age + Gender DP")
        logger.info(' '.join(map(str, max_diff)))    

        # Wrist and Fingers DP
        male_wrist     = get_vals(male_mask)[:,:18].reshape(-1)
        male_fingers   = get_vals(male_mask)[:,18:].reshape(-1)
        female_wrist   = get_vals(female_mask)[:,:18].reshape(-1)
        female_fingers = get_vals(female_mask)[:,18:].reshape(-1)

        male_wrist     = (male_wrist     < radius).sum() * 100 / male_wrist.size
        male_fingers   = (male_fingers   < radius).sum() * 100 / male_fingers.size
        female_wrist   = (female_wrist   < radius).sum() * 100 / female_wrist.size
        female_fingers = (female_fingers < radius).sum() * 100 / female_fingers.size

        logger.info("Gender DP Wrist + Finger")
        logger.info(f"{np.abs(male_wrist - female_wrist)} {np.abs(male_fingers - female_fingers)}")


        # Wrist and Fingers DP Age + Gender
        old_male_wrist     = get_vals(old_male_mask)[:,:18].reshape(-1)
        old_male_fingers   = get_vals(old_male_mask)[:,18:].reshape(-1)
        old_female_wrist   = get_vals(old_female_mask)[:,:18].reshape(-1)
        old_female_fingers = get_vals(old_female_mask)[:,18:].reshape(-1)
        young_male_wrist     = get_vals(young_male_mask)[:,:18].reshape(-1)
        young_male_fingers   = get_vals(young_male_mask)[:,18:].reshape(-1)
        young_female_wrist   = get_vals(young_female_mask)[:,:18].reshape(-1)
        young_female_fingers = get_vals(young_female_mask)[:,18:].reshape(-1)

        old_male_wrist     = (old_male_wrist     < radius).sum() * 100 / old_male_wrist.size
        old_male_fingers   = (old_male_fingers   < radius).sum() * 100 / old_male_fingers.size
        old_female_wrist   = (old_female_wrist   < radius).sum() * 100 / old_female_wrist.size
        old_female_fingers = (old_female_fingers < radius).sum() * 100 / old_female_fingers.size

        young_male_wrist     = (young_male_wrist     < radius).sum() * 100 / young_male_wrist.size
        young_male_fingers   = (young_male_fingers   < radius).sum() * 100 / young_male_fingers.size
        young_female_wrist   = (young_female_wrist   < radius).sum() * 100 / young_female_wrist.size
        young_female_fingers = (young_female_fingers < radius).sum() * 100 / young_female_fingers.size


        arrays_wrist = [old_male_wrist, old_female_wrist, young_male_wrist, young_female_wrist]
        arrays_finger = [old_male_fingers, old_female_fingers, young_male_fingers, young_female_fingers]

        stacked_wrist = np.stack(arrays_wrist)        
        stacked_finger = np.stack(arrays_finger) 
        
        max_diff_wrist = stacked_wrist.max(axis=0) - stacked_wrist.min(axis=0)
        max_diff_finger = stacked_finger.max(axis=0) - stacked_finger.min(axis=0)


        logger.info("Age + Gender DP Wrist + Finger")
        logger.info(f"{max_diff_wrist} {max_diff_finger}") 

    
    def cal_metrics(self, logger):
        # calculate MRE SDR
        all_re = np.array(self.RE_list) 

        if self.meta != []:
            self.get_dp(logger, all_re)

        Mean_RE_land = all_re.mean(axis=0) # Calculate MRE per landmark (across images)

        self.mre = Mean_RE_land.mean() # Calculates the MRE over all images and over all landmarks
        self.std = all_re.std(axis=(0, 1))   # Overall std

        logger.info("----------------------- INFERENCE -----------------------")
        logger.info("Payer: MRE: {:.3f}+-{:.3f} mm".format(self.mre, self.std))      
        logger.info(f"SDR: {self.recall_radius}")

        self.sdr = []
        for radius in self.recall_radius:
            total = all_re.size
            shot = (all_re < radius).sum()
            self.sdr.append(shot * 100 / total)

        logger.info(
            "SDR: " + ", ".join(f"{x:.3f}" for x in self.sdr)
        )

        if self.eres != []:
            self.eres = np.array(self.eres) * self.pixel_to_mm
            logger.info("Payer: ERE: {:.3f}+-{:.3f} mm".format(self.eres.mean(), self.eres.std())) 
            logger.info(f"{self.mre:.3f} {self.std:.3f} {self.sdr[0]:.3f} {self.sdr[3]:.3f} {self.sdr[6]:.3f} {self.eres.mean():.3f} {self.eres.std():.3f}")

        else:
            logger.info(f"{self.mre:.3f} {self.std:.3f} {self.sdr[0]:.3f} {self.sdr[3]:.3f} {self.sdr[6]:.3f}")


