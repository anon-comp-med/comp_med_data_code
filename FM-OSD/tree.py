"""
Random forest classifier for predicting metadata
"""

import argparse
import numpy as np
import random
import save_utils

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def parse_args():
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')

    parser.add_argument("--save_name", type=str, default="", required=True, help="For preprending to the start of file names") # For pre-pending to file names
    parser.add_argument('--save_dir', type=str, default = None, required=True, help="For saving models")  # Changed 

    parser.add_argument('--csv', type=str, default = None, required=True, help="Path to CSV with features")

    parser.add_argument('--predict', type=str, default = None, required=True, help="What to predict: age/gender")

    parser.add_argument('--filter', type=str, default = None, required=True, help="What to filter: M, F, Old, Young")
    
    parser.add_argument('--exp', type=str, required=False, default="tree")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()    

    # Ensure save directory exists
    paths, logger = save_utils.prepare_output_and_logger(args.save_dir, args.exp, args.save_name, True)

    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # random seed
    random_seed = 2026
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Read in CSV
    # Gender is column 37, age is column 38
    data  = pd.read_csv(args.csv, header=None)

    # Filter by args.filter
    if args.filter in ["M", "F"]:
        data = data[data.iloc[:, 37] == args.filter]
    else:
        data = data[data.iloc[:, 38] == args.filter]

   
    # Drop one of the final columns depending on args.predict
    if args.predict == "age": 
        # Drop column 37
        data = data.drop(data.columns[37], axis=1) 
    else:
        # Drop 38
        data = data.drop(data.columns[38], axis=1)  
        
    X = data.drop(data.columns[-1], axis=1)  
    y = data.iloc[:, -1]

    dt_model = DecisionTreeClassifier(random_state=2026)

    cv_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')

    logger.info("Cross-validation scores for each fold:", cv_scores)

    logger.info("Average 5-fold cross-validation accuracy:", cv_scores.mean() * 100)


    
