"""
Helper functions for logging
"""

from pathlib import Path
import os
import time
import logging
from config import get_cfg_defaults
import argparse


def prepare_output_and_logger(save_dir, stage, save_name, make_log = True):

    save_dir = Path(save_dir)

    # Root directory
    save_dir.mkdir(exist_ok=True, parents=True)

    # Stage-specific directory
    stage_dir = save_dir / stage
    stage_dir.mkdir(exist_ok=True, parents=True)

    # Model filename
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    model_name = f"{save_name}_{time_str}_{stage}_model.pth"
    proto_name = f"{save_name}_{time_str}_{stage}_proto.pth"
    ora_name = f"{save_name}_{time_str}_{stage}_ora.pth"
    csv_name = f"{save_name}_{time_str}_{stage}_mres.csv"

    model_path = stage_dir / model_name
    proto_path = stage_dir / proto_name
    ora_path = stage_dir / ora_name
    csv_path = stage_dir / csv_name

    # Create logger
    log_file = f"{save_name}_{time_str}_{stage}.log"    
    log_path = stage_dir / log_file

    logger = None
    if make_log:
        # setup the logger
        logging.basicConfig(filename=log_path,
                            format='%(message)s')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()   # For sending log messages to the terminal
        # Add console logger to the root logger
        # Log messages sent to both the console and log file
        logging.getLogger('').addHandler(console) 

    paths = {
        "model_path" : model_path,
        "proto_path" : proto_path, 
        "ora_path" : ora_path,
        "csv_path" : csv_path
    }      

    return paths, logger



def prepare_config(cfg_path):
    cfg = get_cfg_defaults()    # Get config pattern
    cfg.merge_from_file(cfg_path)   # Add config details from yaml file
    cfg.freeze()   
    
    return cfg


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')