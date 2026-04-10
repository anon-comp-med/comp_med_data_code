"""
Logger helper functions
"""

import os
import time
import logging

from config import get_cfg_defaults


def prepare_config(cfg_path):
    cfg = get_cfg_defaults()    # Get config pattern
    cfg.merge_from_file(cfg_path)   # Add config details from yaml file
    cfg.freeze()   
    return cfg



def prepare_config_output_and_logger(cfg_path, log_prefix, txt_file_path, log_name="", make_log=True):
    # get config
    cfg = get_cfg_defaults()    # Get config pattern
    cfg.merge_from_file(cfg_path)   # Add config details from yaml file
    cfg.freeze()   

    txt_path = os.path.splitext(os.path.basename(txt_file_path))[0]

    # Make an output directory for saving logs to
    split_cfg_path = cfg_path.split("/")
    yaml_file_name = os.path.splitext(split_cfg_path[-1])[0]    
    output_path = os.path.join('output', yaml_file_name)
    output_path = os.path.join('output', txt_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    # Path for the log file
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    log_file = '{}_{}_{}_{}.log'.format(yaml_file_name, log_name, log_prefix, time_str)
    log_path = os.path.join(output_path, log_file)
    save_model_path = os.path.join(output_path, log_name + yaml_file_name + time_str + "_model.pth")

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

    return cfg, logger, output_path, save_model_path


