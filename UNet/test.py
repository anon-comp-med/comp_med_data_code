"""
UNet test code
"""

import argparse
import sys
import utils
import torch
import model
import torch.nn as nn
from dataset import LandmarkDataset
from torchinfo import summary
import os
import numpy as np
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image as to_pil
from model import two_d_softmax, nll_across_batch
import statistic
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cfg',
                    help='The path to the configuration file for the experiment',
                    required=True,
                    type=str)

    parser.add_argument('--images',
                        help='The path to a directory containing all the images (in jpg)',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the csv file containing annotations',
                        type=str,
                        required=True,
                        default='')
    
    parser.add_argument('--test',
                    help='Path to the text file listing the test images',
                    type=str,
                    required=True,
                    default='')
    
   
    parser.add_argument('--weight',
                help='Path to weights',
                type=str,
                required=True,
                default=None)    
    

    parser.add_argument('--log_name',
            help='name of log file',
            type=str,
            required=False,
            default='')

    
    parser.add_argument('--no_scale',
        help='scale the statistics to the original resolution',
        action='store_true',
        required=False),   


    args = parser.parse_args()

    return args


def main():
    
    # Get arguments
    args = parse_args()    

    cfg, logger, _, save_model_path = utils.prepare_config_output_and_logger(args.cfg, "train_and_val", args.test, args.log_name)
    
    # print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")    

    model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS).cuda()             
    loaded_state_dict = torch.load(args.weight)
    model.load_state_dict(loaded_state_dict, strict=True)

    test_dataset = LandmarkDataset(args.images, args.annotations, args.test, cfg.DATASET,
                                       is_test=True, perform_augmentation=False) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


    logger.info(f"----------- Begin Test -----------")        
    

    mm_radial_errors = []
    px_radial_errors = []
    mm_ere = []
    px_ere = []

    with torch.no_grad():
        for idx, (image, channels, metas) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            image = image.cuda()
            channels = channels.cuda()
            outputs = model(image.float())
            outputs = two_d_softmax(outputs)

            metas = [
                {key: metas[key][i].item() for key in metas if key != "cached_name"}
                for i in range(outputs.shape[0])
            ]

            for i in range(outputs.shape[0]):
                output = outputs[i].cpu().numpy()
                channel = channels[i].cpu().numpy()
                meta = metas[i]
                mm_radial_errors.append(statistic.get_radial_error(output, channel, meta, scale=not(args.no_scale)))
                mm_ere.append(statistic.get_ere(output, meta, scale=not(args.no_scale)))
                px_radial_errors.append(statistic.get_radial_error(output, channel, meta, mm=False, scale=not(args.no_scale)))                       
                px_ere.append(statistic.get_ere(output, meta, mm=False, scale=not(args.no_scale)))

    
    mm_radial_errors = np.array(mm_radial_errors)
    mm_ere = np.array(mm_ere)
    px_radial_errors = np.array(px_radial_errors)
    px_ere = np.array(px_ere)

    stats = ""

    logger.info(f"----------- Combined Statistics -----------")    
    
    mm_M_RE = np.mean(mm_radial_errors, axis=(0, 1))   
    mm_SD_RE = np.std(mm_radial_errors, axis=(0, 1))   
    logger.info("Payer: MRE: {:.3f}+-{:.3f} mm".format(mm_M_RE, mm_SD_RE))     

    px_M_RE = np.mean(px_radial_errors, axis=(0, 1))   
    px_SD_RE = np.std(px_radial_errors, axis=(0, 1))   
    logger.info("Pixel: MRE: {:.3f}+-{:.3f} px".format(px_M_RE, px_SD_RE))    

    mm_sdr = statistic.get_sdr(mm_radial_errors.flatten(), [2.0, 4.0, 10.0])
    logger.info("Successful Detection Rate (SDR) for 2mm, 4mm, and 10mm respectively: "
        "{:.3f}% {:.3f}% {:.3f}%".format(*mm_sdr))                

    px_sdr = statistic.get_sdr(px_radial_errors.flatten(), [2.0, 4.0, 10.0])
    logger.info("Successful Detection Rate (SDR) for 2px, 4px, and 10px respectively: "
        "{:.3f}% {:.3f}% {:.3f}%".format(*px_sdr))            

    mm_ere_mean = np.mean(mm_ere, axis=(0,1))
    mm_ere_std = np.std(mm_ere, axis=(0,1))
    logger.info("Payer ERE: {:.3f}+-{:.3f} px".format(mm_ere_mean, mm_ere_std))

    px_ere_mean = np.mean(px_ere, axis=(0,1))
    px_ere_std = np.std(px_ere, axis=(0,1))
    logger.info("Pixel ERE: {:.3f}+-{:.3f} px".format(px_ere_mean, px_ere_std))            


    stats += f"{mm_M_RE:.3f} {mm_SD_RE:.3f} "
    stats += f"{mm_sdr[0]:.3f} {mm_sdr[1]:.3f} {mm_sdr[2]:.3f} "
    stats += f"{px_M_RE:.3f} {px_SD_RE:.3f} "
    stats += f"{px_sdr[0]:.3f} {px_sdr[1]:.3f} {px_sdr[2]:.3f} "
    stats += f"{mm_ere_mean:.3f} {mm_ere_std:.3f} "
    stats += f"{px_ere_mean:.3f} {px_ere_std:.3f}\n"
 
    # Make data collection easier
    logger.info(stats)


if __name__ == '__main__':
    main()


