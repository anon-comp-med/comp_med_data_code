"""
UNet train code
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


def parse_args():

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cfg',
                    help='The path to the configuration file for the experiment',
                    required=False,
                    type=str)

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
    
    parser.add_argument('--train',
                    help='Path to the text file listing the train images',
                    type=str,
                    required=False,
                    default='')
    
    parser.add_argument('--val',
                help='Path to the text file listing the train images',
                type=str,
                required=False,
                default='')
    
    parser.add_argument('--log_name',
            help='name of log file',
            type=str,
            required=False,
            default='')
    
    parser.add_argument('--patience',
            help='patience for early stopping',
            type=int,
            required=False,
            default=5)

    parser.add_argument('--no_scale',
        help='scale the statistics to the original resolution',
        action='store_true',
        required=False),

    args = parser.parse_args()

    return args      
   

def main():
    
    # Get arguments
    args = parse_args()    

    cfg, logger, _, save_model_path = utils.prepare_config_output_and_logger(args.cfg, "train_and_val", args.train, args.log_name)
   

    # print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")        
 
    model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS).cuda() 

    logger.info("-----------Model Summary-----------")
    model_summary = summary(model, (1, 1, 512, 512), verbose=0)
    logger.info(model_summary)
    
    # load the train dataset and put it into a loader
    training_dataset = LandmarkDataset(args.images, args.annotations, args.train, cfg.DATASET, 
                                       is_train=True, perform_augmentation=True)    
    
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    validation_dataset = LandmarkDataset(args.images, args.annotations, args.val, cfg.DATASET, 
                                         is_val=True, perform_augmentation=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg.TRAIN.BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 6, 8], gamma=0.1)

    best_val_loss = float('inf')
    save_mre = 0
    save_sdre = 0
    epochs_no_improve = 0
    patience = args.patience 

    for epoch in range(cfg.TRAIN.EPOCHS):

        logger.info(f"-----------Epoch {epoch} Training-----------")

        losses_per_epoch = []

        for batch, (image, channels, _) in enumerate(training_loader):
        
            image = image.cuda()
            channels = channels.cuda()

            output = model(image.float())
            output = two_d_softmax(output)

            optimizer.zero_grad()
            loss = nll_across_batch(output, channels)
            loss.backward()

            optimizer.step()

            losses_per_epoch.append(loss.item())

            if (batch + 1) % 5 == 0:
                logger.info("[{}/{}]\tLoss: {:.3f}".format(batch + 1, \
                            len(training_loader), np.mean(losses_per_epoch)))


        scheduler.step()

        logger.info(f"-----------Epoch {epoch} Validation -----------")


        validation_losses = []
        mm_radial_errors = []
        px_radial_errors = []
        mm_ere = []
        px_ere = []


        with torch.no_grad():
            
            for idx, (image, channels, metas) in enumerate(validation_loader):
                
                image = image.cuda()
                channels = channels.cuda()

                outputs = model(image.float())
                outputs = two_d_softmax(outputs)

                loss = nll_across_batch(outputs, channels)
                validation_losses.append(loss.item())

                metas = [
                    {key: metas[key][i].item() for key in metas if key != "cached_name"}
                    for i in range(outputs.shape[0])
                ]


                for i in range(outputs.shape[0]):
                   output = outputs[i].cpu().numpy()
                   channel = channels[i].cpu().numpy()
                   meta = metas[i]

                   mm_radial_errors.append(statistic.get_radial_error(output, channel, meta, scale=not(args.no_scale)))

                   if epoch == cfg.TRAIN.EPOCHS - 1:                       

                       mm_ere.append(statistic.get_ere(output, meta, scale=not(args.no_scale)))
                       
                       px_radial_errors.append(statistic.get_radial_error(output, channel, meta, mm=False, scale=not(args.no_scale)))                       
                       px_ere.append(statistic.get_ere(output, meta, mm=False, scale=not(args.no_scale)))
            

            mean_val_loss = np.mean(validation_losses)
            logger.info("Loss: {:.3f}".format(mean_val_loss))


            mm_radial_errors = np.array(mm_radial_errors)
            mm_M_RE = np.mean(mm_radial_errors, axis=(0, 1))   
            mm_SD_RE = np.std(mm_radial_errors, axis=(0, 1))   
            logger.info("Payer: MRE: {:.3f}+-{:.3f} mm".format(mm_M_RE, mm_SD_RE))            


            if epoch == cfg.TRAIN.EPOCHS - 1:
                
                px_radial_errors = np.array(px_radial_errors)
                mm_ere = np.array(mm_ere)
                px_ere = np.array(px_ere)

                # Output MRE in pixels
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
                logger.info("Payer ERE: {:.3f}+-{:.3f} mm".format(mm_ere_mean, mm_ere_std))


                px_ere_mean = np.mean(px_ere, axis=(0,1))
                px_ere_std = np.std(px_ere, axis=(0,1))
                logger.info("Pixel ERE: {:.3f}+-{:.3f} px".format(px_ere_mean, px_ere_std))


            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_no_improve = 0
                save_mre = mm_M_RE
                save_sdre = mm_SD_RE
                # Optionally save the best model
                torch.save(model.state_dict(), save_model_path)
                logger.info("Validation loss improved. Saving model.")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                logger.info("Saved Payer: MRE: {:.3f}+-{:.3f} mm".format(save_mre, save_sdre)) 
                logger.info("-----------Training Complete-----------")
                sys.exit()
                break                

    if mean_val_loss < best_val_loss:           
        logger.info("Saving Model's State Dict to {}".format(save_model_path))
        torch.save(model.state_dict(), save_model_path)
        
    logger.info("-----------Training Complete-----------")



if __name__ == '__main__':
    main()




