"""
Global branch training
"""

import argparse
import torch
from extractor_gpu import ViTExtractor
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader
from eval import *
from post_net import *
import torch.optim as optim
import save_utils
from save_utils import str2bool
import dataset
import landmarks


def make_heatmap(landmark, size, var=5.0): 
    length, width = size
    x,y = torch.meshgrid(torch.arange(0, length),
                         torch.arange(0,width))
    p = torch.stack([x,y], dim=2).float()
    inner_factor = -1/(2*(var**2))
    mean = torch.as_tensor(landmark).float()
    heatmap = (p-mean).pow(2).sum(dim=-1)
    heatmap = torch.exp(heatmap*inner_factor)
    return heatmap

# Return MSE loss with the predictions and GT
def heatmap_mse_loss(features, landmarks, var = 5.0, criterion = torch.nn.MSELoss()):
    lab = [] # Stores B heatmaps in shape (N, W, H)
    for i in range(len(landmarks)):     # batchsize
        labels = landmarks[i]
        labtemp = []
        for l in range(labels.shape[0]):
            labtemp.append(make_heatmap(labels[l], [features.shape[-2], features.shape[-1]], var=var))
        labtemp2 = torch.stack(labtemp, dim = 0)
        lab.append(labtemp2)

    label = torch.stack(lab,dim=0)  # (B, N, W, H)
    label = label.to(features.device)

    pred = []
    for i in range(len(landmarks)):  # batchsize
        feature_temp = features[i] # (N, H, W)
        pred_temp = []
        for j in range(labels.shape[0]): # number of landmarks
            gt = feature_temp[:, landmarks[i,j,0], landmarks[i,j,1]].unsqueeze(1).unsqueeze(2) # get GT feature vector (C, 1, 1)
            similarity = torch.nn.CosineSimilarity(dim=0)(gt, feature_temp).unsqueeze(0) # cosine similarity with feature and feature map (1, H, W)
            pred_temp.append(similarity)
        
        pred_temp = torch.cat(pred_temp, dim = 0).unsqueeze(0) # (1, N, H, W)
        pred.append(pred_temp)
    
    pred = torch.cat(pred, dim = 0) # (B, N, H, W)

    loss = criterion(pred, label) # Compute MSE
    return loss

def parse_args():
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')

    parser.add_argument("--save_name", type=str, default="", required=True, help="For preprending to the start of file names") # For pre-pending to file names
    parser.add_argument('--save_dir', type=str, default = None, required=True, help="For saving models")  # Changed 

    parser.add_argument('--csv_train', type=str, default = None, required=True, help="Path to CSV with augmented GT") 
    parser.add_argument('--csv_infer', type=str, default = None, required=True, help="Path to CSV with base GT") 

    parser.add_argument('--imgs_train', type=str, default = None, required=True, help="Path to augmented training (template) images")
    parser.add_argument('--imgs_infer', type=str, default = None, required=True, help="Path to base validation images")

    parser.add_argument('--txt_train', type=str, default = None, required=True, help="Path to txt file with image names used for inference")
    parser.add_argument('--txt_infer', type=str, default = None, required=True, help="Path to txt file with inference image names")

    parser.add_argument('--cfg', help='The path to the configuration file for the experiment', required=False, type=str)

    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                 small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                           Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                           vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=8, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='True', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--thresh', default=0.05, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--topk', default=5, type=int, help='Final number of correspondences.')

    parser.add_argument('--id_shot', default=125, type=int, help='template id') # TODO: Need to change
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8, 10], help='radius')

    parser.add_argument('--bs', default=4, type=int, help='batch size.')
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=20000)
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate.')
    parser.add_argument('--exp', default='global', help='exp name.')

    parser.add_argument('--no_early_stop', help='No early stopping?', action='store_true', required=False)
    parser.add_argument("--patience", default=5, type=int, help="Number of validation runs until early stopping")

    parser.add_argument('--grn', help='Use GRN in UpNet?', action='store_true', required=False)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()    

    # Ensure save directory exists
    paths, logger = save_utils.prepare_output_and_logger(args.save_dir, args.exp, args.save_name, True)
    save_model_pth = paths["model_path"]
    cfg = save_utils.prepare_config(args.cfg)

    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # random seed
    random_seed = 2026
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # For extracting descriptors
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    # Used for training the decoder
    train_dataset = dataset.TrainDatasetNew(args.imgs_train, args.csv_train, args.load_size, cfg.DATASET)
        
    # Used for getting paths to use for inference
    val_dataset = dataset.TrainDatasetNew(args.imgs_infer, args.csv_infer, args.load_size, cfg.DATASET, img_name_txt=args.txt_infer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=1)    
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)  

    # Use this for initialising Upnet (image is downsampled so shortest side is 224px)
    image, *_ = train_dataset.__getitem__(0)
    image_size = (image.shape[-2], image.shape[-1])

    print(image_size)
    
    model_post = Upnet_v3(image_size, 6528, 256, grn=args.grn).cuda() # Corse stage decoder
    model_post.train()

    for n,p in model_post.named_parameters():
        print(n)
    
    optimizer = optim.Adam(model_post.parameters(), lr=args.lr)

    best_val_loss = 10000
    
    iter_num = 0
    val_no_improve = 0 # For early stopping
    early_stop = False
    max_iterations = args.max_iterations
       
    # Save the initial model
    torch.save(model_post.state_dict(), save_model_pth)

    model_post.train()

    for epoch in np.arange(0, args.max_epoch) + 1:

        if early_stop:
            break

        model_post.train()

        for images, labs, lab_smalls, image_paths, _ in train_dataloader:

            if early_stop:
                break

            iter_num = iter_num + 1

            # Get the feature descriptor(s) (using the downsampled image(s))
            with torch.no_grad():
                descriptors, num_patches, load_size = landmarks.get_feature(
                    extractor, device, images, 
                    args.layer, args.facet, args.bin
                )

            # Run through decoder
            descriptors_post = model_post(descriptors, num_patches)  
        
            optimizer.zero_grad()
            loss = heatmap_mse_loss(descriptors_post, lab_smalls)   # Calculates MSE loss
            loss.backward()
            optimizer.step()

            #writer.add_scalar('info/loss', loss, iter_num)
            if iter_num < 50 or iter_num % 100 == 0:
                logger.info('iter: {}, loss: {}'.format(iter_num, loss))

            # regular testing and saving
            if iter_num % 2000 == 0:
                model_post.eval()

                val_loss = 0.0
                num_batches = 0

                for images, labs, lab_smalls, image_paths, _ in tqdm(val_dataloader):
                    with torch.no_grad():
                        descriptors, num_patches, load_size = landmarks.get_feature(
                            extractor, device, images, 
                            args.layer, args.facet, args.bin
                    )

                    # Run through decoder
                    descriptors_post = model_post(descriptors, num_patches)  

                    val_loss += float(heatmap_mse_loss(descriptors_post, lab_smalls).item())   # Calculates MSE loss 
                    num_batches += 1    

                val_loss /= num_batches       

                logger.info('Validation loss: {}'.format(val_loss))    

                if not args.no_early_stop:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model_post.state_dict(), save_model_pth)
                        val_no_improve = 0
                    else:
                        val_no_improve += 1

                    if val_no_improve >= args.patience:
                        logger.info(f"Early stopping triggered after {iter_num} iterations")
                        early_stop = True
                else:
                    torch.save(model_post.state_dict(), save_model_pth)


                model_post.train()
            
            if iter_num >= max_iterations:
               break
        
        if iter_num >= max_iterations:
                break
        


