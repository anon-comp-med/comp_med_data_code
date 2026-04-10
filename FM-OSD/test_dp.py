"""
Inference code
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
import landmarks
from save_utils import str2bool
import dataset


def parse_args():

    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')

    parser.add_argument("--weights_global", type=str, default="", required=True, help="Path to weights for global decoder") # For pre-pending to file names
    parser.add_argument('--weights_local', type=str, default = None, required=True, help="Path to weights for local decoder")  # Changed 

    parser.add_argument("--save_name", type=str, default="", required=True, help="For preprending to the start of file names") # For pre-pending to file names
    parser.add_argument('--save_dir', type=str, default = None, required=True, help="For saving models")  # Changed 

    parser.add_argument('--csv_train', type=str, default = None, required=True, help="Path to CSV with augmented GT") 
    parser.add_argument('--csv_infer', type=str, default = None, required=True, help="Path to CSV with base GT") 

    parser.add_argument('--imgs_train', type=str, default = None, required=True, help="Path to augmented training (template) images")
    parser.add_argument('--imgs_infer', type=str, default = None, required=True, help="Path to base validation images")

    parser.add_argument('--txt_train', type=str, default = None, required=True, help="Path to txt file with image names used for inference")
    parser.add_argument('--txt_infer', type=str, default = None, required=True, help="Path to txt file with inference image names")

    parser.add_argument('--num_templates', default=None, type=int, help="Number of templates to use for testing")


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
    parser.add_argument('--topk', default=3, type=int, help='Final number of correspondences.')

    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--id_shot', default=125, type=int, help='template id')
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8, 10], help='radius')

    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate.')
    parser.add_argument('--exp', default='direct_up_mse_20231226', help='learning rate.')

    parser.add_argument('--grn', help='Use GRN in UpNet?', action='store_true', required=False)

    parser.add_argument("--meta", type=str, default="", required=True, help="Path to metadata csv") # For pre-pending to file names

    parser.add_argument('--store_mre', help='Store MREs in a CSV', action='store_true', required=False)

    
    args = parser.parse_args()
    return args


def read_csv_meta(csv_file):
    data = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = int(row[0].split()[0])
            data[int(image_name)] = {"gender" : row[1], "age": row[2]}  

    return data

if __name__ == "__main__":

    args = parse_args()

    # Ensure save directory exists
    paths, logger = save_utils.prepare_output_and_logger(args.save_dir, "test", args.save_name, True)
    save_model_pth = paths["model_path"]
    if args.store_mre:  # Storing MREs for decision tree
        csv_path = paths["csv_path"]
    else:
        csv_path = None
    cfg = save_utils.prepare_config(args.cfg)

    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # fix random seed
    random_seed = 2026
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    train_dataset = dataset.TrainDatasetNew(args.imgs_train, args.csv_train, args.load_size, cfg.DATASET)

    template_dataset = dataset.InferDataset(args.imgs_infer, args.csv_infer, cfg.DATASET, args.txt_train, num_templates=args.num_templates)
    
    test_dataset = dataset.InferDataset(args.imgs_infer, args.csv_infer, cfg.DATASET, args.txt_infer)

    template_dataloader = DataLoader(template_dataset, batch_size=1, shuffle=False, num_workers=1)  
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)   

    # Get extractor output image size 
    image, *_ = train_dataset.__getitem__(0)
    image_size = (image.shape[-2], image.shape[-1])

    model_post = Upnet_v3_coarsetofine2_tran_new(image_size, 6528, 256, grn=args.grn).cuda()
    model_post.train()

    optimizer = optim.Adam(model_post.parameters(), lr=args.lr)    

    best_performance = 10000
    
    iter_num = 0
    # load models, we need to correctly load out_1 for the global deconder
    model_path = args.weights_local 
    model_post.load_state_dict(torch.load(model_path))  # Load fine model

    model_dict = model_post.state_dict()

    global_model_path = args.weights_global 
    pretrained_dict = torch.load(global_model_path) # Global decoder weights
    
    model_keys = model_dict.keys()
    keys = [k for k in model_keys if 'conv_out1' in k]  # Get out_1 parameters

    values = []
    for k in keys:
        temp = k.split('.')
        key_temp = temp[0][:-1]
        for i in range(len(temp) - 1):
            key_temp += '.' + temp[i+1]
        values.append(pretrained_dict[key_temp])
        
    new_state_dict = {k: v for k, v in zip(keys, values)}   # Make new state dict
    model_dict.update(new_state_dict)

    # Load grn weights also
    if args.grn:
        model_dict['grn_global.gamma'] = pretrained_dict['grn_global.gamma']
        model_dict['grn_global.beta']  = pretrained_dict['grn_global.beta']

    model_post.load_state_dict(model_dict)
    model_post.eval()


    # Run test
    with torch.no_grad():
        pred_all, gt_all, eres = landmarks.inference_find_landmark_all(
            extractor, device, model_post,  
            template_dataloader, test_dataloader, 
            args.load_size, args.layer, args.facet, 
            args.bin, topk = args.topk, get_ere = True
        )
       
    # Add back batch dimension
    pred_all = torch.tensor(pred_all)
    gt_all = torch.tensor(gt_all)

    print(pred_all.shape)

    # Read in CSV with the img name to gender splits
    data = data = read_csv_meta(args.meta)

    # Get image metadata from dataloader
    img_data = []
    for landmark_list, img_path_query, original_size_query in tqdm(test_dataloader):
        name = img_path_query[0].split("/")[-1]
        name = int(name.split(".")[0])
        img_data.append(data[name])

    print(img_data)     

    evaluater = Evaluater(pred_all, gt_all, cfg.DATASET, args.eval_radius, eres, meta=img_data, csv_path=csv_path)
    evaluater.calculate()
    evaluater.cal_metrics(logger)    

    
