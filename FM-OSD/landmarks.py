"""
Helper functions for running through encoder and inference
"""

import torch
from tqdm import tqdm
import numpy as np
from post_net import *
import torch.nn as nn
import eval


# Return the encoder features (assumes pre-processed image)
def get_feature(extractor, device, image1_batch, layer: int = 9, facet: str = 'key', bin: bool = True):
    
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size

    return descriptors1, num_patches1, load_size1


# Return the global decoder features (For "Upnet_v3")
def run_decoder_global(extractor, model_post, image_path1, load_size, device, layer, facet, bin):

    # Resize and normalise image
    image1_batch, _ = extractor.preprocess(image_path1, (load_size, load_size)) 

    # Get encoder features
    descriptors1, num_patches1, _ = get_feature(extractor, device, image1_batch, layer, facet, bin)

    # Run through decoder
    descriptors1_post = model_post(descriptors1, num_patches1) 

    return descriptors1_post


# For "Upnet_v3_coarsetofine2_tran_new"
def run_decoder(extractor, model_post, image_path1, load_size, device, layer, facet, bin, gt=None, is_local=False, with_feature=False, feature=None):

    gt_local, offset, crop_feature = None, None, None
    if is_local and not with_feature:
        image1_batch, _, gt_local, offset = extractor.preprocess_local(
            image_path1, (load_size, load_size), [int(gt[0]), int(gt[1])]
        )    
    elif with_feature:
        image1_batch, _, gt_local, offset, crop_feature = extractor.preprocess_local_withfeature(
            image_path1, (load_size, load_size), [int(gt[0]), int(gt[1])], feature
        )      
    else:
        image1_batch, image1_pil = extractor.preprocess(
            image_path1, (load_size, load_size)
        )
    
    # Get local patch features
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size 
    
    # Run through local/global decoder
    descriptors1_post = model_post(descriptors1, num_patches1, load_size1, (is_local or with_feature)) 

    return descriptors1_post, gt_local, offset, crop_feature


# Original size is assumed to be the same for all images
"""
template_features : N,C,1,1
descriptors1_posts : List with each element [C, H, W], but the H and W's may be different (due to test inference resizing globally)
descriptors2_post : B, C, H, W
"""
def BDMS(template_features, descriptors1_posts, descriptors2_post, gts, original_size_temps, original_size_query, topk, is_local=False, offset=None):

    # Take average cosine similarity over all templates
    similarities_local = torch.nn.CosineSimilarity(dim=1)(template_features, descriptors2_post)
    similarities_local = similarities_local.mean(dim=0)  # [H, W]

    h2, w2 = similarities_local.shape
    similarities_local = similarities_local.reshape(1, -1).squeeze(0) 
    sim_k_local, nn_k = torch.topk(similarities_local, k = topk, dim=-1, largest=True)

    index_matches_all = [] # (K, H, 2) (matching points per possible prediction)
    for index_local in range(topk):   

        index_matches = []

        for idx, descriptors1_post in enumerate(descriptors1_posts):

            # Find query prediction coordinate
            i_y = nn_k[index_local]//w2
            i_x = nn_k[index_local]%w2

            # Find matching point on the template feature map with highest similarity to the predicted query position
            similarities_reverse_local = torch.nn.CosineSimilarity(dim=0)(
                descriptors2_post[0, :, i_y, i_x].unsqueeze(1).unsqueeze(2), 
                descriptors1_post
            )
           
            h1, w1 = similarities_reverse_local.shape
            similarities_reverse_local = similarities_reverse_local.reshape(-1) 
            _, nn_1_local = torch.max(similarities_reverse_local, dim=-1)

            # Matching point coordinates on template
            img1_y_to_show_local = nn_1_local // w1
            img1_x_to_show_local = nn_1_local % w1

            size_y, size_x = descriptors1_post.shape[-2:]

            # Scale back to original resolution for global stage (as using downsampled, whilst local uses crops)
            x1_show = img1_x_to_show_local
            y1_show = img1_y_to_show_local
            if not is_local:
                x1_show = original_size_temps[idx, 1] * x1_show / size_x
                y1_show = original_size_temps[idx, 0] * y1_show / size_y
        
            index_matches.append((y1_show, x1_show))

        index_matches_all.append(index_matches)      


    # Compute average distance to GT for each possible choice and take min for final prediction
    best_distance = 1000000000
    index_best = -1
    for k in range(topk):
        matches = index_matches_all[k]

        distance_temp = 0
        for idx, (y1_show, x1_show) in enumerate(matches):
            distance_temp += pow(y1_show - gts[idx, 0], 2) + pow(x1_show - gts[idx, 1], 2)

        distance_temp = distance_temp / len(matches)

        if distance_temp < best_distance:
            best_distance = distance_temp
            index_best = k

    img2_indices_to_show_local = nn_k[index_best:index_best+1].cpu().item()
    size_y, size_x = descriptors2_post.shape[-2:]
    y2_show = img2_indices_to_show_local // size_x
    x2_show = img2_indices_to_show_local % size_x

    if not is_local:
        y2_show = np.round(y2_show/size_y*original_size_query[0])
        x2_show = np.round(x2_show/size_x*original_size_query[1])
    else:
        y2_show = offset[0] + y2_show
        x2_show = offset[1] + x2_show

    return y2_show, x2_show, best_distance  


# Uses the full pipeline of the local and global decoder
def inference_find_landmark_all(extractor, device, model_post, template_dataloader, infer_dataloader, 
                                load_size: int = 224, layer: int = 9, facet: str = 'key', bin: bool = True, topk = 5, get_ere = False):
    

    init = False

    lab_feature_global_all = []
    lab_feature_local_all = []
    descriptors1_global_post_all = []
    descriptors1_local_post_all = []
    gt_global_all = []
    gt_local_all = []
    original_size_all = []

    for idx, (template_kps, img_path_temp, original_size_temp) in enumerate(tqdm(template_dataloader)):
        
        """
        torch.cuda.synchronize(device)
        start_alloc = torch.cuda.memory_allocated(device)
        """
        
        # Assume batch-size of 1
        template_kps = template_kps.squeeze(0)
        img_path_temp = img_path_temp[0]
        original_size_temp = original_size_temp[0]
        
        if not init:
            descriptors1_local_post_all = [[] for _ in range(len(template_kps))]
            init = True
        
        # Run global decoder on template
        descriptors1_global_post, *_ = run_decoder(
            extractor, model_post, img_path_temp, 
            load_size, device, layer, facet, bin, 
            is_local=False
        )
        descriptors1_global_post_cpu = descriptors1_global_post.squeeze(0)#.detach().cpu()
        descriptors1_global_post_all.append(descriptors1_global_post_cpu)

        # Resize to original dimensions
        temp_size = [original_size_temp[0].item(), original_size_temp[1].item()]
        descriptors1_post_large = torch.nn.functional.interpolate(descriptors1_global_post, temp_size, mode = 'bilinear')
        del descriptors1_global_post

        lab_features_global = []    
        lab_features_local = []
        gt_locals = []  

        for i in range(len(template_kps)):
            lab_y = int(template_kps[i][0])
            lab_x = int(template_kps[i][1])
            size_y, size_x = descriptors1_global_post_cpu.shape[-2:]
            lab_y = int(lab_y/original_size_temp[0]*size_y)  
            lab_x = int(lab_x/original_size_temp[1]*size_x)

            # Feature from global decoder
            lab_feature = descriptors1_global_post_cpu[:, lab_y, lab_x]

            lab_feature = lab_feature.unsqueeze(1).unsqueeze(2) #.detach().cpu() # (C, 1, 1)
            lab_features_global.append(lab_feature)

            # Run local decoder on crop, and get the cropped feature
            descriptors1_post_local, gt_local, _, crop_feature = run_decoder(
                extractor, model_post, img_path_temp, 
                load_size, device, layer, 
                facet, bin, gt = template_kps[i], 
                with_feature=True, feature=descriptors1_post_large
            ) 
                                   
            # Computed fused local features
            descriptors1_post_local = nn.functional.normalize(descriptors1_post_local, dim=1) + nn.functional.normalize(crop_feature, dim=1)
            descriptors1_post_local_cpu = descriptors1_post_local.squeeze(0) #.detach().cpu()
            
            descriptors1_local_post_all[i].append(descriptors1_post_local_cpu)
            gt_locals.append(gt_local)

            # Extracts the fused landmark feature vector
            lab_feature_local = descriptors1_post_local[0, :, gt_local[0], gt_local[1]]
            lab_feature_local = lab_feature_local.unsqueeze(1).unsqueeze(2) #.detach().cpu()
            lab_features_local.append(lab_feature_local)

            del descriptors1_post_local, crop_feature

            torch.cuda.empty_cache()

        del descriptors1_post_large
        torch.cuda.empty_cache()

        gt_global_all.append(template_kps)
        gt_local_all.append(torch.tensor(gt_locals))        
        lab_feature_global_all.append(torch.stack(lab_features_global, dim=0))   
        lab_feature_local_all.append(torch.stack(lab_features_local, dim=0))  
        original_size_all.append(original_size_temp)
        
        """
        torch.cuda.synchronize(device)
        end_alloc = torch.cuda.memory_allocated(device)

        delta_mb = (end_alloc - start_alloc) / 1024**2
        print(f"extra live GPU memory after block: {delta_mb:.2f} MB")
        """
        

    gt_global_all = torch.stack(gt_global_all, dim=0)
    gt_local_all = torch.stack(gt_local_all, dim=0)
    
    lab_feature_global_all = torch.stack(lab_feature_global_all, dim=0)    
    lab_feature_local_all = torch.stack(lab_feature_local_all, dim=0)
    original_size_all = torch.stack(original_size_all, dim=0)


    pred_all = []
    gt_all = []
    eres_all = []

    # iterate over all the testing images
    for landmark_list, img_path_query, original_size_query in tqdm(infer_dataloader):

        landmark_list = landmark_list.squeeze(0)

        # Run query through global decoder
        descriptors2_post, *_ = run_decoder(
            extractor, model_post, img_path_query[0], 
            load_size, device, layer, facet, bin, 
            is_local=False
        )
        descriptors2_post_cpu = descriptors2_post #.detach().cpu()

        # Upsample to original resolution
        size = [original_size_query[0][0].item(), original_size_query[0][1].item()]
        descriptors2_post_large = torch.nn.functional.interpolate(descriptors2_post, size, mode = 'bilinear')

        points1 = []    # Query GT 
        points2 = []    # Final prediction (on fused features)

        # iterate over each point (landmark)
        for i in range(landmark_list.shape[0]):
            points1.append([landmark_list[i][0].item(), landmark_list[i][1].item()])

            # Find prediction point on query
            y2_show, x2_show, _ = BDMS(
                lab_feature_global_all[:, i], descriptors1_global_post_all, 
                descriptors2_post_cpu, gt_global_all[:, i], original_size_all, 
                original_size_query[0], topk, is_local=False
            )
        
            # Run local decoder on crop, and get the cropped feature
            descriptors2_post_local, gt_local, offset2, crop_feature2 = run_decoder(
                extractor, model_post, img_path_query[0], 
                load_size, device, layer, 
                facet, bin, gt = [int(y2_show), int(x2_show)], 
                with_feature=True, feature=descriptors2_post_large
            )

            # Computed fused features
            descriptors2_post_local = nn.functional.normalize(descriptors2_post_local, dim = 1) + nn.functional.normalize(crop_feature2, dim = 1)
            descriptors2_post_local_cpu = descriptors2_post_local #.detach().cpu()
            del descriptors2_post_local
            torch.cuda.empty_cache()

            # Find prediction point on query crop
            y2_show, x2_show, _ = BDMS(
                lab_feature_local_all[:, i], descriptors1_local_post_all[i], 
                descriptors2_post_local_cpu, gt_local_all[:, i], original_size_all, 
                original_size_query[0], topk, is_local=True, offset=offset2
            )

            if get_ere:
                # ERE using local patch
                # Compute cosine-sim map between template vector and query
                similarities_local = torch.nn.CosineSimilarity(dim=1)(lab_feature_local_all[:, i], descriptors2_post_local_cpu[0])
                similarities_local = similarities_local.mean(dim=0)  # [H, W]

                # Get the landmark position on local crop
                y2_show_local = y2_show - offset2[0]  
                x2_show_local = x2_show - offset2[1]

                # ERE for current landmark (batch-size 1)
                eres_all.append(eval.ere(similarities_local, [y2_show_local, x2_show_local]))
                
           
            points2.append([y2_show, x2_show])

            del descriptors2_post_local_cpu
            torch.cuda.empty_cache()

        del descriptors2_post, descriptors2_post_large, descriptors2_post_cpu
        torch.cuda.empty_cache()

        pred_all.append(points2)
        gt_all.append(points1)

    return pred_all, gt_all, eres_all
