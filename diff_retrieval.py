#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import yaml
import io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.data import Dataset
# import models.moco_vits as moco_vits
import utils_ret
from utils_ret import extract_features
import pickle
from PIL import Image, ImageFile
# import natsort
# from models import mae_vits
import timm

import matplotlib.pyplot as plt
import LovelyPlots.utils as lp
import itertools
import json, glob 
import seaborn as sns

from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import pandas as pd
import cv2

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)



class SynthDataset(Dataset):
    def __init__(self, main_dir, transform, capjson=None):
        self.main_dir = main_dir
        self.transform = transform
        if not os.path.exists(f"{main_dir}/prompts.txt"): # TODO: will not work for imagenettemodel
            p = main_dir.split('train')[0]
            if 'laion' in main_dir or 'l100kaion' in main_dir:
                capjson = glob.glob(os.path.join(p, "*combined_captions.json"))[0]
            elif 'imagenette' in main_dir:
                capjson = glob.glob(os.path.join(p, "*blip_captions.json"))[0]

        if capjson is None:
            self.total_imgs = []
            for (root,dirs,files) in os.walk(self.main_dir, topdown=True):
                if len(dirs) > 0:
                    dirs.sort()
                    continue
                else:
                    temp = [x for x in files if x.endswith(('.JPG', '.JPEG', '.jpg','.png'))]
                    temp = sorted(temp)   
                val_files =  [f"{root}/{word}" for word in temp]
                self.total_imgs.append(val_files)
            self.total_imgs = list(itertools.chain(*self.total_imgs))
            
            from natsort import natsorted
            self.total_imgs = natsorted(self.total_imgs)
            with open(f"{main_dir}/prompts.txt") as file_in:
                self.prompts = []
                for line in file_in:
                    self.prompts.append(line)
            
        else:
            with open(capjson) as f:
                all_prompts_dict = json.load(f)
            self.total_imgs = list(all_prompts_dict.keys())
            self.prompts = [all_prompts_dict[k][0] for k in self.total_imgs]
        
        print(self.total_imgs[0])
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            tensor_image = image
        prompt = self.prompts[idx]
        return tensor_image,prompt,idx

def tv_loss(img, tv_weight=1e-4, norm='l1'):
    if norm == 'l2':
      w_variance = torch.sum(torch.pow(img[:,:,:-1] - img[:,:,1:], 2))
      h_variance = torch.sum(torch.pow(img[:,:-1,:] - img[:,1:,:], 2))
    else:
      w_variance = torch.sum((img[:,:,:-1] - img[:,:,1:]).abs())
      h_variance = torch.sum((img[:,:-1,:] - img[:,1:,:]).abs()) 
    loss = tv_weight * (h_variance + w_variance)
    return loss.item()


parser = argparse.ArgumentParser('Generic image retrieval given a path')
parser.add_argument('--query_dir', type=str, required=True, help="The inferences")
parser.add_argument('--val_dir', type=str, required=True, help = "The train data")
parser.add_argument('--pt_style', default='sscd', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--multiscale', default=False, type=utils_ret.bool_flag)

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--similarity_metric', default='dotproduct', type=str)
parser.add_argument('--num_loss_chunks', default=1, type=int)
parser.add_argument('--numpatches', default=1, type=int)
parser.add_argument('--isvit', action='store_true')
parser.add_argument('--layer', default=1, type=int, help="layer from end to create descriptors from.")
parser.add_argument('--stype', default='', type=str,choices=['cross'])
parser.add_argument('--keephead', action='store_true')
parser.add_argument('--keeppredictor', action='store_true')

parser.add_argument('-ssp','--sim_save_path', type=str,default='./similarityscores/')
# parser.add_argument('--extra', default='', type=str)
parser.add_argument('--einsum_chunks', default=30, type=int)
parser.add_argument('--dontsave', action='store_true')

parser.add_argument('--num_matches', default=4, type=int)

## for oxfrd paris
parser.add_argument('--imsize', default=224, type=int, help='Image size')

parser.add_argument('--noeval', action='store_true')

# parser.add_argument('-cj','--captions_json', type=str,default= None)
# best_acc1 = 0


def main():
    args = parser.parse_args()
    assert os.path.isdir(args.query_dir), "Query dir doesnt exist, skipping!"
    if args.similarity_metric == 'splitlosscross':
        args.similarity_metric = 'splitloss'
        args.stype = 'cross'
   
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # utils.init_distributed_mode(args)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    if args.pt_style == 'dino':  
        import dino_vits  
        dinomapping = {
            'vit_base' : 'dino_vitb16',
            'vit_base8' : 'dino_vitb8',
            'vit_small' : 'dino_vits16',
            'resnet50': 'dino_resnet50',
            'vit_base_cifar10' : 'dino_vitb_cifar10'
        }
        if args.similarity_metric == 'splitloss':
            model = dino_vits.__dict__[dinomapping[args.arch]](
            pretrained = True,
            global_pool = ''
        )
            args.isvit = True
        else:
            model = dino_vits.__dict__[dinomapping[args.arch]](
            pretrained = True
        )
    elif args.pt_style == 'clip':
        import clip
        clipmapping = {
            'vit_large' : 'ViT-L/14',
            'vit_base' : 'ViT-B/16',
            'resnet50' : 'RN50x16'
        }
        model, _ = clip.load(clipmapping[args.arch])
 
    elif args.pt_style == 'sscd':
        if args.arch == 'resnet50':
            model = torch.jit.load("./pretrainedmodels/sscd_disc_mixup.torchscript.pt")
        elif args.arch == 'resnet50_im':
            model = torch.jit.load("./pretrainedmodels/sscd_imagenet_mixup.torchscript.pt")
        elif args.arch == 'resnet50_disc':
            model = torch.jit.load("./pretrainedmodels/sscd_disc_large.torchscript.pt")
        else:
            NotImplementedError('This model type does not exist for SSCD')

    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True



    # Data loading code

    ret_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])


    dataset_query = SynthDataset(args.query_dir, ret_transform)
    dataset_values = SynthDataset(args.val_dir, ret_transform) # the train data

    
    dataset_simpl = SynthDataset(args.val_dir, transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        
                    ])) 

    ## creating dataloader
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_values, shuffle=False)
    else:
        sampler = None
    data_loader_values = torch.utils.data.DataLoader(
        dataset_values,
        sampler=sampler,
        batch_size=64,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=64,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    
    print(f"train: {len(dataset_values)} imgs / query: {len(dataset_query)} imgs")

    model.eval()


    ############################################################################
    if not args.multiprocessing_distributed:
        utils_ret.init_distributed_mode(args)
    if args.rank == 0:  # only rank 0 will work from now on

        # dp = os.path.basename(os.path.normpath(args.query_dir))   
        dp  = os.sep.join(os.path.normpath(args.query_dir).split(os.sep)[-3:])
         
        if not args.noeval: 
            import wandb
            wandb.init(project="imsimv2_retrieval",name=f"{dp}_{args.similarity_metric}")
            wandb.config.update(args)
        
        # Step 1: extract features
        values_features = extract_features(args, model, data_loader_values, args.gpu, multiscale=args.multiscale)
        query_features = extract_features(args, model, data_loader_query, args.gpu, multiscale=args.multiscale)
        values_features = nn.functional.normalize(values_features, dim=1, p=2)
        query_features = nn.functional.normalize(query_features, dim=1, p=2)

        ############################################################################
        # Step 2: similarity
        if args.similarity_metric == 'splitloss':
            if args.numpatches > 1:
                args.num_loss_chunks = args.numpatches
            from einops import rearrange, reduce
            v = rearrange(values_features, 'b (c p) -> b c p ', c = args.num_loss_chunks)
            q = rearrange(query_features, 'b (c p) -> b c p ', c = args.num_loss_chunks)
            chunk_dp = torch.einsum('ncp,mcp->nmc', [v, q])
            sim = reduce(chunk_dp, 'n m c -> n m', 'max')
        else:
            sim = torch.mm(values_features, query_features.T)
            sim2 = torch.mm(values_features, values_features.T)

        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        ######################
        ret_savepath = f'ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/'
        os.makedirs(ret_savepath,exist_ok=True)

        simscores = sim.T 
        bg_simscores = sim2.T

        torch.save(simscores.cpu(), os.path.join(ret_savepath, "similarity.pth"))
        torch.save(bg_simscores.cpu(), os.path.join(ret_savepath, "similarity_wtrain.pth"))

        main_v,main_l = simscores.topk(1,axis=1,largest=True)
        bg_v,bg_l = bg_simscores.topk(2,axis=1,largest=True)
        bg_v = bg_v[:,-1] #remove the first one since it is to self.

        print(main_v.shape, bg_v.shape)
        plt.figure(figsize=(6,4))
        

        x0 =  main_v.cpu().numpy()
        x1 = bg_v.cpu().numpy()
        bin_width= 0.005
        import math
        nbins = math.ceil(1 / bin_width)
        bins = np.linspace(0,1, nbins)

        fig = plt.hist(x0, bins, alpha=0.4, label='sim(gen,train)',density=True)
        fig = plt.hist(x1, bins, alpha=0.6, label='sim(train,train)',density=True)
        plt.legend(loc='upper right')

        plt.savefig(f"ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/histogram.png")
        # plt.close(fig)
        plt.figure()

        #### Computing mean similarity and FID scores of generations ######

        sim_mean = np.mean(x0)
        sim_std = np.std(x0)
        sim_75pc = np.percentile(x0, 75)
        sim_90pc = np.percentile(x0, 90)
        sim_95pc = np.percentile(x0, 95)

        bg_mean = np.mean(x1)
        bg_std = np.std(x1)
        bg_75pc = np.percentile(x1, 75)
        bg_90pc = np.percentile(x1, 90)
        bg_95pc = np.percentile(x1, 95)
        
        sim_gt_05pc = np.sum(x0 > 0.5)/(x0.shape[0])

        wandb.log({
            'sim_mean':sim_mean,
            'sim_std':sim_std,
            'sim_75pc' : sim_75pc,
            'sim_90pc' : sim_90pc,
            'sim_95pc' : sim_95pc,
            'sim_gt_05pc' : sim_gt_05pc,
            'bg_mean':bg_mean,
            'bg_std' : bg_std,
            'bg_75pc' : bg_75pc,
            'bg_90pc' : bg_90pc,
            'bg_95pc' : bg_95pc
        })

        print('Simscores @x% part done')
        print({
            'sim_mean':sim_mean,
            'sim_std':sim_std,
            'sim_75pc' : sim_75pc,
            'sim_90pc' : sim_90pc,
            'sim_95pc' : sim_95pc,
            'sim_gt_05pc' : sim_gt_05pc,
            'bg_mean':bg_mean,
            'bg_std' : bg_std,
            'bg_75pc' : bg_75pc,
            'bg_90pc' : bg_90pc,
            'bg_95pc' : bg_95pc
        })
        ### clip alignment scores between images and the captions
        from utils_ret import gen_clipscore
        clipscore = gen_clipscore(data_loader_query)
        clipscore_bg = gen_clipscore(data_loader_values)
        wandb.log({
            'clipscore':clipscore,
            'clipscore_bg':clipscore_bg
            })
        print({
            'clipscore':clipscore,
            'clipscore_bg':clipscore_bg
            })
        
        ##  Computing the complexity of the matched images vs simscores
        dblocs = main_l.squeeze().numpy()
        dbsims = main_v.squeeze().numpy()
        
        entropies = []
        crs = []
        tvls = []
        for i in range(len(dblocs)):
            loc = dblocs[i]
            torchim = dataset_simpl.__getitem__(loc)[0]*255
            rgbImg = (torchim).permute(1, 2, 0).numpy().astype(np.uint8)
            ent = entropy(img_as_ubyte(color.rgb2gray(rgbImg)))
            # if ent < 1 and dbsims[i] < 0.6:
            #     import ipdb; ipdb.set_trace()
            entropies.append(ent)
            # https://stackoverflow.com/questions/44328645/interpreting-cv2-imencode-result
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encimg = cv2.imencode('.jpg', rgbImg, encode_param)
            crs.append(len(encimg)/(1024))
            tvls.append(tv_loss(torchim))
        entropies = np.array(entropies)
        crs = np.array(crs)
        tvls = np.array(tvls)
        # import ipdb; ipdb.set_trace()
        torch.save(entropies, os.path.join(ret_savepath, "entropies.pth"))
        torch.save(tvls, os.path.join(ret_savepath, "totvar.pth"))
        torch.save(crs, os.path.join(ret_savepath, "compressions.pth"))
        torch.save(dbsims, os.path.join(ret_savepath, "dbsims.pth"))
        from scipy import stats
        cc_ent,pval_ent = stats.pearsonr(entropies, dbsims)
        cc_comp,pval_comp = stats.pearsonr(crs, dbsims)
        cc_tvl,pval_tvl = stats.pearsonr(tvls, dbsims)
        cc_mixed, pval_mixed = stats.pearsonr(entropies*crs**(0.5), dbsims)
        wandb.log({
            'cc_ent':cc_ent,
            'pval_ent':pval_ent,
            'cc_comp' : cc_comp,
            'pval_comp' : pval_comp,
            'cc_tvl' : cc_tvl,
            'pval_tvl' : pval_tvl,
            'cc_mixed' : cc_mixed,
            'pval_mixed' : pval_mixed

            })
        
        ax = sns.scatterplot(data=pd.DataFrame({'simplicity':entropies, 'sims':dbsims}), x="simplicity", y="sims")
        plt.title(f'CC={cc_ent}, pval={pval_ent}')
        plt.savefig(f"ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/simplicityscatter_entropies.png")
        plt.figure()

        ax1 = sns.scatterplot(data=pd.DataFrame({'simplicity':tvls, 'sims':dbsims}), x="simplicity", y="sims", color = 'green')
        plt.title(f'CC={cc_tvl}, pval={pval_tvl}')
        plt.savefig(f"ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/simplicityscatter_tvls.png")
        plt.figure()
        ax2 = sns.scatterplot(data=pd.DataFrame({'simplicity':crs, 'sims':dbsims}), x="simplicity", y="sims", color = 'hotpink')
        plt.title(f'CC={cc_comp}, pval={pval_comp}')
        plt.savefig(f"ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/simplicityscatter_crs.png")
        plt.figure()

        ax2 = sns.scatterplot(data=pd.DataFrame({'simplicity':entropies*crs**(0.5), 'sims':dbsims}), x="simplicity", y="sims", color = 'red')
        plt.title(f'CC={cc_mixed}, pval={pval_mixed}')
        plt.savefig(f"ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/simplicityscatter_crs.png")
        plt.figure()

       ## simscores of repeated images
        if "nodup" not in args.query_dir:
            main_v = main_v.squeeze().numpy()
            main_l = main_l.squeeze().numpy()
            try:
                with open(f"{args.val_dir}/weights_0.05_5_seedNone.pickle",'rb') as file:
                    weights = pickle.load(file)
                file.close()
            except:
                print('This case should have a weights file, check the path')

            is_weighted = []
            for loc in main_l:
                if weights[loc] > 1:
                    val = 1
                else:
                    val = 0
                is_weighted.append(val)
            
            sns.barplot(x='is_weighted', y='sims', data={'is_weighted':is_weighted, 'sims': main_v},palette=['tomato','limegreen'])
            plt.savefig(f"ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/weightplot.png")
            # plt.close(fig)
            plt.figure()
        

        ### fid and precision, recall stuff
        from metrics.ipr import IPR
        from  metrics.fid import calculate_fid_given_paths
        from glob import glob
        totnum = len(dataset_query)

        realroot = args.val_dir
        genroot = args.query_dir
        


        fid = calculate_fid_given_paths([realroot,genroot],
                                        50,
                                        "cuda",
                                        2048, 4)
        wandb.log({
            # 'precision':precision,
            # 'recall':recall,
            'fid':fid,
        })


        # Plotting code also here
        showtill = 200
        toshow = 10
        topn = 10
        simscores = sim.T
        vals,_ = simscores.topk(1,axis=1,largest=True)
        vals = vals.squeeze().cpu().numpy()
        locas = np.argsort(vals)[::-1]

        nm = 0 #start location, change it for later ranked images
        for nm in range(0,showtill + 10,10):
            topcopies = locas[nm:nm+toshow]
            # print(topcopies)
            _,topk_indices = simscores.topk(topn,axis=1,largest=True)

            
            plt.figure(figsize = (20,2*toshow))
            allarray = []
            for whichloc in topcopies:
                matches = topk_indices[whichloc,:].detach().numpy()
                match_array = np.expand_dims(np.asarray(dataset_query.__getitem__(whichloc)[0]), axis=0).transpose(0,2,3,1)
                _,h,w,c = match_array.shape
                for m in matches:
                    v_im = np.expand_dims(np.asarray(dataset_values.__getitem__(m)[0]), axis=0).transpose(0,2,3,1)
                    match_array = np.concatenate((match_array,v_im),axis=0)
                allarray.append(match_array)
            allarray = np.array(allarray).reshape(11*toshow,h,w,c)
            result = gallery(allarray,toshow)
            plt.imshow(result)
            plt.axis('off')
            os.makedirs(f'ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/',exist_ok=True)
            plt.savefig(f'ret_plots/{dp}/images/{args.pt_style}_{args.arch}_{args.similarity_metric}{args.stype}/{nm}.png',bbox_inches='tight')
            plt.close()

    
def einsum_in_chunks(v,q,stype='cross',nchunks=100):
    from einops import rearrange, reduce
    n = v.shape[0]
    sim_list = []
    tchunks = torch.chunk(v,nchunks, dim=0)
    count = 0
    for val in tchunks:
        print(f'In chunk {count}')
        # val = v[i,:,:]
        # val = torch.unsqueeze(val, 0)
        if stype=='cross':
            chunk_dp = torch.einsum('ncp,mdp->nmcd', [val,q])
            sim = reduce(chunk_dp, 'n m c d -> n m', 'max').clone()
        else:
            chunk_dp = torch.einsum('ncp,mcp->nmc', [val,q])
            sim = reduce(chunk_dp, 'n m c -> n m', 'max').clone()
        sim_list.append(sim)
        count+=1

    return torch.cat(sim_list,dim=0)



def gallery(array, nrows=1):
    nindex, height, width, intensity = array.shape
    # print(nindex, height, width, intensity,nrows)
    array = array*0.5 + 0.5
    ncols = nindex//nrows
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

if __name__ == '__main__':
    main()
