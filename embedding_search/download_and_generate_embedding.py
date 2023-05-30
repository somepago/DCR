import argparse
import os
import pandas as pd
from img2dataset import download
import torch
import pickle as pkl
import time
import shutil

from glob import glob
from utils import extract_features_custom, get_dataloader, get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet-fname', type=str,
            default=None)
    parser.add_argument('--image-folder', type=str,
            default=None)
    parser.add_argument('--tars', nargs='+', default=[])
    parser.add_argument('--dump-path', type=str,
                        default='./data/'
                        'laion_sd_v2p1/data')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--skip-img-embed', action='store_true')
    parser.add_argument('--skip-image-delete', action='store_true')

    # fixed params, don't change unless you wan't to change similarity metric for search
    parser.add_argument('--pt-style', default='sscd', type=str)
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--similarity-metric', type=str, default='d')
    parser.add_argument('--data-dir', type=str, default=None)
    return parser.parse_args()

def main(args):
    assert (args.parquet_fname is not None) or \
           (args.image_folder is not None) or \
           (args.tars is not None and args.skip_download), "Specify parquet/tar list or image folder with pngs"
    if args.skip_download:
        assert args.tars is not None, "Must specify tar files if skipping download"
    
    if args.parquet_fname:
        temp_img_folder = os.path.join(args.dump_path, 
                        'images')
        os.makedirs(temp_img_folder, exist_ok=True)

    embedding_dump_folder = args.dump_path


    # 1. download files from urls using img2dataset
    if not args.skip_download and (args.parquet_fname is not None):
        print("Download starting, this may take a while.....")
        start = time.time()
        download(
            processes_count=16,
            thread_count=32,
            url_list = args.parquet_fname,
            image_size=256,
            resize_only_if_bigger=True,
            resize_mode="keep_ratio",
            skip_reencode=True,
            output_folder=temp_img_folder,
            output_format="webdataset",
            input_format="parquet",
            url_col="URL",
            caption_col="TEXT",
            enable_wandb=args.wandb,
            number_sample_per_shard=10000, # keep decent sized for multiprocess
            distributor="multiprocessing",
            save_additional_columns=["URL", "SAMPLE_ID"],
            oom_shard_count=6,
        )
        end = time.time()

        print(f"Image download + dumping took: {end-start:.2f}s")

        # all images are in tar files, collect the tars for dataloader
        tar_files = glob(temp_img_folder+'/*.tar')
        idx_end = len(tar_files)-1
        args.tars = temp_img_folder+'/{000000..'+'{idx_end:06d}'.format(idx_end=idx_end)+'}.tar'


    # 2. generate embeddings
    if not args.skip_img_embed:
        start = time.time()
        model = get_model(args)
        dataloader = get_dataloader(args)
        features, indexes = extract_features_custom(args, model, dataloader,
                                args.gpu, args.multiscale)
        dump_data = {'features': features.cpu().numpy(), 'indexes': indexes}
        with open(os.path.join(embedding_dump_folder, 'embedding.pkl'), 'wb') as f:
            pkl.dump(dump_data, f)
        end = time.time()
        print(f"Embedding processing + dumping took: {end-start:.2f}s")

    # 3. clean up, tar files
    if not args.skip_image_delete:
        for tar_file in tar_files:
            os.remove(tar_file)
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)