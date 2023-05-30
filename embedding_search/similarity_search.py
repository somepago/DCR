import torch
import pickle as pkl
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--laion-embedding-folder', type=str)
    parser.add_argument('--generation-embedding-path', type=str)
    parser.add_argument('--dump-path', type=str)
    parser.add_argument('--num-chunks', type=int, default=100)
    return parser.parse_args()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the embeddings
    with open(args.generation_embedding_path, 'rb') as f:
        data = pkl.load(f)
        gen_embeddings = data['features']
        gen_images_fname = data['indexes']

    print(f"Number of generated images to test: {gen_embeddings.shape[0]}")

    # load the metadata
    laion_folders = [x for x in os.listdir(args.laion_embeddings_folders)]
    laion_folders.sort()
    print(f"Number of LAION chunks to test: {len(laion_folders)}")

    # create chunks of the generated embeddings
    gen_embedding_tensor = torch.Tensor(gen_embeddings).to(device)
    chunks = args.num_chunks
    chunked_gen_tensor = torch.chunk(gen_embedding_tensor, chunks)
    
    top_scores = []
    top_keys = []
    start_time = time.time()
    for batch_gen_tensor in chunked_gen_tensor:
        batch_top_scores = -np.ones(batch_gen_tensor.shape[0])
        batch_top_keys = np.zeros(batch_gen_tensor.shape[0])
       
        for laion_folder in sorted(laion_folders):
            try:
                with open(os.path.join(laion_folder, 'embedding.pkl'), 'rb') as f:
                    data = pkl.load(f)
            except Exception as e:
                print(e)
                continue
            features = data['features']
            features_tensor = torch.Tensor(features).to(device)
            keys = data['indexes']
            
            # find the top-k from current laion folder
            dist_matrix = features_tensor @ batch_gen_tensor.T
            curr_top_scores, curr_top_index = dist_matrix.max(dim=0)
            curr_top_scores = curr_top_scores.cpu().numpy()
            curr_top_index = curr_top_index.cpu().numpy().astype(int)
            curr_top_keys = np.array([keys[index] for index in curr_top_index])
            curr_top_keys = np.array([laion_folder+ ':' + x for x in curr_top_keys])
            
            # keep actual top from previous top and current top
            temp_scores = np.vstack([batch_top_scores, curr_top_scores])
            temp_keys = np.vstack([batch_top_keys, curr_top_keys])
            max_index = temp_scores.argmax(axis=0).reshape(1, -1)
            batch_top_scores = np.take_along_axis(temp_scores, max_index, axis=0).squeeze()
            batch_top_keys = np.take_along_axis(temp_keys, max_index, axis=0).squeeze()
            
            # print(batch_top_scores.shape)
            
        top_scores.append(batch_top_scores)
        top_keys.append(batch_top_keys)
        # break
    print(f"Matching took: {time.time() - start_time:.2f} secs")

    top_scores_all = np.concatenate(top_scores)
    top_keys_all = np.concatenate(top_keys)
    
    dump_dict = {'scores': top_scores_all,
                'keys': top_keys_all,
                'gen_images': gen_images_fname}

    with open(dump_dict, 'wb') as f:
        pkl.dump(f, dump_dict)
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
