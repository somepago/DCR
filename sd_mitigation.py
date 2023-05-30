from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datasets import insert_rand_word
class Newpipe(StableDiffusionPipeline):
    def _encode_prompt(self,*args, **kwargs):
        embedding = super()._encode_prompt(*args,**kwargs)
        return embedding + self.noiselam * torch.randn_like(embedding)

def resize(w_val,l_val,img):
#   img = Image.open(img)
  img = img.resize((w_val,l_val), Image.Resampling.LANCZOS)
  return img


def prompt_augmentation(prompt, aug_style,tokenizer=None, repeat_num=2):
    if aug_style =='rand_numb_add':
        for i in range(repeat_num):
            randnum  = np.random.choice(100000)
            prompt = insert_rand_word(prompt,str(randnum))
    elif aug_style =='rand_word_add':
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_rand_word(prompt,randword)
    elif aug_style =='rand_word_repeat':
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt,randword)
    else:
        raise Exception('This style of prompt augmnentation is not written')
    return prompt

import torch
import argparse
import os
import numpy as np
import json
import ast
from transformers import AutoTokenizer
import glob 
import itertools
from PIL import Image

def main(args):

    # checkpath = "stabilityai/stable-diffusion-2-1" 
    checkpath = "CompVis/stable-diffusion-v1-4"
    
    device = "cuda"
    if args.rand_noise_lam is not None: 
        
        pipe = Newpipe.from_pretrained(
            checkpath,safety_checker=None
        ) # .to("cuda")
        pipe.noiselam = args.rand_noise_lam
    else:
        pipe = StableDiffusionPipeline.from_pretrained(checkpath, use_auth_token=True)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
            checkpath,
            subfolder="tokenizer",
            # revision=args.revision,
            use_fast=False,
        )
    # generator = torch.Generator("cuda").manual_seed(42)
    savepath = f"./mitigationSD/inf_{args.seed}/gen"
    if args.rand_noise_lam is not None:
        savepath = f'{savepath}_ginfer{args.rand_noise_lam}'
    elif args.rand_augs is not None:
        savepath = f'{savepath}_auginfer_{args.rand_augs}_{args.rand_aug_repeats}'
    else:
        savepath = f'{savepath}_nomit'
        
    os.makedirs(savepath,exist_ok=True)
    os.makedirs(f'{savepath}/generations',exist_ok=True)

    prompt_list = ["Wall View 002" ,"Wall View 003", "Chamberly - Alloy 5 Piece Sectional","Hopped-Up Gaming: East", "Pantomine - Driftwood 4 Piece Sectional", "Cresson - Pewter 4 Piece Sectional", "Jinllingsly - Chocolate 3 Piece Sectional", "Maier - Charcoal 2 Piece Sectional", "Classic Cars for Sale", "Mothers influence on her young hippo", "Living in the Light with Ann Graham Lotz", "The No Limits Business Woman Podcast" ]

    if args.rand_augs is not None:
        final_prompt_list = []
        for prompt in prompt_list:
            newprompt = prompt_augmentation(prompt, args.rand_augs, tokenizer, args.rand_aug_repeats)
            final_prompt_list.append(newprompt)
        prompt_list = final_prompt_list

    # save the prompt list
    with open(f'{savepath}/prompts.txt', 'w') as f:
        for line in prompt_list:
            f.write(f"{line}\n")
    count = 0
    for i in range(len(prompt_list)):
        if prompt_list is not None:
            prompt = prompt_list[i]
        else:
            raise "no prompt list!"
        
        # if args.modelpath is None:
        images = pipe(prompt, num_inference_steps=50, generator=generator).images
        # else:
        #     images = pipe(prompt=prompt, height=args.resolution, width=args.resolution,
        #                                 num_inference_steps=50, num_images_per_prompt=args.im_batch).images
        
        for j in range(len(images)):
            image = images[j]
            if image.size[0] > args.resolution:
                image = resize(args.resolution, args.resolution, image)
            # import ipdb; ipdb.set_trace()
            image.save(f"{savepath}/generations/{count}.png")
            count+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess images')


    # parser.add_argument('--synset_map', type=str, default=None)
    # parser.add_argument("-nb","--nbatches", type=int, required=True)
    # parser.add_argument("-imb","--im_batch", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    # parser.add_argument("--iternum", default=None, type=int)
    parser.add_argument("--rand_noise_lam", type=float, default=None)
    parser.add_argument("--rand_augs", type=str, default=None)
    parser.add_argument("--rand_aug_repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")


    args = parser.parse_args()


    
    # if args.modelpath is None:
    print('Default SD generations will be done')

    main(args)
