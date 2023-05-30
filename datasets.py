from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.datasets import ImageFolder
import json
import ast
from itertools import chain, repeat, islice
import torch
import numpy as np
import pickle
from pathlib import Path

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def get_classnames(datasetpath):
    if "imagenette_2class" in datasetpath:
        return [ 'church', 'garbage truck']
    else:
        return [ 'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        

class ObjectAttributeDataset(ImageFolder):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        class_prompt=None,
        size=320,
        center_crop=False,
        random_flip = False,
        prompt_json = None,
        duplication = "nodup",
        args = None
    ):
        super().__init__(instance_data_root)
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        self.duplication = duplication
        self.objects = get_classnames(instance_data_root)
        self.trainspecial = args.trainspecial
        self.trainspecial_prob = args.trainspecial_prob
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if self.center_crop else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.class_prompt = class_prompt
        if class_prompt in ['instancelevel_blip','instancelevel_ogcap','instancelevel_random']:
            assert prompt_json != None
            
            with open(prompt_json) as f:
                self.prompts = json.load(f)

            print(len(self.prompts))
        if self.duplication in ['dup_both','dup_image']:
            sw_path = f"{instance_data_root}/weights_{args.weight_pc}_{args.dup_weight}_seed{args.seed}.pickle"
            if Path(sw_path).exists():
                with open(sw_path, 'rb') as handle:
                    samplingweights = pickle.load(handle)
            else:
                samplingweights = [1]*len(self.samples)
                ow_samples = np.random.choice(len(self.samples), int(args.weight_pc*len(self.samples)),replace=False)
                for i in ow_samples:
                    samplingweights[i] = samplingweights[i]*args.dup_weight//1
                with open(sw_path, 'wb') as handle:
                    pickle.dump(samplingweights, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.samplingweights = samplingweights
            print(len(self.samplingweights))    
            
    def __getitem__(self, index):
        instance_image,label = super().__getitem__(index)
        path_img,_ = self.samples[index]
        example = {}
        # instance_image = Image.open(img)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        if self.trainspecial is not None:
            if self.trainspecial in ['allcaps']:
                instance_prompt = np.random.choice(self.prompts[path_img], 1)[0]
            elif self.trainspecial in ['randrepl']:
                instance_prompt = self.prompts[path_img][0]
                rand = np.random.uniform()
                if rand <= self.trainspecial_prob :
                    instance_prompt = list(np.random.randint(49400, size=4))
                    instance_prompt = self.tokenizer.decode(instance_prompt)
            elif self.trainspecial in ['randwordadd']: # 2 random words get added
                instance_prompt = self.prompts[path_img][0]
                rand = np.random.uniform()
                if rand <= self.trainspecial_prob:
                    randword = self.tokenizer.decode(list(np.random.randint(49400, size=1)))
                    instance_prompt = insert_rand_word(instance_prompt,randword) 
                    randword = self.tokenizer.decode(list(np.random.randint(49400, size=1)))
                    instance_prompt = insert_rand_word(instance_prompt,randword) 
            elif self.trainspecial in ['wordrepeat']:
                instance_prompt = self.prompts[path_img][0]
                wordlist = instance_prompt.split(" ")
                rand = np.random.uniform()
                if rand <= self.trainspecial_prob:
                    randword = np.random.choice(wordlist)
                    instance_prompt = insert_rand_word(instance_prompt,randword) 
                    randword = np.random.choice(wordlist)
                    instance_prompt = insert_rand_word(instance_prompt,randword) 

        else:
            if self.class_prompt == 'nolevel':
                instance_prompt = "An image"
            elif self.class_prompt == 'classlevel':
                instance_prompt = f"An image of {self.objects[label]}"
            elif self.class_prompt in ['instancelevel_blip','instancelevel_random','instancelevel_ogcap']:
                if self.duplication in ['nodup','dup_both']:
                    instance_prompt = self.prompts[path_img][0]
                elif self.duplication == 'dup_image':
                    if self.samplingweights[index] > 1:
                        instance_prompt = np.random.choice(self.prompts[path_img], 1)[0]
                    else:
                        instance_prompt = self.prompts[path_img][0]
            if self.class_prompt in ['instancelevel_random']:
                instance_prompt = ast.literal_eval(instance_prompt)
                instance_prompt = self.tokenizer.decode(instance_prompt)
        # print(path_img, instance_prompt)
        example["instance_prompt_ids"] = self.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

def insert_rand_word(sentence,word):
    import random
    sent_list = sentence.split(' ')
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = ' '.join(sent_list)
    return new_sent
