import torch
import torchvision
import webdataset as wds
import os 

from torchvision.datasets import ImageFolder
from torchvision import transforms
from einops import rearrange
from PIL import ImageFile, Image
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_model(args):
    # Load model
    if args.pt_style == 'sscd':
        if args.arch == 'resnet50':
            model = torch.jit.load("./pretrained_models/sscd_disc_mixup.torchscript.pt")
        elif args.arch == 'resnet50_im':
            model = torch.jit.load("./pretrained_models/sscd_imagenet_mixup.torchscript.pt")
        elif args.arch == 'resnet50_disc':
            model = torch.jit.load("./pretrained_models/sscd_disc_large.torchscript.pt")
        else:
            NotImplementedError('This model type does not exist for SSCD')
    else:
        NotImplementedError('Wrong model style')
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("CUDA is not available. Using CPU instead.")
    model.eval()
    return model

def get_transform(do_normalize = True):
    if do_normalize:
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            )
    else:
        normalize = transforms.Normalize(
                mean = 0., std = 1.
                )
    ret_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])
    return ret_transform

def get_dataloader(args, do_normalize = True):
    # Data loading code
    ret_transform = get_transform(do_normalize)
    json_preproc = lambda x: x['key']

    if args.tars is not None:
        dataset = (wds.WebDataset(args.url)
                    .decode("pil")
                    .rename(image="jpg", json="json")
                    .map_dict(image=ret_transform, json=json_preproc)
                    .to_tuple("image", "json")
                    )
    elif args.image_folder is not None:
        dataset = ImageswithFilename(args.image_folder, ret_transform)
    else:
        raise RuntimeError('Either tar files or image folder must be specified')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader


@torch.no_grad()
def extract_features_custom(args, model, data_loader, use_cuda=True):
    features = []
    # count = 0
    index_all = []
    for samples, index in data_loader:
        samples = samples.cuda(non_blocking=True)
        
        if args.similarity_metric == 'splitloss':
            if  args.pt_style == 'dino' and args.layer > 1:
                feats = model.module.get_intermediate_layers(samples,args.layer)[0].clone()
            elif args.pt_style == 'vicregl':
                feats = model(samples)[0]
            else:  
                feats = model(samples)
            args.numpatches = feats.shape[1]
            feats = rearrange(feats, 'b h w -> b (h w)').clone()
        else:
            if  args.pt_style == 'dino' and args.layer > 1:
                feats = model.module.get_intermediate_layers(samples,args.layer)[0][:,0,:].clone()
            elif args.pt_style == 'vicregl':
                feats = model(samples)[1].clone()
            elif args.pt_style == 'multigrain':
                feats = model(samples)['normalized_embedding'].clone()
            else:
                feats = model(samples).clone()

        feats = feats.cpu()
        # append features
        features.append(feats)
        # get indexes
        index_all.extend(index)

    # concatenate features
    features = torch.cat(features, axis=0)
    return features, index_all


class ImageswithFilename(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)
        self.image_files = [x for x in self.image_files if (x.endswith('.png')
                                                        or x.endswith('.jpg'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]