'''
Code elements borrowed from 
https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
'''

import numpy as np
import torch
import torch.distributed as dist
import subprocess
import os
import sys
import PIL
from collections import defaultdict, deque
import time, datetime
from pathlib import Path

import builtins
from torch._six import inf

from torchvision import transforms
from einops import rearrange

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(images,cutmix_beta=1):
    input = images.detach().clone()
    if cutmix_beta > 0:
            # generate mixed sample
        lam = np.random.beta(cutmix_beta,cutmix_beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input = input[rand_index]
        input[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]
    return input

# def cutmix_bg(images,cutmix_beta=1):
#     input = images.detach().clone()
#     if cutmix_beta > 0:
#             # generate mixed sample
#         lam = np.random.beta(cutmix_beta,cutmix_beta)
#         rand_index = torch.randperm(input.size()[0]).cuda()
#         bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
#         input = input[rand_index]
#         images[:, :, bbx1:bbx2, bby1:bby2] = input[:, :, bbx1:bbx2, bby1:bby2]
#     return images

# def cutmix_alltypes(images,cutmix_beta=1): 
#     input = images.detach().clone()
#     lam = np.random.beta(cutmix_beta,cutmix_beta)
#     rand_index = torch.randperm(input.size()[0]).cuda()
#     bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
#     if lam > 0.5:
#         input = input[rand_index]
#         input[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]
#         return input
#     else:
#         input = input[rand_index]
#         images[:, :, bbx1:bbx2, bby1:bby2] = input[:, :, bbx1:bbx2, bby1:bby2]
#         return images

def cutmix_alltypes(images,cutmix_beta):
    n = images.shape[0]
    input = images.detach().clone()
    im1 = images[0,:,:,:]
    im_rep = im1.repeat(n, 1, 1, 1)
    # generate mixed sample
    lam = np.random.beta(cutmix_beta,cutmix_beta)
    rand_index = torch.randperm(input.size()[0]).cuda()
    input = input[rand_index]

    fglam = np.random.uniform(0, 1)
    if fglam > 0.5:
        for i in range(n):
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[i, :, bbx1:bbx2, bby1:bby2] = im_rep[i, :, bbx1:bbx2, bby1:bby2]
        return input[1:,:,:,:], im1
    else:
        for i in range(n):
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            im_rep[i, :, bbx1:bbx2, bby1:bby2] = input[i, :, bbx1:bbx2, bby1:bby2]
        return im_rep[1:,:,:,:], im1

def segmix(images,targets,bg_images,cutoff=0):

    n = bg_images.shape[0]
    nc = bg_images.shape[1]
    h,w = targets[0].shape[0],targets[0].shape[1]
    im1 = images[0]
    uq_classes,classcounts = torch.unique(targets[0],return_counts=True) 
    classcounts = classcounts/(h*w)
    ok1 = torch.where(uq_classes > -100)[0]
    ok2 = torch.where(classcounts > cutoff)[0]
    oklocs = np.intersect1d(ok1, ok2)
    uq_classes = uq_classes[oklocs]
    if len(uq_classes) <= 0:
        return 0,0
    
    for i in range(n):
        c = np.random.choice(uq_classes, 1)
        mask = torch.eq(targets[0],torch.Tensor(c)).repeat(nc, 1, 1)
        bg_images[i][mask] = im1[mask]
    im1 = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(im1)
    bg_images = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(bg_images)
    return bg_images, im1


def segmix_hard(images,targets,bg_images,cutoff=0,resize_range=(0.75, 1.0), rotate_range=(-10, 10)):
    
    n = bg_images.shape[0]
    nc = bg_images.shape[1]
    h,w = targets[0].shape[0],targets[0].shape[1]
    im1 = images[0]
    uq_classes,classcounts = torch.unique(targets[0],return_counts=True) 
    classcounts = classcounts/(h*w)
    # ok1 = torch.where(uq_classes > 0)[0]
    ok1 = torch.where(uq_classes > -100)[0] # this is to add background as a class, -100 since VOC has -1 as bg class
    ok2 = torch.where(classcounts > cutoff)[0]
    oklocs = np.intersect1d(ok1, ok2)
    uq_classes = uq_classes[oklocs]
    if len(uq_classes) <= 0:
        return 0,0
   
    for i in range(n):
        c = np.random.choice(uq_classes, 1)
        im1 = images[0]
        target1 = targets[0]
        augmentation = [
        transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 4)),
                transforms.RandomGrayscale(p=0.3),
                transforms.RandomSolarize(threshold=0.9),
                transforms.RandomAutocontrast(p=0.3),
            ]
        im_augm = transforms.Compose(augmentation)(im1)
        hflip_prob = np.random.uniform(0,1)
        if hflip_prob >= 0.5:
            im1_flip =  transforms.RandomHorizontalFlip(p=1)(im_augm)
            target_flip  = transforms.RandomHorizontalFlip(p=1)(target1)
        else:
            im1_flip = im_augm.detach().clone()
            target_flip = target1.detach().clone()

        vflip_prob = np.random.uniform(0,1)
        if vflip_prob >= 0.8:
            im1_flip =  transforms.RandomVerticalFlip(p=1)(im1_flip)
            target_flip  = transforms.RandomVerticalFlip(p=1)(target_flip)
        else:
            im1_flip = im1_flip.detach().clone()
            target_flip = target_flip.detach().clone()

        if c not in [0,-1]:
            # resize the image and mask
            resize_ratio = np.random.uniform(resize_range[0], resize_range[1])
            new_h, new_w = int(im1_flip.shape[1] * resize_ratio), int(im1_flip.shape[2] * resize_ratio)
            im1_resize = transforms.Resize((new_h, new_w))(im1_flip)
            target_resize = transforms.Resize((new_h, new_w), interpolation=PIL.Image.NEAREST)(target_flip.unsqueeze(0))

            # rotate the image and mask
            angle = np.random.uniform(rotate_range[0], rotate_range[1])
            im1_rotate = transforms.functional.rotate(im1_resize, angle, expand=True)
            target_rotate = transforms.functional.rotate(target_resize, angle, resample=PIL.Image.NEAREST, expand=True)


            # select a random class, and create its mask
        
            mask = torch.eq(target_rotate,torch.Tensor(c))

            # retrieve the bounding box of the mask
            y, x = torch.where(mask.squeeze())
            try:
                x_min, y_min = torch.min(x), torch.min(y)
            except:
                return 0,0
            x_max, y_max = torch.max(x), torch.max(y)
            im_masked = im1_rotate[:, y_min:y_max, x_min:x_max]
            mask = mask[:, y_min:y_max, x_min:x_max]
            mask = mask.repeat(nc, 1, 1)

            paste_image = bg_images[i]
            x1, y1 = np.random.randint(0, max(1, paste_image.shape[2] - x_max + x_min)), np.random.randint(0, max(1, paste_image.shape[1] - y_max + y_min))
            x2, y2 = min(paste_image.shape[2], x1+x_max-x_min), min(paste_image.shape[1], y1+y_max-y_min)
            mask = mask[:, :y2-y1, :x2-x1]
            im_masked = im_masked[:, :y2-y1, :x2-x1]
            
            bg_images[i][:,y1:y2, x1:x2][mask] = im_masked[mask]
        else:
            mask = torch.eq(target_flip,torch.Tensor(c))
            mask = mask.repeat(nc, 1, 1)
            bg_images[i][mask] = im1_flip[mask]

    im1 = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(im1)
    bg_images = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(bg_images)       
    return bg_images, im1

def segmix_resized(images, targets, resize_range=(0.5, 1.0), rotate_range=(-10, 10)):
    input = images.detach().clone()
    n = images.shape[0]
    nc = images.shape[1]
    # im1 = images[0,:,:,:]
    target1 = targets[0]
    uq_classes = torch.unique(targets[0]).cpu().numpy()
    uq_classes = uq_classes[uq_classes > 0]
    if len(uq_classes) <= 0:
        return 0,0
   
    for i in range(1,n):
        # if i >1:
        # import ipdb; ipdb.set_trace()
        im1 = images[0,:,:,:]
        # print(f'Segmixing {i}th image')
        #adding random augmentations to data
        augmentation = [
        transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 4)),
                transforms.RandomGrayscale(p=0.3),
                transforms.RandomSolarize(threshold=0.8),
                transforms.RandomAutocontrast(p=0.5),
            ]
        im_augm = transforms.Compose(augmentation)(im1)

        #random flip - horizontal (TODO: vertical flip to add)
        hflip_prob = np.random.uniform(0,1)
        if hflip_prob >= 0.5:
            im1_flip =  transforms.RandomHorizontalFlip(p=1)(im_augm)
            target_flip  = transforms.RandomHorizontalFlip(p=1)(target1)
        else:
            im1_flip = im_augm.detach().clone()
            target_flip = target1.detach().clone()

        vflip_prob = np.random.uniform(0,1)
        if vflip_prob >= 0.8:
            im1_flip =  transforms.RandomVerticalFlip(p=1)(im1_flip)
            target_flip  = transforms.RandomVerticalFlip(p=1)(target_flip)
        else:
            im1_flip = im1_flip.detach().clone()
            target_flip = target_flip.detach().clone()

        im1_flip = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(im1_flip)
        input = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(input)

        # resize the image and mask
        resize_ratio = np.random.uniform(resize_range[0], resize_range[1])
        new_h, new_w = int(im1_flip.shape[1] * resize_ratio), int(im1_flip.shape[2] * resize_ratio)
        im1_resize = transforms.Resize((new_h, new_w))(im1_flip)
        target_resize = transforms.Resize((new_h, new_w), interpolation=PIL.Image.NEAREST)(target_flip.unsqueeze(0))

        # rotate the image and mask
        angle = np.random.uniform(rotate_range[0], rotate_range[1])
        im1_rotate = transforms.functional.rotate(im1_resize, angle, expand=True)
        target_rotate = transforms.functional.rotate(target_resize, angle, resample=PIL.Image.NEAREST, expand=True)

        # select a random class, and create its mask
        c = np.random.choice(uq_classes, 1)
        mask = torch.eq(target_rotate, torch.Tensor(c))

        # retrieve the bounding box of the mask
        y, x = torch.where(mask.squeeze())
        try:
            x_min, y_min = torch.min(x), torch.min(y)
        except:
            # import ipdb; ipdb.set_trace()
            return 0,0
        x_max, y_max = torch.max(x), torch.max(y)
        im_masked = im1_rotate[:, y_min:y_max, x_min:x_max]
        mask = mask[:, y_min:y_max, x_min:x_max]
        mask = mask.repeat(nc, 1, 1)

        # replace the input image, random region with the selected mask
        paste_image = input[i]
        x1, y1 = np.random.randint(0, max(1, paste_image.shape[2] - x_max + x_min)), np.random.randint(0, max(1, paste_image.shape[1] - y_max + y_min))
        x2, y2 = min(paste_image.shape[2], x1+x_max-x_min), min(paste_image.shape[1], y1+y_max-y_min)
        mask = mask[:, :y2-y1, :x2-x1]
        im_masked = im_masked[:, :y2-y1, :x2-x1]
        input[i] = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(input[i])
        input[i][:,y1:y2, x1:x2][mask] = im_masked[mask]
        
    return input[1:,:,:,:], im1

# from https://github.com/facebookresearch/dino/blob/main/utils.py
def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0
    mrr = 0
    rec_array = []
    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]
       
        mrr+= 1/(np.min(pos) + 1)
        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap
        
        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

        recalls = []
        len_true_matches = len(qgnd)
        for k in kappas:
            rec = (pos < k+1).sum() / len_true_matches # since pos is 1-based now
            recalls.append(rec)
        rec_array.append(recalls)

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    
    # import ipdb; ipdb.set_trace()
    rec_array = np.array(rec_array)
    recs = np.mean(rec_array,axis=0)
    mrr = mrr/nq
    # return map, aps, pr, prs
    return map,  pr , recs, mrr


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

import argparse
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    args.distributed = True
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def multi_scale(samples, model,args):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = torch.nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)

        if args.pt_style == 'vicregl':
            feats = model(inp)[-1].clone()
        elif args.pt_style == 'clip':
            feats = model.module.encode_image(samples).to(torch.float32).clone()
        else:
            feats = model(inp).clone()
        feats = torch.squeeze(feats)
        feats = torch.unsqueeze(feats,0)
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


###### this is from https://github.com/facebookresearch/dino/blob/main/eval_knn.py


@torch.no_grad()
def extract_features(args, model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = MetricLogger(delimiter="  ")
    features = None
    # count = 0
    for samples,_,index in metric_logger.log_every(data_loader, 100):
        # print(f'At the index {index}')
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        
        if multiscale:
            feats = multi_scale(samples, model,args)
        else:
            # if args.pt_style == 'vicregl':
            #     if args.similarity_metric == 'splitloss':
            #         feats = model(samples)[0]
            #         if args.numpatches <= 1:
            #             _,n,_ = feats.shape
            #             args.numpatches = n
            #         feats = rearrange(feats, 'b h w -> b (h w)').clone()
            #     else:
            #         feats = model(samples)[-1].clone()
            # elif args.pt_style == 'clip':
            #     feats = model.module.encode_image(samples).to(torch.float32).clone()
            # elif args.pt_style in ['supervised','moco','mae','dino']:
            if args.similarity_metric == 'splitloss':
                # import ipdb; ipdb.set_trace()
                if  args.pt_style == 'dino' and args.layer > 1:
                    feats = model.module.get_intermediate_layers(samples,args.layer)[0].clone()
                elif args.pt_style == 'vicregl':
                    feats = model(samples)[0]
                else:  
                    feats = model(samples)
                args.numpatches = feats.shape[1]
                feats = rearrange(feats, 'b h w -> b (h w)').clone()
            #     else:
            #         feats = model(samples).clone()
            else:
                # import ipdb; ipdb.set_trace()
                if  args.pt_style == 'dino' and args.layer > 1:
                    # import ipdb; ipdb.set_trace()
                    feats = model.module.get_intermediate_layers(samples,args.layer)[0][:,0,:].clone()
                elif args.pt_style == 'vicregl':
                    feats = model(samples)[1].clone()
                elif args.pt_style == 'multigrain':
                    feats = model(samples)['normalized_embedding'].clone()
                else:
                    feats = model(samples).clone()
            # feats = torch.squeeze(feats)
            # feats = torch.unsqueeze(feats,0)
            # print(feats.shape)
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


### following code from https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py



def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        args.start_epoch = 800 #hardcoded since that's the only model available
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

################


def micro_average_precision(gnds, probas_pred):
    from sklearn.metrics import average_precision_score
    num = probas_pred.shape[0]
    # num_vals = probas_pred.shape[1]
    microAPs = []
    for i in range(num):
        y_true = np.zeros_li= ke(probas_pred[i,:])
        loca = gnds[i]['ok']
        y_true[loca] = 1
    # ap = average_precision_score(y_true, probas_pred[i,:])
    # microAPs.append(ap)

    return np.mean(microAPs)




#### my code
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import natsort


class SynthDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        all_imgs = natsort.natsorted(all_imgs)
        self.total_imgs = [x for x in all_imgs if x.endswith(('.JPG', '.JPEG', '.jpg','.png','.PNG'))]

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image,idx



class TwoLeveldataset(Dataset):
    def __init__(self, img_dir,split, transform=None,numvals = 4):
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.numvals = numvals
        flevel_dirs = os.listdir(img_dir)
        flevel_dirs = sorted(flevel_dirs)
        if split == 'query':
            self.namelist = []
            
            for dir in flevel_dirs:
                cdir = os.path.join(img_dir,dir)
                temp = [x for x in os.listdir(cdir) if x.endswith(('.JPG', '.JPEG', '.jpg'))]
                temp = sorted(temp)
                self.namelist.append(f"{dir}/{temp[0]}")
               
        elif split == 'value':
            # import ipdb; ipdb.set_trace()
            self.namelist = []
            num = self.numvals+1
            for dir in flevel_dirs:
                cdir = os.path.join(img_dir,dir)
                temp = [x for x in os.listdir(cdir) if x.endswith(('.JPG', '.JPEG', '.jpg'))]
                temp = sorted(temp)
                if num > len(temp):
                    val_files =  [f"{dir}/{word}" for word in temp[1:]]
                else:
                    val_files =  [f"{dir}/{word}" for word in temp[1:num]]
                self.namelist.append(val_files)
            self.namelist = list(itertools.chain(*self.namelist))

    def __len__(self):
        return len(self.namelist)
               
    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.namelist[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx



class SD_Gens(Dataset):
    def __init__(self, img_dir,split, transform=None, numvals = 4):
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.numvals = numvals
        if split == 'query':
            self.namelist = []
            for (root,dirs,files) in os.walk(self.img_dir, topdown=True):
                if len(dirs) > 0:
                    dirs.sort()
                    continue
                else:
                    temp = [x for x in files if x.endswith(('.JPG', '.JPEG', '.jpg','.png'))]
                    temp = sorted(temp)
                    self.namelist.append(f"{root}/{temp[0]}")   
                
        elif split == 'value':
            self.namelist = []
            for (root,dirs,files) in os.walk(self.img_dir, topdown=True):
                if len(dirs) > 0:
                    dirs.sort()
                    continue
                else:
                    num = self.numvals+1
                    temp = [x for x in files if x.endswith(('.JPG', '.JPEG', '.jpg','.png'))]
                    temp = sorted(temp)   
                val_files =  [f"{root}/{word}" for word in temp[1:num]]
                self.namelist.append(val_files)
            self.namelist = list(itertools.chain(*self.namelist))

    def __len__(self):
        return len(self.namelist)
               
    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.namelist[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx
    

# from torchmetrics.multimodal import CLIPScore



# from transformers import CLIPModel,CLIPTokenizer,CLIPFeatureExtractor
# import torch

# version = "openai/clip-vit-large-patch14"
# tokenizer = CLIPTokenizer.from_pretrained(version)
# model = CLIPModel.from_pretrained(version)
# feature_extractor = CLIPFeatureExtractor.from_pretrained(version)

# @torch.no_grad()
# def clip_score(image,text):
    
#     txt_features = model.get_text_features(tokenizer(text,return_tensors="pt",max_length=77, truncation=True,padding=True)["input_ids"])
#     img_features = model.get_image_features(image)
    
#     img_features, txt_features = [
#         x / torch.linalg.norm(x, axis=-1, keepdims=True)
#         for x in [img_features, txt_features]
#     ]
    
#     return (img_features * txt_features).sum(axis=-1).numpy()

import clip
def gen_clipscore(dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)

    scores = []
    for i, (images, caps, _) in enumerate(dataloader):
        images = images.to(device)
        caps = clip.tokenize(caps,77,True).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(caps)
            image_features, text_features = [
                x / torch.linalg.norm(x, axis=-1, keepdims=True)
                for x in [image_features, text_features]
                ]
            sims = (image_features * text_features).sum(axis=-1).cpu().numpy()
        scores+=list(sims) 
        if i%50==0:
            print(f"CLIPevals at {i}/{len(dataloader)}")
            
    return np.mean(scores)
