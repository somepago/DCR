import argparse
import hashlib
import itertools
import math
import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from datasets import ObjectAttributeDataset, get_classnames
from utils.draw_utils import concat_h

logger = get_logger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    # parser.add_argument(
    #     "--class_data_dir",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="A folder containing the training data of class images.",
    # )
    parser.add_argument(
        "--instance_prompt_loc",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="nolevel",
        choices = ["nolevel","classlevel","instancelevel_blip","instancelevel_random"],
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    # parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    # parser.add_argument(
    #     "--num_class_images",
    #     type=int,
    #     default=100,
    #     help=(
    #         "Minimal class images for prior preservation loss. If there are not enough images already present in"
    #         " class_data_dir, additional images will be sampled with class_prompt."
    #     ),
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--generation_seed", type=int, default=1024, help="A seed for generation images.")
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save images every X updates steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("-j","--num_workers", type=int, default=4)
    parser.add_argument("--modelsavesteps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--duplication",
        type=str,
        default= "nodup",
        choices = ['nodup','dup_both','dup_image'],
        help="Duplicate the training data or not",
    )

    parser.add_argument(
        "--unet_from_scratch",
        type=str,
        default= "no",
        choices = ['no','yes'],
        help="Duplicate the training data or not",
    )
    parser.add_argument(
        "--weight_pc", type=float, default=0.05, help="Percentage of points to sample more."
    )
    parser.add_argument(
        "--dup_weight", type=int, default=5, help="How likely dup points are, over the others."
    )
    parser.add_argument(
        "--rand_noise_lam", type=float, default=0, help="How much gaussian noise to add to text encoder embedding during training"
    )
    parser.add_argument(
        "--mixup_noise_lam", type=float, default=0, help="How much mixup noise to add to text encoder embedding during training"
    )
    
    parser.add_argument(
        "--trainspecial", type=str, default=None, choices = ['allcaps','randrepl','randwordadd','wordrepeat'],help="which caps to use"
    )
    parser.add_argument(
        "--trainspecial_prob", type=float, default=0.1, help="for special training, intervention probability"
    )
    parser.add_argument(
        "--trainsubset", type=float, default=None, help="percentage of training data to use"
    )


    ################ End of args #####################

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]


    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    OBJECTS = get_classnames(args.instance_data_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(f"{args.output_dir}/generations", exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import ipdb; ipdb.set_trace()
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    if args.unet_from_scratch == 'no':
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )
    else:
        import json
        unet_config = json.loads(open('./unet_config.json').read())
        unet = UNet2DConditionModel(**unet_config)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = ObjectAttributeDataset(
        instance_data_root=args.instance_data_dir,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        random_flip = args.random_flip,
        prompt_json = args.instance_prompt_loc,
        duplication = args.duplication,
        args = args
    )

    temp = list(train_dataset.prompts.values())
    choicelist = [x[0] for x in temp]
    # choicelist  = list(itertools.chain(*))

    if args.trainsubset is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, int(len(train_dataset)*args.trainsubset))))
        choicelist = choicelist[:int(len(train_dataset)*args.trainsubset)]
        
    if args.duplication in ['dup_both','dup_image']:
        sampler = torch.utils.data.WeightedRandomSampler(train_dataset.samplingweights, len(train_dataset), replacement=True) 
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=lambda examples: collate_fn(examples),
            num_workers=args.num_workers,
            sampler = sampler
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples),
            num_workers=args.num_workers,
        )


    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # LINEAR = "linear"
    # COSINE = "cosine"
    # COSINE_WITH_RESTARTS = "cosine_with_restarts"
    # POLYNOMIAL = "polynomial"
    # CONSTANT = "constant"
    # CONSTANT_WITH_WARMUP = "constant_with_warmup"


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    import wandb
    if accelerator.is_main_process:
        # accelerator.init_trackers("diffrep_ft", 
        #                           init_kwargs={"wandb":{"name":f"{args.output_dir}_{args.class_prompt}"}, "settings":wandb.Settings(start_method="fork")},
        #                           config=vars(args))
        init_kwargs = {"wandb":{"name":f"{args.output_dir}_{args.class_prompt}","settings":{"console": "off"}}}

        accelerator.init_trackers("diffrep_ft", 
                                  init_kwargs=init_kwargs,
                                  config=vars(args))
        
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision, safety_checker=None,
        ).to(accelerator.device)
        pipeline.set_use_memory_efficient_attention_xformers(True)
        genseed = args.generation_seed
        if args.class_prompt in ['instancelevel_blip','instancelevel_ogcap']: # and args.duplication == 'dup_image':
            # choicelist  = list(itertools.chain(*list(train_dataset.prompts.values())))
            rand_prompts = np.random.choice(choicelist,len(OBJECTS))

        if args.class_prompt in ['instancelevel_random']:
            
            temp = np.random.choice(choicelist,len(OBJECTS))
            # temp = np.random.choice(list(train_dataset.prompts.values()),len(OBJECTS))
            rand_prompts = []
            for p in temp:
                rand_prompts.append(tokenizer.decode(eval(p)))
            print(rand_prompts)
            
        for count,object in enumerate(OBJECTS):
            if count > 2:
                break
            if args.class_prompt == 'nolevel':
                genseed+=1
                prompt = f"An image"
            elif args.class_prompt == 'classlevel':
                prompt = f"An image of {object}"
            elif args.class_prompt in ['instancelevel_blip','instancelevel_ogcap','instancelevel_random']:
                # import ipdb; ipdb.set_trace()
                prompt = rand_prompts[count]
                print('first one------------')
                print(prompt)

            save_path = os.path.join(args.output_dir,"generations", f"{global_step:04d}_{object}.png")
            generator = torch.Generator("cuda").manual_seed(genseed)
            images = pipeline(prompt=prompt, height=args.resolution, width=args.resolution,
                                num_inference_steps=50, num_images_per_prompt=4, generator=generator).images
            concat_h(*images, pad=4).save(save_path)

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # import ipdb; ipdb.set_trace()
                encoder_hidden_states = text_encoder(batch["input_ids"])[0] 
                if args.rand_noise_lam > 0:
                    encoder_hidden_states = encoder_hidden_states + args.rand_noise_lam*torch.randn_like(encoder_hidden_states)
                if args.mixup_noise_lam > 0:
                    lam = np.random.beta(args.mixup_noise_lam, 1)
                    index = torch.randperm(encoder_hidden_states.shape[0]).cuda()
                    encoder_hidden_states = lam*encoder_hidden_states + (1-lam)*encoder_hidden_states[index,:]
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            revision=args.revision, safety_checker=None,
                        ).to(accelerator.device)

                        genseed = args.generation_seed
                        for count,object in enumerate(OBJECTS):
                            if count > 2:
                                break
                            if args.class_prompt == 'nolevel':
                                genseed+=1
                                prompt = f"An image"
                            elif args.class_prompt == 'classlevel':
                                prompt = f"An image of {object}"
                            elif args.class_prompt in ['instancelevel_blip','instancelevel_ogcap','instancelevel_random']:
                                prompt = rand_prompts[count]
                                print('_______in the loop______-')
                                print(prompt)
                                
                            save_path = os.path.join(args.output_dir,"generations", f"{global_step:04d}_{object}.png")
                            generator = torch.Generator("cuda").manual_seed(genseed)
                            images = pipeline(prompt=prompt, height=args.resolution, width=args.resolution,
                                              num_inference_steps=50, num_images_per_prompt=4,
                                              generator=generator).images
                            concat_h(*images, pad=4).save(save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            if accelerator.is_main_process and global_step % args.modelsavesteps == 0:
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    revision=args.revision,
                )
                pipeline.save_pretrained(os.path.join(args.output_dir, f'checkpoint_{global_step}'))

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )
        pipeline.save_pretrained(os.path.join(args.output_dir, 'checkpoint'))

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    assert not (args.duplication == 'dup_image' and args.class_prompt == 'instancelevel_ogcap'), "Duplicating just the image in original captions scenario is not acceptable"

    if args.trainspecial:
        if args.class_prompt != 'instancelevel_blip':
            raise Exception("Cant train special without blip captions")
    
    if args.trainsubset is not None:
         args.output_dir = f"{args.output_dir}_{args.trainsubset}subset"
    if args.unet_from_scratch == 'no':
        args.output_dir = f"{args.output_dir}_{args.class_prompt}_{args.duplication}"
    else:
        args.output_dir = f"{args.output_dir}_{args.class_prompt}_{args.duplication}_unetfromscr"

    if args.duplication in ['dup_both','dup_image']:
        args.output_dir  = f"{args.output_dir}_{args.weight_pc}_{args.dup_weight}"
    
    if args.rand_noise_lam > 0:
        args.output_dir  = f"{args.output_dir}_glam{args.rand_noise_lam}"
    if args.mixup_noise_lam > 0:
        args.output_dir  = f"{args.output_dir}_mixlam{args.mixup_noise_lam}"
    if args.trainspecial is not None:
        args.output_dir  = f"{args.output_dir}_special_{args.trainspecial}_{args.trainspecial_prob}"


    # TODO: adding noise in text case
    main(args)
