import argparse
import glob
import logging
import numpy as np
import os
import torch
from collections import OrderedDict
from copy import deepcopy
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from annotator.canny import CannyDetector
from dataloader.load_imagenet_canny import ImageNetCanny
from diffusion import create_diffusion
from models.dit import DiT_models
from models.dit_cldm import DiTControlNet


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = 1 # len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    latent_size = args.image_size // 8
    model = DiTControlNet()
    model.load_state_dict(torch.load("E:/DiT-ControlNet/DiTControlNet-XL-2-256x256.pt"))
    model = model.to(device)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"sd-vae-ft-{args.vae}").to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageNetCanny(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    # update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0

    for epoch in range(args.epochs):
        for x, c, y in loader:
            x = x.to(device)
            c = c.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, c=c)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                running_loss = 0
                log_steps = 0

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = model.module.state_dict()
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)

    model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
