"""Optional reconstruction-only SSL, followed separately by supervised training."""
from pathlib import Path
import argparse
import csv
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from .models import MODEL_REGISTRY, build_model
from .train import seed_everything, seed_worker, autocast_context, save_json, save_history

class ImagesOnly(Dataset):
    def __init__(self, manifest):
        with Path(manifest).open(newline='') as f:
            self.paths = [Path(row['image_path']) for row in csv.DictReader(f)]
        if not self.paths or any(not p.is_file() for p in self.paths):
            raise ValueError('SSL manifest has no images or contains missing paths')
        if len(set(self.paths)) != len(self.paths):
            raise ValueError('Duplicate image paths in SSL manifest')
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = TF.pil_to_tensor(image.convert('RGB')).float()/255
        image = TF.resize(image, [224,224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if random.random() < .5:
            image = TF.hflip(image)
        if random.random() < .5:
            image = TF.vflip(image)
        return torch.rot90(image, random.randrange(4), (-2,-1)).contiguous()

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-manifest', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--model', choices=list(MODEL_REGISTRY), default='TransUNetMAESizeDeepHeads')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--mae-mask-ratio', type=float, default=.30)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--gradient-clip', type=float, default=.5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=False)
    return p

def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.mae_mask_ratio < 1:
        raise ValueError('Require positive epochs/batch size and 0 < masking < 1')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f'Use a new SSL output directory: {args.output_root}')
    seed_everything(args.seed)
    dataset = ImagesOnly(args.image_manifest)
    device = torch.device(args.device)
    model, _ = build_model(args.model, pretrained=args.pretrained)
    model.to(device)
    # Train shared encoder and reconstruction modules; freeze every supervised head.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('model.transformer.') or not name.startswith(
            ('model.','classifier.','tumor_size_regressor.')))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == 'cuda')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(annotations_used=False, images=len(dataset),
        trainable_parameters=sum(p.numel() for p in parameters),
        masked_patches=round(196*args.mae_mask_ratio), patch_size=16,
        frozen_branches=['segmentation_decoder','classification','area_estimation'])
    save_json(args.output_root/'config.json', config)
    history = []
    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        seen = 0
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                predicted, mask = model.forward_reconstruction(images, args.mae_mask_ratio)
                loss = model.compute_reconstruction_loss(predicted.float(), images, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite SSL loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite SSL gradients')
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()*len(images)
            seen += len(images)
        mean_loss = total/seen
        history.append({'epoch':epoch+1,'reconstruction_loss':mean_loss,
                        'learning_rate':optimizer.param_groups[0]['lr']})
        if mean_loss < best:
            best = mean_loss
            torch.save({'model':model.state_dict(),'epoch':epoch,'best_train_loss':best,
                        'model_key':args.model,'config':config}, args.output_root/'best_model.pt')
        save_history(args.output_root/'history.csv', history)
        scheduler.step()
        print(f'SSL epoch {epoch+1}/{args.epochs}: MSE={mean_loss:.6f}', flush=True)
    (args.output_root/'COMPLETE').write_text('completed\n')

if __name__ == '__main__':
    main()
