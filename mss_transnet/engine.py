"""Validation-selected SSL/SML training; no held-out test access during optimization."""
from pathlib import Path
from contextlib import nullcontext
import csv
import hashlib
import json
import platform
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import UltrasoundDataset, read_manifest
from .model import MSS_TransNet
from .losses import class_weights, joint_losses, reconstruction_loss

SML_WEIGHTS = {"segmentation":.9,"classification":1/30,"area":1/30,"reconstruction":1/30}


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed);np.random.seed(seed)


def save_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def autocast(device, enabled):
    return torch.autocast("cuda",dtype=torch.float16) if enabled else nullcontext()


def make_loader(manifest, split, classes, batch, workers, seed, image_only=False):
    dataset = UltrasoundDataset(manifest,split,classes,image_only,augment=split=="train")
    return DataLoader(dataset,batch_size=batch,shuffle=split=="train",num_workers=workers,
                      generator=torch.Generator().manual_seed(seed),worker_init_fn=seed_worker,
                      pin_memory=torch.cuda.is_available(),drop_last=False)


def objective_epoch(model, loader, device, phase, weights, class_weight,
                    mask_ratio, optimizer=None, scaler=None, amp=False,
                    frozen_encoder=False, validation_mask_seed=1042):
    training = optimizer is not None
    model.train(training)
    if frozen_encoder:model.encoder.eval()
    mask_generator = None if training else torch.Generator().manual_seed(validation_mask_seed)
    sums,denominators = {},{}
    skipped_steps = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            if training:optimizer.zero_grad(set_to_none=True)
            with autocast(device,amp):
                rec = None
                if phase=="ssl" or "reconstruction" in weights:
                    prediction,pixels = model.reconstruct(batch["image"],mask_ratio,mask_generator)
                    rec = reconstruction_loss(prediction,batch["image"],pixels)
                if phase=="ssl":
                    loss,components = rec,{"reconstruction":rec}
                else:
                    outputs = model(batch["image"],tasks=weights)
                    loss,components = joint_losses(outputs,batch,weights,class_weight,rec)
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite objective")
            if training:
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
                if not torch.isfinite(norm):
                    if not amp:raise FloatingPointError("Non-finite gradients")
                    # FP16 overflow is recoverable. Do not write invalid model
                    # weights; lower the loss scale and clear this batch's grads.
                    scaler.update(new_scale=scaler.get_scale()/2)
                    optimizer.zero_grad(set_to_none=True)
                    skipped_steps += 1
                else:
                    scaler.step(optimizer);scaler.update()
            for name,value in components.items():
                denominator = (float(class_weight[batch["class_index"]].sum())
                               if name=="classification" else len(batch["image"]))
                sums[name] = sums.get(name,0.)+float(value.detach())*denominator
                denominators[name] = denominators.get(name,0.)+denominator
    means = {k:v/denominators[k] for k,v in sums.items()}
    total = means["reconstruction"] if phase=="ssl" else sum(weights[k]*means[k] for k in weights)
    return {"objective":total,**means,"amp_skipped_steps":skipped_steps}


def resize_binary_classifier(model):
    old = model.classifier[-1]
    new = nn.Linear(old.in_features,2,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight[[1,2]]);new.bias.copy_(old.bias[[1,2]])
    model.classifier[-1] = new


def run_experiment(manifest, output, config, mode="busi",
                   pretrained=None, source=None, random_init=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new experiment directory: {output}")
    if mode=="limited" and source is None:raise ValueError("Limited adaptation requires one BUSI checkpoint")
    if mode!="limited" and source is not None:raise ValueError("BUSI/full-target must not load BUSI-trained weights")
    if mode!="limited" and pretrained is None and not random_init:
        raise ValueError("Training requires the ImageNet initialization (--pretrained); use --random-init only for quick checks")
    seed = config["seed"];seed_everything(seed)
    device = torch.device(config["device"])
    amp = config["amp"] and device.type=="cuda"
    records = read_manifest(manifest)
    classes = ("normal","benign","malignant") if mode=="busi" else ("benign","malignant")
    model_config = dict(config["model"],classes=len(classes))
    if mode=="limited":
        payload = torch.load(source,map_location="cpu",weights_only=False)
        if payload["classes"]!=["normal","benign","malignant"]:
            raise ValueError("Limited adaptation must start from a three-class BUSI fold model")
        model_config = dict(payload["model_config"])
        model = MSS_TransNet(**model_config)
        model.load_state_dict(payload["model"])
        resize_binary_classifier(model);model_config["classes"] = 2
    else:
        model = MSS_TransNet(**model_config,pretrained_path=pretrained)
    model.to(device)
    weights = SML_WEIGHTS
    # Only optimization-training labels define inverse-frequency weights.
    labels = [classes.index(r["class_name"]) for r in records if r["split"]=="train"]
    class_weight = class_weights(labels,len(classes)).to(device)
    metadata = dict(manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        mode=mode,classes=list(classes),config=config,
        model_config=model_config,random_initialization=bool(random_init and mode!="limited"),
        pretrained_sha256=hashlib.sha256(Path(pretrained).read_bytes()).hexdigest() if pretrained else None,
        source_checkpoint_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
        partition_images={s:sum(r["split"]==s for r in records) for s in ("train","val","test")},
        train_class_weights=class_weight.cpu().tolist(),python=platform.python_version(),torch=torch.__version__,
        hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.machine(),
        test_used_for_checkpoint_selection=False)
    save_json(output/"experiment.json",metadata)
    phases = [] if mode=="limited" else [("ssl",config["ssl_epochs"],{"reconstruction":1.})]
    phases.append(("sml",config["sml_epochs"],weights))
    final = None
    for phase,epochs,phase_weights in phases:
        if epochs<1:raise ValueError("Every active phase needs at least one epoch")
        freeze = mode=="limited"
        model.set_trainable(phase,phase_weights,freeze)
        train = make_loader(manifest,"train",classes,config["batch_size"],config["workers"],seed,image_only=phase=="ssl")
        val = make_loader(manifest,"val",classes,config["batch_size"],config["workers"],seed,image_only=phase=="ssl")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config["learning_rate"],weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
        scaler = torch.amp.GradScaler("cuda",enabled=amp,init_scale=config.get("amp_initial_scale",1024.))
        history = [];best = float("inf");stale = 0
        phase_dir = output/phase;phase_dir.mkdir(parents=True,exist_ok=True)
        for epoch in range(1,epochs+1):
            arguments = dict(model=model,device=device,phase=phase,weights=phase_weights,
                class_weight=class_weight,mask_ratio=config["mask_ratio"],amp=amp,
                frozen_encoder=freeze,validation_mask_seed=seed+1000)
            train_result = objective_epoch(loader=train,optimizer=optimizer,scaler=scaler,**arguments)
            validation = objective_epoch(loader=val,**arguments)
            improved = validation["objective"]<best
            if improved:
                best = validation["objective"];stale = 0
                torch.save(dict(model=model.state_dict(),model_config=model_config,classes=list(classes),
                    phase=phase,epoch=epoch,validation_loss=best,
                    mask_ratio=config["mask_ratio"],experiment=metadata),phase_dir/"best.pt")
            else:stale += 1
            history.append(dict(epoch=epoch,learning_rate=optimizer.param_groups[0]["lr"],
                                train=train_result,validation=validation))
            save_json(phase_dir/"history.json",history)
            print(f"{phase} {epoch}/{epochs}: train={train_result['objective']:.6f} val={validation['objective']:.6f}",flush=True)
            scheduler.step()
            # SSL uses its full epoch allocation; SML stops early without validation improvement.
            if phase=="sml" and stale>=config["patience"]:break
        final = phase_dir/"best.pt"
        payload = torch.load(final,map_location=device,weights_only=False)
        model.load_state_dict(payload["model"])
        save_json(phase_dir/"selection.json",dict(selected_epoch=payload["epoch"],
            validation_loss=best,completed_epochs=len(history),test_access=False))
    (output/"COMPLETE").write_text("completed\n")
    return final


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parent/"configs/default.json")
    p.add_argument("--mode",choices=["busi","full-target","limited"],default="busi")
    p.add_argument("--pretrained",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--random-init",action="store_true")
    a = p.parse_args()
    final = run_experiment(a.manifest,a.output,json.loads(a.config.read_text()),a.mode,
                           a.pretrained,a.source,a.random_init)
    print(final)

if __name__=="__main__":main()
