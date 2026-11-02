"""Image-only SSL access and three-way manifests with held-out validation."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

CLASSES = ("normal","benign","malignant")


def read_manifest(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["split"] not in {"train","val","test"} for r in rows):
        raise ValueError("Expected a non-empty train/val/test manifest")
    for r in rows:
        for k in ["image_path","mask_paths"]:
            r[k] = ";".join(str((Path(path).parent/part).resolve()) if not Path(part).is_absolute()
                             else part for part in r[k].split(";"))
    ids = [r["sample_id"] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError("Sample IDs must be unique within a manifest")
    sets = {s:{r["group_id"] for r in rows if r["split"]==s} for s in ("train","val","test")}
    if any(sets[a]&sets[b] for a,b in [("train","val"),("train","test"),("val","test")]):
        raise ValueError("Group overlap between optimization, validation or test")
    if not all(sets.values()):
        raise ValueError("All three partitions are required")
    return rows


class UltrasoundDataset(Dataset):
    def __init__(self, manifest, split, classes=CLASSES, image_only=False, augment=False):
        self.records = [r for r in read_manifest(manifest) if r["split"]==split]
        self.classes, self.image_only, self.augment = tuple(classes), image_only, augment
        self.class_to_index = {c:i for i,c in enumerate(classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        r = self.records[index]
        with Image.open(r["image_path"]) as f:
            image = np.asarray(f.convert("RGB").resize((224,224),Image.Resampling.BILINEAR)).copy()
        image = torch.from_numpy(image).permute(2,0,1).float()/255
        sample = {"image":image,"sample_id":r["sample_id"]}
        if not self.image_only:
            union = None
            for p in r["mask_paths"].split(";"):
                with Image.open(p) as f:
                    array = np.asarray(f.convert("L"))>0
                union = array if union is None else np.logical_or(union,array)
            area = float(union.mean()*100.)
            mask = Image.fromarray(union.astype(np.uint8)*255).resize((224,224),Image.Resampling.NEAREST)
            sample.update(mask=torch.from_numpy((np.asarray(mask)>0).copy()).float()[None],
                class_index=torch.tensor(self.class_to_index[r["class_name"]]),
                class_name=r["class_name"], log_area=torch.tensor(np.log1p(area),dtype=torch.float32))
        if self.augment:
            horizontal,vertical = torch.rand(2)<.5
            rotation = int(torch.randint(0,4,()))
            for key in ("image","mask"):
                if key in sample:
                    value = sample[key]
                    if horizontal:value = value.flip(-1)
                    if vertical:value = value.flip(-2)
                    sample[key] = value.rot90(rotation,(-2,-1)).contiguous()
        return sample


def scan_busi(root):
    rows = []
    root = Path(root)
    for name in CLASSES:
        for image in sorted((root/name).glob("*.png")):
            if "_mask" in image.stem:continue
            masks = sorted(image.parent.glob(image.stem+"_mask*.png"))
            if not masks:raise FileNotFoundError(f"Missing mask for {image}")
            rows.append(dict(dataset="BUSI",sample_id=f"{name}/{image.stem}",
                group_id=f"{name}/{image.stem}",class_name=name,
                image_path=str(image.resolve()),mask_paths=";".join(str(p.resolve()) for p in masks)))
    if not rows:raise ValueError("No BUSI images found")
    return rows


def scan_busbra(root):
    root = Path(root)
    with (root/"bus_data.csv").open(newline="") as f:metadata = list(csv.DictReader(f))
    rows = []
    for row in metadata:
        name = row["ID"]
        image = root/"Images"/(name+".png")
        mask = root/"Masks"/(name.replace("bus_","mask_",1)+".png")
        if not image.is_file() or not mask.is_file():raise FileNotFoundError(name)
        label = row["Pathology"].lower()
        if label not in {"benign","malignant"}:raise ValueError(label)
        rows.append(dict(dataset="BUS-BRA",sample_id="BUS-BRA/"+name,
            group_id=row["Case"],class_name=label,image_path=str(image.resolve()),mask_paths=str(mask.resolve())))
    return rows


def save_manifest(rows, destination):
    path = Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    read_manifest(path)
    counts = {s:{c:sum(r["split"]==s and r["class_name"]==c for r in rows)
                 for c in sorted({r["class_name"] for r in rows})} for s in ("train","val","test")}
    audit = dict(images=counts,groups={s:len({r["group_id"] for r in rows if r["split"]==s})
                                    for s in counts},sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(audit,indent=2)+"\n")


def split_validation(indices, labels, fraction, seed):
    # Approximately 593/31/156 for complete BUSI; stratification requires at
    # least one validation example per class in tiny synthetic checks.
    nval = max(len(set(labels)),round(len(indices)*fraction))
    return train_test_split(indices,test_size=nval,stratify=labels,random_state=seed)


def prepare_busi(root, destination, seed=42):
    rows = scan_busi(root);labels = [r["class_name"] for r in rows]
    splitter = StratifiedKFold(5,shuffle=True,random_state=seed)
    heldout = []
    for fold,(development,test) in enumerate(splitter.split(rows,labels),1):
        train,val = split_validation(list(development),[labels[i] for i in development],.05,seed+fold)
        partition = {i:s for s,indices in [("train",train),("val",val),("test",test)] for i in indices}
        records = [dict(r,split=partition[i]) for i,r in enumerate(rows)]
        save_manifest(records,Path(destination)/f"fold_{fold:02d}.csv")
        heldout.extend(r["sample_id"] for r in records if r["split"]=="test")
    assert len(heldout)==len(set(heldout))==len(rows)


def prepare_busbra(root, destination, seed=42):
    rows = scan_busbra(root)
    group_labels = {}
    for r in rows:
        previous = group_labels.setdefault(r["group_id"],r["class_name"])
        if previous!=r["class_name"]:raise ValueError("Patient has conflicting labels")
    groups = sorted(group_labels)
    development,test = train_test_split(groups,test_size=.2,random_state=seed,
                                        stratify=[group_labels[g] for g in groups])
    limited,_ = train_test_split(development,train_size=.2,random_state=seed+1,
                                 stratify=[group_labels[g] for g in development])
    for name,pool in [("full",development),("limited",limited)]:
        train,val = split_validation(pool,[group_labels[g] for g in pool],.05,seed+2)
        partition = {g:s for s,items in [("train",train),("val",val),("test",test)] for g in items}
        subset = [dict(r,split=partition[r["group_id"]]) for r in rows if r["group_id"] in partition]
        save_manifest(subset,Path(destination)/(name+".csv"))


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["BUSI","BUS-BRA"],required=True)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    a = p.parse_args()
    (prepare_busi if a.dataset=="BUSI" else prepare_busbra)(a.root,a.output,a.seed)

if __name__=="__main__":main()
