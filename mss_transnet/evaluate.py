from pathlib import Path
import argparse
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import UltrasoundDataset
from .model import MSS_TransNet
from .metrics import segmentation_metrics,segmentation_summary,classification_summary
from .engine import save_json


def csv_write(path, rows):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate(checkpoint, manifest, output, device="cuda:0", batch_size=8, binary=False):
    payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
    model = MSS_TransNet(**payload["model_config"])
    model.load_state_dict(payload["model"]);model.to(device).eval()
    source_classes = tuple(payload["classes"])
    classes = ("benign","malignant") if binary else source_classes
    dataset = UltrasoundDataset(manifest,"test",classes)
    segmentation,classification = [],[]
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=batch_size):
            result = model(batch["image"].to(device),tasks=["segmentation","classification"])
            metrics = segmentation_metrics(result["segmentation"],batch["mask"].to(device))
            logits = result["classification"].float()
            if binary and len(source_classes)==3:
                # Algebraically p_M/(p_B+p_M), evaluated without normal-class underflow.
                logits = logits[:,[source_classes.index("benign"),source_classes.index("malignant")]]
            probability = logits.softmax(1).cpu()
            for i,sample in enumerate(batch["sample_id"]):
                segmentation.append(dict(sample_id=sample,class_name=batch["class_name"][i],
                    **{k:float(v[i].cpu()) for k,v in metrics.items()}))
                classification.append(dict(sample_id=sample,true_index=int(batch["class_index"][i]),
                    **{"probability_"+c:float(probability[i,j]) for j,c in enumerate(classes)}))
    output = Path(output)
    csv_write(output/"segmentation.csv",segmentation)
    csv_write(output/"classification.csv",classification)
    result = dict(segmentation=segmentation_summary(segmentation),
                  classification=classification_summary(classification,classes),
                  checkpoint=str(checkpoint),binary_mapping=bool(binary and len(source_classes)==3))
    save_json(output/"summary.json",result)
    return result


def pool_busi(directories, output, expected_images=780, expected_lesions=647):
    if len(directories)!=5:raise ValueError("Pooling requires all five held-out BUSI folds")
    segmentation,classification = [],[]
    classification_present=[(Path(d)/"classification.csv").is_file() for d in directories]
    if any(classification_present) and not all(classification_present):
        raise ValueError("Only some folds have an active classification endpoint")
    for directory in directories:
        for name,dest in [("segmentation",segmentation),("classification",classification)]:
            if name=="classification" and not all(classification_present):continue
            with (Path(directory)/(name+".csv")).open(newline="") as f:dest.extend(csv.DictReader(f))
    for rows in [segmentation,classification]:
        if not rows:continue
        ids = [r["sample_id"] for r in rows]
        if len(ids)!=len(set(ids)) or len(ids)!=expected_images:
            raise ValueError("Each BUSI image must contribute exactly once")
    if classification and {r["sample_id"] for r in segmentation}!={r["sample_id"] for r in classification}:
        raise ValueError("Classification and segmentation cohorts differ")
    result = dict(segmentation=segmentation_summary(segmentation),
                  classification=classification_summary(classification,("normal","benign","malignant")) if classification else None)
    if result["segmentation"]["images"]!=expected_lesions:raise ValueError("Unexpected lesion-image count")
    output = Path(output)
    csv_write(output/"segmentation.csv",segmentation)
    if classification:csv_write(output/"classification.csv",classification)
    save_json(output/"pooled_summary.json",result)
    return result


def average_target(directories,output):
    if len(directories)!=5:raise ValueError("Target reporting requires five separate source-fold models")
    summaries = [json.loads((Path(d)/"summary.json").read_text()) for d in directories]
    ids = []
    for directory in directories:
        with (Path(directory)/"segmentation.csv").open(newline="") as f:
            ids.append({r["sample_id"] for r in csv.DictReader(f)})
    if not all(s==ids[0] for s in ids):raise ValueError("Target test cohort changed across models")
    x = np.linspace(0,1,101)
    # Average ROC curves and row-normalized confusion matrices, NOT probabilities.
    rocs = [np.interp(x,s["classification"]["roc_fpr"],s["classification"]["roc_tpr"]) for s in summaries]
    result = dict(models=5,test_images=len(ids[0]),
        segmentation_mean_of_model_means={k:float(np.mean([s["segmentation"]["mean"][k] for s in summaries]))
                                         for k in summaries[0]["segmentation"]["mean"]},
        confusion_row_percent_mean=np.mean([s["classification"]["confusion_row_percent"] for s in summaries],axis=0).tolist(),
        roc_fpr=x.tolist(),roc_tpr_mean=np.mean(rocs,axis=0).tolist(),
        malignant_auc_mean=float(np.mean([s["classification"]["malignant_auc"] for s in summaries])),
        probabilities_ensembled=False)
    save_json(Path(output)/"target_summary.json",result)
    return result


def main():
    p = argparse.ArgumentParser(description="Evaluate one held-out checkpoint or aggregate five models")
    sub = p.add_subparsers(dest="command",required=True)
    single = sub.add_parser("single")
    single.add_argument("--checkpoint",type=Path,required=True)
    single.add_argument("--manifest",type=Path,required=True)
    single.add_argument("--output",type=Path,required=True)
    single.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    single.add_argument("--batch-size",type=int,default=8)
    single.add_argument("--binary",action="store_true")
    for name in ["pool-busi","average-target"]:
        aggregate = sub.add_parser(name)
        aggregate.add_argument("--directories",nargs=5,type=Path,required=True)
        aggregate.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    if a.command=="single":evaluate(a.checkpoint,a.manifest,a.output,a.device,a.batch_size,a.binary)
    elif a.command=="pool-busi":pool_busi(a.directories,a.output)
    else:average_target(a.directories,a.output)

if __name__=="__main__":main()
