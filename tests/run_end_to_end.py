"""Optional CPU integration check; synthetic images and temporary checkpoints only."""
from pathlib import Path
import argparse
import csv
import gc
import json
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mss_transnet.engine import run_experiment
from mss_transnet.evaluate import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--amp",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/synthetic_check.json").read_text())
    config.update(device=args.device,amp=args.amp)
    proof={}
    with tempfile.TemporaryDirectory(prefix="mss_transnet_integration_") as tmp:
        tmp=Path(tmp);generator=np.random.default_rng(42);records=[]
        for split,repeats in [("train",2),("val",1),("test",1)]:
            for name in ["normal","benign","malignant"]:
                for index in range(repeats):
                    image=tmp/f"{split}_{name}_{index}.png"
                    mask=tmp/f"{split}_{name}_{index}_mask.png"
                    pixels=(generator.random((32,32,3))*255).astype(np.uint8)
                    foreground=np.zeros((32,32),dtype=np.uint8)
                    if name!="normal":foreground[7:24,9:22]=255
                    Image.fromarray(pixels).save(image);Image.fromarray(foreground).save(mask)
                    records.append(dict(dataset="BUSI",sample_id=f"{split}/{name}/{index}",
                        group_id=f"{split}/{name}/{index}",class_name=name,
                        image_path=str(image),mask_paths=str(mask),split=split))
        manifest=tmp/"busi.csv"
        with manifest.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        original_open=Image.open
        def guarded(path,*args,**kwargs):
            if Path(path).name.startswith("test_"):
                raise AssertionError("Held-out test pixels were loaded during optimization/selection")
            return original_open(path,*args,**kwargs)
        with patch("PIL.Image.open",side_effect=guarded):
            checkpoint=run_experiment(manifest,tmp/"source",config,random_init=True)
        source=torch.load(checkpoint,map_location="cpu",weights_only=False)
        history=json.loads((tmp/"source/sml/history.json").read_text())
        assert history[-1]["train"]["amp_skipped_steps"]<3, "Every synthetic optimizer update was skipped"
        encoder={k:v.clone() for k,v in source["model"].items() if k.startswith("encoder.")}
        assert (tmp/"source/ssl/selection.json").exists()
        assert source["phase"]=="sml" and source["experiment"]["test_used_for_checkpoint_selection"] is False
        summary=evaluate(checkpoint,manifest,tmp/"test",device=args.device,batch_size=2)
        assert summary["segmentation"]["images"]==2
        assert summary["segmentation"]["normal_images_excluded"]==1
        assert summary["classification"]["images"]==3
        proof.update(ssl_then_sml=True,test_pixels_never_loaded_during_training=True,
                     validation_checkpoint_selected=True,lesion_only_segmentation=True)
        target_records=[dict(r,dataset="BUS-BRA") for r in records if r["class_name"]!="normal"]
        target=tmp/"target.csv"
        with target.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(target_records[0]));writer.writeheader();writer.writerows(target_records)
        del source;gc.collect()
        zero=evaluate(checkpoint,target,tmp/"zero",device=args.device,batch_size=2,binary=True)
        assert zero["classification"]["classes"]==["benign","malignant"]
        assert zero["binary_mapping"]
        with patch("PIL.Image.open",side_effect=guarded):
            adapted=run_experiment(target,tmp/"limited",config,mode="limited",source=checkpoint)
        payload=torch.load(adapted,map_location="cpu",weights_only=False)
        assert payload["classes"]==["benign","malignant"]
        assert all(torch.equal(payload["model"][k],v) for k,v in encoder.items())
        assert not (tmp/"limited/ssl").exists()
        limited=evaluate(adapted,target,tmp/"limited_test",device=args.device,batch_size=2,binary=True)
        assert limited["classification"]["images"]==2
        try:
            run_experiment(target,tmp/"invalid_full",config,mode="full-target",source=checkpoint)
        except ValueError:
            proof["full_target_rejects_busi_checkpoint"]=True
        else:raise AssertionError("Full-target accepted BUSI weights")
        proof.update(zero_shot_binary_mapping=True,limited_encoder_state_unchanged=True,
                     limited_has_no_extra_ssl=True,limited_binary_classifier=True)
    print(json.dumps(dict(passed=True,synthetic_only=True,device=args.device,amp=args.amp,**proof),indent=2))

if __name__=="__main__":main()
