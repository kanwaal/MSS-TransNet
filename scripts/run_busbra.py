"""Fixed target hold-out: five-source zero-shot/limited adaptation or independent full target."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mss_transnet.engine import run_experiment
from mss_transnet.evaluate import evaluate,average_target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setting",choices=["zero-shot","limited","full-target"],required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source-checkpoints",nargs=5,type=Path)
    p.add_argument("--pretrained",type=Path)
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/default.json")
    a = p.parse_args();config=json.loads(a.config.read_text())
    if a.setting=="full-target":
        if a.source_checkpoints:raise ValueError("Full-target excludes BUSI checkpoints")
        if a.pretrained is None:raise ValueError("Full-target requires ImageNet initialization")
        checkpoint=run_experiment(a.manifest,a.output/"training",config,
                                  mode="full-target",pretrained=a.pretrained)
        evaluate(checkpoint,a.manifest,a.output/"test",config["device"],config["batch_size"],True)
    else:
        if a.pretrained:raise ValueError("Transfer uses BUSI checkpoints rather than fresh ImageNet weights")
        if not a.source_checkpoints:raise ValueError("Supply all five BUSI checkpoints")
        directories=[]
        for i,source in enumerate(a.source_checkpoints,1):
            directory=a.output/f"source_fold_{i:02d}"
            checkpoint=(run_experiment(a.manifest,directory/"training",config,mode="limited",source=source)
                        if a.setting=="limited" else source)
            evaluate(checkpoint,a.manifest,directory/"test",config["device"],config["batch_size"],True)
            directories.append(directory/"test")
        average_target(directories,a.output)

if __name__=="__main__":main()
