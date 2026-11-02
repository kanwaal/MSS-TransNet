"""Run five independent SSL/SML folds and pool held-out predictions once per image."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mss_transnet.engine import run_experiment
from mss_transnet.evaluate import evaluate,pool_busi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pretrained",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/default.json")
    a = p.parse_args();config = json.loads(a.config.read_text());directories=[]
    for fold in range(1,6):
        manifest=a.manifests/f"fold_{fold:02d}.csv"
        directory=a.output/f"fold_{fold:02d}"
        checkpoint=run_experiment(manifest,directory,config,pretrained=a.pretrained)
        evaluation=directory/"test"
        evaluate(checkpoint,manifest,evaluation,config["device"],config["batch_size"])
        directories.append(evaluation)
    pool_busi(directories,a.output/"pooled")

if __name__=="__main__":main()
