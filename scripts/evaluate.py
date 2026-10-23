"""Evaluate a user-supplied checkpoint using the packaged experiment metrics."""
from pathlib import Path
import argparse
import sys
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from breast_benchmark.models import MODEL_REGISTRY, build_model, configure_classification_classes
from breast_benchmark.data import BUSIDataset
from breast_benchmark.train import evaluate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--model', choices=list(MODEL_REGISTRY), default='TransUNetMAESizeDeepHeads')
    p.add_argument('--classes', nargs='+', default=['normal','benign','malignant'])
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    model, _ = build_model(args.model, pretrained=False)
    configure_classification_classes(model, tuple(args.classes))
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'] if 'model' in payload else payload)
    device = torch.device(args.device)
    model.to(device)
    dataset = BUSIDataset(args.manifest, 'test', class_names=tuple(args.classes))
    evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device, False, args.output_root)

if __name__ == '__main__':
    main()
