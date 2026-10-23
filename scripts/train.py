"""Run joint training from the bundled JSON preset, with normal CLI overrides."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from breast_benchmark.train import parser, train_one, run_all

def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=ROOT/'configs/joint_mask30.json')
    known, remaining = config_parser.parse_known_args()
    cli = parser()
    config = json.loads(known.config.read_text())
    unknown = set(config) - {a.dest for a in cli._actions}
    if unknown:
        raise ValueError(f'Unknown training config options: {sorted(unknown)}')
    cli.set_defaults(**config)
    args = cli.parse_args(remaining)
    if args.model == 'all':
        run_all(args)
    else:
        train_one(args)

if __name__ == '__main__':
    main()
