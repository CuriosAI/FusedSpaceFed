"""Registry extension only: call the unchanged Digits scientific runner.

The first two runs retain their original config/entry point. Additional seeds
use an expanded registry whose only different fields are run_seeds and the
device map. No training implementation or hyperparameter is modified.
"""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from research.feature_shift_digits.run_digits import run, torch


def scientific_configuration(config):
    return {key:value for key,value in config.items() if key not in ('run_seeds','per_seed_device')}


def validate_extension(original,extended):
    if scientific_configuration(original)!=scientific_configuration(extended):
        raise ValueError('Extension may only change the seed/device registry')
    if extended['run_seeds']!=[42,43,44,45,46] or extended['per_seed_device']!={'42':'cuda:1','43':'cuda:0','44':'cuda:1','45':'cuda:0','46':'cuda:1'}:
        raise ValueError('Wrong five-run registry')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--partition',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(44,45,46),required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    original=json.loads((REPO/'research/feature_shift_digits/config.json').read_text())
    config=json.loads(args.config.read_text())
    validate_extension(original,config)
    if args.device!=config['per_seed_device'][str(args.seed)]:
        raise ValueError('Assigned GPU differs from registry')
    run(config,args.partition,args.output,args.seed,torch.device(args.device),resume=args.resume)
