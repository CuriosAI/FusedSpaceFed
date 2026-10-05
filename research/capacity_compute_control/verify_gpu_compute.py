"""Synthetic-only GPU spot check of the frozen CPU FLOP signatures."""
import argparse
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from research.capacity_compute_control.compute import profile_step,torch
from research.feature_shift_digits.data import atomic_json

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--device',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Verification output exists')
    profile=json.loads(args.profile.read_text());checks={};started=time.perf_counter()
    for batch in (2,32):
        observed=profile_step(batch,profile['settings'],torch.device(args.device))
        assert observed==profile['batches'][str(batch)],'GPU/CPU training cost signature differs'
        checks[str(batch)]=observed
    atomic_json(args.output,{'status':'passed','input':'synthetic only','device':args.device,
                           'profile_sha256':profile['profile_sha256'],'batches':checks,'wall_seconds':time.perf_counter()-started})
    print(json.dumps({'status':'passed','device':args.device,'batches':[2,32],'wall_seconds':time.perf_counter()-started}))
