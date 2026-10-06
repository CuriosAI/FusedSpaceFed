"""PathMNIST original/fused gradients; reuse audited Digits BN/Gram controls."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import torch
from research.pathmnist_five_seed.fp32.precision import PUBLIC,force_fp32
from fusedspacefed_core import ResNet20V2,UNetSmallAE,seed_everything
from research.pathmnist_five_seed.data import cache,verify_cache
from research.pathmnist_five_seed.fp32.diagnostic_common import ANCHORS,probe,load_anchor,source_identity,state_hash
from research.pathmnist_pathological.run import assert_finite,file_hash,write_json
from research.digits_mechanism_diagnostics.phase3.probe_gradients import paired_gradients
from research.digits_mechanism_diagnostics.phase3.gradient_stats import decompose,dispersion


def measure_state(state,batches,mode,device):
    model=ResNet20V2(9,3).to(device);model.load_state_dict(state['classifier'])
    ae=UNetSmallAE(3,16).to(device);ae.load_decoder_state(state['decoder'])
    before={'classifier':state_hash(model.state_dict()),'decoder':state_hash(ae.decoder_state())}
    rng=torch.get_rng_state();crng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    cdim=sum(p.numel() for p in model.parameters());ddim=sum(p.numel() for p in ae.decoder_parameters())
    cids=list(batches);count=len(cids);n=len(batches[cids[0]])
    om=[torch.zeros(cdim,dtype=torch.float64) for _ in cids];fm=[v.clone() for v in om]
    dm=[torch.zeros(ddim,dtype=torch.float64) for _ in cids];per_batch=[];per_client={cid:[] for cid in cids}
    for b in range(n):
        originals=[];fused=[];decoders=[]
        for i,cid in enumerate(cids):
            ae.load_encoder_state(state['encoders'][cid]);x,y=batches[cid][b]
            o,f,d,m=paired_gradients(model,ae,x,y,batch_norm_mode=mode)
            originals.append(o);fused.append(f);decoders.append(d)
            om[i].add_(o.double(),alpha=1/n);fm[i].add_(f.double(),alpha=1/n);dm[i].add_(d.double(),alpha=1/n)
            m.update(batch=b+1,samples=len(y));per_client[cid].append(m)
        per_batch.append({'batch':b+1,'classifier':decompose(originals,fused),'decoder':dispersion(decoders)})
    after={'classifier':state_hash(model.state_dict()),'decoder':state_hash(ae.decoder_state())}
    unchanged=torch.equal(rng,torch.get_rng_state()) and (crng is None or torch.equal(crng,torch.cuda.get_rng_state(device)))
    if before!=after or not unchanged:raise AssertionError('Probe mutated parameters, BN buffers or RNG')
    return {'classifier':decompose(om,fm),'decoder':dispersion(dm),'per_batch':per_batch,'per_client_batches':per_client,
            'state_before':before,'state_after':after,'rng_unchanged':unchanged,'batch_norm_mode':mode,
            'gradient_definition':f'Mean of {n} equal batch CE gradients/client, uniform population dispersion over {count} clients'}


def run(seed,physical_device,output):
    force_fp32();os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1];device=torch.device('cuda:0')
    torch.cuda.set_device(device);torch.set_num_threads(2);seed_everything(seed)
    verify_cache();spec=probe(seed)
    if output.exists():raise FileExistsError('Preserve gradient attempt')
    output.mkdir(parents=True);started=time.perf_counter();images,labels=cache(device);batches={}
    for cid,row in spec['clients'].items():
        ids=torch.tensor(row['indices'],device=device);x,y=images[ids],labels[ids]
        batches[cid]=[(x[i:i+128],y[i:i+128]) for i in range(0,640,128)]
    del images,labels
    result={'seed':seed,'status':'running','physical_device':physical_device,'test_access':'none','precision':'FP32; no AMP/scaler/TF32',
            'probe_sha256':file_hash(PUBLIC/'probe.json'),'anchors':{},
            'source':source_identity(('research/pathmnist_five_seed/fp32/phase3/probe_gradients.py',
                                     'research/digits_mechanism_diagnostics/phase3/probe_gradients.py',
                                     'research/digits_mechanism_diagnostics/phase3/gradient_stats.py'))}
    for anchor in ANCHORS:
        state,parent=load_anchor(seed,anchor);result['anchors'][anchor]={'source_checkpoint':parent,'modes':{}}
        for mode in ('eval','batch-stateless'):
            value=measure_state(state,batches,mode,device);assert_finite(value)
            result['anchors'][anchor]['modes'][mode]=value;write_json(output/'results.json',result)
            print(json.dumps({'seed':seed,'anchor':anchor,'mode':mode,'Gamma_original':value['classifier']['Gamma_original'],
                              'Gamma_fused':value['classifier']['Gamma_fused']}),flush=True)
        del state
    torch.cuda.synchronize(device)
    result.update(status='completed',wall_seconds=time.perf_counter()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert_finite(result);write_json(output/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.seed,a.device,a.output)
