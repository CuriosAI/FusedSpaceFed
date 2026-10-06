"""Training-only decoder probes and persistent-optimizer local replay on copies."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import torch
from research.pathmnist_five_seed.fp32.precision import PUBLIC,force_fp32,check_client_precision
from fusedspacefed_core import seed_everything
from research.pathmnist_pathological.run import (restore_client,client_snapshot,restore_rng,rng_state,
                                                 file_hash,write_json,atomic_save,assert_finite)
from research.pathmnist_five_seed.client import PathClient
from research.pathmnist_five_seed.data import cache,loaders,partition,verify_cache
from research.pathmnist_five_seed.fp32.diagnostic_common import (ANCHORS,load_anchor,probe,source_identity,
                                                           measure,component_states,deltas,state_hash)


@torch.no_grad()
def visuals(client,images,indices):
    old=client.autoencoder.training;client.autoencoder.eval()
    try:
        x=images[torch.tensor(indices,device=images.device)];d,_=client.autoencoder(x)
        assert_finite([x,d]);return {'input':x.cpu(),'decoder':d.cpu(),'fused':(x+d).cpu()}
    finally:client.autoencoder.train(old)


def figure(path,cid,anchor,stages,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[('before','input')]+[(s,k) for s in ('before','after_warmup','after_classification') for k in ('decoder','fused')]
    fig,axes=plt.subplots(7,4,figsize=(8,11))
    for r,(stage,signal) in enumerate(rows):
        for c,label in enumerate(labels):
            axes[r,c].imshow(stages[stage][signal][c].permute(1,2,0).clamp(0,1).numpy())
            axes[r,c].set_xticks([]);axes[r,c].set_yticks([])
            if r==0:axes[r,c].set_title(f'class {label}')
            if c==0:axes[r,c].set_ylabel(f'{stage}\n{signal}',fontsize=8)
    fig.suptitle(f'PathMNIST seed 42, client {cid}, {anchor}\nFixed [0,1] display; raw amplitudes recorded separately',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(path,dpi=110);plt.close(fig)


def run(seed,physical_device,output):
    # The controller passes repository-relative destinations; checkpoint
    # metadata below is serialized relative to the absolute repository root.
    output=output.resolve()
    force_fp32();os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1];device=torch.device('cuda:0')
    torch.cuda.set_device(device);torch.set_num_threads(2);seed_everything(seed)
    verify_cache();parts,_=partition(seed);spec=probe(seed);settings=json.loads((PUBLIC/'plan.json').read_text())['settings']
    if output.exists():raise FileExistsError('Preserve diagnostic attempt')
    output.mkdir(parents=True);started=time.perf_counter();images,labels=cache(device)
    result={'seed':seed,'status':'running','physical_device':physical_device,'configuration':settings,
            'source':source_identity(('research/pathmnist_five_seed/fp32/phase1/diagnose.py',)),
            'probe_sha256':file_hash(PUBLIC/'probe.json'),'test_access':'none','precision':'FP32; no AMP/scaler/TF32','anchors':{}}
    for anchor in ANCHORS:
        state,parent=load_anchor(seed,anchor);anchor_folder=output/anchor;anchor_folder.mkdir()
        rows={};before_source={k:state_hash(state[k]) for k in ('classifier','decoder')}
        for cid in range(10):
            loader=loaders(images,labels,{'clients':{str(cid):parts['clients'][str(cid)]}},settings,seed)[str(cid)]
            c=PathClient(cid,loader,settings,device);restore_client(c,state['clients'][str(cid)]);check_client_precision(c)
            # Native models contain no dropout or random augmentation. Restore process
            # RNG too; original seed42 stored a separate RNG for each GPU worker.
            saved_rng=state.get('rng')
            if saved_rng is None:
                worker='0' if cid in (1,5,7,8,9) else '1';saved_rng=state['workers'][worker]['rng']
            restore_rng(saved_rng)
            cp={'precision':'FP32; no AMP/scaler/TF32','configuration':settings,'source':result['source'],'seed':seed,'anchor':anchor,'client_id':cid,'source_checkpoint':parent,
                'partition':parts,'probe_indices':spec['clients'][str(cid)],'stages':{}}
            before=component_states(c);cp['stages']['before']={'client':client_snapshot(c),'rng':rng_state(True)}
            ids=spec['clients'][str(cid)]['indices'];metrics={'before':measure(c,images,labels,ids)}
            grids={'before':visuals(c,images,spec['clients'][str(cid)]['visual_indices'])} if seed==42 else {}
            c.phase='warmup';warm_loss=c._warmup(1);warm=component_states(c)
            metrics['after_warmup']=measure(c,images,labels,ids)
            cp['stages']['after_warmup']={'client':client_snapshot(c),'rng':rng_state(True),'actual_optimizer_steps':c.actual.copy()}
            if seed==42:grids['after_warmup']=visuals(c,images,spec['clients'][str(cid)]['visual_indices'])
            c.phase='classification';ce_loss=c._joint_train(3);after=component_states(c)
            metrics['after_classification']=measure(c,images,labels,ids)
            cp['stages']['after_classification']={'client':client_snapshot(c),'rng':rng_state(True),'actual_optimizer_steps':c.actual.copy()}
            change={'warmup':deltas(before,warm),'classification':deltas(warm,after)}
            if change['warmup']['classifier']['all_state_l2'] or change['warmup']['decoder']['all_state_l2']:
                raise AssertionError('Warm-up changed shared state')
            path=anchor_folder/f'client-{cid}-phases.pt';assert_finite(cp);atomic_save(path,cp)
            if seed==42:
                grids['after_classification']=visuals(c,images,spec['clients'][str(cid)]['visual_indices'])
                figure(output/f'{anchor}-client-{cid}.png',cid,anchor,grids,spec['clients'][str(cid)]['visual_labels'])
            metrics.update(state_l2_changes=change,warmup_loss=statistics.mean(warm_loss),classification_loss=statistics.mean(ce_loss),
                           warmup_steps=len(warm_loss),classification_steps=len(ce_loss),actual_optimizer_steps=c.actual.copy(),
                           replay_checkpoint={'path':str(path.relative_to(ROOT)),'sha256':file_hash(path),'bytes':path.stat().st_size})
            check_client_precision(c);assert_finite(metrics);rows[str(cid)]=metrics;del c,cp,before,warm,after
        if before_source!={k:state_hash(state[k]) for k in ('classifier','decoder')}:raise AssertionError('Source mutated')
        result['anchors'][anchor]={'source_checkpoint':parent,'clients':rows};write_json(output/'results.json',result)
        print(json.dumps({'seed':seed,'anchor':anchor,'clients':10}),flush=True);del state
    torch.cuda.synchronize(device)
    result.update(status='completed',wall_seconds=time.perf_counter()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert_finite(result);write_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.seed,a.device,a.output)
