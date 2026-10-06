"""Each spawned client keeps its images, labels, and feature cache locally."""
import copy
import json
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fusedspacefed_core import ResNet20V2,UNetSmallAE,seed_everything
from research.pathmnist_pathological.run import file_hash,write_json,atomic_save,rng_state
from research.digits_mechanism_diagnostics.common import state_hash
from research.pathmnist_federated_head.protocol import (PARAMETERS,read_vector,reply,head_from,install,flatten)

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'_local/pathmnist_pathological/prepared'


class LocalObjective:
    def __init__(self,features,labels,initial):
        self.features=features.detach();self.labels=labels.detach();self.head=head_from(initial,features.device)
    def value_gradient(self,vector):
        install(self.head,vector.to(self.features.device));self.head.zero_grad(set_to_none=True)
        # Same per-example CE and local mean as the centralized weighted_ce.
        loss=F.cross_entropy(self.head(self.features),self.labels,reduction='none').mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite private CE')
        loss.backward();gradient=torch.cat([p.grad.detach().reshape(-1) for p in self.head.parameters()])
        return float(loss.detach()),gradient.cpu()


@torch.no_grad()
def extract_features(body,ae,images,batch_size=128):
    values=[]
    for begin in range(0,len(images),batch_size):
        x=images[begin:begin+batch_size];d,_=ae(x);values.append(body(x+d))
    result=torch.cat(values)
    if result.shape!=(len(images),64) or not torch.isfinite(result).all():raise FloatingPointError('Bad private features')
    return result.detach()


def logits_metrics(a,b,tolerance):
    delta=(a-b).double();violations=(a-b).abs()>tolerance['atol']+tolerance['rtol']*b.abs()
    return {'entries':a.numel(),'max_abs':float(delta.abs().max()),'squared_error_sum':float(delta.square().sum()),
            'sum_abs':float(delta.abs().sum()),'entries_outside_tolerance':int(violations.sum()),
            'prediction_disagreements':int((a.argmax(1)!=b.argmax(1)).sum()),'samples':len(a)}


def merge_metrics(values):
    return {'entries':sum(v['entries'] for v in values),'max_abs':max(v['max_abs'] for v in values),
            'rms':(sum(v['squared_error_sum'] for v in values)/sum(v['entries'] for v in values))**.5,
            'mean_abs':sum(v['sum_abs'] for v in values)/sum(v['entries'] for v in values),
            'squared_error_sum':sum(v['squared_error_sum'] for v in values),'sum_abs':sum(v['sum_abs'] for v in values),
            'entries_outside_tolerance':sum(v['entries_outside_tolerance'] for v in values),
            'prediction_disagreements':sum(v['prediction_disagreements'] for v in values),
            'samples':sum(v['samples'] for v in values)}


def worker(cid,pipe,config,output,physical_device):
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1];device=torch.device('cuda:0')
        torch.cuda.set_device(device);torch.set_num_threads(2);seed_everything(config['seed'])
        started=time.perf_counter();folder=output/'clients'/f'client-{cid}';folder.mkdir(parents=True,exist_ok=False)
        path=ROOT/config['checkpoint']
        if file_hash(path)!=config['checkpoint_sha256']:raise ValueError('Changed original source')
        state=torch.load(path,map_location='cpu',weights_only=False)
        for name,digest in state['data_manifest']['files_sha256'].items():
            if file_hash(DATA/name)!=digest:raise ValueError('Changed source cache '+name)
        original_client=state['clients'][str(cid)];ids=state['partitions']['clients'][str(cid)]
        body=ResNet20V2(9,3).to(device);body.load_state_dict(state['classifier']);body.fc=nn.Identity()
        body.eval().requires_grad_(False)
        ae=UNetSmallAE(3,16).to(device);ae.load_state_dict(original_client['autoencoder']);ae.eval().requires_grad_(False)
        original_head=torch.cat([state['classifier']['fc.weight'].reshape(-1),state['classifier']['fc.bias']])
        frozen={'body':state_hash(body.state_dict()),'autoencoder':state_hash(ae.state_dict())}
        del state
        # Only this client's training rows are materialized; never build a pooled cache.
        images=torch.from_numpy(np.load(DATA/'train-images.npy',mmap_mode='r')[ids].copy()).to(device)
        labels=torch.from_numpy(np.load(DATA/'train-labels.npy',mmap_mode='r')[ids].copy()).to(device)
        phase=time.perf_counter();features=extract_features(body,ae,images);feature_seconds=time.perf_counter()-phase
        del images
        cache_path=folder/'private_features.pt'
        atomic_save(cache_path,{'features':features.cpu(),'labels':labels.cpu(),'origin_indices':ids})
        local=LocalObjective(features,labels,original_head);calls=0
        pipe.send_bytes(b'R')  # status only; counts/data remain in private client files
        while True:
            message=pipe.recv_bytes()
            if message[:1]==b'G':
                vector=read_vector(message[1:]);loss,gradient=local.value_gradient(vector);calls+=1
                pipe.send_bytes(reply(loss,gradient))
            elif message[:1]==b'V':
                if len(message)!=1+2*PARAMETERS*4:raise ValueError('Wrong final audit message')
                fed=read_vector(message[1:1+PARAMETERS*4]);central=read_vector(message[1+PARAMETERS*4:])
                ha,hb,ho=[head_from(v,device).eval() for v in (fed,central,original_head)]
                with torch.no_grad():
                    train_metrics=logits_metrics(ha(features),hb(features),config['tolerances']['final_logits'])
                    test_images=torch.from_numpy(np.load(DATA/'test-images.npy')).to(device)
                    test_labels=torch.from_numpy(np.load(DATA/'test-labels.npy')).to(device)
                    comparisons=[];correct={'original':0,'centralized':0,'federated':0}
                    for begin in range(0,len(test_images),128):
                        x,y=test_images[begin:begin+128],test_labels[begin:begin+128]
                        d,_=ae(x);f=body(x+d);a,b,o=ha(f),hb(f),ho(f)
                        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all() & torch.isfinite(o).all()):
                            raise FloatingPointError('Nonfinite comparison logits')
                        comparisons.append(logits_metrics(a,b,config['tolerances']['final_logits']))
                        for name,value in (('original',o),('centralized',b),('federated',a)):
                            correct[name]+=int((value.argmax(1)==y).sum())
                after={'body':state_hash(body.state_dict()),'autoencoder':state_hash(ae.state_dict())}
                if frozen!=after:raise AssertionError('Client changed body/E/D/BN')
                # Original optimizers/scaler/generator retained; only fc changes.
                saved=copy.deepcopy(original_client)
                saved['classifier']['fc.weight']=fed[:9*64].reshape(9,64).clone()
                saved['classifier']['fc.bias']=fed[9*64:].clone()
                atomic_save(folder/'final.pt',{'seed':42,'round':50,'client':saved,'phase_rng':rng_state(True),
                                             'training_indices':ids,'configuration':config})
                atomic_save(folder/'phase_rng.pt',rng_state(True))
                torch.cuda.synchronize(device)
                result={'status':'completed','client_id':cid,'train_samples':len(ids),'test_samples':len(test_labels),
                        'feature_dimensions':list(features.shape),'feature_extraction_seconds':feature_seconds,
                        'loss_gradient_calls':calls,'train_logits_comparison':merge_metrics([train_metrics]),
                        'test_logits_comparison':merge_metrics(comparisons),'correct_counts':correct,
                        'accuracy_percent':{name:100*value/len(test_labels) for name,value in correct.items()},
                        'frozen_before':frozen,'frozen_after':after,'feature_cache_sha256':file_hash(cache_path),
                        'private_checkpoint_sha256':file_hash(folder/'final.pt'),
                        'wall_seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),
                        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),
                        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pid':os.getpid()}
                write_json(folder/'audit.json',result);pipe.send_bytes(b'K');return
            elif message==b'Q':return
            else:raise ValueError('Unknown head-only message')
    except BaseException:
        try:pipe.send_bytes(b'E'+traceback.format_exc().encode('utf8'))
        except (BrokenPipeError,EOFError):pass
        raise
    finally:pipe.close()
