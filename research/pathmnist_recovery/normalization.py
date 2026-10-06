"""Shared BN calibration from fit-only images; parameters and RNG unchanged."""
import math
import numpy as np
import torch
from fusedspacefed_core import ResNet20V2, UNetSmallAE, clone_state_dict
from research.pathmnist_pathological.run import rng_state, restore_rng, assert_finite


def assignments(partition, cross):
    rows = [i for cid in range(10) for i in partition['bn_calibration'][str(cid)]]
    if cross:
        rows = np.asarray(rows)[np.random.default_rng(20261006).permutation(len(rows))].tolist()
        groups = {str(cid): rows[cid::10] for cid in range(10)}
    else:
        groups = partition['bn_calibration']
    if len(set(rows)) != len(rows):
        raise ValueError('BN pool duplicates examples')
    fit = {i for x in partition['fit'].values() for i in x}
    if not set(rows) <= fit:
        raise ValueError('BN calibration accessed non-fit images')
    return groups


@torch.no_grad()
def fused_pool(decoder, encoders, images, partition, device, cross, batch_size=128):
    ae = UNetSmallAE(3,16).to(device);ae.load_decoder_state(decoder);ae.eval()
    blocks=[]
    for cid,rows in assignments(partition,cross).items():
        ae.load_encoder_state(encoders[cid])
        for begin in range(0,len(rows),batch_size):
            x=images[torch.tensor(rows[begin:begin+batch_size],device=device)]
            d,_=ae(x);blocks.append(x+d)
    result=torch.cat(blocks)
    order=np.random.default_rng(20261006).permutation(len(result))
    return result[torch.tensor(order,device=device)]


@torch.no_grad()
def calibrate_layerwise(model, inputs, batch_size=128):
    """Topological calibration: preceding BN layers already use final eval stats."""
    model.eval()
    bns=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        total=torch.zeros(bn.num_features,device=inputs.device,dtype=torch.float64)
        square=torch.zeros_like(total);count=0
        def collect(module,args):
            nonlocal count
            x=args[0].detach().to(torch.float64)
            dims=(0,)+tuple(range(2,x.ndim))
            total.add_(x.sum(dims));square.add_((x*x).sum(dims))
            count+=x.numel()//x.shape[1]
        hook=bn.register_forward_pre_hook(collect)
        try:
            for start in range(0,len(inputs),batch_size):model(inputs[start:start+batch_size])
        finally:hook.remove()
        if count<2:raise ValueError('Insufficient BN calibration data')
        mean=total/count
        variance=(square/count-mean*mean).clamp_min(0)*count/(count-1)
        bn.running_mean.copy_(mean.to(bn.running_mean.dtype))
        bn.running_var.copy_(variance.to(bn.running_var.dtype))
        bn.num_batches_tracked.fill_(math.ceil(len(inputs)/batch_size))
    model.eval()


@torch.no_grad()
def calibrate(classifier, decoder, encoders, images, partition, device, mode):
    if mode=='native':return clone_state_dict(classifier)
    if mode not in ('owner-cumulative','owner-layerwise','cross-cumulative','cross-layerwise'):
        raise ValueError('Undeclared normalization')
    rng=rng_state(cuda=device.type=='cuda')
    try:
        model=ResNet20V2(9,3).to(device);model.load_state_dict(classifier)
        fused=fused_pool(decoder,encoders,images,partition,device,mode.startswith('cross'))
        if mode.endswith('layerwise'):
            calibrate_layerwise(model,fused)
        else:
            for bn in model.modules():
                if isinstance(bn,torch.nn.modules.batchnorm._BatchNorm):
                    bn.reset_running_stats();bn.momentum=None
            model.train()
            for start in range(0,len(fused),128):model(fused[start:start+128])
        result=clone_state_dict(model.state_dict());assert_finite(result)
        for key,value in classifier.items():
            if not key.endswith(('running_mean','running_var','num_batches_tracked')):
                if not torch.equal(value,result[key]):raise AssertionError('Changed trained weights')
        return result
    finally:restore_rng(rng)
