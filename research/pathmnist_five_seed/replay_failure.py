import sys, json, os, time
from pathlib import Path
sys.path.insert(0,'/mnt/data/codex/FusedSpaceFed')
import torch
import numpy as np
from fusedspacefed_core import FusedSpaceFedClient, seed_everything
from research.pathmnist_five_seed.data import cache,loaders,partition,PUBLIC
from research.pathmnist_pathological.run import restore_client,write_json,file_hash,cpu_copy

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    if Path('_local/pathmnist_five_seed/failure43_diagnosis.json').exists():
        raise FileExistsError('Preserve the original forensic output; copy harness to fresh paths for a replay')
    seed_everything(43);torch.set_num_threads(2);device=torch.device('cuda:0')
    path=Path('_local/pathmnist_five_seed/full/seed-43/latest.pt')
    s=torch.load(path,map_location='cpu',weights_only=False);images,labels=cache(device)
    settings=s['configuration']['settings'];parts,_=partition(43);all_loaders=loaders(images,labels,parts,settings,43)
    started=time.perf_counter();report={'checkpoint':str(path),'checkpoint_sha256':file_hash(path),'round_completed':s['round'],'settings':settings,'test_access':'none','phase':None,'client':None,'batch':0,'status':'replaying'}
    class Found(Exception):pass
    for cid in range(10):
        c=FusedSpaceFedClient(cid,all_loaders[str(cid)],9,3,16,'multiclass',device,use_amp=True)
        restore_client(c,s['clients'][str(cid)]);c.set_classifier_state(s['classifier']);c.set_decoder_state(s['decoder'])
        report['client']=cid
        def start(m,args):
            if m is c.autoencoder:report['batch']+=1
        pre=c.autoencoder.register_forward_pre_hook(start)
        hooks=[]
        def check(name):
            def hook(m,args,out):
                values=[out] if isinstance(out,torch.Tensor) else [v for v in out if isinstance(v,torch.Tensor)] if isinstance(out,tuple) else []
                for v in values:
                    if not bool(torch.isfinite(v).all()):
                        report.update(status='nonfinite_forward',first_nonfinite_module=name,dtype=str(v.dtype),shape=list(v.shape),nan=int(torch.isnan(v).sum()),inf=int(torch.isinf(v).sum()),max_finite_abs=float(v[torch.isfinite(v)].abs().max()) if torch.isfinite(v).any() else None)
                        snapshot={'classifier':cpu_copy(c.classifier.state_dict()),'autoencoder':cpu_copy(c.autoencoder.state_dict()),'ae_optimizer':cpu_copy(c.ae_optimizer.state_dict()),'classifier_optimizer':cpu_copy(c.classifier_optimizer.state_dict()),'scaler':c.scaler.state_dict(),'inputs_to_failing_module':cpu_copy(args),'report':report.copy()}
                        torch.save(snapshot,'_local/pathmnist_five_seed/failure43_forward.pt')
                        raise Found
            return hook
        for name,module in c.autoencoder.named_modules():hooks.append(module.register_forward_hook(check('autoencoder.'+name)))
        for name,module in c.classifier.named_modules():hooks.append(module.register_forward_hook(check('classifier.'+name)))
        try:
            for phase,fn,epochs in [('warmup',c._warmup,1),('classification',c._joint_train,3)]:
                report.update(phase=phase,batch=0);values=fn(epochs)
                if not np.isfinite(values).all():report.update(status='nonfinite_loss',values=values);raise Found
        except Found:
            report['seconds']=time.perf_counter()-started
            write_json(Path('_local/pathmnist_five_seed/failure43_diagnosis.json'),report)
            print(json.dumps(report),flush=True);break
        finally:
            pre.remove()
            for h in hooks:h.remove()
        del c
    else:
        report.update(status='finite_replay',seconds=time.perf_counter()-started)
        write_json(Path('_local/pathmnist_five_seed/failure43_diagnosis.json'),report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
