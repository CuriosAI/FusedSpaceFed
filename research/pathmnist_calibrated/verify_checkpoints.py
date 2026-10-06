"""CPU-only full-model, buffer, optimizer, partition and initialization audit."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from fusedspacefed_core import ResNet20V2,UNetSmallAE,clone_state_dict
from research.pathmnist_pathological.run import restore_client, resident_loader, assert_finite, file_hash,write_json
from research.pathmnist_calibrated.client import CalibratedClient
from research.pathmnist_calibrated.data import PRIVATE,PUBLIC,verify


def main():
    torch.set_num_threads(1);selected=json.loads((PUBLIC/'selection.json').read_text());partition=verify()
    checks={}
    for seed in selected['final_seeds']:
        root=PRIVATE/'final'/f'seed-{seed}';rows={}
        for name,round_index in (('initial.pt',0),('native-final.pt',selected['rounds']),('final.pt',selected['rounds'])):
            s=torch.load(root/name,map_location='cpu',weights_only=False)
            assert_finite(s)
            assert s['round']==round_index and s['seed']==seed and s['partition']['partition_sha256']==partition['partition_sha256']
            assert s['training_indices']==partition['full']
            assert set(s['clients'])==set(s['encoders'])=={str(i) for i in range(10)}
            assert s['rng']['torch_cpu'].numel()>0 and len(s['rng']['torch_cuda'])==1
            model=ResNet20V2(9,3);model.load_state_dict(s['classifier'],strict=True);model.eval()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);expected_c=clone_state_dict(ResNet20V2(9,3).state_dict())
                torch.manual_seed(seed+1000000);expected_ae=UNetSmallAE(3,16)
                if not round_index:
                    assert all(torch.equal(v,s['classifier'][k]) for k,v in expected_c.items())
                    assert all(torch.equal(v,s['decoder'][k]) for k,v in expected_ae.decoder_state().items())
            for key in s['clients']:
                ae=UNetSmallAE(3,16);ae.load_state_dict({**s['encoders'][key],**s['decoder']},strict=True);ae.eval()
                x=torch.zeros(2,3,32,32)
                with torch.no_grad():
                    d,_=ae(x);logits=model(x+d)
                assert d.shape==x.shape and logits.shape==(2,9);assert_finite((d,logits))
                cfg={'seed':seed,'batch_size':128}
                loader=resident_loader(x,torch.zeros(2,dtype=torch.long),[0,1],cfg,int(key))
                client=CalibratedClient(int(key),loader,s['configuration']['settings'],torch.device('cpu'))
                restore_client(client,s['clients'][key])
                assert torch.equal(client.loader.generator.get_state(),s['clients'][key]['loader_generator'])
                for parameter,state in client.ae_optimizer.state.items():
                    assert state['exp_avg'].shape==state['exp_avg_sq'].shape==parameter.shape
                assert not client.classifier_optimizer.state
                if not round_index:
                    assert not client.ae_optimizer.state
                    assert all(torch.equal(v,s['encoders'][key][k]) for k,v in expected_ae.encoder_state().items())
                else:
                    assert len(client.ae_optimizer.state)==26
                assert all(torch.equal(v,s['clients'][key]['classifier'][k]) for k,v in s['classifier'].items())
                assert all(torch.equal(v,s['clients'][key]['decoder'][k]) for k,v in s['decoder'].items())
            bn=sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in s['classifier'])
            assert bn==57
            rows[name]={'round':round_index,'sha256':file_hash(root/name),'bytes':(root/name).stat().st_size,
                        'ten_encoders_strictly_loaded':True,'optimizer_moments_loaded_shapes_valid':True,
                        'batchnorm_buffers':bn,'rng_loader_and_device_mapping_valid':True,
                        'initialization_exact_original_seed_recipe':True if not round_index else None}
        checks[str(seed)]=rows
    write_json(PRIVATE/'checkpoint_verification.json',{'status':'passed','seeds':checks})
    print(json.dumps({'status':'passed','complete_checkpoint_count':3*len(selected['final_seeds']),'seed_count':len(selected['final_seeds'])},indent=2))


if __name__=='__main__':main()
