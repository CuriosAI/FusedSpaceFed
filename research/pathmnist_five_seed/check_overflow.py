import os,sys,json
from pathlib import Path
sys.path.insert(0,'/mnt/data/codex/FusedSpaceFed')
import torch
from fusedspacefed_core import UNetSmallAE
from research.pathmnist_pathological.run import write_json

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    if json.loads(Path('_local/pathmnist_five_seed/failure43_diagnosis.json').read_text()).get('same_layer_float32'):
        raise FileExistsError('Preserve the original Float32 proof; reproduce with fresh paths')
    s=torch.load('_local/pathmnist_five_seed/failure43_forward.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2);m=UNetSmallAE(3,16).cuda();m.load_state_dict(s['autoencoder'])
    layer=dict(m.named_modules())['bott.2'];x=s['inputs_to_failing_module'][0].cuda().float()
    with torch.no_grad(),torch.autocast('cuda',enabled=False):out=layer(x)
    r=json.loads(Path('_local/pathmnist_five_seed/failure43_diagnosis.json').read_text())
    r['same_layer_float32']={'finite':bool(torch.isfinite(out).all()),'max_abs':float(out.abs().max()),'float16_max':torch.finfo(torch.float16).max,'count_above_float16_max':int((out.abs()>torch.finfo(torch.float16).max).sum()),'input_finite':bool(torch.isfinite(x).all())}
    write_json(Path('_local/pathmnist_five_seed/failure43_diagnosis.json'),r);print(json.dumps(r))


if __name__=='__main__':
    main()
