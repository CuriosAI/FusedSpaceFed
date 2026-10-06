"""One fixed federated-simulated head refit; no training data at the server."""
import argparse
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from fusedspacefed_core import seed_everything
from research.pathmnist_pathological.run import (file_hash,write_json,atomic_save,assert_finite,rng_state,cpu_copy)
from research.pathmnist_head_only.run import (replace_head_only,check_complete,tree_equal,validate)
from research.pathmnist_federated_head.client import worker,merge_metrics
from research.pathmnist_federated_head.protocol import (WireLedger,RemoteOracle,flatten,optimize,central_gradient_anchor,vector_bytes)

PUBLIC=ROOT/'research/pathmnist_federated_head'
SOURCES=('fusedspacefed_core.py','research/pathmnist_pathological/run.py',
         'research/pathmnist_head_only/run.py','research/pathmnist_head_only/config.json',
         'research/digits_mechanism_diagnostics/common.py',
         'research/pathmnist_federated_head/protocol.py','research/pathmnist_federated_head/client.py',
         'research/pathmnist_federated_head/run.py','research/pathmnist_federated_head/config.json')


def head_vector(state):return torch.cat([state['fc.weight'].reshape(-1),state['fc.bias']])


def scalar_comparison(actual,reference,tolerance):
    error=abs(actual-reference);bound=tolerance['atol']+tolerance['rtol']*abs(reference)
    return {'actual':actual,'reference':reference,'abs_error':error,'allowed_error':bound,'within_tolerance':error<=bound}


def gradient_comparison(actual,reference,tolerance):
    a,b=actual.double(),reference.double();d=a-b;bound=tolerance['atol']+tolerance['rtol']*b.abs()
    return {'max_abs_error':float(d.abs().max()),'l2_error':float(d.norm()),'reference_l2_norm':float(b.norm()),
            'actual_l2_norm':float(a.norm()),'relative_l2_error':float(d.norm()/b.norm()) if b.norm() else None,
            'entries_outside_tolerance':int((d.abs()>bound).sum()),'within_tolerance':bool((d.abs()<=bound).all())}


def main(config_path,output,physical_device):
    started=time.perf_counter();os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1]
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2);seed_everything(42)
    config=json.loads(config_path.read_text());source=ROOT/config['checkpoint'];central_path=ROOT/config['centralized_checkpoint']
    if file_hash(source)!=config['checkpoint_sha256'] or file_hash(central_path)!=config['centralized_checkpoint_sha256']:
        raise ValueError('Changed frozen original/reference checkpoints')
    if file_hash(ROOT/config['centralized_results'])!=config['centralized_results_sha256']:raise ValueError('Changed reference results')
    original=torch.load(source,map_location='cpu',weights_only=False);central=torch.load(central_path,map_location='cpu',weights_only=False)
    validate(config,original,{'full':original['partitions']['clients']});check_complete(original,central)
    if config['client_count']!=10 or config['head_parameters']!=585:raise ValueError('Wrong original head/client count')
    central_results=json.loads((ROOT/config['centralized_results']).read_text())
    central_opt=central['head_only_refit']['head_optimizer'];central_stats=central['head_only_refit']['statistics']
    if central_stats['actual_iterations']!=100 or central_stats['final_uniform_client_training_ce']!=central_stats['objective_history'][-1]:
        raise ValueError('Reference last-step anchor is not an accepted terminal update')
    initial=head_vector(original['classifier']);central_final=head_vector(central['classifier'])
    pre_last,central_gradient,central_loss=central_gradient_anchor(central_final,central_opt)
    if central_loss!=central_stats['objective_history'][-2]:raise ValueError('Cannot locate stored central gradient point')
    code={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_sha256':{name:file_hash(ROOT/name) for name in SOURCES},'config_sha256':file_hash(config_path)}
    if output.exists():raise FileExistsError('Preserve previous attempt')
    output.mkdir(parents=True);shutil.copyfile(source,output/'before.pt')
    shutil.copyfile(ROOT/'_local/pathmnist_pathological/seed-42/initial.pt',output/'training-initial.pt')
    processes=[];pipes=[];ledger=WireLedger();ctx=mp.get_context('spawn')
    try:
        for cid in range(10):
            parent,child=ctx.Pipe();process=ctx.Process(target=worker,args=(cid,child,config,output,physical_device),
                                                      name=f'pathmnist-head-client-{cid}')
            process.start();child.close();pipes.append(parent);processes.append(process)
        for cid,pipe in enumerate(pipes):
            if not pipe.poll(240):raise TimeoutError('Client feature initialization timeout')
            message=pipe.recv_bytes();ledger.record('control','up',message)
            if message!=b'R':raise RuntimeError('Client initialization failure: '+message.decode('utf8',errors='replace'))
            print(json.dumps({'client_ready':cid,'pid':processes[cid].pid}),flush=True)
        feature_setup_seconds=time.perf_counter()-started
        oracle=RemoteOracle(pipes,device,ledger)
        def progress(number,row):
            if number==1 or number%10==0:
                write_json(output/'progress.json',{'stage':'head-refit','closure':number,'uniform_client_ce':row['loss']})
                print(json.dumps({'closure':number,'uniform_client_ce':row['loss']}),flush=True)
        began=time.perf_counter();vector,optimizer,trace=optimize(initial,oracle,config['lbfgs'],config['head_max_iter'],device,progress)
        optimization_seconds=time.perf_counter()-began
        opt_state=next(iter(optimizer['state'].values()))
        # Fixed training-only audits, after the optimizer has finished; no further update.
        final_loss,final_gradient=oracle.evaluate(vector,'verification')
        c_loss,c_gradient=oracle.evaluate(central_final,'verification')
        p_loss,p_gradient=oracle.evaluate(pre_last,'verification')
        comparisons={
            'initial_loss':scalar_comparison(trace[0]['loss'],central_stats['objective_history'][0],config['tolerances']['same_point_loss']),
            'central_final_loss_same_head':scalar_comparison(float(c_loss),central_stats['final_uniform_client_training_ce'],config['tolerances']['same_point_loss']),
            'central_pre_last_loss':scalar_comparison(float(p_loss),central_loss,config['tolerances']['same_point_loss']),
            'central_pre_last_gradient':gradient_comparison(p_gradient.cpu(),central_gradient,config['tolerances']['stored_gradient_anchor']),
            'final_optimized_loss':scalar_comparison(float(final_loss),float(c_loss),config['tolerances']['final_loss']),
            'final_optimized_gradient':gradient_comparison(final_gradient.cpu(),c_gradient.cpu(),config['tolerances']['final_gradient'])}
        atomic_save(output/'gradient_audit.pt',{'central_pre_last_theta':pre_last,'central_pre_last_gradient_stored':central_gradient,
                                             'central_pre_last_gradient_federated':p_gradient.cpu(),
                                             'central_final_gradient_federated':c_gradient.cpu(),'federated_final_gradient':final_gradient.cpu()})
        atomic_save(output/'closure_trace.pt',trace)
        fitted=dict(original['classifier']);fitted['fc.weight']=vector[:9*64].reshape(9,64).clone();fitted['fc.bias']=vector[9*64:].clone()
        final=replace_head_only(original,fitted)
        statistics={'actual_iterations':opt_state['n_iter'],'function_evaluations':opt_state['func_evals'],
                    'closure_evaluations':len(trace),'objective_history':[t['loss'] for t in trace],
                    'final_uniform_client_training_ce':float(final_loss),'final_gradient_l2_norm':float(final_gradient.norm()),
                    'penalty':0.,'max_iter':100}
        final['federated_head_refit']={'configuration':config,'code':code,'head_optimizer':cpu_copy(optimizer),
                                     'phase_rng':rng_state(True),'statistics':statistics,
                                     'communication_before_test':ledger.summary(),'training_aggregations':dict(oracle.aggregations),
                                     'formula':'x+D(E_i(x))','source_checkpoint_sha256':file_hash(source),
                                     'reference_checkpoint_sha256':file_hash(central_path)}
        atomic_save(output/'final.pt',final);check_complete(original,torch.load(output/'final.pt',map_location='cpu',weights_only=False))
        print(json.dumps({'checkpoint_frozen':str(output/'final.pt'),'before_test':True}),flush=True)
        # The optimizer server receives only a done status. Numeric audit metadata
        # is written locally and read separately by the report assembler.
        message=b'V'+vector_bytes(vector)+vector_bytes(central_final);began=time.perf_counter()
        for pipe in pipes:pipe.send_bytes(message);ledger.record('inference_audit','down',message)
        for pipe in pipes:
            if not pipe.poll(240):raise TimeoutError('Client final audit timeout')
            response=pipe.recv_bytes();ledger.record('inference_audit','up',response)
            if response!=b'K':raise RuntimeError('Client audit failure: '+response.decode('utf8',errors='replace'))
        for process in processes:
            process.join(timeout=15)
            if process.exitcode!=0:raise RuntimeError('Client process failed')
        audit_seconds=time.perf_counter()-began
        clients=[json.loads((output/'clients'/f'client-{cid}'/'audit.json').read_text()) for cid in range(10)]
        train_logits=merge_metrics([c['train_logits_comparison'] for c in clients]);test_logits=merge_metrics([c['test_logits_comparison'] for c in clients])
        correct={name:sum(c['correct_counts'][name] for c in clients) for name in ('original','centralized','federated')}
        accuracy={name:100*n/71800 for name,n in correct.items()}
        if [c['correct_counts']['original'] for c in clients]!=[r['correct'] for r in central_results['metrics_before']['pipeline_metrics']]:
            raise AssertionError('Original accuracy not reproduced')
        if [c['correct_counts']['centralized'] for c in clients]!=[r['correct'] for r in central_results['metrics_after']['pipeline_metrics']]:
            raise AssertionError('Centralized 75.33% counts not reproduced')
        accuracy_error=abs(accuracy['federated']-accuracy['centralized'])
        comparisons['final_test_accuracy']={'actual_percent':accuracy['federated'],'reference_percent':accuracy['centralized'],
                                            'abs_error_percentage_points':accuracy_error,
                                            'within_tolerance':accuracy_error<=config['tolerances']['accuracy_percentage_points_atol']}
        comparisons['final_train_logits']={'within_tolerance':train_logits['entries_outside_tolerance']==0,**train_logits}
        comparisons['final_test_logits']={'within_tolerance':test_logits['entries_outside_tolerance']==0,**test_logits}
        disagreement=test_logits['prediction_disagreements']/71800
        comparisons['test_predictions']={'disagreement_fraction':disagreement,
                                         'within_tolerance':disagreement<=config['tolerances']['test_prediction_disagreement_fraction_max']}
        # Preserve phase RNGs in the complete checkpoint without changing weights.
        final['federated_head_refit']['client_phase_rng']={str(cid):torch.load(output/'clients'/f'client-{cid}'/'phase_rng.pt',weights_only=False,map_location='cpu') for cid in range(10)}
        final['federated_head_refit']['communication']=ledger.summary();atomic_save(output/'final.pt',final)
        check_complete(original,torch.load(output/'final.pt',map_location='cpu',weights_only=False))
        for name,digest in code['source_sha256'].items():
            if file_hash(ROOT/name)!=digest:raise ValueError('Sources changed during refit')
        result={'status':'completed','seed':42,'source_round':50,'configuration':config,'code':code,
                'accuracy_percent':accuracy,'correct_counts':correct,'predictions_total':71800,'comparisons':comparisons,
                'all_declared_tolerances_met':all(v['within_tolerance'] for v in comparisons.values()),
                'head_statistics':statistics,'communication':ledger.summary(),'aggregations':dict(oracle.aggregations),
                'clients':clients,'feature_setup_wall_seconds':feature_setup_seconds,'optimization_seconds':optimization_seconds,
                'inference_audit_seconds':audit_seconds,'wall_seconds':time.perf_counter()-started,
                'server_peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),
                'server_peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),
                'server_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'physical_device':physical_device,
                'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                           'gpu':torch.cuda.get_device_name(device),'python':sys.version},
                'checkpoint_sha256':{p.name:file_hash(p) for p in output.glob('*.pt')},
                'client_exit_codes':[p.exitcode for p in processes],'completed_utc':datetime.now(timezone.utc).isoformat()}
        assert_finite(result);write_json(output/'results.json',result)
        print(json.dumps({'status':'completed','accuracy_percent':accuracy,'all_tolerances_met':result['all_declared_tolerances_met'],
                          'aggregations':result['aggregations'],'bytes':result['communication']['payload_total_bytes']}),flush=True)
        return result
    finally:
        for process,pipe in zip(processes,pipes):
            if process.is_alive():
                try:pipe.send_bytes(b'Q');ledger.record('control','down',b'Q')
                except (BrokenPipeError,EOFError,OSError):pass
            pipe.close()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():process.terminate();process.join()  # only this attempt's own child


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True)
    a=p.parse_args();main(a.config,a.output,a.device)
