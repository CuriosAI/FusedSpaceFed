"""Audit and losslessly archive original runs and three diagnostic phases."""
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics
import torch
from research.pathmnist_five_seed.data import partition
from research.pathmnist_five_seed.fp32.precision import ROOT,PUBLIC,PRIVATE
from research.pathmnist_five_seed.fp32.diagnostic_common import (run_path,stats,ANCHORS,STAGES,state_hash)
from research.pathmnist_pathological.run import assert_finite,file_hash,write_json,json_hash


def compress(source,destination):
    raw=source.read_bytes();destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(gzip.compress(raw,mtime=0))
    if gzip.decompress(destination.read_bytes())!=raw:raise AssertionError('Lossy numeric archive')


def source_check(code):
    for name,digest in code['source_sha256'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Changed scientific source '+name)


def audit_run(seed,variant):
    path=run_path(seed,variant);raw=json.loads((path/'results.json').read_text());assert_finite(raw)
    old=False;parts,spec=partition(seed)
    if raw['status']!='completed' or raw['seed']!=seed or raw.get('rounds',raw.get('completed_rounds'))!=50:
        raise ValueError('Incomplete original/ablation run')
    if raw['test_evaluations']!=1 or raw['test_round']!=50:raise ValueError('Changed test protocol')
    if [r['round'] for r in raw['history']]!=list(range(1,51)):raise ValueError('Missing/duplicate round')
    if old:
        rows=[{'client_id':r['client_id'],'correct':r['correct'],'total':r['samples'],'accuracy_percent':100*r['accuracy']} for r in raw['client_test_metrics']]
        accuracy=raw['test_metrics']['accuracy']*100
    else:
        if raw['configuration']['variant']!=variant:raise ValueError('Wrong intervention')
        rows=raw['metrics']['pipeline_metrics'];accuracy=raw['metrics']['uniform_pipeline_accuracy_percent']
        source_check(raw['code'])
    if [r['client_id'] for r in rows]!=list(range(10)) or any(r['total']!=7180 for r in rows):raise ValueError('Wrong pipelines/counts')
    if any(abs(r['accuracy_percent']-100*r['correct']/r['total'])>1e-10 for r in rows):raise ValueError('Wrong client accuracy')
    expected=100*sum(r['correct'] for r in rows)/71800
    if abs(accuracy-expected)>1e-10:raise ValueError('Incorrect primary metric')
    cps={}
    initial=None
    for filename,number in (('initial.pt',0),('final.pt',50)):
        p=path/filename;state=torch.load(p,map_location='cpu',weights_only=False);assert_finite(state)
        if file_hash(p)!=raw['checkpoint_sha256'][filename]:raise ValueError('Changed checkpoint')
        if state['seed']!=seed or state['round']!=number or state['partitions']!=parts or state['partition_sha256']!=spec['sha256']:
            raise ValueError('Wrong checkpoint/indices')
        cfgkey='config' if old else 'configuration'
        if state['config_sha256']!=json_hash(state[cfgkey]):raise ValueError('Wrong config hash')
        source_check(state['code'])
        expected_clients={str(i) for i in range(10)}
        if set(state['clients'])!=expected_clients or set(state['encoders'])!=expected_clients:raise ValueError('Incomplete clients')
        if len(state['clients'])!=10 or len(state['encoders'])!=10:raise ValueError('Incomplete checkpoint')
        for cid,c in state['clients'].items():
            for name in ('classifier','decoder','encoder','autoencoder','classifier_optimizer','ae_optimizer','scaler','loader_generator'):
                if name not in c:raise ValueError('Incomplete client state')
            if state_hash(c['classifier'])!=state_hash(state['classifier']) or state_hash(c['decoder'])!=state_hash(state['decoder']):
                raise ValueError('Unsynchronized shared state in initial/final checkpoint')
            if state_hash(c['encoder'])!=state_hash(state['encoders'][cid]):raise ValueError('Wrong private encoder')
            if number==50 and not c['ae_optimizer']['state']:raise ValueError('Missing persistent Adam moments')
            if c['scaler'] is not None or state['configuration']['settings']['use_amp'] is not False:
                raise ValueError('Non-FP32 checkpoint')
            for component in ('classifier','autoencoder'):
                if any(v.is_floating_point() and v.dtype!=torch.float32 for v in c[component].values()):
                    raise ValueError('Non-FP32 state tensor')
        if filename=='initial.pt':initial=state
        cps[str(p.relative_to(ROOT))]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    if True: # Full reruns also require exact historical round-zero weights and loader RNG.
        cfg=raw['configuration'];paired=torch.load(ROOT/cfg['paired_initial_checkpoint'],map_location='cpu',weights_only=False)
        if file_hash(ROOT/cfg['paired_initial_checkpoint'])!=cfg['paired_initial_sha256']:raise ValueError('Changed paired initialization')
        for name in ('classifier','decoder'):
            if state_hash(initial[name])!=state_hash(paired[name]):raise ValueError('Different model initialization')
        if any(state_hash(initial['encoders'][cid])!=state_hash(paired['encoders'][cid]) for cid in initial['encoders']):
            raise ValueError('Different private initialization')
        for cid in initial['clients']:
            if not torch.equal(initial['clients'][cid]['loader_generator'],paired['clients'][cid]['loader_generator']):
                raise ValueError('Different loader RNG initialization')
            if initial['clients'][cid]['ae_optimizer']['state'] or initial['clients'][cid]['classifier_optimizer']['state']:
                raise ValueError('Run did not start with empty original optimizer states')
    if raw['precision']!={'use_amp':False,'scaler':None,'matmul_tf32':False,'cudnn_tf32':False}:
        raise ValueError('Wrong precision policy')
    for row in raw['history']:
        if [c['client_id'] for c in row['clients']]!=list(range(10)):raise ValueError('Missing client participation')
        for c in row['clients']:
            batches=math.ceil(len(parts['clients'][str(c['client_id'])])/128)
            warm=0 if variant=='no-warmup' else batches
            if c['warmup_steps']!=warm or c['classification_steps']!=3*batches:
                raise ValueError('Changed local update budget')
            if c['actual_optimizer_steps']!={'warmup_ae':warm,'classification_ae':3*batches,'classification_classifier':3*batches}:
                raise ValueError('Skipped FP32 optimizer update')
    return {'seed':seed,'variant':variant,'accuracy_percent':accuracy,'correct_total':sum(r['correct'] for r in rows),
            'predictions_total':71800,'pipeline_metrics':rows,'partition_sha256':spec['sha256'],
            'training_seconds':raw.get('training_seconds',raw.get('elapsed_training_seconds')),
            'session_seconds':raw['session_seconds'],'source_code':state['code'],'private_checkpoints':cps,
            'peak_cuda_allocated_bytes':raw.get('peak_cuda_allocated_bytes'),
            'peak_cuda_reserved_bytes':raw.get('peak_cuda_reserved_bytes'),'optimizer_step_counts':raw.get('optimizer_step_counts')}


def archive_runs(variants,destination):
    if destination.exists():raise FileExistsError('Preserve previous result archive')
    rows={};private={}
    for variant in variants:
        rows[variant]=[]
        for seed in range(42,47):
            row=audit_run(seed,variant);rows[variant].append(row);private.update(row['private_checkpoints'])
    destination.mkdir(parents=True)
    for variant in variants:
        for seed in range(42,47):
            folder=destination/variant/f'seed-{seed}';compress(run_path(seed,variant)/'results.json',folder/'results.json.gz')
            shutil.copyfile(run_path(seed,variant)/'timings.jsonl',folder/'timings.jsonl')
    summary={'seeds':list(range(42,47)),'ddof':1,'metric':'uniform mean of ten private pipelines on 7180 official test examples each',
             'runs':rows,'accuracy_percent':{v:stats([r['accuracy_percent'] for r in rs]) for v,rs in rows.items()},
             'private_checkpoints':private}
    write_json(destination/'summary.json',summary);return summary


def manifest(folder):
    files={str(p.relative_to(folder)):{'sha256':file_hash(p),'bytes':p.stat().st_size}
           for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    write_json(folder/'manifest.json',{'files':files,'checkpoint_policy':'private thanos _local only; no dataset or checkpoint uploaded'})


def originals():
    summary=archive_runs(['full'],PUBLIC/'originals/artifacts')
    receipt=PRIVATE/'full_campaign.json';campaign=json.loads(receipt.read_text())
    if campaign['status']!='completed' or len(campaign['runs'])!=5 or any(r['exit_code']!=0 for r in campaign['runs']):
        raise ValueError('Original training still active or failed')
    shutil.copyfile(receipt,PUBLIC/'originals/artifacts/campaign.json')
    if (PRIVATE/'failed_launch_campaign.json').exists():
        shutil.copyfile(PRIVATE/'failed_launch_campaign.json',PUBLIC/'originals/artifacts/failed_launch_campaign.json')
    s=summary['accuracy_percent']['full'];lines=['# Five original PathMNIST runs','',
        f"FusedSpaceFed with original settings, FP32: **{s['mean']:.6f} ± {s['sd_sample_ddof1']:.6f}%** (sample SD, five seeds 42–46).",
        '', '| Seed | Accuracy (%) | Correct / predictions | Training (s) | Session (s) |','|---:|---:|---:|---:|---:|']
    for r in summary['runs']['full']:
        lines.append(f"| {r['seed']} | {r['accuracy_percent']:.9f} | {r['correct_total']} / 71800 | {r['training_seconds']:.3f} | {r['session_seconds']:.3f} |")
    lines+=['', 'All seeds 42–46 are rerun from their exact immutable historical round-zero states in FP32; none reuses a trained checkpoint. Three run slots on GPU 1 and two on GPU 0. No tuning, head refit, BN recalibration, adaptation or checkpoint selection. Each seed controls its own frozen pathological partition, as in the original runner; the three ablations will be paired within each seed.', '',
            'The primary accuracy is the uniform mean of ten private-encoder pipelines; each predicts the same official test set (7180 distinct test images). All raw numerators/denominators reconstruct the reported values. Exactly 50 rounds and one terminal test per run. Full checkpoints contain all private encoders, shared classifier/decoder, BN buffers, local persistent optimizers, RNGs and indices; scaler=None explicitly. AMP and TF32 are disabled everywhere.', '',
            f"Fresh five-run FP32 campaign calendar: {campaign['wall_seconds']:.3f} s; sum of process wall times: {campaign['process_wall_seconds_sum']:.3f} s. Previous FP16 attempts and their costs remain separately archived.", '',
            'The historical Table 4 contains 50.94%. These five audited runs are the evidence for a future paper update; the manuscript is intentionally unchanged. They do not establish statistical equivalence to the historical number. Between-seed SD includes partition and initialization variation; five seeds are not a confidence interval.', '',
            'The first launch exited during Python imports because a new module named profile.py shadowed the standard library. The module was renamed before any training update; three direct-CLI regression tests were added. The failed receipt and logs remain separately preserved (failed_launch_campaign.json and logs/import-failure/); successful run times do not include that attempt.', '',
            'Raw JSON is gzip-compressed without loss; timing logs, source hashes, configuration identity and private checkpoint hashes are in artifacts/. Configuration is ../plan.json. Full data, checkpoints and complete logs remain on thanos. Existing experiments and manuscript are unchanged.']
    (PUBLIC/'originals/ORIGINALS_REPORT.md').write_text('\n'.join(lines)+'\n');manifest(PUBLIC/'originals');return summary


def phase1():
    folder=PUBLIC/'phase1';dest=folder/'artifacts'
    if dest.exists():raise FileExistsError('Preserve archive')
    dest.mkdir();records=[];checkpoints={}
    campaign=json.loads((PRIVATE/'phase1/campaign.json').read_text())
    if campaign['status']!='completed' or len(campaign['runs'])!=5 or any(r['exit_code'] for r in campaign['runs']):raise ValueError('Incomplete diagnostics')
    for seed in range(42,47):
        source=PRIVATE/'phase1'/f'seed-{seed}';r=json.loads((source/'results.json').read_text());assert_finite(r);source_check(r['source'])
        if r['status']!='completed' or r['test_access']!='none' or r['seed']!=seed:raise ValueError('Wrong probe')
        for a in ANCHORS:
            if len(r['anchors'][a]['clients'])!=10:raise ValueError('Missing clients')
            for cid,v in r['anchors'][a]['clients'].items():
                if any(v[s]['samples']!=640 for s in STAGES):raise ValueError('Wrong probe size')
                expected=math.ceil(len(partition(seed)[0]['clients'][cid])/128)
                if v['warmup_steps']!=expected or v['classification_steps']!=expected*3:raise ValueError('Wrong replay epochs')
                if any(v['state_l2_changes']['warmup'][c]['all_state_l2'] for c in ('decoder','classifier')):raise ValueError('Shared warmup update')
                cp=v['replay_checkpoint'];p=ROOT/cp['path']
                if file_hash(p)!=cp['sha256'] or p.stat().st_size!=cp['bytes']:raise ValueError('Changed replay')
                checkpoints[cp['path']]={k:cp[k] for k in ('sha256','bytes')}
        records.append(r);compress(source/'results.json',dest/f'seed-{seed}.json.gz')
    metrics=('reconstruction_mse','relative_reconstruction_mse','decoder_to_input_rms_ratio','decoder_max_abs',
             'decoder_fraction_outside_input_range','fused_to_input_rms_ratio','fused_cross_entropy')
    summary={'seeds':list(range(42,47)),'ddof':1,'aggregation':'uniform clients within seed, then mean/sample SD over five seeds',
             'anchors':{a:{s:{k:stats([statistics.mean(r['anchors'][a]['clients'][cid][s][k] for cid in map(str,range(10))) for r in records])
                              for k in metrics} for s in STAGES} for a in ANCHORS},'private_checkpoints':checkpoints,
             'costs':[{'seed':r['seed'],'wall_seconds':r['wall_seconds'],'peak_cuda_allocated_bytes':r['peak_cuda_allocated_bytes'],
                       'peak_cuda_reserved_bytes':r['peak_cuda_reserved_bytes']} for r in records]}
    write_json(dest/'summary.json',summary);shutil.copyfile(PRIVATE/'phase1/campaign.json',dest/'campaign.json')
    failed=PRIVATE/'phase1_failed_relative_path/campaign.json'
    if failed.exists():shutil.copyfile(failed,dest/'failed_relative_path_campaign.json')
    figures=folder/'figures';figures.mkdir()
    for p in (PRIVATE/'phase1/seed-42').glob('*.png'):shutil.copyfile(p,figures/p.name)
    if len(list(figures.glob('*.png')))!=20:raise ValueError('Missing preselected visual grids')
    lines=['# Phase 1 — PathMNIST decoder and warm-up','',
           'Five original seeds, initial/final round-50 anchors. Fixed 640 training images/client; each anchor is copied and independently replayed on the entire local training set: one encoder-only warm-up epoch, then three joint classification epochs. Restore saved persistent SGD/Adam moments, loader RNG; AMP/scaler/TF32 are disabled; no test, tuning, BN recalibration or head refit. Complete per-client intermediate states remain private. This hypothetical extra local update is not a recovered historical trajectory.', '',
           'Rows summarize client-uniform means within each seed, then five-seed means and sample SD. Clients own different class pairs across seeds, so client identifier is not a stable semantic group. UNet decoder has a linear output; values outside [0,1] are allowed. Probe runs in Float32 eval with unchanged native BN. Visual grids use a fixed [0,1] display, clipping only for rendering; numeric measurements use raw values.', '']
    for a in ANCHORS:
        lines+=['## '+a,'','| Stage | MSE ± SD | Decoder/input RMS ± SD | Max abs ± SD | Fused training CE ± SD |','|---|---:|---:|---:|---:|']
        for s in STAGES:
            fmt=lambda k:f"{summary['anchors'][a][s][k]['mean']:.6f} ± {summary['anchors'][a][s][k]['sd_sample_ddof1']:.6f}"
            lines.append(f"| {s} | {fmt('reconstruction_mse')} | {fmt('decoder_to_input_rms_ratio')} | {fmt('decoder_max_abs')} | {fmt('fused_cross_entropy')} |")
        warm=[statistics.mean(r['anchors'][a]['clients'][c]['after_warmup']['reconstruction_mse']-r['anchors'][a]['clients'][c]['before']['reconstruction_mse'] for c in map(str,range(10))) for r in records]
        ce=[statistics.mean(r['anchors'][a]['clients'][c]['after_classification']['reconstruction_mse']-r['anchors'][a]['clients'][c]['after_warmup']['reconstruction_mse'] for c in map(str,range(10))) for r in records]
        lines+=['',f"Mean MSE change: warm-up {statistics.mean(warm):+.6f}; classification {statistics.mean(ce):+.6f}.",'']
    lines+=['## Controls and limits','',
            'All measured values/checkpoints are finite. Shared classifier/decoder states are bitwise unchanged by warm-up; encoder updates are recorded. Classification changes encoder/decoder/classifier; weight distances and BN-buffer distances are separate. Every attempted FP32 optimizer update is executed; no scaler or skipped AMP step exists. A lower MSE does not imply better classification, and CE training need not preserve faithful reconstruction. The final training probe has already been seen during training. No generalization or causal claim follows from a local replay.', '',
            'Digits used five domains, dz=64, DigitCNN, optimizer reset per round, one classification epoch and calibrated settings; PathMNIST uses ten label-skewed clients, dz=16, ResNet20V2, persistent optimizers and three classification epochs. Raw MSE scales also differ ([−1,1] versus [0,1]). Interpret qualitative patterns without treating these as controlled cross-benchmark effect sizes.', '',
            f"Phase calendar {campaign['wall_seconds']:.3f} s; process sum {campaign['process_wall_seconds_sum']:.3f} s. Measured per-seed costs and memory are in artifacts/summary.json; executed commands/exit codes in artifacts/campaign.json.", '',
            'Twenty preselected seed-42 visual grids are in figures/: two anchors × ten clients, two training examples per assigned class. All five seeds contribute to numeric statistics. Configuration/indices/hashes are ../plan.json, ../probe.json and ../anchors.json.']
    lines+=['', 'An earlier replay attempt stopped after saving the first client copy because repository-relative output paths were passed to absolute-path metadata serialization. The output path was canonicalized; no model/loss/optimizer change was made. All five diagnostics restarted from the same immutable anchors. Earlier checkpoints, receipt and logs are preserved in phase1_failed_relative_path/ and logs/phase1-relative-path-failure/; their cost is separate from the successful phase.']
    (folder/'PHASE1_REPORT.md').write_text('\n'.join(lines)+'\n');manifest(folder);return summary


def phase2():
    folder=PUBLIC/'phase2';summary=archive_runs(['full','no-warmup','shared-encoder','decoder-only'],folder/'artifacts')
    campaign=json.loads((PRIVATE/'phase2/campaign.json').read_text())
    if campaign['status']!='completed' or len(campaign['runs'])!=15 or any(r['exit_code'] for r in campaign['runs']):raise ValueError('Incomplete ablations')
    shutil.copyfile(PRIVATE/'phase2/campaign.json',folder/'artifacts/campaign.json')
    full=summary['accuracy_percent']['full']['values'];paired={v:stats([a-b for a,b in zip(summary['accuracy_percent'][v]['values'],full)])
                                                       for v in ('no-warmup','shared-encoder','decoder-only')}
    summary['paired_accuracy_delta_percentage_points']=paired;write_json(folder/'artifacts/summary.json',summary)
    lines=['# Phase 2 — Five-seed paired PathMNIST component ablations','',
           'Each ablation starts from zero, restoring the corresponding original full run initial checkpoint, exact frozen partition and loader generators. Fifty rounds, all ten clients, batch 128, SGD 0.01 and Adam 0.001, persistent local optimizer states, FP32 without AMP/scaler/TF32, native BN. One final test, no tuning or adaptation. The full original campaign is reused; no tuned or head-refitted state enters these comparisons.', '',
           '| Method | Seed 42 | 43 | 44 | 45 | 46 | Mean ± sample SD (%) | Paired delta ± SD (pp) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,s in summary['accuracy_percent'].items():
        delta='reference' if v=='full' else f"{paired[v]['mean']:+.6f} ± {paired[v]['sd_sample_ddof1']:.6f}"
        lines.append('| '+v+' | '+' | '.join(f'{x:.6f}' for x in s['values'])+f" | {s['mean']:.6f} ± {s['sd_sample_ddof1']:.6f} | {delta} |")
    lines+=['',
            'No-warmup removes only the reconstruction phase; classification remains three epochs. This changes compute, Adam history and shuffle progression inherently; it is not compute-matched. Shared-encoder additionally averages encoder model state uniformly after every round, retaining separate persistent local Adam moments. Decoder-only removes the additive raw input from CE and test; warm-up is unchanged. Neither optimizer moments nor BN statistics are specially recalibrated. Full and ablated FP32 runs each use one GPU process, identical original initial weights and loader states. Changing precision can change the trajectory relative to the preserved FP16 campaign.', '',
            'All accuracies are reconstructed from the ten saved client correct/total counts (7180 images/pipeline). All five seeds are shown, no best-seed selection. Paired deltas are calculated within seed before reporting mean/sample SD; this is descriptive, with five pairs and no significance claim. Between-seed variation includes the run-seeded pathological partition.', '',
            'Digits component results for context only: full 85.862359 ± 0.846930%, no-warmup 85.714598 ± 0.789950%, shared-encoder 86.178544 ± 0.468069%, decoder-only 84.859627 ± 0.580810%. Different domains/classes, model, preprocessing, local epochs, optimizer persistence and calibrated hyperparameters prevent interpreting between-benchmark differences as a single controlled effect.', '',
            f"New 15-run campaign calendar {campaign['wall_seconds']:.3f} s; process sum {campaign['process_wall_seconds_sum']:.3f} s. Per-run timing logs, actual/attempted optimization steps and CUDA memory are archived. No old run is charged as newly executed. All initial/final complete checkpoints, configs, indices, optimizer/scaler/RNG and full logs remain on thanos, with paths/hashes in artifacts/summary.json.", '',
            'The original full-method mean belongs to a future Table 4 update. Ablations remain a separate analysis. Manuscript and previous experiments unchanged.']
    (folder/'PHASE2_REPORT.md').write_text('\n'.join(lines)+'\n');manifest(folder);return summary


def phase3():
    folder=PUBLIC/'phase3';dest=folder/'artifacts'
    if dest.exists():raise FileExistsError('Preserve archive')
    dest.mkdir();records=[];campaign=json.loads((PRIVATE/'phase3/campaign.json').read_text())
    if campaign['status']!='completed' or len(campaign['runs'])!=5 or any(r['exit_code'] for r in campaign['runs']):raise ValueError('Incomplete gradient phase')
    for seed in range(42,47):
        path=PRIVATE/'phase3'/f'seed-{seed}'/'results.json';r=json.loads(path.read_text());assert_finite(r);source_check(r['source'])
        if r['status']!='completed' or r['seed']!=seed or r['test_access']!='none':raise ValueError('Wrong gradient probe')
        for a in ANCHORS:
            for mode in ('eval','batch-stateless'):
                v=r['anchors'][a]['modes'][mode]
                if not v['rng_unchanged'] or v['state_before']!=v['state_after']:raise ValueError('Gradient mutated state')
                if len(v['per_batch'])!=5 or len(v['per_client_batches'])!=10:raise ValueError('Wrong gradient budget')
                if v['classifier']['client_count']!=10 or abs(v['classifier']['relative_identity_residual'])>1e-10:raise ValueError('Wrong algebra')
        records.append(r);compress(path,dest/f'seed-{seed}.json.gz')
    summary={'seeds':list(range(42,47)),'ddof':1,'dispersion_divisor':10,'anchors':{}}
    lines=['# Phase 3 — Five-seed fixed-state PathMNIST gradient diagnostics','',
           'Original full models at initialization and round 50; identical classifier/decoder/private encoder states for the original x and fused x+D(E_i(x)) branches. Five fixed training batches of 128 per client. Float32 raw gradients, CPU Float64 means/Gram reductions, no optimization or test access. Primary mode: native eval BN. Sensitivity mode: batch-stateless BN, buffers restored before/between/after branches. Neither changes stored BN statistics.', '',
           'gᵒᵢ=∂θ CE(C(x),y), gᶠᵢ=∂θ CE(C(x+D(E_i(x))),y), bᵢ=gᶠᵢ−gᵒᵢ. Γ=(1/10)Σ||gᵢ−mean(g)||²; B=(1/10)Σ||bᵢ−mean(b)||²; Φ=(2/10)Σ〈gᵒᵢ−mean(gᵒ),bᵢ−mean(b)〉. Exact identity Γf=Γo+B+Φ is audited. The five batch gradients are averaged **before** client dispersion; per-batch dispersion is separately retained. The population client divisor is 10; seed SD uses ddof=1.', '']
    for a in ANCHORS:
        summary['anchors'][a]={}
        for mode in ('eval','batch-stateless'):
            vals=[r['anchors'][a]['modes'][mode] for r in records]
            keys=('Gamma_original','Gamma_fused','B','Phi','fused_original_ratio')
            value={k:stats([v['classifier'][k] for v in vals]) for k in keys}
            value['Gamma_decoder']=stats([v['decoder']['Gamma_decoder'] for v in vals])
            for branch in ('original','fused'):
                value[branch]={k:stats([v['classifier'][branch][k] for v in vals])
                               for k in ('normalized_dispersion','mean_pairwise_cosine','mean_client_squared_gradient_norm','mean_gradient_norm')}
            summary['anchors'][a][mode]=value
            lines+=['## '+a+' — '+mode,'','| Seed | Γ original | Γ fused | B | Φ | Γf/Γo | Γ decoder |','|---:|---:|---:|---:|---:|---:|---:|']
            for r,v in zip(records,vals):
                c=v['classifier'];lines.append(f"| {r['seed']} | {c['Gamma_original']:.9e} | {c['Gamma_fused']:.9e} | {c['B']:.9e} | {c['Phi']:.9e} | {c['fused_original_ratio']:.6f} | {v['decoder']['Gamma_decoder']:.9e} |")
            ratio=value['fused_original_ratio'];reduced=sum(v['classifier']['Gamma_fused']<v['classifier']['Gamma_original'] for v in vals)
            lines+=['',f"Γf/Γo {ratio['mean']:.6f} ± {ratio['sd_sample_ddof1']:.6f}; reduction in {reduced}/5 seeds. Normalized dispersion original→fused {value['original']['normalized_dispersion']['mean']:.6f}→{value['fused']['normalized_dispersion']['mean']:.6f}; mean pairwise cosine {value['original']['mean_pairwise_cosine']['mean']:.6f}→{value['fused']['mean_pairwise_cosine']['mean']:.6f}.",'']
    summary['costs']=[{'seed':r['seed'],'wall_seconds':r['wall_seconds'],'peak_cuda_allocated_bytes':r['peak_cuda_allocated_bytes'],
                      'peak_cuda_reserved_bytes':r['peak_cuda_reserved_bytes']} for r in records]
    write_json(dest/'summary.json',summary);shutil.copyfile(PRIVATE/'phase3/campaign.json',dest/'campaign.json')
    lines+=['## Interpretation, controls and comparison with Digits','',
            'All parameters and BN buffers are unchanged by paired measurement; RNG unchanged. Raw gradient norms/decoder dispersion, per-client/per-batch records and Gram matrices are archived. A raw Γ decrease can follow smaller gradients rather than better angular alignment: inspect normalized dispersion and cosines together. Batch-stateless BN couples examples, so it is a distinct batch-conditioned gradient objective; native eval remains the inference-aligned measurement.', '',
            'Digits final models: native-eval Γf/Γo 0.791656 ± 0.253408, batch-stateless 0.148367 ± 0.140864. Native normalized dispersion 0.794160→0.799588 and mean pairwise cosine 0.090559→0.043966 already showed why raw scale reduction alone does not establish alignment. PathMNIST uses a different classifier, label-skewed clients, persistent optimizers and uncalibrated original settings. These are descriptive cross-benchmark observations, not a controlled causal comparison.', '',
            'Two anchors and finite training probes do not describe every round or the population objective. Five seeds do not justify a universal reduction claim. No tuning or additional generalization evaluation was performed; the manuscript is unchanged.', '',
            f"Phase calendar {campaign['wall_seconds']:.3f} s; sum process time {campaign['process_wall_seconds_sum']:.3f} s. Per-seed costs/memory in artifacts/summary.json. Configuration, exact probe IDs and anchor hashes: ../plan.json, ../probe.json, ../anchors.json. Executed commands and exit codes: artifacts/campaign.json."]
    (folder/'PHASE3_REPORT.md').write_text('\n'.join(lines)+'\n');manifest(folder);return summary


if __name__=='__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('originals','1','2','3'))
    a=p.parse_args();value={'originals':originals,'1':phase1,'2':phase2,'3':phase3}[a.phase]()
    print(json.dumps({'archived':a.phase,'seeds':value['seeds']}))
