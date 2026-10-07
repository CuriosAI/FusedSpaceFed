"""Assemble the audited four-part FP32 campaign report without further evaluation."""
from datetime import datetime
import gzip
import json
import statistics
import subprocess
from research.pathmnist_five_seed.fp32.precision import ROOT, PUBLIC, PRIVATE
from research.pathmnist_five_seed.fp32.archive import manifest
from research.pathmnist_pathological.run import file_hash, write_json


def finish():
    originals=json.loads((PUBLIC/'originals/artifacts/summary.json').read_text())
    decoder=json.loads((PUBLIC/'phase1/artifacts/summary.json').read_text())
    ablation=json.loads((PUBLIC/'phase2/artifacts/summary.json').read_text())
    gradients=json.loads((PUBLIC/'phase3/artifacts/summary.json').read_text())
    zero_encoder={}
    for mode in ('eval','batch-stateless'):
        cases=[]
        for seed in range(42,47):
            raw=json.loads(gzip.decompress((PUBLIC/f'phase3/artifacts/seed-{seed}.json.gz').read_bytes()))
            batches=raw['anchors']['final-round50']['modes'][mode]['per_client_batches']
            cases.extend({'seed':seed,'client_id':int(cid)} for cid,rows in batches.items()
                         if all(r['encoder_fused_norm']==0 for r in rows))
        zero_encoder[mode]=cases
    write_json(PUBLIC/'encoder_gradient_zero_cases.json',{'anchor':'final-round50','batch_count':5,'modes':zero_encoder,
        'interpretation':'zero encoder CE gradients on these fixed training probes; not a universal population claim'})
    for s in (originals,decoder,ablation,gradients):
        if s['seeds']!=list(range(42,47)) or s['ddof']!=1:raise ValueError('Incomplete five-seed summary')
    previous=json.loads((PRIVATE/'preservation.json').read_text())
    changes=[name for name,digest in {**previous['versioned_sha256'],**previous['previous_run_files_sha256']}.items()
             if not (ROOT/name).is_file() or file_hash(ROOT/name)!=digest]
    if changes:raise ValueError('Previous experiment or manuscript changed: '+str(changes))
    verification={'previous_versioned_files_unchanged':len(previous['versioned_sha256']),
                  'previous_private_run_files_unchanged':len(previous['previous_run_files_sha256']),
                  'start_commit':previous['start_commit'],'scientific_source_change':'precision only; historical sources unchanged'}
    write_json(PUBLIC/'preservation_verification.json',verification)
    histories={name:json.loads((PUBLIC/name/'artifacts/campaign.json').read_text())
               for name in ('originals','phase1','phase2','phase3')}
    if any(c['status']!='completed' or any(r['exit_code']!=0 for r in c['runs']) for c in histories.values()):
        raise ValueError('Active or failed campaign')
    expected={'originals':5,'phase1':5,'phase2':15,'phase3':5}
    if any(len(histories[k]['runs'])!=n for k,n in expected.items()):raise ValueError('Missing processes')
    costs={k:{'campaign_wall_seconds':c['wall_seconds'],'sum_process_wall_seconds':c['process_wall_seconds_sum']}
           for k,c in histories.items()}
    memory_rows={'originals':originals['runs']['full'],'phase1':decoder['costs'],
                 'phase2':[r for name,rs in ablation['runs'].items() if name!='full' for r in rs],
                 'phase3':gradients['costs']}
    for name,rows in memory_rows.items():
        costs[name]['max_process_peak_cuda_allocated_bytes']=max(r['peak_cuda_allocated_bytes'] for r in rows)
        costs[name]['max_process_peak_cuda_reserved_bytes']=max(r['peak_cuda_reserved_bytes'] for r in rows)
    costs['sum_phase_campaign_wall_seconds']=sum(c['campaign_wall_seconds'] for c in costs.values())
    costs['sum_process_wall_seconds']=sum(c['process_wall_seconds_sum'] for c in histories.values())
    begin=datetime.fromisoformat(histories['originals']['started_utc']);end=datetime.fromisoformat(histories['phase3']['ended_utc'])
    costs['elapsed_first_successful_launch_to_last_diagnostic_seconds']=(end-begin).total_seconds()
    costs['preserved_failed_attempts']={}
    for name,path in (('import_only',PRIVATE/'failed_launch_campaign.json'),
                      ('diagnostic_relative_path',PRIVATE/'phase1_failed_relative_path/campaign.json')):
        if path.exists():
            c=json.loads(path.read_text())
            costs['preserved_failed_attempts'][name]={k:c[k] for k in ('wall_seconds','process_wall_seconds_sum')}
    full=originals['accuracy_percent']['full']
    fmt=lambda s:f"{s['mean']:.6f} ± {s['sd_sample_ddof1']:.6f}"
    lines=['# PathMNIST original settings: five-seed FP32 campaign and mechanism diagnostics','',
           f"Full FusedSpaceFed: **{fmt(full)}%** (five fresh seeds 42–46; sample SD, ddof1).",
           '', '## Scope and exact pairing','',
           'All five full runs and all fifteen ablations start at round 0. Historical initial C/D/private-E weights, empty local optimizers and client shuffle-generator states are restored exactly; frozen per-seed partitions are unchanged. Precision is FP32 throughout training, replay, terminal inference and gradient computation: AMP/autocast, GradScaler and matmul/cuDNN TF32 are disabled. Float64 is used only for statistical/Gram reductions. Architecture, data, batch 128, 50 rounds, full participation, native BN, persistent SGD0.01/Adam0.001, warm-up 1 and CE 3 remain fixed. No tuning, baseline, head refit, BN recalibration, checkpoint selection or manuscript edit.', '',
           'Seed 42 previously used two workers. This campaign uses one process per run; exact model and per-client data-order initialization is retained, while its process RNG mapping is explicitly recorded. Different precision and device scheduling need not reproduce historical trajectories bitwise. The preserved FP16 seed 43 overflow did not justify changing any learning rate or adding clipping.', '',
           'Each terminal test is performed once at round 50: 7180 official images through each of ten private encoder pipelines; uniform client mean, 71800 predictions on 7180 distinct images. Correct/total counts independently reconstruct all scores. Seed affects both model and frozen pathological partition; every intervention is paired within seed. SD over five seeds is descriptive, not a confidence interval.', '',
           '## Full method and separate component ablations','',
           '| Method | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± SD (%) | Paired delta ± SD (pp) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in ablation['accuracy_percent'].items():
        delta='reference' if name=='full' else fmt(ablation['paired_accuracy_delta_percentage_points'][name])
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in s['values'])+f" | {fmt(s)} | {delta} |")
    lines+=['', 'For each intervention, paired signs and all five differences are retained:', '']
    for name,s in ablation['paired_accuracy_delta_percentage_points'].items():
        lines.append(f"- {name}: below full in {sum(x<0 for x in s['values'])}/5 pairs, above in {sum(x>0 for x in s['values'])}/5. No selection follows from these test results.")
    lines+=['',
            f"The FP32 full-method mean is {50.94-full['mean']:.6f} percentage points below the historical Table 4 value of 50.94%; these runs do not recover that result. The original five Table 4 training artifacts were unavailable: the initial states reused here are from the preceding fixed-setting campaign, not recovered original paper runs. This is new evidence for a future update, separate from the component interventions; the manuscript is unchanged. No-warmup reduces compute and changes Adam history/shuffle progression; it is not FLOP-matched. Shared-encoder averages only encoder model weights, retaining local optimizer moments. Decoder-only removes the raw-input additive branch during CE and test. No ablation hyperparameter calibration was performed.", '',
            '## Decoder and warm-up: training-only replay','',
            'Initial/final complete checkpoints were copied independently per client and replayed for one warm-up epoch followed by three CE epochs on the full local training set. Restored persistent moments and shuffle RNG are retained. Fixed 640 training examples/client are measured before/after each phase. This is a hypothetical extra local update, not a recovered historical within-round trajectory. Warm-up is audited to leave all shared C/D parameters and BN buffers unchanged. Complete before/warm/CE snapshots and twenty predetermined seed 42 visual grids are preserved.', '',
            '| Anchor | Stage | Reconstruction MSE ± SD | Decoder/input RMS ± SD | Raw max abs ± SD | Fused training CE ± SD |',
            '|---|---|---:|---:|---:|---:|']
    for anchor,stages in decoder['anchors'].items():
        for stage,values in stages.items():
            lines.append(f"| {anchor} | {stage} | "+' | '.join(fmt(values[k]) for k in ('reconstruction_mse','decoder_to_input_rms_ratio','decoder_max_abs','fused_cross_entropy'))+' |')
    lines+=['',
            'Summaries first average clients uniformly within seed, then average five seeds. Client IDs have different class pairs across seeds. Decoder output is linear, not restricted to[0,1]. Raw amplitudes, out-of-range fractions and input/decoder cosine are retained; only visual display clips to[0,1]. Reduced reconstruction error alone does not establish better classification.', '',
            f"At the final anchor the mean relative MSE is {decoder['anchors']['final-round50']['before']['relative_reconstruction_mse']['mean']:.6f}, where 1 is the error of returning zeros (an analytic reference, not a trained baseline). The small output RMS and representative grids do not support calling the decoder a faithful image reconstruction. The encoder is exactly unchanged after both replay phases in {len(decoder['state_updates']['final-round50']['encoder_unchanged_in_both_phases'])}/50 client copies: {decoder['state_updates']['final-round50']['encoder_unchanged_in_both_phases']}. This is a measured absence of parameter updates, not proof that all encoder gradients are zero; phase3 records those gradients separately. All 50 initial encoders did update. BN-buffer distance includes num_batches_tracked, so a large aggregate buffer norm cannot be interpreted as a large normalization-statistic drift.", '',
            '## Paired gradients at identical model states','',
            'Five fixed training batches of 128/client, original x versus x+D(E_i(x)), initial/final anchors. Native-eval BN is the inference-aligned primary probe; batch-stateless BN is a separate batch-conditioned sensitivity probe. All weights, buffers and RNG are audited unchanged. No optimizer step is performed. Client gradients are averaged over batches before computing population dispersion with divisor 10; Γf=Γo+B+Φ is checked to relative tolerance 1e-10. Raw vectors are computed in Float32; CPU means/Gram algebra in Float64.', '',
            '| Anchor | BN mode | Γf/Γo ± SD | Original normalized Γ | Fused normalized Γ | Original pair cosine | Fused pair cosine |',
            '|---|---|---:|---:|---:|---:|---:|']
    for anchor,modes in gradients['anchors'].items():
        for mode,v in modes.items():
            lines.append(f"| {anchor} | {mode} | {fmt(v['fused_original_ratio'])} | {v['original']['normalized_dispersion']['mean']:.6f} | {v['fused']['normalized_dispersion']['mean']:.6f} | {v['original']['mean_pairwise_cosine']['mean']:.6f} | {v['fused']['mean_pairwise_cosine']['mean']:.6f} |")
    lines+=['',
            'A raw Γ decrease can reflect smaller gradients rather than angular alignment. Inspect normalized dispersion, cosines, B, Φ and decoder dispersion together. The trained classifier and its native BN reflect fused inputs; the raw-input branch is a counterfactual at identical weights, not a separately trained classifier. Input-distribution shift or loss calibration can also change gradient scale. Batch-stateless BN is a distinct batch-conditioned objective. Full per-client/per-batch norms, Gram matrices and decomposition identities are archived in phase3/. Two anchors and finite training probes cannot establish a theorem or explain every intermediate round.', '',
            'At the final anchor raw dispersion falls in 4/5 native-eval seeds, but seed 42 has Γf/Γo 2.697930. Native normalized dispersion increases 0.700936→0.822183 and pairwise cosine falls 0.221775→0.093693. Batch-stateless raw Γ falls in 5/5 seeds, while normalized dispersion also increases 0.888618→0.894538. Thus these probes do not establish improved angular alignment or universal dispersion reduction. At initialization native Γ increases in all 5 seeds.', '',
            f"Encoder fused-CE gradients are exactly zero in all five fixed batches for {zero_encoder['eval']} under native eval and for {zero_encoder['batch-stateless']} under batch-stateless BN. These are the same three client cases with no encoder parameter change in either replay phase. The observation concerns the measured training probes at the saved state; it does not prove zero gradients on every possible input.", '',
            '## Comparison with existing Digits diagnostics','',
            'Digits full 85.862359±0.846930%, no-warmup 85.714598±0.789950%, shared-encoder 86.178544±0.468069%, decoder-only 84.859627±0.580810%. Final native Γf/Γo 0.791656±0.253408; batch-stateless 0.148367±0.140864. Native normalized dispersion 0.794160→0.799588 and cosine 0.090559→0.043966 demonstrate that smaller raw gradients need not mean improved angular alignment. PathMNIST differs in classifier, latent size 16 versus 64, input scale[0,1] versus[-1,1], label-skew versus domain shift, ten versus five clients, three versus one CE epochs, persistent versus reset optimizers, and fixed original versus calibrated settings. Cross-benchmark comparisons are descriptive. Every PathMNIST result above is a five-seed statistic, separate from older single-seed investigations.', '',
            '## Measured cost and preservation','',
            '| Phase | Campaign elapsed (s) | Sum process elapsed (s) | Maximum process allocated / reserved (MiB) |',
            '|---|---:|---:|---:|---:|']
    for name in expected:
        c=costs[name];lines.append(f"| {name} | {c['campaign_wall_seconds']:.3f} | {c['sum_process_wall_seconds']:.3f} | {c['max_process_peak_cuda_allocated_bytes']/2**20:.3f} / {c['max_process_peak_cuda_reserved_bytes']/2**20:.3f} |")
    lines+=['',f"Successful first launch→last diagnostic: {costs['elapsed_first_successful_launch_to_last_diagnostic_seconds']:.3f}s including intervening verification/archiving/commits. Sum phase campaign elapsed: {costs['sum_phase_campaign_wall_seconds']:.3f}s; sum process elapsed: {costs['sum_process_wall_seconds']:.3f}s. Concurrent process times are not exclusive GPU-hour charges. Per-seed training/evaluation/session durations, actual optimizer counts, peak allocated/reserved CUDA bytes and RSS are retained in the corresponding archives.", '',
            'A Python standard-library name collision caused a failed import-only launch, before any training. It was fixed by renaming the precision module and adding three direct-CLI regression tests. The first decoder replay also stopped on relative-path metadata serialization after updating only client copies. Its output path was canonicalized; all five diagnostics restarted from immutable anchors. Both failed receipts, logs and partial diagnostic checkpoints remain preserved, with measured costs separately recorded in cost_summary.json. Neither was a numerical failure or altered the full trained models. Phase archives require every declared process to complete.', '',
            f"Hash verification confirms all {verification['previous_versioned_files_unchanged']} pre-existing tracked files and {verification['previous_private_run_files_unchanged']} files of the previous PathMNIST runs unchanged. Frozen partition/probe/source/checkpoint SHA256 manifests are retained. Full checkpoints (initial/latest/final for 20 training runs, plus 100 decoder replay snapshots) remain on thanos in `_local/pathmnist_five_seed/fp32/`; scaler is explicitly None, not missing. Dataset/checkpoints/full logs are not published.", '',
            '## Reproduction and commits','',
            'Commands/configurations: README.md, plan.json, initial_sources.json, full_queue.json and phase1–3/queue.json. Successful commands, physical GPU, exit codes and process durations: each artifacts/campaign.json. Numeric result JSON is gzip-compressed losslessly; timings.jsonl are uncompressed. Source version per process is in each archived result/checkpoint. Tests and exact output are in verification_tests.log, verification_cli_tests.log and verification_diagnostic_tests.log.', '',
            'Commits for this task before the final report commit:', '']
    log=subprocess.check_output(['git','log','--reverse','--format=%H %s',previous['start_commit']+'..HEAD'],cwd=ROOT,text=True)
    lines+=['- '+entry for entry in log.strip().splitlines()]
    test_lines=(PUBLIC/'final_verification_tests.log').read_text().splitlines()
    lines+=['', 'Final full repository test suite: '+next(line for line in reversed(test_lines) if ' passed in ' in line)+'. Exact output: final_verification_tests.log.']
    lines+=['', 'The commit containing this assembled report is identified by Git history; no self-referential commit hash is embedded. All substantive steps are pushed to main. The manuscript remains unchanged.']
    text='\n'.join(lines)+'\n';(PUBLIC/'FP32_CAMPAIGN_REPORT.md').write_text(text)
    (ROOT/'_local/report_pathmnist_fp32.md').write_text(text)
    write_json(PUBLIC/'cost_summary.json',costs)
    manifest(PUBLIC/'originals');manifest(PUBLIC/'phase1');manifest(PUBLIC/'phase2');manifest(PUBLIC/'phase3')


if __name__=='__main__':finish()
