"""Assemble the audited four-part FP32 campaign report without further evaluation."""
from datetime import datetime
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
    full=originals['accuracy_percent']['full']
    fmt=lambda s:f"{s['mean']:.6f} ± {s['sd_sample_ddof1']:.6f}"
    lines=['# PathMNIST original settings: five-seed FP32 campaign and mechanism diagnostics','',
           f"Full FusedSpaceFed: **{fmt(full)}%** (five fresh seeds42–46; sample SD, ddof1).",
           '', '## Scope and exact pairing','',
           'All five full runs and all fifteen ablations start at round0. Historical initial C/D/private-E weights, empty local optimizers and client shuffle-generator states are restored exactly; frozen per-seed partitions are unchanged. Precision is FP32 throughout training, replay, terminal inference and gradient computation: AMP/autocast, GradScaler and matmul/cuDNN TF32 are disabled. Float64 is used only for statistical/Gram reductions. Architecture, data, batch128, 50 rounds, full participation, native BN, persistent SGD0.01/Adam0.001, warm-up1 and CE3 remain fixed. No tuning, baseline, head refit, BN recalibration, checkpoint selection or manuscript edit.', '',
           'Seed42 previously used two workers. This campaign uses one process per run; exact model and per-client data-order initialization is retained, while its process RNG mapping is explicitly recorded. Different precision and device scheduling need not reproduce historical trajectories bitwise. The preserved FP16 seed43 overflow did not justify changing any learning rate or adding clipping.', '',
           'Each terminal test is performed once at round50: 7180 official images through each of ten private encoder pipelines; uniform client mean, 71800 predictions on7180 distinct images. Correct/total counts independently reconstruct all scores. Seed affects both model and frozen pathological partition; every intervention is paired within seed. SD over five seeds is descriptive, not a confidence interval.', '',
           '## Full method and separate component ablations','',
           '| Method | Seed42 | Seed43 | Seed44 | Seed45 | Seed46 | Mean ± SD (%) | Paired delta ± SD (pp) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in ablation['accuracy_percent'].items():
        delta='reference' if name=='full' else fmt(ablation['paired_accuracy_delta_percentage_points'][name])
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in s['values'])+f" | {fmt(s)} | {delta} |")
    lines+=['',
            'The full-method mean is the evidence for a future Table4 update; it must be kept separate from interventions. The historical manuscript number50.94% is not replaced here. No-warmup reduces compute and changes Adam history/shuffle progression; it is not FLOP-matched. Shared-encoder averages only encoder model weights, retaining local optimizer moments. Decoder-only removes the raw-input additive branch during CE and test. No ablation hyperparameter calibration was performed.', '',
            '## Decoder and warm-up: training-only replay','',
            'Initial/final complete checkpoints were copied independently per client and replayed for one warm-up epoch followed by three CE epochs on the full local training set. Restored persistent moments and shuffle RNG are retained. Fixed640 training examples/client are measured before/after each phase. This is a hypothetical extra local update, not a recovered historical within-round trajectory. Warm-up is audited to leave all shared C/D parameters and BN buffers unchanged. Complete before/warm/CE snapshots and twenty predetermined seed42 visual grids are preserved.', '',
            '| Anchor | Stage | Reconstruction MSE ± SD | Decoder/input RMS ± SD | Raw max abs ± SD | Fused training CE ± SD |',
            '|---|---|---:|---:|---:|---:|']
    for anchor,stages in decoder['anchors'].items():
        for stage,values in stages.items():
            lines.append(f"| {anchor} | {stage} | "+' | '.join(fmt(values[k]) for k in ('reconstruction_mse','decoder_to_input_rms_ratio','decoder_max_abs','fused_cross_entropy'))+' |')
    lines+=['',
            'Summaries first average clients uniformly within seed, then average five seeds. Client IDs have different class pairs across seeds. Decoder output is linear, not restricted to[0,1]. Raw amplitudes, out-of-range fractions and input/decoder cosine are retained; only visual display clips to[0,1]. Reduced reconstruction error alone does not establish better classification.', '',
            '## Paired gradients at identical model states','',
            'Five fixed training batches128/client, original x versus x+D(E_i(x)), initial/final anchors. Native-eval BN is the inference-aligned primary probe; batch-stateless BN is a separate batch-conditioned sensitivity probe. All weights, buffers and RNG are audited unchanged. No optimizer step is performed. Client gradients are averaged over batches before computing population dispersion with divisor10; Γf=Γo+B+Φ is checked to relative tolerance1e-10. Raw vectors are computed in Float32; CPU means/Gram algebra in Float64.', '',
            '| Anchor | BN mode | Γf/Γo ± SD | Original normalized Γ | Fused normalized Γ | Original pair cosine | Fused pair cosine |',
            '|---|---|---:|---:|---:|---:|---:|']
    for anchor,modes in gradients['anchors'].items():
        for mode,v in modes.items():
            lines.append(f"| {anchor} | {mode} | {fmt(v['fused_original_ratio'])} | {v['original']['normalized_dispersion']['mean']:.6f} | {v['fused']['normalized_dispersion']['mean']:.6f} | {v['original']['mean_pairwise_cosine']['mean']:.6f} | {v['fused']['mean_pairwise_cosine']['mean']:.6f} |")
    lines+=['',
            'A raw Γ decrease can reflect smaller gradients rather than angular alignment. Inspect normalized dispersion, cosines, B, Φ and decoder dispersion together. Full per-client/per-batch norms, Gram matrices and decomposition identities are archived in phase3/. Two anchors and finite training probes cannot establish a theorem or explain every intermediate round.', '',
            '## Comparison with existing Digits diagnostics','',
            'Digits full85.862359±0.846930%, no-warmup85.714598±0.789950%, shared-encoder86.178544±0.468069%, decoder-only84.859627±0.580810%. Final native Γf/Γo0.791656±0.253408; batch-stateless0.148367±0.140864. Native normalized dispersion0.794160→0.799588 and cosine0.090559→0.043966 demonstrate that smaller raw gradients need not mean improved angular alignment. PathMNIST differs in classifier, latent size16 versus64, input scale[0,1] versus[-1,1], label-skew versus domain shift, ten versus five clients, three versus one CE epochs, persistent versus reset optimizers, and fixed original versus calibrated settings. Cross-benchmark comparisons are descriptive. Every PathMNIST result above is a five-seed statistic, separate from older single-seed investigations.', '',
            '## Measured cost and preservation','',
            '| Phase | Campaign elapsed (s) | Sum process elapsed (s) | Maximum process allocated / reserved (MiB) |',
            '|---|---:|---:|---:|---:|']
    for name in expected:
        c=costs[name];lines.append(f"| {name} | {c['campaign_wall_seconds']:.3f} | {c['sum_process_wall_seconds']:.3f} | {c['max_process_peak_cuda_allocated_bytes']/2**20:.3f} / {c['max_process_peak_cuda_reserved_bytes']/2**20:.3f} |")
    lines+=['',f"Successful first launch→last diagnostic: {costs['elapsed_first_successful_launch_to_last_diagnostic_seconds']:.3f}s including intervening verification/archiving/commits. Sum phase campaign elapsed: {costs['sum_phase_campaign_wall_seconds']:.3f}s; sum process elapsed: {costs['sum_process_wall_seconds']:.3f}s. Concurrent process times are not exclusive GPU-hour charges. Per-seed training/evaluation/session durations, actual optimizer counts, peak allocated/reserved CUDA bytes and RSS are retained in the corresponding archives.", '',
            'A Python standard-library name collision caused a failed import-only launch, before any training. It was fixed by renaming the precision module and adding three direct-CLI regression tests; the failed receipt/logs remain preserved. It is not a numerical failure, and its cost is separate. No later numerical failure is silently dropped: phase archives require every declared process to complete.', '',
            f"Hash verification confirms all {verification['previous_versioned_files_unchanged']} pre-existing tracked files and {verification['previous_private_run_files_unchanged']} files of the previous PathMNIST runs unchanged. Frozen partition/probe/source/checkpoint SHA256 manifests are retained. Full checkpoints (initial/latest/final for20 training runs, plus100 decoder replay snapshots) remain on thanos in `_local/pathmnist_five_seed/fp32/`; scaler is explicitly None, not missing. Dataset/checkpoints/full logs are not published.", '',
            '## Reproduction and commits','',
            'Commands/configurations: README.md, plan.json, initial_sources.json, full_queue.json and phase1–3/queue.json. Successful commands, physical GPU, exit codes and process durations: each artifacts/campaign.json. Numeric result JSON is gzip-compressed losslessly; timings.jsonl are uncompressed. Source version per process is in each archived result/checkpoint. Tests and exact output are in verification_tests.log and verification_cli_tests.log; the final suite result is recorded in final_verification_tests.log.', '',
            'Commits for this task before the final report commit:', '']
    log=subprocess.check_output(['git','log','--reverse','--format=%H %s',previous['start_commit']+'..HEAD'],cwd=ROOT,text=True)
    lines+=['- '+entry for entry in log.strip().splitlines()]
    lines+=['', 'The commit containing this assembled report is identified by Git history; no self-referential commit hash is embedded. All substantive steps are pushed to main. The manuscript remains unchanged.']
    text='\n'.join(lines)+'\n';(PUBLIC/'FP32_CAMPAIGN_REPORT.md').write_text(text)
    (ROOT/'_local/report_pathmnist_fp32.md').write_text(text)
    write_json(PUBLIC/'cost_summary.json',costs)
    manifest(PUBLIC/'originals');manifest(PUBLIC/'phase1');manifest(PUBLIC/'phase2');manifest(PUBLIC/'phase3')


if __name__=='__main__':finish()
