"""Archive every calibration/final result losslessly and generate the report."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.feature_shift_digits.data import DOMAINS, atomic_json, canonical_hash, file_hash
from research.feature_shift_digits_calibrated.audit import check_result, load_result, rivals
from research.feature_shift_digits_calibrated.prepare_plan import PRIVATE, PUBLIC
from research.feature_shift_digits_calibrated.runner import SOURCES, validate_config


def stats(values):
    return {'values': values, 'mean': statistics.mean(values),
            'sample_standard_deviation_ddof1': statistics.stdev(values) if len(values) > 1 else None, 'unit': 'percent'}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(value)


def validate_receipt(queue, receipt, queue_path):
    if receipt['status'] != 'completed' or receipt['queue_sha256'] != file_hash(queue_path):
        raise ValueError('Campaign incomplete or queue changed')
    names = [row['name'] for row in receipt['runs']]
    if sorted(names) != sorted(row['name'] for row in queue['jobs']) or any(row['exit_code'] != 0 for row in receipt['runs']):
        raise ValueError('Workers did not all exit successfully')
    if any('--resume' in row['command'] for row in receipt['runs']):
        raise ValueError('Campaign expected fresh runs; review interruptions explicitly')
    for job in queue['jobs']:
        execution = next(row for row in receipt['runs'] if row['name'] == job['name'])
        if execution['config'] != job['config'] or execution['output'] != job['output']:
            raise ValueError('Executed command differs from registration')


def build():
    destination = PUBLIC / 'artifacts'
    if destination.exists() or (PUBLIC / 'CALIBRATED_DIGITS_REPORT.md').exists():
        raise FileExistsError('Archive/report already exists')
    plan = json.loads((PUBLIC / 'search_plan.json').read_text())
    shortlist = json.loads((PUBLIC / 'shortlist.json').read_text())
    selection = json.loads((PUBLIC / 'selection.json').read_text())
    expected_source = {name: file_hash(REPO / name) for name in SOURCES}
    all_runs, campaigns = {}, {}
    for phase in ('screening', 'confirmation', 'final'):
        queue_path = PUBLIC / (phase + '_queue.json')
        queue = json.loads(queue_path.read_text())
        receipt_path = PRIVATE / (phase + '_campaign.json')
        receipt = json.loads(receipt_path.read_text())
        validate_receipt(queue, receipt, queue_path)
        campaigns[phase] = receipt
        for job in queue['jobs']:
            root = Path(job['output'])
            result_path = root / 'results.json'
            raw = result_path.read_bytes()
            result = json.loads(raw)
            config = json.loads(Path(job['config']).read_text())
            validate_config(config)
            check_result(result, config, expected_source=expected_source)
            lines = [json.loads(line) for line in (root / 'timings.jsonl').read_text().splitlines()]
            if lines != result['history']:
                raise ValueError('Timings JSONL differs from saved history')
            execution = next(row for row in receipt['runs'] if row['name'] == job['name'])
            if result['identity']['code']['commit'] != receipt['code_commit'] or result['identity']['device'] != execution['device']:
                raise ValueError('Code/device differs from execution receipt')
            if len(result['sessions']) != 1 or result['sessions'][0]['start_round'] != 1:
                raise ValueError('Not a fresh initialization; review explicitly')
            relative = Path(phase) / job['name']
            write_new(destination / relative / 'results.json.gz', gzip.compress(raw, compresslevel=9, mtime=0))
            write_new(destination / relative / 'timings.jsonl', (root / 'timings.jsonl').read_bytes())
            all_runs[str(relative)] = {'result': result, 'process_wall_seconds': execution['process_wall_seconds'],
                                      'raw_result_sha256': file_hash(result_path), 'private_checkpoint_sha256': file_hash(root / 'checkpoint.pt')}
        write_new(destination / (phase + '_campaign.json'), (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    final = sorted([value for key, value in all_runs.items() if key.startswith('final/')], key=lambda row: row['result']['identity']['seed'])
    if [row['result']['identity']['seed'] for row in final] != [42, 43, 44, 45, 46]:
        raise ValueError('Final seeds differ from registration')
    domain_stats = {domain: stats([row['result']['evaluations'][0]['domains'][domain]['accuracy_percent'] for row in final]) for domain in DOMAINS}
    aggregates = {metric: stats([row['result']['evaluations'][0][metric] for row in final])
                  for metric in ('uniform_domain_accuracy_percent', 'sample_weighted_accuracy_percent')}
    published = rivals()
    comparisons = [{'method': row['method'], 'domain': row['domain'], 'published_mean_percent': float(row['mean_accuracy_percent']),
                    'published_sd_percent': float(row['published_sd_percent']), 'published_sd_ddof': 'unspecified',
                    'ours_mean_percent': domain_stats[row['domain']]['mean'],
                    'ours_minus_published_percentage_points': domain_stats[row['domain']]['mean'] - float(row['mean_accuracy_percent']),
                    'comparison': 'descriptive, published results; no common baseline reruns'} for row in published]
    pilot_path = REPO / 'research/feature_shift_digits/artifacts/summary.json'
    pilot = json.loads(pilot_path.read_text())
    if pilot['partition_sha256'] != plan['parent_partition_sha256'] or any(pilot['domains'][domain]['seeds'] != [42, 43, 44, 45, 46] for domain in DOMAINS):
        raise ValueError('Prior pilot uses another partition/seed set')
    pilot_comparison = {domain: {'pilot_mean_percent': pilot['domains'][domain]['mean'],
                                  'pilot_sample_sd_ddof1': pilot['domains'][domain]['std_ddof1'],
                                  'new_minus_pilot_percentage_points': domain_stats[domain]['mean'] - pilot['domains'][domain]['mean'],
                                  'paired_seed_differences_percentage_points': [new - old for new, old in zip(domain_stats[domain]['values'], pilot['domains'][domain]['values'])]}
                        for domain in DOMAINS}
    costs, attempts = [], []
    for key, row in all_runs.items():
        result = row['result']
        entry = {'run': key, 'seed': result['identity']['seed'], 'candidate_id': result['configuration']['candidate_id'],
                 'phase': result['identity']['phase'], 'uniform_accuracy_percent': result['evaluations'][0]['uniform_domain_accuracy_percent'],
                 'per_domain_accuracy_percent': {domain: result['evaluations'][0]['domains'][domain]['accuracy_percent'] for domain in DOMAINS},
                 'process_wall_seconds': row['process_wall_seconds'], 'total_session_wall_seconds': result['total_session_wall_seconds'],
                 'peak_cuda_allocated_mib': result['peak_cuda_allocated_mib'], 'peak_cuda_reserved_mib': result['peak_cuda_reserved_mib'],
                 'peak_rss_mib': result['peak_rss_mib'], 'counted_training_flops': result['total_counted_training_flops'],
                 'dense_training_flops': result['total_dense_training_flops'],
                 'warmup_steps': sum(m['warmup_steps'] for h in result['history'] for m in h['clients'].values()),
                 'classification_steps': sum(m['classification_steps'] for h in result['history'] for m in h['clients'].values()),
                 'logical_communication_bytes': result['costs']['total_logical_communication_bytes'],
                 'device': result['identity']['device'], 'commit': result['identity']['code']['commit'],
                 'raw_result_sha256': row['raw_result_sha256'], 'private_checkpoint_sha256': row['private_checkpoint_sha256']}
        costs.append(entry)
        if entry['phase'] != 'final':
            attempts.append(entry)
    interval_seconds = (datetime.fromisoformat(campaigns['final']['ended_utc']) - datetime.fromisoformat(campaigns['screening']['started_utc'])).total_seconds()
    summary = {'method': 'FusedSpaceFed', 'seeds': [42, 43, 44, 45, 46], 'result_origin': 'ours',
               'evaluation': 'single fixed test at300, no adaptation, all five fresh seeds reported',
               'selected_configuration': selection, 'plan_sha256': plan['plan_sha256'],
               'partition_sha256': plan['parent_partition_sha256'], 'validation_sha256': plan['validation_sha256'],
               'per_domain': domain_stats, 'aggregates': aggregates, 'published_comparisons': comparisons,
               'prior_pilot_comparison': pilot_comparison, 'prior_pilot_summary_sha256': file_hash(pilot_path),
               'published_reference_csv_sha256': file_hash(REPO / 'research/feature_shift_digits/fedbn_table11.csv'),
               'calibration_attempts': attempts, 'all_run_costs': costs, 'campaigns': campaigns,
               'runtime_cost': {'campaign_interval_wall_seconds': interval_seconds,
                                'phase_wall_seconds_sum': sum(row['wall_seconds'] for row in campaigns.values()),
                                'worker_process_seconds_sum': sum(row['process_wall_seconds_sum'] for row in campaigns.values()),
                                'counted_training_flops_sum': sum(row['counted_training_flops'] for row in costs),
                                'dense_training_flops_sum': sum(row['dense_training_flops'] for row in costs),
                                'includes_warmup': True},
               'parameters': final[0]['result']['costs']['parameters'], 'scientific_source_sha256': expected_source,
               'prior_campaigns_preserved': ['research/feature_shift_digits', 'research/capacity_compute_control'],
               'limits': plan['limitations'] + ['Published baseline comparisons are not controlled reruns; see PROTOCOL.md.',
                                              'Published standard deviations use an unspecified ddof; ours are sample SD ddof=1.',
                                              'FLOPs are conventional dense/update counts and exclude operations listed in the source profile.']}
    atomic_json(destination / 'summary.json', summary)
    with (destination / 'final_accuracy.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', *DOMAINS, 'uniform_domain_percent', 'sample_weighted_percent', 'origin'])
        for item in final:
            result = item['result']
            evaluation = result['evaluations'][0]
            writer.writerow([result['identity']['seed'], *[evaluation['domains'][domain]['accuracy_percent'] for domain in DOMAINS],
                             evaluation['uniform_domain_accuracy_percent'], evaluation['sample_weighted_accuracy_percent'], 'ours'])
    report = make_report(summary, plan, shortlist)
    write_new(PUBLIC / 'CALIBRATED_DIGITS_REPORT.md', report.encode())
    private_report = PRIVATE / 'CALIBRATED_DIGITS_REPORT.md'
    write_new(private_report, report.encode())
    manifest = {'schema': 1, 'partition_sha256': plan['parent_partition_sha256'], 'validation_sha256': plan['validation_sha256'],
                'selection_sha256': selection['selection_sha256'], 'run_seeds': summary['seeds'],
                'training_commits': sorted({row['commit'] for row in costs}),
                'package_commit': 'recorded by the Git commit that archives this manifest',
                'excluded': ['datasets', 'checkpoints', 'complete stdout/stderr', 'private reviews/credentials'], 'files': {}}
    for path in sorted(PUBLIC.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name != 'artifact_manifest.json' and path.suffix != '.pyc':
            manifest['files'][str(path.relative_to(PUBLIC))] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    manifest['manifest_sha256'] = canonical_hash(manifest)
    atomic_json(PUBLIC / 'artifact_manifest.json', manifest)
    print(json.dumps({'final_uniform': aggregates['uniform_domain_accuracy_percent'], 'selection': selection['selected_training_settings'],
                      'archived_runs': len(all_runs), 'manifest_sha256': manifest['manifest_sha256']}, indent=2))


def make_report(summary, plan, shortlist):
    selected = summary['selected_configuration']
    params = summary['parameters']
    all_costs = summary['all_run_costs']
    final_costs = [row for row in all_costs if row['phase'] == 'final']
    rows = ['# FusedSpaceFed Digits: calibrazione e cinque nuove run', '',
            'Campagna completata: dieci screening, conferme a300 round, una configurazione congelata e cinque nuove inizializzazioni sul training completo. Il pilota resta conservato. Nessuna baseline è stata rieseguita e il manoscritto non è stato modificato.', '',
            '## Risultati finali e competitività', '',
            'Accuratezze percentuali al solo round300. Media e deviazione standard campionaria (ddof=1) comprendono tutti i seed42–46.', '',
            '| Dominio | 42 | 43 | 44 | 45 | 46 | Media ± SD | FedAvg pubblicato | FedProx pubblicato | FedBN pubblicato |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    published = {(row['method'], row['domain']): row for row in summary['published_comparisons']}
    for domain in DOMAINS:
        value = summary['per_domain'][domain]
        references = [published[method, domain] for method in ('FedAvg', 'FedProx', 'FedBN')]
        rows.append(f"| {domain} | " + ' | '.join(f'{v:.4f}' for v in value['values']) + f" | {value['mean']:.4f} ± {value['sample_standard_deviation_ddof1']:.4f} | " + ' | '.join(f"{r['published_mean_percent']:.2f} ± {r['published_sd_percent']:.2f}" for r in references) + ' |')
    rows.extend(['', '| Dominio | Δ FedAvg (pp) | Δ FedProx (pp) | Δ FedBN (pp) |', '|---|---:|---:|---:|'])
    for domain in DOMAINS:
        rows.append(f'| {domain} | ' + ' | '.join(f"{published[method, domain]['ours_minus_published_percentage_points']:+.4f}" for method in ('FedAvg', 'FedProx', 'FedBN')) + ' |')
    above = {method: [domain for domain in DOMAINS if published[method, domain]['ours_minus_published_percentage_points'] >= 0] for method in ('FedAvg', 'FedProx', 'FedBN')}
    rows.extend(['', 'Rispetto alle medie pubblicate, i domini con media Fused almeno pari sono: ' + '; '.join(f"{method}: {', '.join(domains) if domains else 'nessuno'}" for method, domains in above.items()) + '. Questo è un confronto descrittivo, non una dimostrazione di significatività o una replica esatta.', '',
                 'Baseline trascritte e verificate nel CSV originale: [appendice FedBN, tabella11, pagina PDF9/numero stampato21](https://michaelkamp.org/wp-content/uploads/2021/05/FedBN_appendix.pdf). Le loro SD sono pubblicate con ddof non specificato. Non si sono sostituite medie con i seed migliori.', ''])
    rows.extend(['| Dominio | Pilota conservato, media ± SD (%) | Δ nuovo−pilota (pp) |', '|---|---:|---:|'])
    for domain in DOMAINS:
        comparison = summary['prior_pilot_comparison'][domain]
        rows.append(f"| {domain} | {comparison['pilot_mean_percent']:.4f} ± {comparison['pilot_sample_sd_ddof1']:.4f} | {comparison['new_minus_pilot_percentage_points']:+.4f} |")
    rows.extend(['', 'Il confronto con il pilota usa gli stessi cinque seed, dati e protocollo. È riportato soltanto a campagna conclusa; non entra nel selettore della configurazione.', ''])
    for label, key in [('Media uniforme fra domini', 'uniform_domain_accuracy_percent'), ('Media pesata per immagini test', 'sample_weighted_accuracy_percent')]:
        value = summary['aggregates'][key]
        rows.append(f"- {label}: {value['mean']:.4f} ± {value['sample_standard_deviation_ddof1']:.4f}%; seed: " + ', '.join(f'{v:.4f}' for v in value['values']) + '.')
    rows.extend(['', 'La tabella FedBN riporta ogni dominio. La media uniforme è una nostra sintesi; la media pesata è dominata da SynthDigits (97.791 dei147.509 test) e non sostituisce i risultati per dominio.', '',
                 '## Calibrazione e congelamento', '',
                 f"Piano registrato prima delle nuove run, SHA256 `{plan['plan_sha256']}`. Partizione training-validation congelata: `{plan['validation_sha256']}`; 594 fit e149 validation per ciascuno dei cinque domini. Nessuna immagine test usata per il ranking.", '',
                 'La ricerca precedente a60 round favoriva LR/clip al bordo superiore della griglia. Sono stati registrati dieci screening a120 round (seed142): L9 su LR CNN {0.02,0.05,0.1}, LR AE {0.0001,0.0003,0.001}, clip {2,5,10}, più il pilota. Le due migliori configurazioni, più entrambi i riferimenti deduplicati, sono state confermate da zero a300 round sui seed142 e143. Il ranking finale usa la media delle due validation al round300; parità per LR CNN/clip/LR AE crescenti. Nessuna scelta del checkpoint.', '',
                 '| Fase | Configurazione | Seed | LR CNN | LR AE | Clip | Validation uniforme (%) |', '|---|---|---:|---:|---:|---:|---:|'])
    candidates = {row['id']: row['training'] for row in plan['candidates']}
    for row in summary['calibration_attempts']:
        settings = candidates[row['candidate_id']]
        rows.append(f"| {row['phase']} | {row['candidate_id']} | {row['seed']} | {settings['classifier_lr']} | {settings['autoencoder_lr']} | {settings['gradient_clip_norm']} | {row['uniform_accuracy_percent']:.4f} |")
    rows.extend(['', f"Congelata **{selected['selected_candidate_id']}**: LR CNN={selected['selected_training_settings']['classifier_lr']}, LR AE={selected['selected_training_settings']['autoencoder_lr']}, clip={selected['selected_training_settings']['gradient_clip_norm']}; media validation confermata={selected['uniform_validation_accuracy_percent']:.4f}%. SHA256 selezione `{selected['selection_sha256']}`. `selection.json` e le cinque configurazioni sono stati pubblicati prima delle nuove valutazioni test. Nessuna riapertura del tuning dopo il test.", '',
                 '## Configurazione finale, dati e verifiche', '',
                 'Stesso protocollo bilanciato D.2/tabella8:743 training per dominio,300 round, tutti i5 client, aggregazione uniforme, batch32, un’epoca warm-up MSE encoder e un’epoca CE encoder/decoder/CNN. CNN benchmark invariata; UNetSmallAE3 canali/base16/dz64; encoder privato persistente, D/C condivisi (inclusi BN), fusione additiva. SGD CNN senza momentum/decadimento; Adam AE betas(0.9,0.999), epsilon1e-8, senza decadimento. Stati optimizer resettati per partecipazione, Adam continuo fra le fasi. Float32, determinismo, niente AMP/TF32. Una sola valutazione test finale, senza adattamento o ricalibrazione BN.', '',
                 f"Dati `{plan['parent_partition_sha256']}`. Test: MNIST14.000, SVHN19.858, USPS1.860, SynthDigits97.791, MNIST-M14.000. Tutte le confusion matrix10×10, corrette/totali, medie, sequenze1–300, passi/budget, hash, sorgenti, exit code e sessioni sono verificati dall'audit senza una nuova valutazione del modello.", '',
                 '260 test CPU superati prima della campagna, compresi tutti i test esistenti: separazione training/validation, protocollo e riferimenti, assenza di test in calibrazione, configurazioni non registrate e finali non congelate respinte, persistenza dei privati e ripresa esatta di stati/RNG, controlli sui conteggi e sui tempi totali di ripresa. Le verifiche aggiuntive dell’archivio sono riportate nei file `verification*.log`.', '',
                 '## Tempi, memoria e calcolo', '',
                 '| Seed finale | GPU | Processo (s) | Sessione runner (s) | CUDA allocata/reservata (MiB) | RSS (MiB) |', '|---:|---|---:|---:|---:|---:|'])
    for row in sorted(final_costs, key=lambda row: row['seed']):
        rows.append(f"| {row['seed']} | {row['device']} | {row['process_wall_seconds']:.3f} | {row['total_session_wall_seconds']:.3f} | {row['peak_cuda_allocated_mib']:.3f}/{row['peak_cuda_reserved_mib']:.3f} | {row['peak_rss_mib']:.3f} |")
    rows.extend(['', f"Parametri per client: CNN {params['classifier']:,}, E privato {params['private_encoder_per_client']:,}, D condiviso {params['shared_decoder']:,}; totale attivo {sum(params.values()):,}. Ogni run finale registra36.000 passi warm-up e36.000 CE (72.000 encoder,36.000 decoder e36.000 CNN). Il warm-up è incluso nel calcolo.", ''])
    for phase, receipt in summary['campaigns'].items():
        rows.append(f"- {phase}: {len(receipt['runs'])} run; tempo controller {receipt['wall_seconds']/60:.3f} min; somma tempi processi {receipt['process_wall_seconds_sum']/60:.3f} min; exit code tutti0.")
    cost = summary['runtime_cost']
    rows.extend(['', f"Intervallo reale dall'inizio dello screening alla conclusione dei cinque seed: {cost['campaign_interval_wall_seconds']/60:.3f} min, compresi intervalli per selezione/commit. Somma processi: {cost['worker_process_seconds_sum']/3600:.4f} ore. Somma FLOP convenzionali training: {cost['counted_training_flops_sum']/1e15:.6f} PFLOP. Per ogni finale: {final_costs[0]['counted_training_flops']/1e12:.6f} TFLOP (dense forward/backward più costo dichiarato optimizer/clipping). BN, attivazioni, pooling, loss, copie, controlli finiti, aggregazione e overhead kernel non sono compresi: non è energia misurata o conteggio di istruzioni GPU.", '',
                 'Nessuna interruzione/ripresa o errore numerico nelle run archiviate. I tempi processo includono import/startup; quelli del runner includono verifica dati, addestramento, checkpoint e valutazione. Due processi propri al massimo, uno per GPU; nessuna azione sui lavori esterni. GPU0 è condivisa con `tesi_giovanni` per autorizzazione dell’utente.', '',
                 '## Riproduzione e artefatti', '',
                 'Piano, shortlist, selezione, configurazioni ed esatti comandi sono versionati. `artifacts/` contiene tutti i risultati JSON compressi senza perdita, i timings JSONL, i receipt dei controller, `summary.json` e `final_accuracy.csv`. `artifact_manifest.json` contiene hash/byte e commit di training. Dataset, checkpoint e log completi restano privati; SHA dei checkpoint sono registrati, i checkpoint non sono pubblicati.', '',
                 'Comandi eseguiti dalla radice, Python general_ml; per ogni fase:', '', '```bash',
                 '/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/controller.py \\',
                 '  --queue research/feature_shift_digits_calibrated/FASE_queue.json \\',
                 '  --receipt _local/feature_shift_digits_calibrated/FASE_campaign.json \\',
                 '  --logs _local/feature_shift_digits_calibrated/logs/FASE',
                 '# FASE = screening, confirmation, final; fra le fasi:',
                 '/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/select_stage.py --stage confirmation',
                 '/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/select_stage.py --stage final',
                 '/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/archive.py --verify', '```', '',
                 'Commit scientifici della campagna: ' + ', '.join(f'`{value}`' for value in sorted({row['commit'] for row in all_costs})) + '. Il commit finale di archivio è identificato dalla cronologia Git del manifesto.', '',
                 '## Limiti e interpretazione', '',
                 'Il pilota e i risultati del precedente controllo erano già noti: questa è una ricerca retrospettiva, con selezione numerica training-validation separata dal nuovo test. Una validation di745 esempi riutilizzata e due seed di conferma non escludono sovradattamento della ricerca. La griglia L9 non esaurisce le interazioni; nessuna affermazione di ottimalità degli iperparametri.', '',
                 'Il confronto con gli avversari pubblicati è descrittivo: nessuna baseline comune rieseguita, seed/split originali e modalità esatta della statistica non certificati. Dati da mirror successivo collegato al repository degli autori; identificatori di riga sono disgiunti nei file disponibili, non ricostruiscono gli ID originali prima del resplit. MNIST-M deriva da MNIST, quindi i domini non sono sorgenti totalmente indipendenti. Fused ha encoder/decoder e computazione warm-up aggiuntivi; stesso classificatore non implica pari costo. Il BN condiviso di Fused differisce dal BN locale di FedBN. Eccezioni, medie e tutti i seed rimangono visibili, senza selezione dopo il test.', ''])
    return '\n'.join(rows)


def verify_archive():
    manifest_path = PUBLIC / 'artifact_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    content = manifest.copy()
    expected = content.pop('manifest_sha256')
    if canonical_hash(content) != expected:
        raise ValueError('Manifest identity differs')
    for name, identity in manifest['files'].items():
        path = PUBLIC / name
        if path.stat().st_size != identity['bytes'] or file_hash(path) != identity['sha256']:
            raise ValueError(f'Archive file changed: {name}')
    expected_source = {name: file_hash(REPO / name) for name in SOURCES}
    summary = json.loads((PUBLIC / 'artifacts/summary.json').read_text())
    final = []
    for phase in ('screening', 'confirmation', 'final'):
        queue_path = PUBLIC / (phase + '_queue.json')
        queue = json.loads(queue_path.read_text())
        receipt = json.loads((PUBLIC / 'artifacts' / (phase + '_campaign.json')).read_text())
        validate_receipt(queue, receipt, queue_path)
        for job in queue['jobs']:
            directory = PUBLIC / 'artifacts' / phase / job['name']
            result = load_result(directory / 'results.json.gz')
            config = json.loads(Path(job['config']).read_text())
            validate_config(config)
            check_result(result, config, expected_source=expected_source)
            if [json.loads(line) for line in (directory / 'timings.jsonl').read_text().splitlines()] != result['history']:
                raise ValueError('Archived history/timings mismatch')
            expected_raw = next(row['raw_result_sha256'] for row in summary['all_run_costs'] if row['run'] == f"{phase}/{job['name']}")
            if hashlib.sha256(gzip.decompress((directory / 'results.json.gz').read_bytes())).hexdigest() != expected_raw:
                raise ValueError('Lossless result compression differs')
            if phase == 'final':
                final.append(result)
    final.sort(key=lambda row: row['identity']['seed'])
    if [row['identity']['seed'] for row in final] != [42, 43, 44, 45, 46]:
        raise ValueError('Wrong definitive seeds')
    for domain in DOMAINS:
        if stats([row['evaluations'][0]['domains'][domain]['accuracy_percent'] for row in final]) != summary['per_domain'][domain]:
            raise ValueError('Summary mean/SD differs from counts')
    for metric in summary['aggregates']:
        if stats([row['evaluations'][0][metric] for row in final]) != summary['aggregates'][metric]:
            raise ValueError('Summary aggregate mean/SD differs')
    print(json.dumps({'verified_files': len(manifest['files']), 'verified_runs': sum(len(json.loads((PUBLIC / (phase + '_queue.json')).read_text())['jobs']) for phase in ('screening', 'confirmation', 'final')),
                      'final_seeds': [row['identity']['seed'] for row in final], 'manifest_sha256': expected}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify_archive() if args.verify else build()
