"""Archive two new runs and reconstruct the five-seed comparison from counts."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.capacity_compute_control.audit_results import (
    DOMAINS, audit_run, check_identity, check_metrics, finite, sha, stats, write,
)
from research.capacity_compute_control_five_seed.protocol import canonical_hash, validate_final_config

PUBLIC = REPO / 'research/capacity_compute_control_five_seed'
PRIVATE = REPO / '_local/capacity_compute_control_five_seed'
SEEDS = (42, 43, 44, 45, 46)
METHODS = ('FedAvg', 'FusedSpaceFed')


def result_path(method, seed, archived):
    if method == 'FusedSpaceFed':
        return REPO / f'research/feature_shift_digits_calibrated/artifacts/final/expanded-2-1-seed-{seed}/results.json.gz'
    if seed < 45:
        return REPO / f'research/capacity_compute_control/artifacts/final/FedAvg-seed-{seed}/results.json.gz'
    root = PUBLIC / 'artifacts' if archived else PRIVATE
    return root / f'FedAvg-seed-{seed}' / ('results.json.gz' if archived else 'results.json')


def read_result(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(raw)


def gather(*, archived):
    profile = json.loads((REPO / 'research/capacity_compute_control/flop_profile.json').read_text())
    partition = json.loads((REPO / 'research/feature_shift_digits/partition_manifest.json').read_text())
    counts = {domain: partition['domains'][domain]['test']['count'] for domain in DOMAINS}
    reference = json.loads((REPO / 'research/capacity_compute_control/configs/final/FedAvg-seed-42.json').read_text())
    records = {method: {} for method in METHODS}
    costs = {method: {} for method in METHODS}
    runtime = None
    for method in METHODS:
        for seed in SEEDS:
            path = result_path(method, seed, archived)
            result = read_result(path)
            config = result['configuration']
            identity = result['identity']
            check_identity(result, config)
            if identity['method'] != method or identity['seed'] != seed or config['seed'] != seed:
                raise ValueError('Wrong registered method/seed')
            if identity['partition_sha256'] != partition['partition_sha256'] or identity['profile_sha256'] != profile['profile_sha256'] or identity['validation_sha256'] != reference['validation_sha256']:
                raise ValueError('Frozen data/compute identity differs')
            if method == 'FedAvg':
                if config != {**reference, 'seed': seed}:
                    raise ValueError('FedAvg protocol changed across seeds')
                if seed >= 45:
                    validate_final_config(config)
                cost = audit_run(result, profile, 743, 300, 'final', counts)
            else:
                expected_training = {**reference['training'], 'classifier_lr': .1,
                                     'autoencoder_lr': .0003, 'gradient_clip_norm': 10.0}
                if config['training'] != expected_training or config['phase'] != 'final':
                    raise ValueError('Wrong previously calibrated Fused configuration')
                if config['selection_sha256'] != 'f43e6cb2034792de61a416901388df75a778610783c43891bf7bfc69c7c83edc':
                    raise ValueError('Wrong calibrated Fused selection')
                cost = audit_run(result, profile, 743, 300, 'test', counts, reused=True)
                for kind in ('counted', 'dense'):
                    executed = sum(c[kind + '_training_flops'] for r in result['history'] for c in r['clients'].values())
                    if executed != result['total_' + kind + '_training_flops'] or executed != cost[kind + '_training_flops']:
                        raise ValueError('Fused saved/independently counted FLOPs differ')
            if result['evaluations'][0]['split'] != ('final' if method == 'FedAvg' else 'test'):
                raise ValueError('Evaluation split differs')
            for domain, metric in result['evaluations'][0]['domains'].items():
                if [sum(row) for row in metric['confusion_matrix']] != partition['domains'][domain]['test']['labels']:
                    raise ValueError('Test class counts differ')
            if len(result['sessions']) != 1 or result['sessions'][0]['start_round'] != 1 or result['sessions'][0]['end_round'] != 300:
                raise ValueError('Expected uninterrupted initialization-to-300 run')
            if not math.isclose(sum(s['wall_seconds'] for s in result['sessions']), result['total_session_wall_seconds'], rel_tol=0, abs_tol=1e-8):
                raise ValueError('Incorrect total session time')
            timings = path.parent / 'timings.jsonl'
            if [json.loads(line) for line in timings.read_text().splitlines()] != result['history']:
                raise ValueError('Timings differ from saved history')
            current_runtime = {key: value for key, value in result['runtime'].items() if key != 'device'}
            if runtime is not None and current_runtime != runtime:
                raise ValueError('Environment differs beyond GPU assignment')
            runtime = current_runtime
            records[method][seed] = result
            costs[method][str(seed)] = cost
    return records, costs


def summarize(records, costs):
    output = {'seeds': list(SEEDS), 'ddof': 1, 'primary_metric': 'uniform_domain_accuracy_percent',
              'methods': {}, 'paired_fused_minus_fedavg': {}, 'costs': costs,
              'comparison_limit': 'Matched parameters and counted training FLOPs; unequal tuning effort. Descriptive, not an equal-tuning causal control.'}
    for method in METHODS:
        output['methods'][method] = {
            metric: stats([records[method][seed]['evaluations'][0][metric] for seed in SEEDS])
            for metric in ('uniform_domain_accuracy_percent', 'sample_weighted_accuracy_percent')}
        output['methods'][method]['domains'] = {
            domain: stats([records[method][seed]['evaluations'][0]['domains'][domain]['accuracy_percent'] for seed in SEEDS])
            for domain in DOMAINS}
    for metric in ('uniform_domain_accuracy_percent', 'sample_weighted_accuracy_percent'):
        output['paired_fused_minus_fedavg'][metric] = stats([
            records['FusedSpaceFed'][seed]['evaluations'][0][metric] - records['FedAvg'][seed]['evaluations'][0][metric]
            for seed in SEEDS])
    output['paired_fused_minus_fedavg']['domains'] = {
        domain: stats([records['FusedSpaceFed'][seed]['evaluations'][0]['domains'][domain]['accuracy_percent'] -
                       records['FedAvg'][seed]['evaluations'][0]['domains'][domain]['accuracy_percent'] for seed in SEEDS])
        for domain in DOMAINS}
    return output


def archive_new_runs():
    artifacts = PUBLIC / 'artifacts'
    if artifacts.exists():
        raise FileExistsError('Archive already exists; verify instead of overwriting')
    receipt = json.loads((PRIVATE / 'campaign.json').read_text())
    if receipt['status'] != 'completed' or len(receipt['runs']) != 2 or any(r['exit_code'] != 0 for r in receipt['runs']):
        raise ValueError('Workers not successfully completed')
    if {(r['name'], r['device']) for r in receipt['runs']} != {('FedAvg-seed-45', 'cuda:1'), ('FedAvg-seed-46', 'cuda:0')}:
        raise ValueError('Wrong two-worker campaign')
    if receipt['queue_sha256'] != sha(PUBLIC / 'queue.json'):
        raise ValueError('Queue changed after launch')
    records, costs = gather(archived=False)
    artifacts.mkdir()
    raw_sources = {}
    for seed in (45, 46):
        source = PRIVATE / f'FedAvg-seed-{seed}'
        target = artifacts / source.name
        target.mkdir()
        raw = (source / 'results.json').read_bytes()
        with (target / 'results.json.gz').open('wb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                compressed.write(raw)
        if gzip.decompress((target / 'results.json.gz').read_bytes()) != raw:
            raise ValueError('Compression is not lossless')
        shutil.copyfile(source / 'timings.jsonl', target / 'timings.jsonl')
        raw_sources[str(seed)] = {'raw_result_sha256': sha(source / 'results.json'),
                                 'private_checkpoint_sha256': sha(source / 'checkpoint.pt'),
                                 'private_checkpoint_bytes': (source / 'checkpoint.pt').stat().st_size}
    write(artifacts / 'new_run_sources.json', raw_sources)
    shutil.copyfile(PRIVATE / 'campaign.json', artifacts / 'campaign.json')
    output = summarize(records, costs)
    write(artifacts / 'summary.json', output)
    with (artifacts / 'accuracy.csv').open('w') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['method', 'seed', *DOMAINS, 'uniform_domain_percent', 'sample_weighted_percent'])
        for method in METHODS:
            for seed in SEEDS:
                e = records[method][seed]['evaluations'][0]
                writer.writerow([method, seed, *[e['domains'][d]['accuracy_percent'] for d in DOMAINS],
                                 e['uniform_domain_accuracy_percent'], e['sample_weighted_accuracy_percent']])
    return output


def manifest():
    files = {str(p.relative_to(PUBLIC)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in PUBLIC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'artifact_manifest.json'}
    references = {}
    for method in METHODS:
        for seed in SEEDS:
            if method == 'FedAvg' and seed >= 45:
                continue
            path = result_path(method, seed, True)
            for p in (path, path.parent / 'timings.jsonl'):
                references[str(p.relative_to(REPO))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    for p in (REPO / 'research/capacity_compute_control/artifact_manifest.json',
              REPO / 'research/feature_shift_digits_calibrated/artifact_manifest.json',
              REPO / 'research/capacity_compute_control/configs/final/FedAvg-seed-42.json'):
        references[str(p.relative_to(REPO))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    value = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
             'existing_immutable_references': references, 'new_run_training_base_commit': '76a338ccd3115b57a084b830067d00c341b0a435',
             'source_provenance': 'The two new run identities include hashes of the isolated extension files, uncommitted at launch and committed with this package; the original scientific sources are unchanged.'}
    value['manifest_sha256'] = canonical_hash(value)
    write(PUBLIC / 'artifact_manifest.json', value)


def verify_archive():
    value = json.loads((PUBLIC / 'artifact_manifest.json').read_text())
    if value['manifest_sha256'] != canonical_hash({key: item for key, item in value.items() if key != 'manifest_sha256'}):
        raise ValueError('Manifest identity differs')
    actual = {str(p.relative_to(PUBLIC)) for p in PUBLIC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'artifact_manifest.json'}
    if set(value['files']) != actual:
        raise ValueError('Archive file set differs')
    for root, entries in ((PUBLIC, value['files']), (REPO, value['existing_immutable_references'])):
        for name, identity in entries.items():
            path = root / name
            if sha(path) != identity['sha256'] or path.stat().st_size != identity['bytes']:
                raise ValueError('Archived or referenced bytes differ: ' + name)
    records, costs = gather(archived=True)
    reconstructed = summarize(records, costs)
    if reconstructed != json.loads((PUBLIC / 'artifacts/summary.json').read_text()):
        raise ValueError('Saved means/SD differ from reconstructed counts')
    with (PUBLIC / 'artifacts/accuracy.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if [(row['method'], int(row['seed'])) for row in rows] != [(m, s) for m in METHODS for s in SEEDS]:
        raise ValueError('CSV seed/method set differs')
    for row in rows:
        evaluation = records[row['method']][int(row['seed'])]['evaluations'][0]
        expected = {**{d: evaluation['domains'][d]['accuracy_percent'] for d in DOMAINS},
                    'uniform_domain_percent': evaluation['uniform_domain_accuracy_percent'],
                    'sample_weighted_percent': evaluation['sample_weighted_accuracy_percent']}
        if any(float(row[key]) != number for key, number in expected.items()):
            raise ValueError('CSV accuracy differs from saved counts')
    raw_sources = json.loads((PUBLIC / 'artifacts/new_run_sources.json').read_text())
    for seed in (45, 46):
        raw = gzip.decompress(result_path('FedAvg', seed, True).read_bytes())
        if hashlib.sha256(raw).hexdigest() != raw_sources[str(seed)]['raw_result_sha256']:
            raise ValueError('Uncompressed raw SHA differs')
    return {'status': 'passed', 'runs_verified': 10, 'seeds_per_method': list(SEEDS),
            'new_completed_runs': 2, 'payload_files': len(actual)}


def report():
    records, costs = gather(archived=True)
    summary = summarize(records, costs)
    campaign = json.loads((PUBLIC / 'artifacts/campaign.json').read_text())
    fused_old = json.loads((REPO / 'research/feature_shift_digits_calibrated/artifacts/summary.json').read_text())
    fused_tuning_flops = sum(row['counted_training_flops'] for row in fused_old['all_run_costs'] if row['phase'] != 'final')
    previous = json.loads((REPO / 'research/capacity_compute_control/artifacts/summary.json').read_text())
    mean_sd = lambda item: f"{item['mean']:.6f} ± {item['sample_sd_ddof1']:.6f}"
    lines = ['# Digits: controllo capacità/FLOPs su cinque seed', '',
             'Sono stati aggiunti solo i seed FedAvg 45 e 46, da zero per 300 round, con esattamente la configurazione dei seed 42–44. Tutte le run precedenti sono conservate. Il confronto usa i cinque FusedSpaceFed calibrati già disponibili, senza nuovi tuning o run Fused.', '',
             'Accuratezze in percentuale; media e SD campionaria tra seed (`ddof=1`). Metrica primaria: media uniforme tra i cinque domini; la metrica pesata per campioni è riportata separatamente.', '',
             '| Seed | FedAvg uniforme | Fused uniforme | Δ Fused−FedAvg (pp) | FedAvg pesata | Fused pesata |',
             '|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        a, f = [records[m][seed]['evaluations'][0] for m in METHODS]
        lines.append(f"| {seed} | {a['uniform_domain_accuracy_percent']:.6f} | {f['uniform_domain_accuracy_percent']:.6f} | {f['uniform_domain_accuracy_percent']-a['uniform_domain_accuracy_percent']:+.6f} | {a['sample_weighted_accuracy_percent']:.6f} | {f['sample_weighted_accuracy_percent']:.6f} |")
    lines += ['', '| Metrica | FedAvg: media ± SD | Fused: media ± SD | Δ abbinato: media ± SD (pp) |', '|---|---:|---:|---:|']
    for key in ('uniform_domain_accuracy_percent', 'sample_weighted_accuracy_percent'):
        lines.append(f"| {key} | {mean_sd(summary['methods']['FedAvg'][key])} | {mean_sd(summary['methods']['FusedSpaceFed'][key])} | {mean_sd(summary['paired_fused_minus_fedavg'][key])} |")
    lines += ['', '| Dominio | FedAvg: media ± SD | Fused: media ± SD | Δ medio (pp) |', '|---|---:|---:|---:|']
    for domain in DOMAINS:
        lines.append(f"| {domain} | {mean_sd(summary['methods']['FedAvg']['domains'][domain])} | {mean_sd(summary['methods']['FusedSpaceFed']['domains'][domain])} | {summary['paired_fused_minus_fedavg']['domains'][domain]['mean']:+.6f} |")
    lines += ['', 'I valori di ogni dominio per ogni seed sono in `artifacts/accuracy.csv` e `artifacts/summary.json`; risultati, conteggi corrette/totali e matrici di confusione permettono la ricostruzione esatta.', '',
              '## Protocollo e risorse', '',
              'Stesso Digits bilanciato: MNIST, SVHN, USPS, SynthDigits, MNIST-M; 743 training/client, 5 client partecipanti a ogni round, aggregazione uniforme, batch 32, Float32 senza AMP/TF32, SGD senza momentum/weight decay e reset a ogni partecipazione. Test una sola volta al round 300, senza adattamento né selezione del checkpoint. Conteggi test: 14.000 / 19.858 / 1.860 / 97.791 / 14.000, totale 147.509. Seed abbinati 42–46; una sola partizione congelata.', '',
              'FedAvg: CNN con larghezza del primo fully connected 2065 anziché 2048, 14.334.589 parametri. Fused: 14.336.285 parametri attivi/client (classifier 14.219.210 + encoder 72.080 + decoder 44.995), scarto FedAvg −0,011830%. Encoder privato persistente, decoder/classifier condivisi, fusione additiva, 1 epoca warm-up + 1 epoca classificazione. I campi AE/warm-up rimasti nel template JSON FedAvg sono inutilizzati dal suo percorso: non esegue un autoencoder.', '',
              'FedAvg mantiene il budget cumulativo esistente, che include il warm-up Fused: copre prima tutti i 743 esempi e poi aggiunge minibatch CE con riporto intero del residuo. Per run: FedAvg 551.896.914.235.800 FLOPs contabilizzati, Fused 551.898.005.448.000; deficit 0,000197720%. Dense-only: 547.381.518.701.000 vs 549.229.594.368.000. FedAvg 63.000 passi CE e 1.956.535 esposizioni CE; Fused 36.000 warm-up + 36.000 CE e 1.114.500 esposizioni CE.', '',
              'La convenzione conta convolution/matmul forward/backward misurati e clipping/optimizer semantici (FMA=2), esclude BN/ReLU/pool/loss, copie, aggregazione e overhead; non è misura di energia o istruzioni hardware. Matching approssimato della capacità e della computazione non rende equivalenti geometria, BN, personalizzazione o esposizioni supervisionate.', '',
              '## Tuning: procedure diverse, nessun nuovo tuning', '',
              'FedAvg conserva la selezione originaria: 9 candidati, 60 round ciascuno, seed 142; griglia LR C {0,005; 0,01; 0,02} × clip {0,5; 1; 2}. Validation derivata esclusivamente dal training: 594 fit e 149 validation/client, stratificata e congelata (seed 20261005). Selezione della media uniforme al round 60; LR 0,02 e clip 2. Il controllo originario Fused aveva lo stesso numero di tentativi, orizzonte e budget per round e resta intatto.', '',
              'I cinque Fused qui confrontati provengono dalla campagna successiva: 10 screening di 120 round, seed 142, seguiti da 4 candidati × 2 seed (142/143) × 300 round di conferma. LR C {0,02; 0,05; 0,1}, LR AE {0,0001; 0,0003; 0,001}, clip {2; 5; 10}, L9 con riferimento pilota aggiuntivo; top-2 più riferimenti obbligatori. Stessa validation. Selezione della media a round 300 sui due seed di conferma: LR C 0,1, LR AE 0,0003, clip 10. Configurazione congelata prima delle cinque run complete; nessuna selezione sul test né riapertura del tuning.', '',
              f"Costo del tuning: FedAvg {previous['validation_training_flops']['FedAvg']:,} FLOPs; campagna Fused successiva {fused_tuning_flops:,} FLOPs (esclusi i finali). **Lo sforzo di tuning non è equivalente**, e l'intervallo LR di FedAvg non include quello scelto per Fused. Il vantaggio descrittivo di questa tabella non dimostra che il solo meccanismo Fused causi la differenza a tuning equivalente.", '',
              'Il controllo precedente a sforzo comparabile rimane in `research/capacity_compute_control/CAPACITY_COMPUTE_REPORT.md`: tre seed, Fused 82,719265 ± 0,762178% vs FedAvg 82,220834 ± 0,062803%, differenza media +0,498431 pp. Non viene sostituito dalla presente campagna.', '',
              '## Esecuzione e verifiche', '', '| Metodo | Seed | GPU | Sessione totale (s) | CUDA alloc/res (MiB) | RSS (MiB) |', '|---|---:|---|---:|---:|---:|']
    for method in METHODS:
        for seed in SEEDS:
            r = records[method][seed]
            lines.append(f"| {method} | {seed} | {r['identity']['device']} | {r['total_session_wall_seconds']:.3f} | {r['peak_cuda_allocated_mib']:.3f}/{r['peak_cuda_reserved_mib']:.3f} | {r['peak_rss_mib']:.3f} |")
    lines += ['', f"Nuovi seed 45/46: {campaign['wall_seconds']:.3f} s di calendario in parallelo, {campaign['process_wall_seconds_sum']:.3f} s somma processi. Entrambi exit 0, una sessione da round 1 a 300, nessuna interruzione/ripresa. GPU 0 condivisa con il processo preesistente autorizzato; nessun processo altrui segnalato/interrotto. Picchi Torch per processo, esclusi contesto/driver e altri job. Tempi delle run riusate sono originali e non nuovo costo.", '',
              '283 test versionati passati (`pytest -q tests research`, 75,77 s), inclusi 12 dell’estensione. Parità byte del driver scientifico originale salvo registrazione dei due seed e metadati; configurazioni identiche salvo seed; parità esatta di due round sintetici (pesi, RNG, carry, metriche) e ripresa; medie, SD campionaria e abbinamento dei cinque seed. Una collisione del nome del nuovo file test è stata corretta senza modifiche al training. Un pytest senza perimetro includeva suite locali di archivio: il controllo che richiede un interprete senza Torch falliva nella suite combinata; i suoi 22 test sono passati eseguiti isolatamente, senza cambiarli. Log della verifica versionata incluso nel pacchetto. Audit: 10 run complete, 300 round consecutivi, un test finale, 5 domini/147.509 esempi, valori finiti, seed/config/data/source hash, conteggi/confusioni, timing e budget ricostruiti.', '',
              'Provenienza: nuovo training al base commit `76a338ccd3115b57a084b830067d00c341b0a435` con estensione isolata non ancora committata al lancio, identificata dagli SHA256 nei risultati e pubblicata nel commit dedicato di questo pacchetto. Le sorgenti scientifiche originali e i vecchi manifesti non sono cambiati. I record conservano i commit originali per tutte le run riusate.', '',
              'Hash congelati: partizione `7a762ffb10da74e3f0dee9a6f519e6c4057e2a5b47995a0546ee5a871eb47b58`; validation `3dfc26c3fda17f266fb2ffa9899f23b4a5b50cd006742bb84d5b215e412a6943`; profilo FLOPs `164f242a968689c6c1a819fd353317c0a858b8c316a80b8244da0ed8c53f6a12`. Fonti/versioni e preprocessing dettagliati sono conservati nel report originale `research/feature_shift_digits/FEATURE_SHIFT_REPORT.md`; non sono stati ricercati o modificati di nuovo.', '',
              '## Riproduzione e archivio', '', 'Comandi effettivi e codici di uscita sono in `artifacts/campaign.json`. Per una nuova directory di output:', '', '```bash']
    for job in json.loads((PUBLIC / 'queue.json').read_text())['jobs']:
        lines.append(f"/home/schroeder/miniconda3/envs/general_ml/bin/python research/capacity_compute_control_five_seed/runner.py --config {job['config']} --partition _local/feature_shift_digits/prepared --output {job['output']}-reproduction --device {job['device']}")
    lines += ['python research/capacity_compute_control_five_seed/audit.py verify', '```', '',
              'Archivio pubblico: risultati lossless e timings dei due nuovi seed, sintesi dei dieci risultati, CSV, receipt, configurazioni, test, driver isolato e manifesto con hash dei file nuovi e riferimenti immutabili ai risultati precedenti. Dataset, checkpoint e log completi restano in `_local/`; i checkpoint non sono pubblicati. Manoscritto, baseline pubblicate e altri esperimenti invariati.', '',
              'Limiti: un setting e una partizione, cinque seed condizionati a configurazioni selezionate, tuning diverso e test del setting già osservati nelle campagne precedenti. Nessuna varianza inventata, selezione del seed migliore o inferenza di significatività. Questo FedAvg ampliato/con budget aumentato è un nostro controllo e non il FedAvg pubblicato nella tabella FedBN.']
    (PUBLIC / 'FIVE_SEED_REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'report', 'manifest', 'verify'))
    args = parser.parse_args()
    operation = {'build': archive_new_runs, 'report': report, 'manifest': manifest, 'verify': verify_archive}[args.mode]
    print(json.dumps(operation(), indent=2))
