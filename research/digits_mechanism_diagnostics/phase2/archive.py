"""Count-based five-seed ablation audit; full checkpoint/logs remain private."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.capacity_compute_control.audit_results import canon, check_metrics, cost, epoch_batches, finite, sha, stats, write

PUBLIC = REPO / 'research/digits_mechanism_diagnostics'
PHASE = PUBLIC / 'phase2'
PRIVATE = REPO / '_local/digits_mechanism_diagnostics/phase2'
DOMAINS = ('MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M')
SEEDS = (42, 43, 44, 45, 46)
METHODS = ('full', 'no-warmup', 'shared-encoder', 'decoder-only')


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def gather(archived):
    plan = read(PHASE / 'plan.json')
    partition = read(REPO / 'research/feature_shift_digits/partition_manifest.json')
    profile = read(REPO / 'research/capacity_compute_control/flop_profile.json')
    if canon({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256'] or canon({k: v for k, v in profile.items() if k != 'profile_sha256'}) != profile['profile_sha256']:
        raise ValueError('Changed plan/profile content')
    counts = {d: partition['domains'][d]['test']['count'] for d in DOMAINS}
    records = {method: {} for method in METHODS}
    for method in METHODS:
        for seed in SEEDS:
            path = REPO / plan['reference_reuse'][str(seed)] if method == 'full' else (PHASE / 'artifacts' if archived else PRIVATE) / f'{method}-seed-{seed}' / ('results.json.gz' if archived else 'results.json')
            result = read(path)
            if not finite(result) or result['status'] != 'completed' or result['completed_rounds'] != 300 or [r['round'] for r in result['history']] != list(range(1, 301)):
                raise ValueError('Incomplete/nonfinite/incorrect round sequence')
            if result['identity']['seed'] != seed or result['configuration']['training'] != plan['training'] or result['identity']['partition_sha256'] != plan['partition_sha256']:
                raise ValueError('Seed/configuration/partition changed')
            if result['identity']['profile_sha256'] != profile['profile_sha256']:
                raise ValueError('Compute identity differs')
            if len(result['evaluations']) != 1 or result['evaluations'][0]['round'] != 300 or result['evaluations'][0]['split'] != 'test':
                raise ValueError('Wrong final evaluation schedule')
            check_metrics(result['evaluations'][0], counts)
            for d, m in result['evaluations'][0]['domains'].items():
                if [sum(row) for row in m['confusion_matrix']] != partition['domains'][d]['test']['labels']:
                    raise ValueError('Changed test distribution')
            if len(result['sessions']) != 1 or result['sessions'][0]['start_round'] != 1 or result['sessions'][0]['end_round'] != 300:
                raise ValueError('Expected fresh uninterrupted final run')
            if method == 'full':
                if result['configuration']['selection_sha256'] != plan['selected_reference_sha256']:
                    raise ValueError('Wrong full-method selection')
                source = result['identity']['code']['source_sha256']
            else:
                config = read(PHASE / f'configs/{method}-seed-{seed}.json')
                canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                if result['configuration'] != config or result['identity']['config_sha256'] != canonical or result['identity']['variant'] != method or result['identity']['plan_sha256'] != plan['plan_sha256']:
                    raise ValueError('Wrong registered variant/configuration')
                source = result['identity']['code']['source_sha256']
            for name, digest in source.items():
                if sha(REPO / name) != digest:
                    raise ValueError('Source bytes changed: ' + name)
            spec = {'warmup_epochs': 1} if method == 'full' else plan['variants'][method]
            phases = ('warmup', 'classification') if spec['warmup_epochs'] else ('classification',)
            per_client_cost = {kind: sum(cost(profile, phase, batch, kind == 'dense') for phase in phases for batch in epoch_batches(743)) for kind in ('counted', 'dense')}
            for row in result['history']:
                if set(row['clients']) != set(DOMAINS):
                    raise ValueError('Missing participant')
                for m in row['clients'].values():
                    expected = {'warmup_steps': 24 * spec['warmup_epochs'], 'classification_steps': 24,
                                'encoder_steps': 24 * (1 + spec['warmup_epochs']), 'decoder_steps': 24,
                                'classifier_steps': 24, 'warmup_samples': 743 * spec['warmup_epochs'], 'classification_samples': 743}
                    if any(m[key] != value for key, value in expected.items()):
                        raise ValueError('Wrong phase sample/update counts')
                    if any(m[kind + '_training_flops'] != value for kind, value in per_client_cost.items()):
                        raise ValueError('Wrong recorded compute')
            if any(result['total_' + kind + '_training_flops'] != value * 5 * 300 for kind, value in per_client_cost.items()):
                raise ValueError('Wrong total FLOPs')
            timing_path = path.parent / 'timings.jsonl'
            if [json.loads(line) for line in timing_path.read_text().splitlines()] != result['history']:
                raise ValueError('Timing/history differs')
            if not math.isclose(result['total_session_wall_seconds'], sum(s['wall_seconds'] for s in result['sessions']), rel_tol=0, abs_tol=1e-8):
                raise ValueError('Wrong whole-run session time')
            records[method][seed] = result
    return records


def summarize(records):
    result = {'seeds': list(SEEDS), 'ddof': 1, 'primary_metric': 'uniform_domain_accuracy_percent',
              'methods': {}, 'paired_full_minus_variant_pp': {}, 'costs_per_seed': {}}
    for method in METHODS:
        rows = [records[method][seed] for seed in SEEDS]
        result['methods'][method] = {key: stats([r['evaluations'][0][key] for r in rows])
                                     for key in ('uniform_domain_accuracy_percent', 'sample_weighted_accuracy_percent')}
        result['methods'][method]['domains'] = {d: stats([r['evaluations'][0]['domains'][d]['accuracy_percent'] for r in rows]) for d in DOMAINS}
        result['costs_per_seed'][method] = [{'seed': r['identity']['seed'], 'counted_flops': r['total_counted_training_flops'],
                                           'dense_flops': r['total_dense_training_flops'], 'session_seconds': r['total_session_wall_seconds'],
                                           'peak_cuda_allocated_mib': r['peak_cuda_allocated_mib'], 'peak_cuda_reserved_mib': r['peak_cuda_reserved_mib'],
                                           'peak_rss_mib': r['peak_rss_mib']} for r in rows]
        if method != 'full':
            result['paired_full_minus_variant_pp'][method] = {
                key: stats([records['full'][seed]['evaluations'][0][key] - records[method][seed]['evaluations'][0][key] for seed in SEEDS])
                for key in ('uniform_domain_accuracy_percent', 'sample_weighted_accuracy_percent')}
            result['paired_full_minus_variant_pp'][method]['domains'] = {
                d: stats([records['full'][seed]['evaluations'][0]['domains'][d]['accuracy_percent'] -
                          records[method][seed]['evaluations'][0]['domains'][d]['accuracy_percent'] for seed in SEEDS]) for d in DOMAINS}
    return result


def build():
    if (PHASE / 'artifacts').exists():
        raise FileExistsError('Preserve an existing archive')
    campaign = read(PRIVATE / 'campaign.json')
    if campaign['status'] != 'completed' or len(campaign['runs']) != 15 or any(r['exit_code'] != 0 for r in campaign['runs']):
        raise ValueError('Workers not all completed successfully')
    records = gather(False)
    # Inspect all actual saved states, independently of the training metrics.
    import torch
    torch.set_num_threads(4)
    for method in METHODS[1:]:
        for seed in SEEDS:
            path = PRIVATE / f'{method}-seed-{seed}/checkpoint.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['results'] != records[method][seed]:
                raise ValueError('Checkpoint/result identity differs')
            states = [checkpoint['classifier'], checkpoint['decoder'], *checkpoint['encoders'].values()]
            if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for state in states for t in state.values()):
                raise ValueError('Non-finite saved weights/buffers')
            first = checkpoint['encoders'][DOMAINS[0]]
            equal = all(all(torch.equal(t, first[k]) for k, t in state.items()) for state in checkpoint['encoders'].values())
            if equal != (method == 'shared-encoder'):
                raise ValueError('Actual private/shared encoder states differ from intervention')
            del checkpoint, states
    artifacts = PHASE / 'artifacts'; artifacts.mkdir()
    private_checkpoints = {}
    for method in METHODS[1:]:
        for seed in SEEDS:
            source = PRIVATE / f'{method}-seed-{seed}'; target = artifacts / source.name; target.mkdir()
            raw = (source / 'results.json').read_bytes(); (target / 'results.json.gz').write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress((target / 'results.json.gz').read_bytes()) != raw:
                raise ValueError('Lossy result compression')
            shutil.copyfile(source / 'timings.jsonl', target / 'timings.jsonl')
            path = source / 'checkpoint.pt'
            private_checkpoints[source.name] = {'path': str(path.relative_to(REPO)), 'sha256': sha(path), 'bytes': path.stat().st_size,
                                                'raw_result_sha256': sha(source / 'results.json')}
    write(artifacts / 'private_checkpoints.json', private_checkpoints)
    write(artifacts / 'summary.json', summarize(records))
    with (artifacts / 'accuracy.csv').open('w') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['variant', 'seed', *DOMAINS, 'uniform_domain_percent', 'sample_weighted_percent'])
        for method in METHODS:
            for seed in SEEDS:
                e = records[method][seed]['evaluations'][0]
                writer.writerow([method, seed, *[e['domains'][d]['accuracy_percent'] for d in DOMAINS], e['uniform_domain_accuracy_percent'], e['sample_weighted_accuracy_percent']])
    shutil.copyfile(PRIVATE / 'campaign.json', artifacts / 'campaign.json')
    shutil.copyfile(PRIVATE.parent / 'phase2_final_repository_tests.log', PHASE / 'verification_tests.log')
    if (PRIVATE / 'estimates.jsonl').exists():
        shutil.copyfile(PRIVATE / 'estimates.jsonl', artifacts / 'estimates.jsonl')
    report(records, campaign)
    files = {str(p.relative_to(PHASE)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in PHASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    refs = [PUBLIC / 'common.py', PUBLIC / 'controller.py', PUBLIC / 'tests/test_component_ablations.py',
            PUBLIC / 'tests/test_ablation_archive.py', REPO / 'research/capacity_compute_control/flop_profile.json',
            REPO / 'research/feature_shift_digits/partition_manifest.json', REPO / 'research/feature_shift_digits_calibrated/selection.json']
    for seed in SEEDS:
        path = REPO / read(PHASE / 'plan.json')['reference_reuse'][str(seed)]; refs += [path, path.parent / 'timings.jsonl']
    write(PHASE / 'manifest.json', {'schema': 1, 'files': files,
                                  'references': {str(p.relative_to(REPO)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in refs}})


def report(records, campaign):
    summary = summarize(records); mean_sd = lambda v: f"{v['mean']:.6f} ± {v['sample_sd_ddof1']:.6f}"
    lines = ['# Fase 2 — Ablation dei componenti su Digits', '',
             'Cinque seed abbinati 42–46 per quattro metodi. Le cinque run FusedSpaceFed complete calibrate sono riusate; 15 nuove run da zero delle tre ablation, ciascuna per 300 round. Dati e classificatore immutati, training completo 743/client; test una volta al round 300 sui cinque domini e 147.509 esempi, nessun adattamento/checkpoint/seed selection.', '',
             '| Variante | Intervento |', '|---|---|',
             '| full | Encoder privato persistente, decoder/classifier condivisi, x+D(Eᵢ(x)), 1 warm + 1 CE |',
             '| no-warmup | 0 epoche warm-up, 1 CE; encoder rimane privato |',
             '| shared-encoder | Encoder iniziale uguale per tutti; encoder aggiornati localmente e aggregati uniformemente insieme a decoder/classifier ogni round |',
             '| decoder-only | Classifica D(Eᵢ(x)) senza aggiunta di x; warm-up e encoder privato mantenuti |', '',
             'Stesse impostazioni già selezionate esclusivamente su validation da training (594 fit/149 validation/client) nella campagna calibrata precedente: LR C=0,1, LR AE=0,0003, clip=10; SGD senza momentum/decay per C, Adam standard per AE, reset a ogni partecipazione, Adam continuo warm→CE, batch 32, Float32, dz=64. Il template `training.warmup_epochs=1` identifica il metodo di riferimento; il campo esplicito `plan.variants.no-warmup.warmup_epochs=0` determina l’intervento nel runner. Nessun nuovo tuning: si tratta di un’ablation a iperparametri fissi selezionati su validation, non di varianti individualmente ottimizzate. Non si usano i test per cambiare le impostazioni.', '',
             '| Variante | Uniforme: media ± SD (%) | Pesata: media ± SD (%) | Full−variante uniforme (pp) |', '|---|---:|---:|---:|']
    for method in METHODS:
        values = summary['methods'][method]
        diff = mean_sd(summary['paired_full_minus_variant_pp'][method]['uniform_domain_accuracy_percent']) if method != 'full' else '—'
        lines.append(f"| {method} | {mean_sd(values['uniform_domain_accuracy_percent'])} | {mean_sd(values['sample_weighted_accuracy_percent'])} | {diff} |")
    lines += ['', 'SD campionaria tra tutti i cinque seed, ddof=1; differenze calcolate sui seed abbinati. Sono statistiche descrittive, non un test di significatività.', '',
              '| Seed | Full | No warm-up | Encoder condiviso | Solo decoder |', '|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(f"{records[m][seed]['evaluations'][0]['uniform_domain_accuracy_percent']:.6f}" for m in METHODS) + ' |')
    lines += ['', '| Dominio | Full | No warm-up | Encoder condiviso | Solo decoder |', '|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        lines.append('| ' + domain + ' | ' + ' | '.join(mean_sd(summary['methods'][m]['domains'][domain]) for m in METHODS) + ' |')
    lines += ['', 'Valori per dominio e per seed, conteggi, confusioni, differenze abbinate e risultati esatti sono nel CSV/JSON. Tutte le eccezioni di segno rimangono visibili: il ranking osservato non viene corretto scegliendo checkpoint o seed.', '',
              '## Contributi osservati', '']
    for method in METHODS[1:]:
        delta = summary['paired_full_minus_variant_pp'][method]['uniform_domain_accuracy_percent']
        wins = sum(v > 0 for v in delta['values'])
        direction = 'superiore' if delta['mean'] > 0 else 'inferiore'
        lines.append(f"- Rispetto a `{method}`, il metodo completo ha media {direction} di {abs(delta['mean']):.6f} pp e supera la variante in {wins}/5 seed. La SD delle differenze abbinate è {delta['sample_sd_ddof1']:.6f} pp.")
    lines += ['', 'Il confronto dell’encoder condiviso verifica la persistenza privata in questo setting bilanciato di soli cinque domini: un risultato competitivo o superiore della variante condivisa limita la necessità empirica dell’encoder privato qui. Non viene trasferito ai setting label-skew del manoscritto. Un vantaggio della fusione additiva rispetto al solo decoder rimane condizionato alla configurazione e alla partizione; un effetto piccolo del warm-up va letto insieme al suo costo.', '',
              '## Costo e lettura causale', '',
              'Le architetture per client e i parametri attivi sono identici (14.336.285), salvo la condivisione dell’encoder sul server. Il numero di epoche CE rimane uno per client/round. La rimozione del warm-up riduce passi di encoder e computazione e cambia lo stato Adam iniziale della CE: è l’effetto complessivo di rimuovere questa fase, non un confronto a FLOPs identici. Non consumando lo shuffle dell’epoca warm-up, questa variante ha lo stesso seed/inizializzazione ma non le stesse permutazioni CE del metodo completo; tutti gli esempi restano coperti una volta in ogni epoca CE. Le altre due ablation mantengono il numero di fasi e la sequenza dei generatori. Il costo dell’addizione è escluso dalla convenzione FLOPs esistente; decoder-only ha lo stesso costo contabile ma un diverso percorso funzionale.', '',
              '| Variante | FLOPs contabili/run | Dense FLOPs/run | Passi CE/run | Passi warm/run |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        r = records[method][42]
        ce = sum(c['classification_steps'] for row in r['history'] for c in row['clients'].values())
        warm = sum(c['warmup_steps'] for row in r['history'] for c in row['clients'].values())
        lines.append(f"| {method} | {r['total_counted_training_flops']} | {r['total_dense_training_flops']} | {ce} | {warm} |")
    lines += ['', 'La convenzione conta forward/backward convolution/matmul (FMA=2) e clipping/optimizer semantici; esclude BN/ReLU/pool/loss, copie, aggregazione, overhead e qui la comunicazione extra dell’encoder condiviso. Non misura energia o istruzioni hardware. Parametri, budget e sorgenti del controllo FedAvg restano separati e invariati.', '',
              f"Fase nuova: {campaign['wall_seconds']:.3f} s calendario, {campaign['process_wall_seconds_sum']:.3f} s somma processi, fino a 3 worker/GPU. Le cinque run full costano zero nuova esecuzione; i loro tempi originali sono registrati in summary.json. Nessun processo esterno interrotto, tutti i 15 exit 0, una sessione da zero per ogni run.", '',
              '| Variante | Durata media nuova/sessione (s) | CUDA alloc massimo (MiB) | CUDA res massimo (MiB) |', '|---|---:|---:|---:|']
    for method in METHODS:
        rows = [records[method][seed] for seed in SEEDS]
        lines.append(f"| {method} | {statistics.mean(r['total_session_wall_seconds'] for r in rows):.3f} | {max(r['peak_cuda_allocated_mib'] for r in rows):.3f} | {max(r['peak_cuda_reserved_mib'] for r in rows):.3f} |")
    lines += ['', '## Verifiche e riproduzione', '',
              'Parità bit per bit degli aggiornamenti del percorso completo con DigitsClient precedente su due round sintetici; input al classificatore verificato; componenti aggiornati e warm-up assente esplicito; encoder persistenti/distinti o identici dopo aggregazione secondo variante; ripresa esatta di classifier, decoder, encoder e RNG; rifiuto di overwrite/configurazioni non registrate. Suite versionata completa e log allegati.', '',
              'Audit indipendente di 20 run: seed/dati/configurazioni/sorgenti, 300 round consecutivi, una valutazione finale, cinque domini, conteggi/confusioni/accuratezze, sample SD, fasi/passaggi e costi. Checkpoint finali e RNG, log e risultati grezzi sono in `_local/digits_mechanism_diagnostics/phase2/`; SHA/byte in artifacts/private_checkpoints.json. Nessun dataset/checkpoint intero viene pubblicato. Manoscritto e risultati precedenti invariati.', '',
              '```bash', '/home/schroeder/miniconda3/envs/general_ml/bin/python research/digits_mechanism_diagnostics/phase2/runner.py --config research/digits_mechanism_diagnostics/phase2/configs/no-warmup-seed-42.json --device cuda:1 --output _local/digits_mechanism_diagnostics/phase2/reproduction-no-warmup-seed-42',
              'python research/digits_mechanism_diagnostics/phase2/archive.py verify', '```', '',
              'Limiti: un solo setting/partizione, cinque seed, configurazione comune scelta per il metodo completo. Una variante può rispondere diversamente al tuning; questo esperimento non ne stima il massimo raggiungibile. Non sono testate interazioni fra interventi combinati, altri dataset o adattamento finale. Le differenze sono condizionate al protocollo e non una prova universale di necessità dei componenti.']
    (PHASE / 'PHASE2_REPORT.md').write_text('\n'.join(lines) + '\n')


def verify():
    manifest = read(PHASE / 'manifest.json')
    actual = {str(p.relative_to(PHASE)) for p in PHASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    if actual != set(manifest['files']):
        raise ValueError('Archive payload set differs')
    for root, entries in ((PHASE, manifest['files']), (REPO, manifest['references'])):
        for name, expected in entries.items():
            path = root / name
            if sha(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
                raise ValueError('Changed archive file: ' + name)
    records = gather(True)
    if summarize(records) != read(PHASE / 'artifacts/summary.json'):
        raise ValueError('Summary differs from reconstructed counts/costs')
    with (PHASE / 'artifacts/accuracy.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if [(r['variant'], int(r['seed'])) for r in rows] != [(m, s) for m in METHODS for s in SEEDS]:
        raise ValueError('CSV row set differs')
    for row in rows:
        e = records[row['variant']][int(row['seed'])]['evaluations'][0]
        expected = {**{d: e['domains'][d]['accuracy_percent'] for d in DOMAINS},
                    'uniform_domain_percent': e['uniform_domain_accuracy_percent'],
                    'sample_weighted_percent': e['sample_weighted_accuracy_percent']}
        if any(float(row[k]) != value for k, value in expected.items()):
            raise ValueError('CSV differs from counts')
    sources = read(PHASE / 'artifacts/private_checkpoints.json')
    for name, identity in sources.items():
        raw = gzip.decompress((PHASE / f'artifacts/{name}/results.json.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest() != identity['raw_result_sha256']:
            raise ValueError('Uncompressed raw result identity differs')
    return {'status': 'passed', 'new_runs': 15, 'reused_full_runs': 5, 'seeds': list(SEEDS)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('build', 'verify'))
    args = parser.parse_args(); print(json.dumps(build() if args.mode == 'build' else verify(), indent=2))
