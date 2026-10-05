"""Validate and publish the decoder probe; keep all full states private."""
import argparse
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
from research.capacity_compute_control.audit_results import finite, sha, stats, write

PUBLIC = REPO / 'research/digits_mechanism_diagnostics'
PHASE = PUBLIC / 'phase1'
PRIVATE = REPO / '_local/digits_mechanism_diagnostics/phase1'
DOMAINS = ('MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M')
ANCHORS = ('initialization', 'final-round300')
STAGES = ('before', 'after_warmup', 'after_classification')
METRICS = ('reconstruction_mse', 'relative_reconstruction_mse', 'decoder_to_input_rms_ratio',
           'decoder_mean', 'decoder_std_population', 'decoder_max_abs',
           'decoder_fraction_outside_input_range', 'fused_to_input_rms_ratio',
           'original_cross_entropy', 'fused_cross_entropy')


def summarize(records):
    return {a: {d: {s: {k: stats([r['anchors'][a]['domains'][d][s][k] for r in records])
                            for k in METRICS} for s in STAGES} for d in DOMAINS} for a in ANCHORS}


def check(records, config):
    if [r['seed'] for r in records] != config['seeds']:
        raise ValueError('Not all five registered seeds')
    for result in records:
        if result['status'] != 'completed' or not finite(result) or result['configuration'] != config:
            raise ValueError('Incomplete, nonfinite or changed configuration')
        for name, digest in result['source']['source_sha256'].items():
            if sha(REPO / name) != digest:
                raise ValueError('Diagnostic source changed')
        if set(result['anchors']) != set(ANCHORS):
            raise ValueError('Missing anchor')
        for anchor in result['anchors'].values():
            if set(anchor['domains']) != set(DOMAINS):
                raise ValueError('Missing client')
            for row in anchor['domains'].values():
                if row['warmup_steps'] != 24 or row['classification_steps'] != 24 or any(row[s]['samples'] != 160 for s in STAGES):
                    raise ValueError('Wrong replay/probe budget')
                if row['parameter_state_l2_changes']['warmup']['classifier'] or row['parameter_state_l2_changes']['warmup']['decoder']:
                    raise ValueError('Shared state changed during warm-up')
                if row['parameter_state_l2_changes']['warmup']['encoder'] <= 0 or any(v <= 0 for v in row['parameter_state_l2_changes']['classification'].values()):
                    raise ValueError('Expected components did not update')


def build():
    if (PHASE / 'artifacts').exists():
        raise FileExistsError('Preserve previous artifacts')
    config = json.loads((PHASE / 'config.json').read_text())
    campaign = json.loads((PRIVATE / 'campaign.json').read_text())
    if campaign['status'] != 'completed' or len(campaign['runs']) != 5 or any(r['exit_code'] != 0 for r in campaign['runs']):
        raise ValueError('Workers unfinished or failed')
    records = [json.loads((PRIVATE / f'seed-{seed}/results.json').read_text()) for seed in config['seeds']]
    check(records, config)
    artifacts = PHASE / 'artifacts'; artifacts.mkdir()
    checkpoints = {}
    for result in records:
        seed = result['seed']; source = PRIVATE / f'seed-{seed}'
        raw = (source / 'results.json').read_bytes()
        (artifacts / f'seed-{seed}.json.gz').write_bytes(gzip.compress(raw, mtime=0))
        if gzip.decompress((artifacts / f'seed-{seed}.json.gz').read_bytes()) != raw:
            raise ValueError('Lossy compression')
        for anchor in ANCHORS:
            saved = result['anchors'][anchor]
            path = source / anchor / 'anchor.pt'
            if sha(path) != saved['checkpoint_sha256']:
                raise ValueError('Anchor checkpoint changed')
            checkpoints[str(path.relative_to(REPO))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
            for domain in DOMAINS:
                path = source / anchor / (domain + '-phases.pt')
                if sha(path) != saved['domains'][domain]['local_replay_checkpoint_sha256']:
                    raise ValueError('Replay checkpoint changed')
                checkpoints[str(path.relative_to(REPO))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    figures = PHASE / 'figures'; figures.mkdir()
    for path in (PRIVATE / 'seed-42').glob('*.png'):
        shutil.copyfile(path, figures / path.name)
    if len(list(figures.glob('*.png'))) != 10:
        raise ValueError('Expected ten predeclared visual grids')
    write(artifacts / 'summary.json', {'seeds': config['seeds'], 'ddof': 1,
                                      'metrics': summarize(records), 'private_checkpoints': checkpoints})
    shutil.copyfile(PRIVATE / 'campaign.json', artifacts / 'campaign.json')
    shutil.copyfile(PRIVATE.parent / 'phase1_repository_tests.log', PHASE / 'verification_tests.log')
    write_report(records, campaign)
    files = {str(p.relative_to(PHASE)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in PHASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    refs = {str(p.relative_to(REPO)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in (PUBLIC / 'common.py', PUBLIC / 'controller.py', PUBLIC / 'probe.json', PUBLIC / 'tests/test_decoder_probe.py')}
    write(PHASE / 'manifest.json', {'schema': 1, 'files': files, 'references': refs,
                                  'checkpoint_policy': 'full states retained only in _local, SHA256 in summary'})


def write_report(records, campaign):
    summary = summarize(records)
    lines = ['# Fase 1 — Decoder e warm-up su Digits', '',
             'Cinque seed (42–46), due ancore: inizializzazione e checkpoint condiviso finale al round 300. Probe fisso di 160 esempi di training/client, 5 batch da 32, senza riuso e senza test. Ogni ancora viene copiata: una nuova partecipazione locale con warm-up 1 epoca (solo encoder) e classificazione 1 epoca (encoder/decoder/classifier). Gli stati intermedi sono replay diagnostici, non recupero degli stati storici del round 300. Optimizer resettati come nel protocollo esistente, Adam continuo tra le fasi.', '',
             'Architettura e configurazione calibrata invariate: UNetSmallAE dz=64, DigitCNN, LR classifier 0,1 SGD, LR AE 0,0003 Adam, clip 10, Float32, batch 32. Parametri scelti precedentemente solo su validation; nessuna nuova selezione. Il decoder termina con Conv2d lineare senza sigmoid/tanh: ampiezze esterne a [-1,1] sono ammesse e misurate.', '',
             'Medie su tutti i cinque seed. MSE, RMS e cross-entropy del probe sono calcolate in modalità eval con BN immutata. La sonda ripristina le modalità dei moduli e non consuma i generatori delle fasi di training. Valori per seed e SD campionarie (`ddof=1`) sono nel JSON, senza inferenze di significatività.', '']
    for anchor in ANCHORS:
        lines += [f'## {anchor}', '', '| Dominio | MSE prima → warm → CE | RMS decoder/input prima → warm → CE | CE fusa prima → warm → CE |', '|---|---:|---:|---:|']
        for domain in DOMAINS:
            values = summary[anchor][domain]
            sequence = lambda key: ' → '.join(f"{values[s][key]['mean']:.6f}" for s in STAGES)
            lines.append(f"| {domain} | {sequence('reconstruction_mse')} | {sequence('decoder_to_input_rms_ratio')} | {sequence('fused_cross_entropy')} |")
        warm_delta = statistics.mean(r['anchors'][anchor]['domains'][d]['after_warmup']['reconstruction_mse'] - r['anchors'][anchor]['domains'][d]['before']['reconstruction_mse'] for r in records for d in DOMAINS)
        ce_delta = statistics.mean(r['anchors'][anchor]['domains'][d]['after_classification']['reconstruction_mse'] - r['anchors'][anchor]['domains'][d]['after_warmup']['reconstruction_mse'] for r in records for d in DOMAINS)
        lines += ['', f'Variazione media MSE warm−prima: {warm_delta:+.6f}; CE−warm: {ce_delta:+.6f}. Sono variazioni locali del probe, non accuratezze di generalizzazione.', '']
    lines += ['## Stabilità e interpretazione', '',
              'Tutti gli output, loss, norme e stati sono finiti. Il warm-up lascia decoder e classifier identici bit per bit; aggiorna l’encoder. La classificazione aggiorna tutti e tre i componenti. Le variazioni L2 degli stati includono eventuali buffer BN, non sono sole norme dei parametri; le norme effettive dei gradienti e gli interventi di clipping sono registrati separatamente. Nessuna variazione del metodo è stata introdotta.', '',
              'L’obiettivo di ricostruzione agisce solo nel warm-up, con decoder congelato; la fase CE non garantisce ricostruzioni fedeli. La misura di ampiezza/coseno/MSE deve quindi distinguere ricostruzione da correzione utile al classificatore. Le sonde non provano un effetto causale del warm-up sulla performance finale: lo misurerà l’ablation riaddestrata della fase 2.', '',
              'Le griglie del checkpoint finale mostrano anche trasformazioni di colore e struttura spaziale nell’output del decoder: va interpretato come segnale appreso insieme al classificatore, non automaticamente come una copia visivamente fedele dell’immagine. Il caso visuale seed 42 è esemplificativo; statistiche su tutti i seed rimangono l’evidenza quantitativa.', '',
              '## Esempi visivi', '',
              'Seed 42 predefinito; quattro classi fissate (0,1,4,7) per dominio, primo esempio disponibile nella sonda. Ogni griglia mostra input, decoder e input+decoder prima/dopo warm-up e CE. Scala visuale fissa [-1,1], clipping soltanto nella resa grafica; nessun riscalamento individuale. Gli output grezzi non vengono tagliati nel training o nelle misure. Un’immagine non rappresenta la distribuzione complessiva: i 160 esempi e tutti i seed determinano le statistiche.', '']
    for domain in DOMAINS:
        lines += [f'### {domain}', '', f'![Inizializzazione {domain}](figures/initialization-{domain}.png)', '',
                  f'![Stato finale {domain}](figures/final-round300-{domain}.png)', '']
    lines += ['## Costi, archivio e riproduzione', '',
              f"Calendario fase: {campaign['wall_seconds']:.3f} s; somma processi: {campaign['process_wall_seconds_sum']:.3f} s. Un worker/GPU, entrambe usate, nessun processo preesistente interrotto. Cinque checkpoint originali riusati; sonde, stati per fase, RNG e checkpoint delle ancore conservati in `_local/digits_mechanism_diagnostics/phase1/`, con hash e dimensioni in summary.json.", '',
              '| Seed | Durata worker (s) | CUDA alloc/res (MiB) | RSS (MiB) |', '|---:|---:|---:|---:|']
    for result in records:
        lines.append(f"| {result['seed']} | {result['wall_seconds']:.3f} | {result['peak_cuda_allocated_mib']:.3f}/{result['peak_cuda_reserved_mib']:.3f} | {result['peak_rss_mib']:.3f} |")
    lines += ['', 'Comandi effettivi, PID, exit code e assegnazioni GPU in `artifacts/campaign.json`; configurazione e sonda congelate in config.json e ../probe.json. Per riprodurre, usare una nuova directory di output:', '',
              '```bash', '/home/schroeder/miniconda3/envs/general_ml/bin/python research/digits_mechanism_diagnostics/phase1/diagnose.py --seed 42 --device cuda:1 --output _local/digits_mechanism_diagnostics/phase1/reproduction-seed-42',
              'python research/digits_mechanism_diagnostics/phase1/archive.py verify', '```', '',
              'Test sintetici: metriche/range/nonfinite; hash e distanze degli stati; sonde senza mutazione di stati/RNG/BN/modi; flusso dei gradienti delle due fasi. Suite versionata completa e log allegati. Manoscritto e vecchi esperimenti invariati.', '',
              'Limiti: due stati e un singolo passo locale di replay, sonda train-only fissata e una partizione; il checkpoint finale ha già visto questi esempi. Non si estrapola una traiettoria di ricostruzione per tutti i 300 round, non si selezionano iperparametri da queste figure e non si identifica ancora una causa del vantaggio.']
    (PHASE / 'PHASE1_REPORT.md').write_text('\n'.join(lines) + '\n')


def verify():
    manifest = json.loads((PHASE / 'manifest.json').read_text())
    for root, entries in ((PHASE, manifest['files']), (REPO, manifest['references'])):
        for name, record in entries.items():
            path = root / name
            if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
                raise ValueError('Changed archive payload: ' + name)
    config = json.loads((PHASE / 'config.json').read_text())
    records = [json.loads(gzip.decompress((PHASE / f'artifacts/seed-{seed}.json.gz').read_bytes())) for seed in config['seeds']]
    check(records, config)
    summary = json.loads((PHASE / 'artifacts/summary.json').read_text())
    if summarize(records) != summary['metrics']:
        raise ValueError('Means/SD differ')
    return {'status': 'passed', 'seeds': config['seeds'], 'replay_clients': 50, 'probe_measurements': 150, 'visual_grids': 10}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'verify'))
    args = parser.parse_args()
    print(json.dumps(build() if args.mode == 'build' else verify(), indent=2))
