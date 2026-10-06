"""Reconstruct dispersion decomposition from saved Gram matrices, no model use."""
import argparse
import gzip
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
PHASE = PUBLIC / 'phase3'
PRIVATE = REPO / '_local/digits_mechanism_diagnostics/phase3'
SEEDS = (42, 43, 44, 45, 46)
ANCHORS = ('initialization', 'final-round300')
MODES = ('eval', 'batch-stateless')
FIELDS = ('Gamma_original', 'Gamma_fused', 'B', 'Phi', 'B_plus_Phi', 'fused_original_ratio')


def reconstructed(gram):
    count = len(gram) // 2
    block = lambda r, c: [row[c:c + count] for row in gram[r:r + count]]
    centered = lambda m: sum(m[i][i] for i in range(count)) / count - sum(map(sum, m)) / count**2
    oo, ff, of = block(0, 0), block(count, count), block(0, count)
    go, gf, covariance = centered(oo), centered(ff), centered(of)
    return {'Gamma_original': go, 'Gamma_fused': gf, 'B': go + gf - 2 * covariance, 'Phi': 2 * (covariance - go)}


def check_decomposition(value):
    count = value['client_count']; gram = value['gram_original_then_fused']
    if count != 5 or len(gram) != 10 or any(len(r) != 10 for r in gram) or value['parameter_count'] != 14219210:
        raise ValueError('Wrong client/parameter/Gram dimensions')
    reconstructed_values = reconstructed(gram)
    scale = max(value['Gamma_original'], value['Gamma_fused'], value['B'], abs(value['Phi']), 1e-30)
    if any(abs(value[k] - v) > 1e-9 * scale for k, v in reconstructed_values.items()):
        raise ValueError('Gram reconstruction differs from saved decomposition')
    residual = value['Gamma_fused'] - value['Gamma_original'] - value['B'] - value['Phi']
    if abs(residual) > 1e-10 * scale or value['B'] < 0:
        raise ValueError('Identity/sign failed')
    for name in ('original', 'fused'):
        if len(value[name]['client_gradient_norms']) != 5:
            raise ValueError('Missing client norms')


def check(records):
    config = json.loads((PHASE / 'config.json').read_text())
    if [r['seed'] for r in records] != list(SEEDS):
        raise ValueError('Missing seeds')
    for result in records:
        if result['status'] != 'completed' or not finite(result) or result['configuration'] != config:
            raise ValueError('Incomplete/nonfinite/configuration differs')
        for name, digest in result['source']['source_sha256'].items():
            if sha(REPO / name) != digest:
                raise ValueError('Diagnostic source changed')
        for anchor in ANCHORS:
            for mode in MODES:
                case = result['anchors'][anchor][mode]
                if case['state_before'] != case['state_after'] or not case['rng_unchanged']:
                    raise ValueError('Model/RNG state mutated')
                if len(case['per_batch']) != 5 or len(case['per_client_batches']) != 5:
                    raise ValueError('Missing batches/clients')
                for point in [case, *case['per_batch']]:
                    check_decomposition(point['classifier'])
                    d = point['decoder']; gram = d['gram']
                    expected = sum(gram[i][i] for i in range(5)) / 5 - sum(map(sum, gram)) / 25
                    if d['parameter_count'] != 44995 or not math.isclose(expected, d['Gamma_decoder'], rel_tol=1e-9, abs_tol=1e-20):
                        raise ValueError('Decoder Gram reconstruction differs')
                for rows in case['per_client_batches'].values():
                    if len(rows) != 5 or any(row['samples'] != 32 for row in rows):
                        raise ValueError('Wrong probe sample budget')


def summarize(records):
    output = {'seeds': list(SEEDS), 'ddof': 1, 'probe_samples_per_client': 160, 'batches_per_client': 5, 'cases': {}}
    for anchor in ANCHORS:
        output['cases'][anchor] = {}
        for mode in MODES:
            cases = [r['anchors'][anchor][mode] for r in records]
            result = {k: stats([case['classifier'][k] for case in cases]) for k in FIELDS}
            result['reduction_seed_count'] = sum(case['classifier']['Gamma_fused'] < case['classifier']['Gamma_original'] for case in cases)
            result['decoder_Gamma'] = stats([case['decoder']['Gamma_decoder'] for case in cases])
            result['decoder_normalized_dispersion'] = stats([case['decoder']['normalized_dispersion'] for case in cases])
            for label in ('original', 'fused'):
                result[label] = {key: stats([case['classifier'][label][key] for case in cases])
                                 for key in ('normalized_dispersion', 'mean_pairwise_cosine', 'mean_gradient_norm', 'mean_client_squared_gradient_norm')}
            output['cases'][anchor][mode] = result
    return output


def build():
    if (PHASE / 'artifacts').exists():
        raise FileExistsError('Archive already exists')
    campaign = json.loads((PRIVATE / 'campaign.json').read_text())
    if campaign['status'] != 'completed' or len(campaign['runs']) != 5 or any(r['exit_code'] != 0 for r in campaign['runs']):
        raise ValueError('Workers not all successfully completed')
    records = [json.loads((PRIVATE / f'seed-{seed}/results.json').read_text()) for seed in SEEDS]
    check(records); summary = summarize(records)
    artifacts = PHASE / 'artifacts'; artifacts.mkdir()
    for seed in SEEDS:
        raw = (PRIVATE / f'seed-{seed}/results.json').read_bytes()
        (artifacts / f'seed-{seed}.json.gz').write_bytes(gzip.compress(raw, mtime=0))
        if gzip.decompress((artifacts / f'seed-{seed}.json.gz').read_bytes()) != raw:
            raise ValueError('Lossy compression')
    write(artifacts / 'summary.json', summary)
    shutil.copyfile(PRIVATE / 'campaign.json', artifacts / 'campaign.json')
    shutil.copyfile(PRIVATE.parent / 'phase3_final_repository_tests.log', PHASE / 'verification_tests.log')
    figure(records); report(records, summary, campaign)
    files = {str(p.relative_to(PHASE)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in PHASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    refs = [PUBLIC / 'common.py', PUBLIC / 'controller.py', PUBLIC / 'probe.json', PUBLIC / 'README.md',
            PUBLIC / 'tests/test_fixed_state_gradients.py', PUBLIC / 'tests/test_gradient_gram_audit.py',
            PUBLIC / 'phase1/manifest.json', PUBLIC / 'phase1/artifacts/summary.json']
    write(PHASE / 'manifest.json', {'schema': 1, 'files': files,
                                  'references': {str(p.relative_to(REPO)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in refs}})


def figure(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mode in zip(axes, MODES):
        all_values = []
        for anchor, marker in zip(ANCHORS, ('o', 's')):
            points = [r['anchors'][anchor][mode]['classifier'] for r in records]
            x = [p['Gamma_original'] for p in points]; y = [p['Gamma_fused'] for p in points]
            ax.scatter(x, y, marker=marker, label=anchor); all_values += x + y
        low, high = min(all_values) * .7, max(all_values) * 1.4
        ax.plot([low, high], [low, high], color='gray', linewidth=1, linestyle='--', label='equal dispersion')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(low, high); ax.set_ylim(low, high)
        ax.set_xlabel('Gamma original'); ax.set_ylabel('Gamma fused'); ax.set_title(mode); ax.legend(fontsize=8)
    fig.suptitle('Same fixed state and training probe; every seed shown')
    fig.tight_layout(); fig.savefig(PHASE / 'gradient_dispersion.png', dpi=160); fig.savefig(PHASE / 'gradient_dispersion.pdf'); plt.close(fig)


def report(records, summary, campaign):
    lines = ['# Fase 3 — Diagnostica dei gradienti su Digits', '',
             'Cinque seed 42–46, stati iniziali e finali al round 300 riusati dalla fase 1. Stessa sonda train-only: 160 esempi/client in 5 batch da 32, stessi input/label per originale e fuso. Parametri classifier θ, decoder φ ed encoder privato ψᵢ sono identici nei due percorsi. Nessun optimizer step, adattamento, accesso al test o selezione di iperparametri.', '',
             'Per ogni client si media il gradiente delle cinque loss CE medie per batch (tutti della stessa dimensione), poi si calcola la dispersione dei cinque gradienti medi dei client. Si conservano anche i risultati per batch; **la media delle dispersioni per batch non viene sostituita alla dispersione dei gradienti medi**. Gradients e modello Float32; accumulo e riduzioni CPU Float64.', '',
             'Definizioni del manoscritto, paper/aistats_2027.tex, sezione Exact Fixed-Round Decomposition: gᵒᵢ=∂θℓ(Cθ(x),y); gᶠᵢ=∂θℓ(Cθ(x+Dφ(Eψᵢ(x))),y); bᵢ=gᶠᵢ−gᵒᵢ. Γᵒ/f=(1/5)Σ||gᵒ/fᵢ−mean(gᵒ/f)||²; B=(1/5)Σ||bᵢ−mean(b)||²; Φ=(2/5)Σ〈gᵒᵢ−mean(gᵒ), bᵢ−mean(b)〉. Γ fusa=Γ originale+B+Φ. Il divisore della dispersione è m=5; la SD tra seed usa ddof=1.', '',
             'Due modalità predefinite: `eval` con statistiche BN globali congelate (principale); `batch-stateless` con statistiche del batch e buffer ripristinati prima/dopo ogni ramo. In quest’ultima modalità l’obiettivo è condizionato ai batch perché BN accoppia i campioni. Tutti i valori dei parametri e dei buffer rimangono uguali; nessuna ricalibrazione delle statistiche. La sensibilità BN distingue allineamento all’inferenza da comportamento del gradiente nella modalità di training.', '']
    for anchor in ANCHORS:
        for mode in MODES:
            lines += [f'## {anchor} — {mode}', '', '| Seed | Γ originale | Γ fusa | B | Φ | Γf/Γo | Γ decoder |', '|---:|---:|---:|---:|---:|---:|---:|']
            for result in records:
                case = result['anchors'][anchor][mode]; c = case['classifier']
                lines.append(f"| {result['seed']} | {c['Gamma_original']:.8e} | {c['Gamma_fused']:.8e} | {c['B']:.8e} | {c['Phi']:.8e} | {c['fused_original_ratio']:.6f} | {case['decoder']['Gamma_decoder']:.8e} |")
            data = summary['cases'][anchor][mode]
            ratio = data['fused_original_ratio']
            lines += ['', f"Riduzione in {data['reduction_seed_count']}/5 seed. Rapporto Γf/Γo medio ± SD: {ratio['mean']:.6f} ± {ratio['sample_sd_ddof1']:.6f}.",
                      f"Dispersione normalizzata Γ/mean(||gᵢ||²), originale → fusa: {data['original']['normalized_dispersion']['mean']:.6f} → {data['fused']['normalized_dispersion']['mean']:.6f}. Coseno medio tra coppie di client: {data['original']['mean_pairwise_cosine']['mean']:.6f} → {data['fused']['mean_pairwise_cosine']['mean']:.6f}.", '']
    maximum = max(abs(r['anchors'][a][m]['classifier']['relative_identity_residual']) for r in records for a in ANCHORS for m in MODES)
    lines += ['## Interpretazione, scale e decoder', '',
              f'Identità verificata anche ricostruendo Γ/B/Φ dai Gram salvati, per 20 casi medi e 100 casi per batch. Residuo relativo massimo dei casi medi: {maximum:.3e}. B è non negativo; la riduzione richiede B+Φ<0, non soltanto Φ<0.', '',
              'Si riportano norme dei gradienti medi per client, norma del gradiente medio globale, energie, coseni e dispersione normalizzata per evitare di leggere un semplice calo di scala del gradiente come maggiore allineamento direzionale. Le norme per batch includono classifier originale/fuso, encoder, decoder e gruppo AE congiunto. Sono gradienti grezzi, prima del clipping; non coincidono con gli aggiornamenti Adam/SGD applicati durante il training.', '',
              'Al finale, in modalità eval la dispersione normalizzata aumenta da 0,794160 a 0,799588 e il coseno medio diminuisce da 0,090559 a 0,043966: la riduzione di Γ assoluta non è accompagnata da migliore allineamento direzionale. In batch-stateless la normalizzazione cambia poco (0,796595 → 0,789012) e il coseno passa da 0,021850 a 0,023454, nonostante un forte calo di Γ grezza. I risultati indicano soprattutto una diversa scala dei gradienti e sensibilità alla BN; non sostengono un miglioramento direzionale ampio o universale.', '',
              'Per il decoder si misura hᵢ=∂φℓ(Cθ(x+Dφ(Eψᵢ(x))),y) al medesimo φ condiviso, mediando prima i cinque batch. ΓD=(1/5)Σ||hᵢ−mean(h)||², norme, normalizzazione e Gram sono salvati. È uno spazio distinto, di 44.995 parametri, rispetto ai 14.219.210 del classifier: i due valori grezzi non vanno confrontati come se avessero la stessa dimensione. Il ramo originale non dipende da φ; non se ne inventa una baseline utile di dispersione decoder.', '',
              '![Dispersioni originali e fuse](gradient_dispersion.png)', '',
              '## Limiti e collegamento alle ablation', '',
              'È una diagnosi condizionata allo stesso stato Fused già allenato: il classifier è stato ottimizzato sull’input fuso, quindi il ramo originale è controfattuale e può avere loss o scala dei gradienti maggiori. La BN eval riflette statistiche apprese su input fusi; la sensibilità batch-stateless espone questo possibile effetto. Le metriche normalizzate e i coseni non sostituiscono Γ ma ne delimitano l’interpretazione. Una riduzione non prova causalità dell’accuratezza, minore drift lungo tutti i round o una garanzia di convergenza.', '',
              'La sonda ha visto training, conserva le proporzioni di classe diverse tra domini e non è un campione di test: la dispersione include anche label-mix, non solo feature shift. Cinque batch riducono il rumore rispetto a uno ma non recuperano l’aspettativa della distribuzione intera. Due stati, una partizione e cinque seed; nessuna estrapolazione alle partizioni label-skew del manoscritto.', '',
              'La fase 2 mostra encoder condiviso 86,178544 ± 0,468069%, full 85,862359 ± 0,846930%, no warm-up 85,714598 ± 0,789950%, decoder-only 84,859627 ± 0,580810%. Qui la persistenza privata non è sostenuta come necessaria, il warm-up ha piccolo effetto medio e maggiore costo, mentre l’aggiunta dell’input migliora in media il solo decoder. Le eccezioni per seed/dominio restano nel report della fase 2; Γ non annulla quei limiti.', '',
              '## Archivio, costi e verifiche', '',
              f"Calendario: {campaign['wall_seconds']:.3f} s; somma processi: {campaign['process_wall_seconds_sum']:.3f} s. Cinque valutazioni indipendenti in parallelo, 3 su GPU 1 e 2 su GPU 0, nessun processo esterno interrotto. Tutti gli exit code sono 0.", '',
              '| Seed | Durata worker (s) | CUDA alloc/res (MiB) | RSS (MiB) |', '|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['seed']} | {r['wall_seconds']:.3f} | {r['peak_cuda_allocated_mib']:.3f}/{r['peak_cuda_reserved_mib']:.3f} | {r['peak_rss_mib']:.3f} |")
    lines += ['', 'Risultati per seed e batch, Gram sufficienti a ricostruire la decomposizione, norme/loss e medie/SD sono in artifacts/. Config, source hash, checkpoint hash e sonda fissati; ancore complete e RNG riusati da `_local/digits_mechanism_diagnostics/phase1/`, risultati/log completi in `_local/digits_mechanism_diagnostics/phase3/`. Dataset e pesi non vengono versionati. Nessuna modifica al manoscritto.', '',
              '302 test versionati passati (`pytest -q tests research`, 78,61 s). Algebra esatta e fattore 2 del termine Φ; peso dei batch; indipendenza dai chunk; valori non finiti; parametri/buffer/RNG immutati in entrambe le modalità BN, nessuna .grad o optimizer step. Log allegato. Audit indipendente dei Gram e dell’identità in tutti i casi, sorgenti e completezza.', '',
              '```bash', '/home/schroeder/miniconda3/envs/general_ml/bin/python research/digits_mechanism_diagnostics/phase3/probe_gradients.py --seed 42 --device cuda:1 --output _local/digits_mechanism_diagnostics/phase3/reproduction-seed-42',
              'python research/digits_mechanism_diagnostics/phase3/archive.py verify', '```']
    (PHASE / 'PHASE3_REPORT.md').write_text('\n'.join(lines) + '\n')


def verify():
    manifest = json.loads((PHASE / 'manifest.json').read_text())
    for root, entries in ((PHASE, manifest['files']), (REPO, manifest['references'])):
        for name, expected in entries.items():
            path = root / name
            if sha(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
                raise ValueError('Changed payload: ' + name)
    records = [json.loads(gzip.decompress((PHASE / f'artifacts/seed-{seed}.json.gz').read_bytes())) for seed in SEEDS]
    check(records)
    if summarize(records) != json.loads((PHASE / 'artifacts/summary.json').read_text()):
        raise ValueError('Means/SD differ')
    return {'status': 'passed', 'seeds': list(SEEDS), 'mean_cases': 20, 'batch_cases': 100, 'model_updates': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('build', 'verify'))
    args = parser.parse_args(); print(json.dumps(build() if args.mode == 'build' else verify(), indent=2))
