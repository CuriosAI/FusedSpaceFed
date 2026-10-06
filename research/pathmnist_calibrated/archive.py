"""Auditable numeric archives, count/identity checks and reproducible reports."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.pathmnist_calibrated.data import PUBLIC,PRIVATE,verify
from research.pathmnist_calibrated.selection_tools import scores,grouped,require_campaign
from research.pathmnist_pathological.run import file_hash,write_json,assert_finite,json_hash


def compressed(source,target):
    raw=source.read_bytes();target.write_bytes(gzip.compress(raw,mtime=0))
    if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossy archive')


def manifest(directory):
    write_json(directory/'manifest.json',{'files_sha256':{str(p.relative_to(directory)):file_hash(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='manifest.json'}})


def calibration():
    selected=json.loads((PUBLIC/'selection.json').read_text())
    destination=PUBLIC/'artifacts/calibration'
    destination.mkdir(parents=True,exist_ok=False)
    campaigns={s:require_campaign(PRIVATE/(s+'_campaign.json')) for s in ('screening','confirmation','extension')}
    all_scores=scores('screening',20)+scores('confirmation',50)+scores('confirmation',100)
    checkpoints={};attempts=[]
    for stage in ('screening','confirmation'):
        for directory in sorted((PRIVATE/stage).iterdir()):
            if not directory.is_dir():continue
            out=destination/stage/directory.name;out.mkdir(parents=True)
            for p in sorted(directory.glob('results-round-*.json')):
                compressed(p,out/(p.name+'.gz'))
            if (directory/'timings.jsonl').exists():compressed(directory/'timings.jsonl',out/'timings.jsonl.gz')
            for p in directory.glob('*.pt'):
                checkpoints[str(p.relative_to(ROOT))]={'sha256':file_hash(p),'bytes':p.stat().st_size}
            attempts.append({'stage':stage,'directory':str(directory.relative_to(ROOT)),
                             'completed_results':[p.name for p in directory.glob('results-round-*.json')]})
    for stage in campaigns:shutil.copyfile(PRIVATE/(stage+'_campaign.json'),destination/(stage+'_campaign.json'))
    shutil.copyfile(PRIVATE/'logs/repository_tests_before_screening.log',destination/'tests.log')
    write_json(destination/'summary.json',{'scores':all_scores,'selection':selected,'attempts':attempts,
                                          'private_checkpoints':checkpoints,'partition_sha256':verify()['partition_sha256']})
    lines=['# Calibrazione PathMNIST patologico','',
           'Partizione dati fissa seed42, due classi/client. 81.002 fit + 8.994 validation disgiunte; nessun nuovo accesso al test. Architecture ResNet20-v2 + UNetSmallAE dz16 invariata; due fasi MSE encoder-only e CE con fusione additiva, stati privati e optimizer persistenti. Il precedente test seed42 (39,62%) era noto prima della ricerca: il benchmark non è un test mai osservato. Tutte le nuove decisioni usano esclusivamente validation.','',
           '24 profili predefiniti, screening20 round sul seed142; top3 e riferimento paper confermati da zero a50 round su seed142/143; top2 estesi a100 con ripresa degli stessi stati. Nessun best-round tra punti non dichiarati e nessun seed scelto. Configurazioni, fallimenti, gradient clipping, precisione, learning rate, schedule e budget in search_plan.json/PROTOCOL.md e nei JSON numerici.','',
           '## Screening — tutti i profili','',
           '| Profilo | Native (%) | BN da fit (%) |','|---|---:|---:|']
    for candidate in json.loads((PUBLIC/'search_plan.json').read_text())['candidates']:
        rows={r['mode']:r['accuracy_percent'] for r in all_scores if r['candidate_id']==candidate and r['rounds']==20}
        lines.append('| '+candidate+' | '+('%.6f'%rows['native'] if 'native' in rows else 'fallita/non disponibile')+' | '+('%.6f'%rows['train-recalibrated'] if 'train-recalibrated' in rows else 'fallita/non disponibile')+' |')
    lines+=['','## Conferme ed estensioni','',
            '| Profilo | Round | BN | Seed142 | Seed143 | Media ± SD ddof1 |','|---|---:|---|---:|---:|---:|']
    for row in grouped([r for r in all_scores if r['rounds'] in (50,100)]):
        vals={r['seed']:r['accuracy_percent'] for r in row['scores']}
        lines.append(f"| {row['candidate_id']} | {row['rounds']} | {row['bn_mode']} | {vals[142]:.6f} | {vals[143]:.6f} | {row['mean_validation_accuracy_percent']:.6f} ± {row['sample_sd_ddof1']:.6f} |")
    lines+=['','## Configurazione congelata','',
            f"**{selected['selected_candidate_id']}, {selected['rounds']} round, BN {selected['bn_mode']}**. Validation media {selected['mean_validation_accuracy_percent']:.6f}% ± {selected['sample_sd_ddof1']:.6f}; SD su due seed. Regola e ranking completi in selection.json, congelata prima della campagna test.",'',
            '```json',json.dumps(selected['training_settings'],indent=2),'```','',
            'La ricalibrazione train-only, se selezionata, è una variante del protocollo del paper: aggiorna solo statistiche BN condivise, con640 immagini fit/client (6.400 forward totali), senza optimizer step. Nessuna inferenza su test o scelta di buffer da test. La stessa sonda fit BN resta fissa anche nelle run su training completo. Precisione, clipping, schedule, epoche locali e round differiscono dal paper solo come indicato nella configurazione scelta. Non si certifica una replica storica.','',
            '## Verifiche e costo','',
            'Test sintetici: parità esatta con il client originale FP32 unclipped; warm-up modifica solo encoder; CE aggiorna decoder/classifier/encoder; optimizer e RNG del loader ripristinati; ricalibrazione BN non modifica pesi/RNG/input; quote e holdout disgiunti; schedule senza reset Adam. Log della suite versionata allegato. Modello, loss e optimizer verificati finiti a ogni round. Overflow del GradScaler nel riferimento FP16 sono gestiti dal codice originale; BF16/FP32 non impiegano scaler. Non sono stati installati pacchetti.','']
    for stage,c in campaigns.items():
        lines.append(f"- {stage}: {c['wall_seconds']:.3f}s calendario; {c['process_wall_seconds_sum']:.3f}s somma worker; {len(c['runs'])} tentativi; {sum(r['exit_code']!=0 for r in c['runs'])} uscite nonzero.")
    lines+=['','La somma dei tempi dei worker concorrenti non è il tempo di computazione fisico GPU. Picchi Torch, RSS, step tentati, clipping e timestamp per run sono nei risultati/timing. Entrambe le GPU sono usate fino a3 worker/GPU; nessun processo altrui interrotto.','',
            'Checkpoint completi iniziali/terminali, optimizer/RNG/partizioni e log restano in _local/pathmnist_calibrated/; hash e percorsi dei checkpoint nel manifesto numerico. Le due modalità BN della calibrazione si ricostruiscono dai checkpoint nativi completi e dagli indici fit congelati; il checkpoint finale conserverà i buffer effettivamente selezionati. Gli studi decoder/gradienti e le ablation sono sospesi.','',
            'Limiti: selezione su due seed e un holdout di immagini da un solo pool/partizione; validation è dello stesso dominio del training, mentre il test ufficiale proviene da un altro centro. Una ricerca finita non certifica un optimum globale. La SD finale isolerà inizializzazione/ottimizzazione sulla partizione fissa, senza variabilità delle partizioni. I risultati nuovi non sono ancora interpretati tramite il test.']
    (PUBLIC/'CALIBRATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest(destination)


def final():
    selected=json.loads((PUBLIC/'selection.json').read_text());campaign=require_campaign(PRIVATE/'final_campaign.json')
    if len(campaign['runs'])!=5 or any(r['exit_code']!=0 for r in campaign['runs']):raise ValueError('Final runs not all complete')
    out=PUBLIC/'artifacts/final';out.mkdir(exist_ok=False)
    records=[];checkpoint_paths={}
    for seed in selected['final_seeds']:
        directory=PRIVATE/'final'/f'seed-{seed}';r=json.loads((directory/'results.json').read_text())
        assert_finite(r)
        if r['status']!='completed' or r['completed_rounds']!=selected['rounds'] or r['seed']!=seed:raise ValueError('Invalid final identity')
        if r['configuration']['selection_sha256']!=file_hash(PUBLIC/'selection.json'):raise ValueError('Changed frozen selection')
        if len(r['milestones'])!=1 or r['milestones'][0]['split']!='test':raise ValueError('Extra/missing test evaluation')
        metric=r['milestones'][0]['evaluations'][selected['bn_mode']]
        if len(metric['pipeline_metrics'])!=10 or any(m['total']!=7180 for m in metric['pipeline_metrics']):raise ValueError('Missing test pipelines')
        if abs(metric['uniform_pipeline_accuracy_percent']-100*metric['correct_total']/71800)>1e-10:raise ValueError('Wrong metric')
        target=out/f'seed-{seed}';target.mkdir();compressed(directory/'results.json',target/'results.json.gz');shutil.copyfile(directory/'timings.jsonl',target/'timings.jsonl')
        for name in ('initial.pt','native-final.pt','final.pt'):
            p=directory/name
            if file_hash(p)!=r['checkpoint_sha256'][name]:raise ValueError('Changed final checkpoint')
            checkpoint_paths[str(p.relative_to(ROOT))]={'sha256':file_hash(p),'bytes':p.stat().st_size}
        records.append({'seed':seed,'accuracy_percent':metric['uniform_pipeline_accuracy_percent'],
                        'pipeline_metrics':metric['pipeline_metrics'],'training_seconds':r['training_seconds'],
                        'session_seconds':r['session_seconds'],'peak_cuda_allocated_bytes':r['peak_cuda_allocated_bytes'],
                        'peak_cuda_reserved_bytes':r['peak_cuda_reserved_bytes'],'peak_rss_kib':r['peak_rss_kib']})
    values=[r['accuracy_percent'] for r in records]
    summary={'seeds':records,'mean_accuracy_percent':statistics.mean(values),'sample_sd_ddof1':statistics.stdev(values),
             'selection':selected,'final_campaign':campaign,'private_checkpoints':checkpoint_paths,
             'paper_table4_mean_accuracy_percent':50.94,'previous_single_seed42_accuracy_percent':39.61977715877437,
             'comparison':'fixed partition, validation-tuned settings/possibly BN variant; no baseline reruns; published mean has no uncertainty; original raw runs unavailable'}
    write_json(out/'summary.json',summary);shutil.copyfile(PRIVATE/'final_campaign.json',out/'campaign.json')
    if (PRIVATE/'logs/final_tests.log').exists():shutil.copyfile(PRIVATE/'logs/final_tests.log',out/'tests.log')
    lines=['# PathMNIST patologico: campagna finale calibrata','',
           f"**Accuratezza {summary['mean_accuracy_percent']:.6f}% ± {summary['sample_sd_ddof1']:.6f}**, media e SD campionaria fra tutti i cinque seed41–45 (ddof=1). Nessun best-seed. Ogni risultato è la media uniforme delle10 pipeline sul medesimo test ufficiale di7.180 immagini; conteggi ricostruibili.",'',
           '| Seed | Accuratezza (%) | Training (s) | Sessione (s) | CUDA alloc/res (MiB) |','|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['seed']} | {r['accuracy_percent']:.6f} | {r['training_seconds']:.3f} | {r['session_seconds']:.3f} | {r['peak_cuda_allocated_bytes']/2**20:.3f}/{r['peak_cuda_reserved_bytes']/2**20:.3f} |")
    lines+=['','## Protocollo e confronto','',
            f"Configurazione selezionata prima del test: **{selected['selected_candidate_id']}, {selected['rounds']} round, BN {selected['bn_mode']}**. Stessi iperparametri per tutte le run da zero, training completo89.996 immagini, partizione originale patologica a2 classi/client congelata con seed dati42. Modello/inizializzazione seed41–45; C seed=run seed, AE seed=run seed+1.000.000. Nessun tuning riaperto, adattamento con dati test, scelta checkpoint o arresto per accuratezza. Protocollo completo in PROTOCOL.md e calibrazione in CALIBRATION_REPORT.md.",'',
            f"Il riferimento paper Tabella4 riporta50,94% medio, senza incertezza disponibile. Differenza descrittiva della nuova media: {summary['mean_accuracy_percent']-50.94:+.6f}pp. La precedente singola run42 con iperparametri originali era39,619777%; il delta della nuova run42 è {next(r['accuracy_percent'] for r in records if r['seed']==42)-39.61977715877437:+.6f}pp. Questi non sono confronti con baseline rieseguite né una replica esatta: dati/partizioni originali delle run pubblicate e varianze non sono recuperate; il tuning e le differenze dichiarate di round, precisione, clipping, LR/schedule/epoche o BN possono contribuire. Nessuna significatività inventata.",'',
            'La ricalibrazione BN train-only, se selezionata, è una variante rispetto al paper: aggiorna buffer condivisi a pesi/optimizer/encoder fissati usando solo una miscela training, senza input o label test. I checkpoint nativi e quelli con la modalità selezionata sono entrambi conservati; solo quest’ultima è valutata sul test. L’architettura, encoder privati persistenti, decoder/classifier condivisi, fusione additiva e le due loss restano invariati.','',
            '## Costo, verifiche e checkpoint','',
            f"Campagna finale: {campaign['wall_seconds']:.3f}s calendario; {campaign['process_wall_seconds_sum']:.3f}s somma worker concorrenti, che non equivale alla computazione fisica GPU. Entrambe le GPU, fino a3 worker/GPU, senza interrompere altri processi. Costi della ricerca sono separati nel report di calibrazione. Tempi, RSS, allocator peak, loss delle fasi, clipping e step nei JSON/timings.",'',
            'Tutte le cinque run complete, terminal round concordante, dieci pipeline/test, conteggi finiti e configurazione/selection coerenti. Caricamento dei checkpoint completi e stati optimizer/RNG/BN verificati separatamente. Checkpoint e dati restano su thanos in _local/pathmnist_calibrated/final/seed-N/: initial.pt, native-final.pt, final.pt, latest.pt e round terminale. Hash/dimensioni nel summary; codice e versione in ciascun risultato.','',
            'Limiti: SD su cinque inizializzazioni con una partizione fissa, nessuna variabilità di partizione o stima indipendente del tuning; precedente test del benchmark già noto; holdout del training non riproduce completamente il cambio di centro clinico del test. Una ricerca finita non prova ottimalità globale. Manoscritto e campagne precedenti invariati; ablation e diagnostiche rimangono sospese.']
    (PUBLIC/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest(out)
    shutil.copyfile(PUBLIC/'FINAL_REPORT.md',PRIVATE/'report_finale.md')


def verify_archives():
    for directory in (PUBLIC/'artifacts').iterdir():
        m=json.loads((directory/'manifest.json').read_text())
        for name,digest in m['files_sha256'].items():
            if file_hash(directory/name)!=digest:raise ValueError('Archive changed '+name)
    print('All archive hashes verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('calibration','final','verify'))
    args=p.parse_args()
    if (PUBLIC/'budget_reduction.json').exists() and args.mode!='verify':
        raise SystemExit('Full-budget archive superseded; use reduced_archive.py')
    {'calibration':calibration,'final':final,'verify':verify_archives}[args.mode]()
