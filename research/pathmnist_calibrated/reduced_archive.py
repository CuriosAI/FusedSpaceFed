"""Archives/reports for the user's 24-screening plus ONE-final-run budget."""
import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.pathmnist_calibrated.data import PUBLIC,PRIVATE,verify
from research.pathmnist_calibrated.selection_tools import scores,require_campaign
from research.pathmnist_calibrated.archive import compressed,manifest,verify_archives
from research.pathmnist_pathological.run import file_hash,write_json,assert_finite


def calibration():
    selected=json.loads((PUBLIC/'selection.json').read_text())
    original=require_campaign(PRIVATE/'screening_campaign.json')
    retry=require_campaign(PRIVATE/'screening_retry_campaign.json')
    rows=scores('screening',20)
    out=PUBLIC/'artifacts/calibration';out.mkdir(parents=True,exist_ok=False)
    checkpoints={};attempts=[]
    for directory in sorted((PRIVATE/'screening').iterdir()):
        if not directory.is_dir():continue
        destination=out/directory.name;destination.mkdir()
        for path in directory.glob('results*.json'):compressed(path,destination/(path.name+'.gz'))
        if (directory/'timings.jsonl').exists():compressed(directory/'timings.jsonl',destination/'timings.jsonl.gz')
        for path in directory.glob('*.pt'):
            checkpoints[str(path.relative_to(ROOT))]={'sha256':file_hash(path),'bytes':path.stat().st_size}
        progress=json.loads((directory/'progress.json').read_text()) if (directory/'progress.json').exists() else None
        attempts.append({'directory':str(directory.relative_to(ROOT)), 'last_progress':progress,
                         'complete20':(directory/'results-round-020.json').exists()})
    for source,name in ((PRIVATE/'screening_campaign.json','initial_campaign.json'),(PRIVATE/'screening_retry_campaign.json','retry_campaign.json')):
        shutil.copyfile(source,out/name)
    shutil.copyfile(PRIVATE/'logs/repository_tests_before_screening.log',out/'tests_before_screening.log')
    if (PRIVATE/'logs/reduced_integrity_tests.log').exists():shutil.copyfile(PRIVATE/'logs/reduced_integrity_tests.log',out/'reduced_tests.log')
    if (PRIVATE/'logs/calibration_final_tests.log').exists():shutil.copyfile(PRIVATE/'logs/calibration_final_tests.log',out/'tests_final.log')
    failures={}
    for campaign,folder in ((original,'screening'),(retry,'screening_retry')):
        for row in campaign['runs']:
            if row['exit_code']:
                source=PRIVATE/'logs'/folder/(row['name']+'.log')
                if source.exists():
                    destination=out/'failure_logs'/folder/source.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
                    failures[str(destination.relative_to(out))]={'exit_code':row['exit_code'],'candidate':row['name'],'output':row['output']}
    elapsed=(datetime.fromisoformat(retry['ended_utc'])-datetime.fromisoformat(original['started_utc'])).total_seconds()
    summary={'selection':selected,'scores':rows,'attempts':attempts,'private_checkpoints':checkpoints,
             'completed20_candidate_count':len({r['candidate_id'] for r in rows}), 'planned_candidate_count':24,
             'training_seed':142,'selection_round':20,'confirmations':False,'extension':False,
             'failure_logs':failures,'initial_campaign':original,'retry_campaign':retry,
             'elapsed_calendar_seconds_including_interstage_gap':elapsed,
             'active_controller_wall_seconds_sum':original['wall_seconds']+retry['wall_seconds'],
             'worker_process_seconds_sum':original['process_wall_seconds_sum']+retry['process_wall_seconds_sum']}
    write_json(out/'summary.json',summary)
    lines=['# Calibrazione PathMNIST — budget ridotto','',
           '**Selezione su un solo seed (142), sul risultato terminale a 20 round. Nessuna conferma e nessuna estensione.** La disposizione dell’utente in BUDGET_REDUCTION.md sostituisce il budget iniziale; i file protetti e gli iperparametri dello screening restano invariati.','',
           f"24 configurazioni pianificate; **{summary['completed20_candidate_count']} completate a 20 round**. Il riferimento originale FP16 è fallito per loss non finita al round 19 (ultimo checkpoint completo 18) e non ha uno score valido a 20. Nessuna correzione silenziosa del candidato: il controllo FP32 con gli stessi LR/epoche è un candidato separato già previsto. I tentativi fermati all’import per collisione del nome select.py sono ripetuti con identici hash scientifici dopo la rinomina dell’helper; tutti i receipt/log e i checkpoint sono conservati.",'',
           'Training fit 81.002 immagini + validation 8.994, disgiunte globalmente, stratificate per client/classe con holdout seed 20261006. Partizione dati originale 42 congelata, due classi/client; nessun dato test usato per calibrazione. Primaria: media uniforme di 10 pipeline sul pooled holdout, ricostruita dai conteggi. Il precedente test seed 42 del benchmark (39,619777%) era noto prima del task; non si presenta questo come un test mai osservato.','',
           '| Candidato | Native (%) | BN da fit (%) |','|---|---:|---:|']
    for candidate in json.loads((PUBLIC/'search_plan.json').read_text())['candidates']:
        s={r['mode']:r['accuracy_percent'] for r in rows if r['candidate_id']==candidate}
        fmt=lambda k:f'{s[k]:.6f}' if k in s else 'fallito/non disponibile'
        lines.append(f"| {candidate} | {fmt('native')} | {fmt('train-recalibrated')} |")
    lines+=['','## Congelamento','',
            f"**{selected['selected_candidate_id']}, modalità {selected['bn_mode']}**, validation {selected['validation_accuracy_percent']:.6f}%. Massimo fra candidati finiti e le modalità già dichiarate; parità esatta nei conteggi: BN nativa, poi ordine originale del candidato. Nessun best-round o best-seed. Configurazione unica congelata in selection.json prima della run finale.",'',
            '```json',json.dumps(selected['training_settings'],indent=2),'```','',
            'Una sola run finale nuova: seed 42, training completo 89.996 immagini, stessa partizione patologica congelata, 50 round e unico test finale. L’orizzonte della schedule resta quello del candidato (100 se cosine), senza comprimerlo a 50. Nessuna riapertura del tuning dopo il test.','',
            '## Metodo, verifiche e costi','',
            'Architettura e capacità invariati: ResNet20-v2 con 9 logits, UNetSmallAE dz16; encoder privati persistenti, decoder/classifier condivisi, fusione additiva, MSE encoder-only nel warm-up e CE-only con aggiornamento di tutti i componenti. Aggregazione uniforme e piena partecipazione. SGD/Adam persistono senza reset. LR, epoche, clipping e schedule sono differenze esplicite rispetto alla configurazione precedente; il paper non specifica la precisione numerica. Il riferimento del codice preesistente usa AMP FP16.','',
            'La modalità train-recalibrated, se selezionata, è una variante rispetto al paper: ricalibra soltanto i buffer BatchNorm condivisi, con 6.400 immagini di fit fuse dai rispettivi encoder. Nessuna label, gradiente, optimizer step, dato test o adattamento degli encoder è usato. Native e train-recalibrated sono alternative d’inferenza della stessa configurazione allenata, selezionate esclusivamente sul holdout.','',
            'Test: parità esatta FP32 unclipped con il client originale, flusso delle due fasi, persistenza/ripresa degli optimizer e generatori, holdout disgiunto, BN senza modifica di pesi/RNG/input, scheduler senza reset Adam e parità della selezione su conteggi. Log allegati. Loss e stato modello/optimizer verificati finiti a ogni checkpoint delle run complete. Nessun pacchetto installato.','',
            f"Primo lotto e tentativi meccanici: {original['wall_seconds']:.3f}s calendario, {original['process_wall_seconds_sum']:.3f}s somma worker. Ripetizioni degli avvii: {retry['wall_seconds']:.3f}s calendario, {retry['process_wall_seconds_sum']:.3f}s somma worker. Entrambe le GPU, fino a 3 worker/GPU, nessun processo esterno interrotto. La somma worker concorrenti non è tempo fisico di calcolo GPU. Costi per profilo, RSS, alloc/res GPU, loss, norm/clipping e step tentati nei JSON/timings.",'',
            f"Tempo calendario dal primo avvio all’ultimo completamento, incluso l’intervallo fra i lotti: {elapsed:.3f}s ({elapsed/60:.3f} minuti). Somma delle durate dei controller attivi: {summary['active_controller_wall_seconds_sum']:.3f}s. Somma dei processi worker, inclusi i fallimenti: {summary['worker_process_seconds_sum']:.3f}s.",'',
            'Checkpoint iniziali e terminali completi, optimizer/scaler/RNG, indici e log restano su thanos in _local/pathmnist_calibrated/screening/. I checkpoint nativi completi e gli indici fit rendono riproducibile la modalità BN alternativa; il finale memorizzerà i buffer effettivamente selezionati. Percorsi/hash nel summary.','',
            'Limiti: screening breve e un solo seed, nessuna conferma, selezione su holdout della distribuzione training e una sola partizione. Possibile sovrastima di validation e ranking diverso a 50 round. Una ricerca finita non prova ottimalità. Il finale è un singolo seed, senza SD fra seed. Manoscritto, baseline, ablation e diagnostiche non sono eseguiti/modificati.']
    (PUBLIC/'CALIBRATION_REPORT.md').write_text('\n'.join(lines)+'\n');manifest(out)


def final():
    selected=json.loads((PUBLIC/'selection.json').read_text());campaign=require_campaign(PRIVATE/'final_campaign.json')
    if selected['final_seeds']!=[42] or len(campaign['runs'])!=1 or campaign['runs'][0]['exit_code']!=0:raise ValueError('Final run not complete')
    source=PRIVATE/'final/seed-42';r=json.loads((source/'results.json').read_text());assert_finite(r)
    if r['seed']!=42 or r['completed_rounds']!=50 or r['status']!='completed' or len(r['milestones'])!=1:raise ValueError('Wrong final protocol')
    point=r['milestones'][0]
    if point['round']!=50 or point['split']!='test' or list(point['evaluations'])!=[selected['bn_mode']]:raise ValueError('Extra/wrong test evaluation')
    if r['configuration']['selection_sha256']!=file_hash(PUBLIC/'selection.json'):raise ValueError('Changed selection')
    metric=point['evaluations'][selected['bn_mode']]
    if len(metric['pipeline_metrics'])!=10 or any(row['total']!=7180 for row in metric['pipeline_metrics']):raise ValueError('Missing test counts')
    if abs(metric['uniform_pipeline_accuracy_percent']-100*metric['correct_total']/71800)>1e-10:raise ValueError('Mean not reconstructed')
    audit=json.loads((PRIVATE/'checkpoint_verification.json').read_text())
    if audit['status']!='passed':raise ValueError('Checkpoint audit failed')
    out=PUBLIC/'artifacts/final';out.mkdir(exist_ok=False)
    compressed(source/'results.json',out/'results.json.gz');shutil.copyfile(source/'timings.jsonl',out/'timings.jsonl')
    shutil.copyfile(PRIVATE/'final_campaign.json',out/'campaign.json');shutil.copyfile(PRIVATE/'checkpoint_verification.json',out/'checkpoint_verification.json')
    if (PRIVATE/'logs/final_tests.log').exists():shutil.copyfile(PRIVATE/'logs/final_tests.log',out/'tests.log')
    summary={'seed':42,'rounds':50,'seed_count':1,'accuracy_percent':metric['uniform_pipeline_accuracy_percent'],
             'pipeline_metrics':metric['pipeline_metrics'],'selection':selected,'checkpoint_verification':audit,
             'training_seconds':r['training_seconds'],'session_seconds':r['session_seconds'],
             'peak_cuda_allocated_bytes':r['peak_cuda_allocated_bytes'],'peak_cuda_reserved_bytes':r['peak_cuda_reserved_bytes'],
             'peak_rss_kib':r['peak_rss_kib'],'campaign':campaign,
             'previous_single_seed42_accuracy_percent':39.61977715877437,'paper_table4_mean_accuracy_percent':50.94,
             'performance_target_accuracy_percent':50.94,
             'above_performance_target':metric['uniform_pipeline_accuracy_percent']>50.94,
             'statistical_note':'one final seed; no between-seed SD; selection used one screening seed at20 rounds; no confirmations or100-round extension'}
    write_json(out/'summary.json',summary)
    lines=['# PathMNIST — report finale, budget ridotto','',
           f"**Run unica seed 42: accuratezza {summary['accuracy_percent']:.6f}%** a 50 round. Nessuna SD fra seed. Metrica: media uniforme di 10 pipeline private sul test ufficiale di 7.180 immagini ciascuna; {metric['correct_total']} corrette su 71.800 predizioni ricostruiscono il risultato. Non è il best encoder/seed/checkpoint.",'',
           f"Obiettivo prioritario dichiarato prima del congelamento: superare il 50,94% del paper. **Obiettivo {'raggiunto' if summary['above_performance_target'] else 'non raggiunto'}**, con differenza {summary['accuracy_percent']-50.94:+.6f} punti percentuali. Questa soglia non ha guidato una selezione sul test o una riapertura della calibrazione. Le 71.800 predizioni riutilizzano gli stessi 7.180 esempi con dieci encoder: non sono 71.800 campioni indipendenti.",'',
           '## Calibrazione e congelamento','',
           f"Selezione su **un solo seed 142 e sul risultato a 20 round**: {selected['selected_candidate_id']}, modalità BN {selected['bn_mode']}, validation {selected['validation_accuracy_percent']:.6f}%. 24 configurazioni previste, fallimenti e ripetizioni meccaniche conservati; tutte le metriche in CALIBRATION_REPORT.md e artifacts/calibration/. **Nessuna conferma su due seed e nessuna estensione a 100 round.** Regola di parità: native BN, poi ordine originale del candidato.",'',
           'Run definitiva nuova su training completo 89.996 immagini e partizione dati42 patologica congelata. Nessun nuovo dato, ottimizzazione/parallelismo o tuning dopo il congelamento; unica valutazione del test al termine, nessun arresto anticipato o selezione di checkpoint. L’orizzonte della schedule rimane quello della configurazione selezionata anche se i round finali sono50.','',
           '```json',json.dumps(selected['training_settings'],indent=2),'```','',
           '## Confronto e differenze dal paper','',
           f"Precedente run seed 42 con i default del paper:39,619777%; differenza descrittiva {summary['accuracy_percent']-39.61977715877437:+.6f}pp. Tabella 4 riporta 50,94% medio, senza incertezza disponibile: differenza descrittiva {summary['accuracy_percent']-50.94:+.6f}pp. Non è una replica esatta o un confronto comune con baseline rieseguite. Una run non verifica la media pubblicata e non permette una SD fra seed o significatività statistica.",'',
           'Restano encoder privati persistenti, decoder e classificatore condivisi, fusione additiva e le due loss originali; architettura ResNet20-v2/UNetSmallAE dz16 invariata. Iperparametri di LR/precisione/clipping/schedule/epoche locali differiscono dalla configurazione precedente soltanto come dichiarato nella configurazione congelata. Riferimento documentato: SGD LR 0,01, Adam LR 0,001, niente clipping/schedule, warm-up 1 e classificazione 3 epoche, 50 round. La precedente implementazione usa AMP FP16; il paper non specifica la precisione numerica.','',
           'Se train-recalibrated è selezionata, è una variante d’inferenza rispetto al paper: reset/re-stima dei soli buffer BN condivisi da6.400 immagini fit fuse, senza label, gradienti, optimizer step o dati test. Encoder/decoder e pesi classifier sono invariati dalla ricalibrazione. Solo la modalità congelata è testata; native-final.pt e final.pt conservano entrambe le versioni.','',
           '## Costo e integrità','',
           f"Run finale: {r['training_seconds']:.3f}s training e salvataggi; {r['session_seconds']:.3f}s sessione totale; {campaign['wall_seconds']:.3f}s calendario controller. GPU {campaign['runs'][0]['device']}, runner esistente. Picco alloc/res Torch {r['peak_cuda_allocated_bytes']/2**20:.3f}/{r['peak_cuda_reserved_bytes']/2**20:.3f}MiB; RSS {r['peak_rss_kib']/1024:.3f}MiB. Precisione/BN, clipping, loss e step nei risultati/timing. Nessuna ripresa del training definitivo; costo dello screening separato nel report calibrazione.",'',
           'Checkpoint completi e caricabili su CPU verificati: initial.pt round0, native-final.pt e final.pt round50. Contengono classifier/decoder,10 encoder privati,57 buffer BN,20 dizionari optimizer, scaler se richiesto dalla precisione, generatori loader e RNG CPU/Python/NumPy/CUDA, configurazione, versione del codice e indici esatti. SGD senza momentum ha stato vuoto previsto; BF16/FP32 non necessitano scaler. Inizializzazione verificata con C seed 42 ed AE seed1.000.042. Tutti gli stati salvati sono finiti. Hash/dimensioni in checkpoint_verification.json.','',
           'Percorsi su thanos: _local/pathmnist_calibrated/final/seed-42/{initial.pt,native-final.pt,final.pt,latest.pt}; log in _local/pathmnist_calibrated/logs/final/. I dati/checkpoint completi restano privati; solo codice, configurazioni, indici, risultati numerici, hash, costi e report sono pubblicati. Test e verifiche allegati.','',
           '## Limiti e chiusura','',
           'Screening a 20 round con un seed può scegliere un profilo diverso dal migliore a 50; nessuna conferma indipendente. Validation ricavata dal training di uno stesso centro, test ufficiale da un altro centro; il test del precedente benchmark era già noto prima della ricerca, ma nessuna nuova accuratezza test è stata usata per scegliere. Una partizione fissa e un solo seed finale non stimano variabilità o ottimalità globale. Tutti i tentativi, compreso il riferimento FP16 numericamente fallito, sono conservati. Manoscritto e altre campagne invariati; nessuna baseline, ablation o diagnostica avviata. Il lavoro si ferma qui.']
    (PUBLIC/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest(out)
    (PRIVATE/'report_finale.md').write_text((PUBLIC/'CALIBRATION_REPORT.md').read_text()+'\n---\n\n'+(PUBLIC/'FINAL_REPORT.md').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('calibration','final','verify'))
    args=p.parse_args();{'calibration':calibration,'final':final,'verify':verify_archives}[args.mode]()
