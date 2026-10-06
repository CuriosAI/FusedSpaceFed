"""Rebuild the recovery report and summary using versioned numeric archives."""
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'research/pathmnist_recovery'
ARTIFACTS=PUBLIC/'artifacts'
FREEZE='704f285bf41ac7ca5a5bbbd07f8beb52b331d654'


def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def union_seconds(campaigns):
    intervals=sorted((datetime.fromisoformat(c['started_utc']),datetime.fromisoformat(c['ended_utc'])) for c in campaigns)
    total=0.;start,end=intervals[0]
    for left,right in intervals[1:]:
        if left<=end:end=max(end,right)
        else:total+=(end-start).total_seconds();start,end=left,right
    return total+(end-start).total_seconds()


def main():
    final=read(ARTIFACTS/'attempt-05/results.json.gz')
    verification=read(ARTIFACTS/'attempt-05/verification.json.gz')
    identity=read(ARTIFACTS/'final-summary/data_and_cost_identity.json')
    qa=read(ARTIFACTS/'final-summary/quality_checks.json')
    selection=read(PUBLIC/'head_selection.json')
    old=read(ROOT/'research/pathmnist_calibrated/artifacts/final/results.json.gz')
    assert final['above_target'] and verification['status']=='passed' and qa['tracked_suite']['failed']==0
    score=final['metrics']['uniform_pipeline_accuracy_percent']
    attempts=[read(ARTIFACTS/f'attempt-{i:02}/results.json.gz') for i in range(1,6)]
    campaigns=[(name,read(ARTIFACTS/name/'campaign.json')) for name in
               ('attempt-01','normalization-validation','attempt-02','continuation-validation',
                'groupnorm-validation','attempt-03','fusion-validation','attempt-04',
                'head-validation','attempt-05')]
    commits=subprocess.check_output(['git','log','--reverse','--format=%H %s',
                                    '4bbc24ba12a98c498197bfe3d3187027f49395a3..'+FREEZE],cwd=ROOT,text=True).splitlines()
    parameters=identity['parameters'];active=parameters['classifier']+parameters['encoder']+parameters['decoder']
    total_parameters=parameters['classifier']+10*parameters['encoder']+parameters['decoder']
    costs={'campaign_wall_seconds_sum':sum(c['wall_seconds'] for _,c in campaigns),
           'campaign_interval_union_seconds':union_seconds([c for _,c in campaigns]),
           'worker_wall_seconds_sum':sum(c['process_wall_seconds_sum'] for _,c in campaigns),
           'elapsed_first_campaign_to_success_seconds':
             (datetime.fromisoformat(campaigns[-1][1]['ended_utc'])-datetime.fromisoformat(campaigns[0][1]['started_utc'])).total_seconds(),
           'reused_full_training_seconds':final['reused_training_seconds'],
           'final_extra_phase_session_seconds':final['session_seconds']}
    summary={'status':'success; search stopped','accuracy_percent':score,'seed':42,'round':50,
             'paper_reference_mean_percent':50.94,'descriptive_delta_pp':score-50.94,
             'sample_sd_between_seeds':None,'metric':'uniform mean of all 10 private-encoder pipeline accuracies',
             'unique_test_images':7180,'prediction_counts':final['metrics'],
             'validation_selection':selection['selected'],'final_configuration':final['config'],
             'final_code':final['code'],'checkpoints':verification['checkpoints'],'quality_checks':qa,
             'parameters_active_per_client':active,'parameters_total_persistent':total_parameters,
             'costs':costs,'attempt_scores_percent':[r['metrics']['uniform_pipeline_accuracy_percent'] for r in attempts],
             'limitations':['one final seed, no seed SD','known test with adaptive threshold stop',
                            'positive-gain BN and extra-head phase variant, not original paper protocol',
                            'validation reused for many candidates','no controlled baseline or component-attribution test'],
             'search_commits':[c.split(' ',1)[0] for c in commits]}
    (ARTIFACTS/'final-summary/summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# PathMNIST patologico: recupero esplorativo concluso','',
      f'**Seed 42, round 50: {score:.6f}% sul test**, {score-50.94:+.6f} punti percentuali rispetto al 50,94% medio riportato nella [Tabella 4 del paper](../../paper/aistats_2027.tex). Ricerca fermata al primo superamento, il 6 ottobre 2026. È una singola run riutilizzata con una fase aggiuntiva: **nessuna SD fra seed**, nessuna replica certificata della media pubblicata. Manoscritto, altri esperimenti e tutti i tentativi precedenti restano invariati.','',
      'La soluzione mantiene encoder privati persistenti, decoder e classificatore condivisi, fusione additiva e le due fasi originali di training. Aggiunge una **variante esplicita**: guadagno positivo 0,1 nella fusione, ricalibrazione dei buffer BN su training e ottimizzazione del solo ultimo strato condiviso usando feature training fisse. Non va descritta come il FusedSpaceFed del paper a iperparametri invariati.','',
      '## Dati, partizione e metrica','',
      'Sorgente [MedMNIST 3.0.2 PathMNIST-64]('+identity['dataset']['source_url']+'). Cache già verificata, PIL Resize 32×32 seguito da ToTensor, RGB Float32 senza augmentation per la configurazione finale. Training 89.996, test ufficiale 7.180. Dieci client patologici, due classi/client; ogni immagine training appartiene a un solo client, stessa partizione dati seed 42. Gli indici originali sono conservati nei checkpoint e nel manifesto congelato. Non vengono aggiunti dati o usata la validation ufficiale.','',
      f"- SHA256 sorgente NPZ: `{identity['dataset']['source_sha256']}`.",
      f"- SHA256 partizione originale: `{identity['source_partition_sha256']}`.",
      f"- SHA256 manifesto holdout: `{identity['partition_sha256']}`.",
      f"- Seed holdout: {identity['holdout_seed']}; 81.002 fit + 8.994 validation, disgiunti globalmente, holdout stratificato per client/classe. BN: 640 immagini fit/client, 6.400 totali, nessuna label.",'',
      '| Client | Classi | Training completo | Fit | Validation |','|---:|---|---:|---:|---:|']
    stats={r['client_id']:r for r in identity['fit_validation_statistics']}
    for r in identity['dataset']['clients']:
        s=stats[r['client_id']];lines.append(f"| {r['client_id']} | {r['classes']} | {r['samples']} | {s['fit']} | {s['validation']} |")
    lines+=['',
      'Metrica primaria: ciascuno dei dieci encoder privati, con lo stesso decoder/classificatore condiviso, è valutato su **tutto** il test o holdout pooled. Si riporta la media uniforme delle dieci accuratezze, non local two-class accuracy, migliore encoder, ensemble di logits o migliore seed. Le 71.800 predizioni test riusano 7.180 immagini dieci volte: non sono 71.800 osservazioni indipendenti.','',
      '## Configurazione finale e fase aggiuntiva','',
      'La base completa seed 42 è stata allenata da zero su tutto il training nella precedente campagna, con C inizializzato seed 42 ed AE seed 1.000.042. Si riutilizza il suo terminale **nativo round 50**, senza ripresa da smoke o checkpoint intermedi. Impostazioni identiche al modello fit-only selezionato seed 142, round 50:','',
      '```json',json.dumps(final['config']['training_settings'],indent=2),'```','',
      'Partecipano tutti i 10 client a ogni round. Warm-up: una epoca MSE encoder-only; classificazione: tre epoche CE, encoder/decoder/classificatore trainabili. SGD e Adam sono persistenti, senza reset; aggregazione uniforme solo degli stati condivisi. BF16 durante training senza GradScaler; ricalibrazione, feature, testa e test Float32. Schedule constant: il nominale orizzonte 100 non cambia i LR nei 50 round effettivi.','',
      'Rispetto agli iperparametri del [paper](../../paper/aistats_2027.tex), restano architettura, dati, 50 round, warm-up 1, CE 3 e batch 128; LR classifier cambia 0,01→0,1 e LR AE/warm-up 0,001→0,0001. Si aggiunge clipping norm 5. Il paper non specifica la precisione; la precedente implementazione di riferimento usava FP16 AMP, la base selezionata BF16. Si aggiungono inoltre le tre modifiche post-training descritte sotto: fusione attenuata, buffer BN ricalibrati e ottimizzazione della testa.','',
      'Dopo i 50 round, senza modificare encoder/decoder o pesi del corpo ResNet:','',
      '1. Usare `x + 0.1 D(E_i(x))`; α=0 è stato escluso dalle soluzioni. La convoluzione finale lineare del decoder consente una riscalatura esatta in copia, senza nuovi parametri.',
      '2. Reset e ricalibrazione delle 19 BN condivise sulla miscela fit fissata, 50 batch 128, shuffle 20261006, cumulative averaging. Ogni immagine usa il suo encoder proprietario. Solo buffer, senza label/gradienti/test.',
      '3. Congelare C body, BN, encoder e decoder; estrarre feature 64 da **tutti** gli 89.996 training attraverso l’encoder del client d’origine. Ottimizzare la testa condivisa 64→9, iniziando dai suoi pesi terminali, con L-BFGS LR 1, history 20, strong-Wolfe, max_iter=100, tolerance_grad1e-7 e tolerance_change1e-10. Penalità L2 selezionata λ=0; bias non penalizzato. Obiettivo: media uniforme delle 10 CE locali full-batch, poi eventuale 0,5λ||W_fc||².',
      '4. Salvare il checkpoint completo della variante prima di caricare/evaluare il test. Un’unica valutazione della configurazione congelata; nessun test-adaptation, selezione checkpoint o riapertura dopo questo esito.','',
      f"Sono state effettuate {identity['head_optimizer_actual_iterations']} iterazioni effettive della testa, {identity['head_optimizer_function_evaluations']} valutazioni dell’obiettivo/gradiente. CE uniforme training: {final['head_statistics']['objective_history'][0]:.6f} → {final['head_statistics']['final_uniform_client_training_ce']:.6f}. Queste misure provengono dal training e non hanno determinato early stopping sul test.",'',
      f"Architettura: ResNet20-v2 {parameters['classifier']:,} parametri, encoder {parameters['encoder']:,}, decoder {parameters['decoder']:,}; **{active:,} parametri attivi/client**, {total_parameters:,} persistenti fra tutti i 10 client. La testa ha {parameters['head']} parametri già inclusi nel ResNet, nessuna nuova capacità. La fase aggiuntiva cambia computazione e algoritmo: non si rivendica costo equivalente al protocollo originale.",'',
      'La media di loss/gradienti della testa equivale matematicamente a una media di contributi locali full-batch. Il simulatore cachea centralmente feature prodotte dagli encoder proprietari. Non è una dimostrazione di privacy, né un’implementazione del traffico di rete distribuito; una realizzazione federata avrebbe aggregazioni/line-search aggiuntive oltre ai 50 round delle due fasi.','',
      'Motivazione: sulle feature fissate la CE della testa lineare è un problema convesso, quindi un refit condiviso economico verifica se il punto terminale ottenuto con aggiornamenti locali è subottimale. L-BFGS evita un nuovo sweep di learning rate della rete intera; il budget fisso e le penalità sono dichiarati prima della validation. Il guadagno positivo attenua il decoder preservando il percorso privato, e il suo valore è selezionato su validation. Nessuna conclusione di causalità sui componenti viene ricavata dal solo test finale.','',
      '## Selezione e tutti i tentativi','',
      'Il primo screening storico: 24 profili, 20 round, seed 142; il congelamento precedente selezionava c0.1/ae0.0001 ma ha prodotto 35,279944% test. Conservato integralmente in `../pathmnist_calibrated/`, senza riscriverne i report. La successiva autorizzazione ha riaperto la ricerca e revocato la deadline. La nuova ricerca ha proceduto per probabilità qualitativa/costo stimato: BN economiche, prolungamento dei checkpoint fit a 50, GN, guadagni positivi, infine refit della testa. Nessuna baseline è rieseguita.','',
      '| Tentativo test seed 42 | Variante | Accuratezza (%) | Sessione aggiuntiva(s) |','|---:|---|---:|---:|']
    descriptions=['Originale, owner-cumulative BN da fit','Originale, cross-layerwise BN da fit',
                  'c0.1/ae0.0001, native diagnostica','Originale, α=0,25 +BN da fit',
                  'c0.1/ae0.0001, α=0,1 +BN da fit +refit fc']
    for i,(r,d) in enumerate(zip(attempts,descriptions),1):
        lines.append(f"| {i} | {d} | {r['metrics']['uniform_pipeline_accuracy_percent']:.6f} | {r['session_seconds']:.3f} |")
    lines+=['',
      'Il tentativo 3 era un controllo diagnostico dell’inferenza nativa, **non** la modalità favorita dalla validation; questa eccezione è dichiarata prima del test nel piano. Tentativi 1/2/4 trasferivano selezioni FP32 fit seed 142 a pesi FP16 completi seed 42; limite esplicito. Il tentativo 5 usa le stesse impostazioni di training/precisione/orizzonte tra fit e full.','',
      'Ricalibrazione BN: cinque modalità su sei checkpoint fit. Quattro erano i profili promettenti, più c0.01/ae0.0003 e il riferimento FP16 finito al round 18, fallito al 19: il suo controllo non è un risultato round 20. Tutti i 30 valori sono negli archivi normalization-validation. Quattro checkpoint sono ripresi esattamente 20→50 con optimizer, generatori e RNG; nessun nuovo seed o best-round.','',
      '| Fit seed 142, round 50 | Native (%) | BN da fit (%) |','|---|---:|---:|']
    for p in sorted((ARTIFACTS/'continuation-validation').glob('*/results.json.gz')):
        r=read(p);values=r['milestones'][-1]['evaluations']
        lines.append(f"| {p.parent.name} | {values['native']['uniform_pipeline_accuracy_percent']:.6f} | {values['train-recalibrated']['uniform_pipeline_accuracy_percent']:.6f} |")
    lines+=['','Due profili GN da zero, 50 round, seed 142:','',
            '| Profilo | Validation nativa (%) | Sessione(s) |','|---|---:|---:|']
    for p in sorted((ARTIFACTS/'groupnorm-validation').glob('*/results.json.gz')):
        r=read(p);lines.append(f"| {p.parent.name} | {r['milestones'][-1]['evaluations']['native']['uniform_pipeline_accuracy_percent']:.6f} | {r['session_seconds']:.3f} |")
    lines+=['','GN sostituisce 19 BN con GN8, stessi parametri affini; il profilo CE1 usa flip/rot90. Esiti negativi su validation; nessun test di questi profili. La ricerca dei guadagni 0/0,1/0,25/0,5/1/2 su quattro checkpoint fit a 50 include 42 modalità; α=0 è soltanto diagnostico. Il massimo positivo senza refit era 52,066934% sul riferimento FP32, α=0,25 con BN da fit.','',
      'Refit fc: **sei profili in parallelo 3+3 GPU**, cinque penalità 0/1e-4/1e-3/1e-2/0,1 ciascuno. Risultati completi, inclusi controlli identici senza refit:','',
      '| Profilo | Prima (%) | λ0 | λ1e-4 | λ1e-3 | λ1e-2 | λ0,1 |','|---|---:|---:|---:|---:|---:|---:|']
    for p in sorted((ARTIFACTS/'head-validation').glob('*/results.json.gz')):
        r=read(p);lines.append('| '+p.parent.name+' | '+f"{r['baseline']['uniform_pipeline_accuracy_percent']:.6f}"+' | '+' | '.join(f"{v['metrics']['uniform_pipeline_accuracy_percent']:.6f}" for v in r['rows'])+' |')
    lines+=['',
      f"Selezione congelata su un solo seed 142, risultato terminale 50 round: **{selection['selected']['accuracy_percent']:.6f}%**,72.137/89.940. Si confrontano tutti i 30 refit e sei controlli; parità: controllo senza refit, ordine dei profili, poi penalità. Ranking integrale: `head_selection.json`; hash selezione `{final['config']['head_selection_sha256']}`. Scelte λ/guadagno/profilo soltanto da validation; nessun seed finale alternativo. L’uso ripetuto dello stesso holdout può sovradattare il tuning.",'',
      '## Risultato completo e interpretazione','',
      '| Encoder/pipeline | Corrette | Test immagini | Accuratezza (%) |','|---:|---:|---:|---:|']
    for r in final['metrics']['pipeline_metrics']:
        lines.append(f"| {r['client_id']} | {r['correct']} | {r['total']} | {r['accuracy_percent']:.6f} |")
    lines += ['',
      f"Totale 54.018/71.800 → **{score:.6f}%**, identico alla media uniforme. Un singolo seed 42: nessuna SD fra seed inventata e nessuna sostituzione con la pipeline migliore. La differenza {score-50.94:+.6f}pp dalla Tabella 4 è descrittiva; il 50,94% del paper è una media di cinque run senza incertezza in tabella e con partizioni/seed originali non completamente recuperati. Non è una dimostrazione di superiorità statistica o una replica esatta.",'',
      'La validation controllata dello stesso checkpoint c0.1/ae0.0001, α=0,1 e stessa BN passa da 47,585057 a 80,205693% cambiando soltanto la testa. Questo documenta che i pesi dell’ultima testa terminale erano subottimali per quelle feature/obiettivo. Non dimostra quale causa di training li abbia prodotti, né il beneficio causale dell’encoder, warm-up o fusione rispetto a un controllo ablation. Non si dichiarano nuove misure Γ o vantaggi di costo sulle baseline.','',
      'Il test era già noto e cinque tentativi sono stati valutati nel recupero. Lo stop al primo superamento della soglia è adattivo: il risultato finale resta **esplorativo**. Le scelte puntuali della fase vincente sono congelate su validation prima del relativo test, ma la campagna nel suo insieme non è una conferma indipendente o test mai osservato. Nessuna ulteriore calibrazione, run, baseline, ablation o diagnostica parte dopo il successo.','',
      '## Tempi, memoria e impiego GPU','',
      '| Fase | Controller calendario(s) | Somma worker(s) | Processi |','|---|---:|---:|---:|']
    for name,c in campaigns:lines.append(f"| {name} | {c['wall_seconds']:.3f} | {c['process_wall_seconds_sum']:.3f} | {len(c['runs'])} |")
    lines+=['',
      f"Unione degli intervalli delle campagne: {costs['campaign_interval_union_seconds']:.3f}s, senza doppio conteggio delle sovrapposizioni. Somma controller {costs['campaign_wall_seconds_sum']:.3f}s; somma worker concorrenti {costs['worker_wall_seconds_sum']:.3f}s. Quest’ultima non misura il tempo fisico di calcolo GPU né FLOPs. Dal primo lancio al successo: {costs['elapsed_first_campaign_to_success_seconds']:.3f}s calendario, inclusi sviluppo, attese e interruzioni fra campagne. Lo screening precedente e i 20 round già eseguiti dei checkpoint ripresi sono costi storici aggiuntivi, non rimisurati come nuovi training.",'',
      f"Run completa riutilizzata: {old['training_seconds']:.3f}s training/salvataggi, {old['session_seconds']:.3f}s sessione, versione `{old['code']['base_commit']}`. Non è stata ripresa: stati terminali compatibili riutilizzati per la fase aggiuntiva. 35.400 warm-up steps encoder-only + 106.200 joint steps per optimizer, Adam totale 141.600 e SGD 106.200; BF16 senza step saltati dal GradScaler.",'',
      f"Fase finale vincente: BN {final['bn_seconds']:.3f}s, feature {final['feature_extraction_seconds']:.3f}s, refit {final['head_refit_seconds']:.3f}s, test {final['evaluation_seconds']:.3f}s; sessione {final['session_seconds']:.3f}s. Processo incluse import/init: {campaigns[-1][1]['runs'][0]['process_wall_seconds']:.3f}s; controller {campaigns[-1][1]['wall_seconds']:.3f}s. Sommare 1804 s storici al solo refit non equivale a misurare una nuova run end-to-end; i contributi sono riportati separatamente.",'',
      f"Picco Torch finale allocato/riservato: {final['peak_cuda_allocated_bytes']/2**20:.3f}/{final['peak_cuda_reserved_bytes']/2**20:.3f}MiB; RSS {final['peak_rss_kib']/1024:.3f}MiB. Training base: {old['peak_cuda_allocated_bytes']/2**20:.3f}/{old['peak_cuda_reserved_bytes']/2**20:.3f}MiB. Picchi per worker delle altre fasi nei JSON. Non si sommano picchi non contemporanei come un picco GPU totale.",'',
      'Host thanos.gasl.unich.it, due RTX 6000 Ada 48 GB, ambiente general_ml senza installazioni. Validation BN 6 worker 3+3, continuazioni 4 worker 2+2 e GN indipendenti, fusion probe 4 worker 2+2, refit 6 worker 3+3. GPU entrambe al 99% nella breve campagna refit; circa 5,5–6 GB usati/GPU in quel controllo. Processo esterno tesi_giovanni PID 4146 su GPU 0 lasciato intatto; nessun processo altrui interrotto. Test congelati sequenziali per poter fermare la ricerca alla soglia. Nessun nostro processo training resta attivo.','',
      '## Checkpoint, verifiche e riproduzione','',
      'Su thanos: `/mnt/data/codex/FusedSpaceFed/_local/pathmnist_recovery/attempt-05/`.','',
      '| File | Round | SHA256 |','|---|---:|---|']
    for name,c in verification['checkpoints'].items():lines.append(f"| {name} | {c['round']} | `{c['sha256']}` |")
    lines+=['',
      'Checkpoint completi: C, D, tutti 10 encoder, copie client, 57 buffer BN, 20 dizionari degli optimizer originali, generatori loader e RNG CPU/Python/NumPy/CUDA, partizione/indici esatti, configurazione, codice e storia 50 round. SGD senza momentum ha stato vuoto previsto, BF16 non richiede scaler. `final.pt` aggiunge optimizer L-BFGS completo, RNG della fase e configurazione effettiva. Audit: tutti gli stati finiti e caricabili; D, encoder, C body, optimizer/RNG originali bit-identici alla base; C fc e BN sincronizzati in tutte le copie client.','',
      '**Attenzione operativa:** top-level `decoder` conserva i pesi allenati originali. Per l’inferenza effettiva usare `inference_decoder(checkpoint)` o moltiplicare D per `recovery.config.fusion_gain`. Non usare direttamente la vecchia valutazione α=1 sul nuovo checkpoint. `precalibration.pt` è lo stato originale completo prima della fase aggiuntiva; `initial.pt` è l’inizializzazione round0, non una nuova inizializzazione post-hoc.','',
      '331 test versionati passati (`pytest -q tests research`), inclusi 18 test di recupero: normalizzazione, gain esatto, warm-up/CE gradient flow, persistenza, ripresa GN+augmentation, obiettivo per client, refit head e stati completi. `verification.py` ricostruisce metriche da conteggi e verifica dati/config/partizioni/checkpoint senza training o altre valutazioni immagini.','',
      'Una raccolta indiscriminata `pytest -q` includeva test privati ignorati in `_local/`: 413 passati, 1 fallito perché un vecchio controllo queue richiede interprete senza import Torch/NumPy, incompatibile con la raccolta della suite scientifica. Non è stato modificato codice scientifico per quel vincolo: tutti 38 test privati queue passano in un processo unittest isolato. Esiti/log/hash in quality_checks.json; log dettagliato fallito conservato localmente.','',
      'Comando effettivamente eseguito, con stdout/stderr e exit 0 nel receipt:','',
      '```bash','/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_recovery/head_final.py \\',
      '  --config research/pathmnist_recovery/attempt-05.json \\',
      '  --output _local/pathmnist_recovery/attempt-05 --device cuda:1','```','',
      'Il runner rifiuta sovrascritture; per una futura replica occorre una nuova directory, senza crearla prima. Tutte le sorgenti/queue del comando sono congelate con hash nel receipt, codice esecuzione `'+FREEZE+'`. Per verificare e rigenerare soltanto il report numerico:','',
      '```bash','/home/schroeder/miniconda3/envs/general_ml/bin/python research/pathmnist_recovery/archive.py verify',
      '/home/schroeder/miniconda3/envs/general_ml/bin/python research/pathmnist_recovery/report.py','```','',
      'Il report generator usa solo JSON pubblicati e cronologia Git; non legge checkpoint o immagini e non esegue esperimenti. Gli artefatti numerici gzip sono lossless, tutti i tentativi negativi inclusi. Manifesti SHA e provenance elencano percorsi/hash/dimensioni dei checkpoint privati. Log e checkpoint completi restano in `_local/pathmnist_recovery/`; base riutilizzata in `_local/pathmnist_calibrated/final/seed-42/`. Nessuna immagine, credenziale, review privata o checkpoint è versionato.','',
      '## Commit e limiti aperti','',
      'Base precedente conservata: `4bbc24ba12a98c498197bfe3d3187027f49395a3`. Commit del recupero fino al congelamento finale, tutti push su main:','']
    for c in commits:lines.append('- `'+c.split(' ',1)[0]+'` — '+c.split(' ',1)[1])
    lines+=['',
      'Il commit che aggiunge questo report e artifacts/attempt-05 archivia l’esito finale; identificabile con `git log -1 -- research/pathmnist_recovery/RECOVERY_REPORT.md`. La copia locale del report registra anche hash/esito del push finale dopo la creazione del commit. Nessun reset/rebase/forcepush.','',
      'Limiti aperti: un solo seed di selezione e finale; holdout ripetutamente consultato; test noto con stop adattivo; differenze di LR/clipping/precisione/fusione/BN e fase aggiuntiva rispetto al paper; comunicazione e privacy della fase head non implementate; nessun confronto controllato con baseline o isolamento causale dei componenti; partizioni/seed storici del 50,94% non interamente recuperati. Nessuna nuova SD, significatività o equivalenza di costo è affermata. Ablation e diagnostiche rimangono sospese; la ricerca termina al successo, senza modifica del manoscritto.']
    text='\n'.join(lines)+'\n'
    (PUBLIC/'RECOVERY_REPORT.md').write_text(text)
    directory=ARTIFACTS/'final-summary'
    (directory/'manifest.json').write_text(json.dumps({'files_sha256':{
        str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='manifest.json'}},indent=2)+'\n')
    return text


if __name__=='__main__':
    main();print('Report rebuilt from numeric archives; no training/test evaluation')
