"""Stdlib-only numerical checks; never runs a model or reads checkpoints."""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import shutil

DOMAINS=('MNIST','SVHN','USPS','SynthDigits','MNIST-M')
METHODS=('FusedSpaceFed','FedAvg')
SEEDS=(42,43,44)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def canon(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value):
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def check_metrics(evaluation,counts):
    if set(evaluation['domains'])!=set(DOMAINS):
        raise ValueError('Missing domain')
    for domain,metric in evaluation['domains'].items():
        matrix=metric['confusion_matrix']
        if len(matrix)!=10 or any(len(row)!=10 for row in matrix):
            raise ValueError('Wrong confusion dimensions')
        if any(type(v) is not int or v<0 for row in matrix for v in row):
            raise ValueError('Invalid confusion entries')
        total=sum(map(sum,matrix));correct=sum(matrix[i][i] for i in range(10))
        if metric['total']!=counts[domain] or total!=metric['total'] or correct!=metric['correct']:
            raise ValueError('Wrong confusion counts')
        if not math.isclose(metric['accuracy_percent'],100*correct/total,abs_tol=1e-10):
            raise ValueError('Wrong domain accuracy')
    expected=statistics.mean(m['accuracy_percent'] for m in evaluation['domains'].values())
    if not math.isclose(expected,evaluation['uniform_domain_accuracy_percent'],abs_tol=1e-10):
        raise ValueError('Wrong uniform accuracy')
    expected=100*sum(m['correct'] for m in evaluation['domains'].values())/sum(m['total'] for m in evaluation['domains'].values())
    if not math.isclose(expected,evaluation['sample_weighted_accuracy_percent'],abs_tol=1e-10):
        raise ValueError('Wrong weighted accuracy')


def cost(profile,phase,batch,dense=False):
    row=profile['batches']['32']['phases'][phase]
    return row['dense']//32*batch+(0 if dense else row['update_and_clip'])


def epoch_batches(count):
    return [32]*(count//32)+([count%32] if count%32 else [])


def audit_run(result,profile,count,rounds,split,test_counts,*,reused=False):
    if not finite(result) or result['status']!='completed' or result['completed_rounds']!=rounds:
        raise ValueError('Incomplete/nonfinite result')
    if [r['round'] for r in result['history']]!=list(range(1,rounds+1)):
        raise ValueError('Wrong round sequence')
    if len(result['evaluations'])!=1 or result['evaluations'][0]['round']!=rounds:
        raise ValueError('Wrong evaluation schedule')
    if not reused and result['evaluations'][0]['split']!=split:
        raise ValueError('Wrong evaluation split')
    check_metrics(result['evaluations'][0],test_counts)
    method=result['identity']['method'];target=sum(cost(profile,phase,batch) for phase in ('warmup','classification') for batch in epoch_batches(count))
    spent=dense_spent=steps=samples=warm_steps=0
    per_domain={domain:0 for domain in DOMAINS}
    for row in result['history']:
        if set(row['clients'])!=set(DOMAINS):
            raise ValueError('Missing participation')
        for domain,client in row['clients'].items():
            if method=='FusedSpaceFed':
                nsteps=len(epoch_batches(count))
                if client['warmup_steps']!=nsteps or client['classification_steps']!=nsteps or client['encoder_steps']!=2*nsteps:
                    raise ValueError('Wrong Fused phase updates')
                if client['warmup_samples']!=count or client['classification_samples']!=count or client['classifier_steps']!=nsteps or client['decoder_steps']!=nsteps:
                    raise ValueError('Wrong Fused sample/shared updates')
                actual=target
                dense_actual=sum(cost(profile,phase,batch,True) for phase in ('warmup','classification') for batch in epoch_batches(count))
                warm_steps+=nsteps
            elif method=='FedAvg':
                plan=client['batch_sizes']
                if any(type(batch) is not int or batch<2 or batch>32 for batch in plan) or client['classification_steps']!=len(plan) or client['classification_samples']!=sum(plan):
                    raise ValueError('Wrong FedAvg executed minibatches')
                actual=sum(cost(profile,'fedavg',batch) for batch in plan)
                dense_actual=sum(cost(profile,'fedavg',batch,True) for batch in plan)
                carry=row['round']*target-per_domain[domain]-actual
                if carry!=client['budget_carry'] or not 0<=carry<cost(profile,'fedavg',2):
                    raise ValueError('Wrong cumulative compute carry')
            else:
                raise ValueError('Wrong method')
            if not reused and (client['counted_flops']!=actual or client['dense_flops']!=dense_actual or client['target_counted_flops']!=target):
                raise ValueError('Saved FLOPs differ from executed batches')
            per_domain[domain]+=actual;spent+=actual;dense_spent+=dense_actual
            steps+=client['classification_steps'];samples+=client['classification_samples']
    if not reused and (result['total_counted_training_flops']!=spent or result['total_dense_training_flops']!=dense_spent):
        raise ValueError('Wrong total training FLOPs')
    return {'counted_training_flops':spent,'dense_training_flops':dense_spent,'target_flops':target*rounds*5,
            'warmup_steps':warm_steps,'classification_steps':steps,'classification_samples':samples,
            'reused':reused,'seed':result['identity']['seed'],'code_commit':result['identity']['code']['commit']}


def stats(values):
    return {'values':values,'mean':statistics.mean(values),'sample_sd_ddof1':statistics.stdev(values)}


def summary(records,costs):
    output={'seeds':list(SEEDS),'ddof':1,'methods':{},'paired_fused_minus_fedavg':{},'costs':costs}
    for method in METHODS:
        output['methods'][method]={key:stats([records[method][seed]['evaluations'][0][key] for seed in SEEDS])
                                   for key in ('uniform_domain_accuracy_percent','sample_weighted_accuracy_percent')}
        output['methods'][method]['domains']={domain:stats([records[method][seed]['evaluations'][0]['domains'][domain]['accuracy_percent'] for seed in SEEDS]) for domain in DOMAINS}
    for key in ('uniform_domain_accuracy_percent','sample_weighted_accuracy_percent'):
        output['paired_fused_minus_fedavg'][key]=stats([records['FusedSpaceFed'][seed]['evaluations'][0][key]-records['FedAvg'][seed]['evaluations'][0][key] for seed in SEEDS])
    return output


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def archive_run(source,target):
    target.mkdir(parents=True)
    raw=(source/'results.json').read_bytes()
    with (target/'results.json.gz').open('wb') as stream:
        with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as compressed:
            compressed.write(raw)
    if gzip.decompress((target/'results.json.gz').read_bytes())!=raw:
        raise ValueError('Lossy compression')
    shutil.copyfile(source/'timings.jsonl',target/'timings.jsonl')
    result=json.loads(raw)
    if [json.loads(line) for line in (target/'timings.jsonl').read_text().splitlines()]!=result['history']:
        raise ValueError('Timing/result histories differ')
    return {'raw_result_sha256':sha(source/'results.json'),'private_checkpoint_sha256':sha(source/'checkpoint.pt'),
            'private_checkpoint_bytes':(source/'checkpoint.pt').stat().st_size}


def check_identity(result,config):
    if result['configuration']!=config or result['identity']['config_sha256']!=canon(config) or result['identity']['seed']!=config['seed'] or result['identity']['method']!=config['method']:
        raise ValueError('Run/config identity differs')
    for name,digest in result['identity']['code']['source_sha256'].items():
        if sha(Path(name))!=digest:
            raise ValueError('Scientific source changed')


def check_campaign(campaign,jobs):
    if campaign['status']!='completed' or len(campaign['runs'])!=len(jobs) or any(row['exit_code']!=0 for row in campaign['runs']):
        raise ValueError('Training workers still running or failed')
    if campaign['max_processes']!=2 or campaign['max_per_gpu']!=1 or campaign['external_actions']!='none':
        raise ValueError('Wrong execution policy')
    if {row['name'] for row in campaign['runs']}!={row['name'] for row in jobs}:
        raise ValueError('Wrong worker set')
    for first in campaign['runs']:
        for second in campaign['runs']:
            if first is second or first['device']!=second['device']:
                continue
            if max(datetime.fromisoformat(first['started_utc']),datetime.fromisoformat(second['started_utc']))<min(datetime.fromisoformat(first['ended_utc']),datetime.fromisoformat(second['ended_utc'])):
                raise ValueError('Overlapping workers on one GPU')


def build(directory):
    destination=directory/'artifacts'
    if destination.exists():
        raise FileExistsError('Numeric archive already exists')
    profile=json.loads((directory/'flop_profile.json').read_text())
    split=json.loads((directory/'validation_split.json').read_text())
    partition=json.loads(Path('research/feature_shift_digits/partition_manifest.json').read_text())
    selection=json.loads((directory/'selection.json').read_text())
    validation_queue=json.loads((directory/'validation_queue_v2.json').read_text())
    final_queue=json.loads((directory/'final_queue.json').read_text())
    validation_campaign=json.loads(Path('_local/capacity_compute_control/validation_v2_campaign.json').read_text())
    final_campaign=json.loads(Path('_local/capacity_compute_control/final_campaign.json').read_text())
    check_campaign(validation_campaign,validation_queue['jobs']);check_campaign(final_campaign,final_queue['jobs'])
    if canon(selection)!=final_queue['selection_sha256'] or selection['validation_campaign_sha256']!=sha('_local/capacity_compute_control/validation_v2_campaign.json'):
        raise ValueError('Selection/freeze receipt differs')
    checked_validation={};validation_totals={method:0 for method in METHODS}
    for job in validation_queue['jobs']:
        source=Path(job['output']);result=json.loads((source/'results.json').read_text());config=json.loads(Path(job['config']).read_text())
        check_identity(result,config)
        cost_result=audit_run(result,profile,594,60,'validation',{domain:149 for domain in DOMAINS})
        for domain,metric in result['evaluations'][0]['domains'].items():
            if [sum(row) for row in metric['confusion_matrix']]!=split['domains'][domain]['validation_counts']:
                raise ValueError('Validation label histogram differs')
        if cost_result['seed']!=142 or cost_result['code_commit']!=validation_campaign['code_commit']:
            raise ValueError('Wrong validation code/seed')
        checked_validation[job['name']]={'costs':cost_result,'evaluation':result['evaluations'][0],
                                         'wall_seconds':result['total_session_wall_seconds']}
        validation_totals[config['method']]+=cost_result['counted_training_flops']
    records={method:{} for method in METHODS};costs={method:{} for method in METHODS};timing=[]
    registry=final_queue['runs_registry']
    if {(row['method'],row['seed']) for row in registry}!={(method,seed) for method in METHODS for seed in SEEDS} or len(registry)!=6:
        raise ValueError('Final registry is not three matched pairs')
    for entry in registry:
        source=Path(entry['source_directory']);result=json.loads((source/'results.json').read_text())
        method=entry['method'];seed=entry['seed'];reused=entry['origin']=='reused_previous'
        if result['identity']['seed']!=seed or result['identity']['method']!=method or result['identity']['partition_sha256']!=partition['partition_sha256']:
            raise ValueError('Wrong final seed/method/partition')
        if reused:
            if method!='FusedSpaceFed' or sha(source/'results.json')!=entry['reused_result_sha256']:
                raise ValueError('Reused source differs')
            if any(result['configuration']['training'][key]!=value for key,value in selection['selected_training_settings'][method].items()):
                raise ValueError('Selected settings not compatible with reused run')
            for name,digest in result['identity']['code']['source_sha256'].items():
                if sha(name)!=digest:
                    raise ValueError('Reused scientific source differs')
        else:
            config=json.loads(Path(entry['config']).read_text());check_identity(result,config)
            if config['selection_sha256']!=canon(selection) or result['identity']['code']['commit']!=final_campaign['code_commit']:
                raise ValueError('Final run preceded freeze or changed code')
        computed=audit_run(result,profile,743,300,'final',{domain:partition['domains'][domain]['test']['count'] for domain in DOMAINS},reused=reused)
        for domain,metric in result['evaluations'][0]['domains'].items():
            if [sum(row) for row in metric['confusion_matrix']]!=partition['domains'][domain]['test']['labels']:
                raise ValueError('Final label histogram differs')
        if len(result['sessions'])!=1 or result['sessions'][0]['start_round']!=1 or result['sessions'][0]['end_round']!=300:
            raise ValueError('Expected fresh final initialization, no resume')
        records[method][seed]=result;costs[method][str(seed)]=computed
        timing.append({'method':method,'seed':seed,'origin':entry['origin'],'device':result['identity']['device'],
                       'wall_seconds':result['total_session_wall_seconds'],'test_seconds':result['evaluations'][0]['seconds'],
                       'peak_cuda_allocated_mib':result['peak_cuda_allocated_mib'],'peak_cuda_reserved_mib':result['peak_cuda_reserved_mib'],
                       'peak_rss_mib':result['peak_rss_mib']})
    result_summary=summary(records,costs)
    common_runtime=records['FusedSpaceFed'][42]['runtime'].copy();common_runtime.pop('device')
    early={}
    for method in METHODS:
        for seed,record in records[method].items():
            current_runtime=record['runtime'].copy();current_runtime.pop('device')
            if current_runtime!=common_runtime:
                raise ValueError('Runtime differs beyond GPU assignment')
            first=record['history'][:5]
            early[f'{method}-seed-{seed}']={'round':5,'median_round_seconds':statistics.median(row['seconds'] for row in first),
                                           'remaining_training_seconds':statistics.median(row['seconds'] for row in first)*295,
                                           'excludes_final_test_and_checkpoint_io':True}
    result_summary.update(status='verified',primary_metric='uniform_domain_accuracy_percent',parameters=profile['parameters'],
                          validation_training_flops=validation_totals,validation_results=checked_validation,
                          timing=timing,selection_sha256=canon(selection),partition_sha256=partition['partition_sha256'],
                          profile_sha256=profile['profile_sha256'],excluded_flop_operations=profile['excluded'],matching_metric=profile['metric'],
                          common_runtime=common_runtime,early_estimates=early)
    destination.mkdir();archived={}
    for job in validation_queue['jobs']:
        archived['validation/'+job['name']]=archive_run(Path(job['output']),destination/'validation'/job['name'])
    for entry in registry:
        name=f"{entry['method']}-seed-{entry['seed']}"
        archived['final/'+name]=archive_run(Path(entry['source_directory']),destination/'final'/name)
    result_summary['archived_source_files']=archived
    write(destination/'summary.json',result_summary)
    for name,campaign in [('validation',validation_campaign),('final',final_campaign)]:
        public={key:campaign[key] for key in ('status','code_commit','started_utc','ended_utc','wall_seconds','process_wall_seconds_sum','max_processes','max_per_gpu','external_actions')}
        public['runs']=[{key:row[key] for key in ('name','device','started_utc','ended_utc','exit_code','process_wall_seconds')} for row in campaign['runs']]
        write(destination/f'{name}_campaign.json',public)
    report_result(result_summary,directory)
    payload=[path for path in sorted(directory.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and path.name!='artifact_manifest.json']
    write(directory/'artifact_manifest.json',{'files':{str(path.relative_to(directory)):{'bytes':path.stat().st_size,'sha256':sha(path)} for path in payload},
                                              'selection_sha256':canon(selection),'code_commits':sorted({row['code_commit'] for method in costs.values() for row in method.values()})})
    print(json.dumps({'status':'verified','summary':result_summary['methods'],'paired':result_summary['paired_fused_minus_fedavg']},indent=2))


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def report_result(result,directory):
    selection=json.loads((directory/'selection.json').read_text())
    rows=[]
    for i,seed in enumerate(SEEDS):
        rows.append([seed,*[f"{result['methods'][method]['uniform_domain_accuracy_percent']['values'][i]:.6f}" for method in METHODS],
                     f"{result['paired_fused_minus_fedavg']['uniform_domain_accuracy_percent']['values'][i]:+.6f}",
                     *[f"{result['methods'][method]['sample_weighted_accuracy_percent']['values'][i]:.6f}" for method in METHODS]])
    metric_table=markdown_table(['Seed','Fused uniforme %','FedAvg uniforme %','Differenza pp','Fused pesata %','FedAvg pesata %'],rows)
    aggregate_table=markdown_table(['Metrica','Fused media ± SD','FedAvg media ± SD','Delta abbinato media ± SD'],[
        [key,*[f"{result['methods'][method][key]['mean']:.6f} ± {result['methods'][method][key]['sample_sd_ddof1']:.6f}" for method in METHODS],
         f"{result['paired_fused_minus_fedavg'][key]['mean']:+.6f} ± {result['paired_fused_minus_fedavg'][key]['sample_sd_ddof1']:.6f}"]
        for key in ('uniform_domain_accuracy_percent','sample_weighted_accuracy_percent')])
    domain_table=markdown_table(['Dominio','Fused media ± SD','FedAvg media ± SD'],[
        [domain,*[f"{result['methods'][method]['domains'][domain]['mean']:.6f} ± {result['methods'][method]['domains'][domain]['sample_sd_ddof1']:.6f}" for method in METHODS]] for domain in DOMAINS])
    parameter=result['parameters'];fused_active=parameter['classifier']+parameter['encoder']+parameter['decoder']
    costs_table=markdown_table(['Metodo/seed','FLOPs contate TF','FLOPs dense TF','Warm-up passi','CE passi','Esposizioni CE'],[
        [method+'/'+seed,f"{value['counted_training_flops']/1e12:.9f}",f"{value['dense_training_flops']/1e12:.9f}",
         value['warmup_steps'],value['classification_steps'],value['classification_samples']]
        for method in METHODS for seed,value in result['costs'][method].items()])
    timing_table=markdown_table(['Metodo/seed','Origine','GPU','Sessione s','Test s','CUDA alloc./ris. MiB','RSS MiB'],[
        [row['method']+'/'+str(row['seed']),row['origin'],row['device'],f"{row['wall_seconds']:.3f}",f"{row['test_seconds']:.3f}",
         f"{row['peak_cuda_allocated_mib']:.3f}/{row['peak_cuda_reserved_mib']:.3f}",f"{row['peak_rss_mib']:.3f}"] for row in result['timing']])
    calibration_table=markdown_table(['Metodo','LR C','LR AE','Clip','Validation uniforme %'],[
        [row['method'],row['classifier_lr'],row['autoencoder_lr'] if row['method']=='FusedSpaceFed' else '—',row['gradient_clip_norm'],f"{row['uniform_validation_accuracy_percent']:.6f}"] for row in selection['scores']])
    cv=json.loads((directory/'artifacts/validation_campaign.json').read_text());final=json.loads((directory/'artifacts/final_campaign.json').read_text())
    deviation=max(abs(result['costs']['FedAvg'][str(seed)]['counted_training_flops']/result['costs']['FusedSpaceFed'][str(seed)]['counted_training_flops']-1) for seed in SEEDS)*100
    delta=result['paired_fused_minus_fedavg']['uniform_domain_accuracy_percent']['mean']
    verdict=('In questo controllo FusedSpaceFed ottiene una media maggiore di FedAvg ampliato e con budget abbinato.' if delta>0 else
             'In questo controllo FusedSpaceFed ottiene una media minore di FedAvg ampliato e con budget abbinato.' if delta<0 else
             'In questo controllo le medie dei due metodi coincidono.')
    report=f'''# Controllo minimale di capacità e calcolo — Digits

{verdict} La differenza primaria è **{delta:+.6f} punti percentuali**. È un confronto diretto fra nostri metodi, distinto dal precedente confronto con baseline pubblicate. Non dimostra che i vantaggi in altri setting dipendano dai soli parametri o dalla sola computazione.

Il seed 42 favorisce FedAvg; 43 e 44 favoriscono FusedSpaceFed. Il delta medio è piccolo rispetto alla sua SD fra seed: tre ripetizioni non sostengono una conclusione causale robusta sull'origine dei vantaggi. Questo controllo indica un residuo vantaggio medio descrittivo con risorse abbinate, con eccezioni visibili; non certifica superiorità generale.

{metric_table}

{aggregate_table}

Media e SD campionaria fra tre seed prefissati, ddof=1. Primaria: media uniforme dei cinque domini, dichiarata prima della calibrazione. Conteggi/confusioni ricostruiscono ogni valore; nessun miglior seed o checkpoint. La media pesata è dominata dal test SynthDigits.

{domain_table}

## Capacità e budget effettivo

Scenario scelto per costo e disponibilità: Digits bilanciato già congelato, 743 immagini/client, cinque client, 300 round, batch massimo 32, tutti i client e BN condivisi, test completo al solo round 300 senza adattamento. Stessa partizione, preprocessing, valutazione e seed 42–44 per entrambi.

Fused attivi/client: **{fused_active:,}** = C {parameter['classifier']:,} + E {parameter['encoder']:,} + D {parameter['decoder']:,}. FedAvg: **{parameter['fedavg']:,}**, CNN con fc1 2065 anziché 2048, scarto **{100*(parameter['fedavg']/fused_active-1):+.6f}%**. Fused memorizza globalmente C+D+5E = {parameter['classifier']+parameter['decoder']+5*parameter['encoder']:,} parametri: capacità attiva abbinata non significa stesso totale memorizzato o stessa memoria.

{costs_table}

Scarto massimo del budget cumulativo FedAvg/Fused: **{deviation:.9f}%**. Il warm-up è incluso. FedAvg vede tutti gli esempi una volta per round, poi minibatch aggiuntivi; resti interi inferiori al costo di due esempi vengono riportati ai round successivi. Passi ed esposizioni sono reali, salvati per client/round. FedAvg qui è un controllo modificato, non la baseline pubblicata a una sola epoca.

Convenzione: [Torch 2.5.1 FlopCounterMode](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/flop_counter.py) sul forward/backward eseguito, FMA=2, comprese convoluzioni trasposte e gradienti attraverso D congelato. Aggiunta convenzione aritmetica SGD+norma/clipping=5P e Adam+norma/clipping=16P per passo. GPU/CPU producono le stesse firme dense. BN, attivazioni, pooling, loss, controlli finite, copie, aggregazione e overhead dei kernel esclusi: **budget abbinato nella metrica dichiarata, non misura esaustiva di istruzioni hardware, energia o tempo**. Le componenti scalari sono una stima semantica esplicita; conteggi dense e totali sono entrambi forniti.

## Calibrazione e congelamento

Train-only: 594 fit e 149 validation per dominio, seed dati 20261005, stratificazione deterministica per ID/quote; seed calibrazione 142. Nove candidati per metodo, 60 round e stesso budget contabile; LR C [.005,.01,.02], clip [.5,1,2], Fused LR AE [.0001,.0003,.001]. L9 bilanciato per Fused, griglia proiettata 3×3 per FedAvg, riferimento precedente incluso. Non è una ricerca esaustiva delle 27 interazioni Fused. L'utente ha esteso il piano prima di qualsiasi worker; il primo piano mai eseguito rimane conservato.

{calibration_table}

Impostazioni selezionate esclusivamente da validation al round 60:

```json
{json.dumps(selection['selected_training_settings'],indent=2)}
```

Parità: LR C più basso, poi clip più basso e LR AE più basso. Configurazioni definitive e selection SHA congelati e pubblicati prima dei nuovi test; nessuna riapertura del tuning. Altri optimizer, architettura e dati invariati. Reuse solo se impostazioni/dati/sorgenti combaciano: l'origine di ogni run è riportata. Nuove run da zero, nessun trasferimento dai checkpoint di calibrazione. I cinque vecchi test Fused erano già osservati: **controllo retrospettivo**, senza pretesa di test mai visto. La selezione usa soltanto la nuova validation da training.

## Tempi e verifiche

{timing_table}

Calibrazione: {cv['wall_seconds']:.3f} s di calendario, {cv['process_wall_seconds_sum']:.3f} s somma processi. Run definitive nuove: {final['wall_seconds']:.3f} s di calendario, {final['process_wall_seconds_sum']:.3f} s somma processi. Le run riusate, se presenti, costano zero nuova esecuzione; i loro tempi originali sono separati nella tabella. Picchi allocator Torch per processo, esclusi contesto/driver/altri job. GPU 0 condivisa con autorizzazione, al massimo un worker per GPU, nessun processo altrui interrotto.

245 test CPU passati in 34,60 s prima del training, inclusi 12 del nuovo controllo; firma GPU sintetica verificata. Cinque ulteriori test stdlib della sintesi numerica passati dopo il training. Checkpoint/ripresa esatta per entrambi, encoder/RNG/carry, split disgiunto/riproducibile, arrotondamenti/budget e CLI dirette. Ulteriori prove sintetiche: selezione/tie/freeze e parità di due round Fused con il driver precedente. Audit indipendente dei 18 tentativi e sei finali: stati completi/exits, round, seed/hash, conteggi, confusioni, costi effettivi e medie/SD. Dataset, checkpoint e log integrali restano privati. Primo avvio fallito prima dei worker per collisione del nome select.py, corretto; primo push HTTP 504, retry HTTP/1.1 riuscito; nessuna modifica dei modelli per questi problemi operativi.

Partizione {result['partition_sha256']}; validation {json.loads((directory/'validation_split.json').read_text())['validation_sha256']}; profilo {result['profile_sha256']}; selection {result['selection_sha256']}. Configurazioni, manifesti, tutti i tentativi numerici e timing sono in questa directory; raw/checkpoint/log in `_local/capacity_compute_control/` e vecchie run conservate. Comandi nel README. Manoscritto e altri esperimenti invariati.

## Limiti

Un solo feature-shift setting e tre seed, partizione fissa, screening corto con un seed di validation; interazioni e variabilità del tuning non stimate. Match congiunto di parametri e calcolo non ne separa gli effetti causali. Parametri privati, geometria del modello, BN e numero di esposizioni supervisionate differiscono. Il risultato non giustifica generalizzazioni ai setting label-skew del paper. Non sono inferite significatività o varianze mancanti; tutte le differenze e le eccezioni restano visibili.
'''
    (directory/'CAPACITY_COMPUTE_REPORT.md').write_text(report)


def verify_archive(directory):
    manifest=json.loads((directory/'artifact_manifest.json').read_text())
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'}
    if set(manifest['files'])!=actual:
        raise ValueError('Manifest file set differs')
    for name,item in manifest['files'].items():
        path=directory/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Manifest file changed: '+name)
    saved=json.loads((directory/'artifacts/summary.json').read_text());profile=json.loads((directory/'flop_profile.json').read_text())
    partition=json.loads(Path('research/feature_shift_digits/partition_manifest.json').read_text());records={method:{} for method in METHODS};costs={method:{} for method in METHODS}
    registry=json.loads((directory/'final_queue.json').read_text())['runs_registry']
    for entry in registry:
        name=f"{entry['method']}-seed-{entry['seed']}";target=directory/'artifacts/final'/name
        raw=gzip.decompress((target/'results.json.gz').read_bytes());record=json.loads(raw)
        identity=saved['archived_source_files']['final/'+name]
        if hashlib.sha256(raw).hexdigest()!=identity['raw_result_sha256']:
            raise ValueError('Lossless raw result SHA differs')
        if [json.loads(line) for line in (target/'timings.jsonl').read_text().splitlines()]!=record['history']:
            raise ValueError('Timing history differs')
        computed=audit_run(record,profile,743,300,'final',{d:partition['domains'][d]['test']['count'] for d in DOMAINS},reused=entry['origin']=='reused_previous')
        records[entry['method']][entry['seed']]=record;costs[entry['method']][str(entry['seed'])]=computed
    reconstructed=summary(records,costs)
    if any(saved[key]!=value for key,value in reconstructed.items()):
        raise ValueError('Summary differs from reconstructed counts/costs')
    for name,value in saved['validation_results'].items():
        target=directory/'artifacts/validation'/name
        raw=gzip.decompress((target/'results.json.gz').read_bytes());record=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=saved['archived_source_files']['validation/'+name]['raw_result_sha256']:
            raise ValueError('Validation raw SHA differs')
        audited=audit_run(record,profile,594,60,'validation',{d:149 for d in DOMAINS})
        if audited!=value['costs'] or record['evaluations'][0]!=value['evaluation']:
            raise ValueError('Validation archived audit differs')
    return {'status':'passed','payload_files':len(actual),'final_pairs':3,'validation_runs':18}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('build','verify'));parser.add_argument('--directory',type=Path,default=Path('research/capacity_compute_control'))
    args=parser.parse_args()
    if args.mode=='build':
        build(args.directory)
    else:
        print(json.dumps(verify_archive(args.directory),indent=2))
