"""Verify complete states/data locality/counts and publish the measured outcome."""
import argparse
import gzip
import json
from pathlib import Path
import re
import shutil
import torch
from research.pathmnist_pathological.run import file_hash,write_json,assert_finite
from research.pathmnist_head_only.run import check_complete,tree_equal
from research.pathmnist_federated_head.run import ROOT,PUBLIC

PRIVATE=ROOT/'_local/pathmnist_federated_head'


def compressed(raw,path):
    path.write_bytes(gzip.compress(raw,mtime=0))
    if gzip.decompress(path.read_bytes())!=raw:raise AssertionError('Lossy archive')


def audit():
    torch.set_num_threads(2);folder=PRIVATE/'seed-42';r=json.loads((folder/'results.json').read_text());assert_finite(r)
    execution=json.loads((PRIVATE/'logs/execution.json').read_text())
    if r['status']!='completed' or execution['exit_code']!=0 or r['client_exit_codes']!=[0]*10:raise ValueError('Incomplete process')
    if len({c['pid'] for c in r['clients']})!=10 or execution['pid'] in {c['pid'] for c in r['clients']}:
        raise ValueError('Clients not separate server processes')
    if sum(c['train_samples'] for c in r['clients'])!=89996 or any(c['test_samples']!=7180 for c in r['clients']):raise ValueError('Changed samples')
    for name,digest in r['code']['source_sha256'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Changed code '+name)
    for name,digest in r['checkpoint_sha256'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Changed checkpoint '+name)
    source=torch.load(ROOT/r['configuration']['checkpoint'],weights_only=False,map_location='cpu')
    final=torch.load(folder/'final.pt',weights_only=False,map_location='cpu');check_complete(source,final)
    for cid,c in enumerate(r['clients']):
        local=folder/'clients'/f'client-{cid}'
        if c['frozen_before']!=c['frozen_after'] or c['loss_gradient_calls']!=sum(r['aggregations'].values()):raise ValueError('Client changed model/budget')
        if file_hash(local/'private_features.pt')!=c['feature_cache_sha256'] or file_hash(local/'final.pt')!=c['private_checkpoint_sha256']:
            raise ValueError('Changed private client cache/checkpoint')
        cp=torch.load(local/'final.pt',weights_only=False,map_location='cpu')
        if cp['training_indices']!=source['partitions']['clients'][str(cid)]:raise ValueError('Wrong private owner indices')
        if not tree_equal(cp['client'],final['clients'][str(cid)]):raise ValueError('Client/global complete checkpoints differ')
    for name,total in r['correct_counts'].items():
        if total!=sum(c['correct_counts'][name] for c in r['clients']) or abs(r['accuracy_percent'][name]-100*total/71800)>1e-10:
            raise ValueError('Accuracy not reconstructed from counts')
    C=r['aggregations']['optimization'];V=r['aggregations']['verification'];comm=r['communication']
    if C!=r['head_statistics']['function_evaluations'] or C!=r['head_statistics']['closure_evaluations'] or V!=3:
        raise ValueError('Incorrect closure/aggregation count')
    if comm['payload_total_bytes']!=(C+V)*46860+46830:raise ValueError('Byte accounting incorrect')
    if comm['with_pipe_framing_total_bytes']!=comm['payload_total_bytes']+(C+V)*80+120:raise ValueError('Pipe frame accounting incorrect')
    for key in ('initial_loss','central_final_loss_same_head','central_pre_last_loss','central_pre_last_gradient'):
        if not r['comparisons'][key]['within_tolerance']:raise ValueError('Same-point objective/gradient equivalence failed')
    if r['all_declared_tolerances_met']!=all(x['within_tolerance'] for x in r['comparisons'].values()):raise ValueError('Wrong equivalence flag')
    guard=json.loads((PRIVATE/'original_campaign_guard.json').read_text())
    if any(file_hash(ROOT/name)!=digest for name,digest in guard.items()):raise ValueError('Prior campaign artifact changed')
    return r,execution,{'status':'passed_protocol_and_checkpoint_checks',
        'terminal_numerical_equivalence':'passed' if r['all_declared_tolerances_met'] else 'not established under declared tolerances',
        'source_fields_unchanged_except_fc':True,'original_BN_buffers_bit_identical':True,
        'all_ten_complete_client_states_optimizer_scaler_rng_indices_preserved':True,'separate_client_processes':10,
        'no_private_features_labels_or_logits_in_head_protocol':True,'samples_and_counts_verified':True,
        'all_bytes_reconstructed_from_messages':True,'prior_campaign_unchanged_files':len(guard)}


def build():
    r,execution,verification=audit();out=PUBLIC/'artifacts'
    if out.exists():raise FileExistsError('Preserve previous archive')
    out.mkdir();folder=PRIVATE/'seed-42'
    compressed((folder/'results.json').read_bytes(),out/'results.json.gz')
    shutil.copyfile(PRIVATE/'logs/execution.json',out/'execution.json')
    shutil.copyfile(ROOT/'_local/pathmnist_federated_head_tests.log',out/'tests.log')
    match=re.search(r'(\d+) passed in ([\d.]+)s',(out/'tests.log').read_text())
    if not match:raise ValueError('Missing passed suite')
    verification['tests']={'passed':int(match[1]),'seconds':float(match[2])};write_json(out/'verification.json',verification)
    private={str((folder/name).relative_to(ROOT)):{'sha256':digest,'bytes':(folder/name).stat().st_size}
             for name,digest in r['checkpoint_sha256'].items()}
    for cid,c in enumerate(r['clients']):
        local=folder/'clients'/f'client-{cid}'
        for name,key in (('private_features.pt','feature_cache_sha256'),('final.pt','private_checkpoint_sha256')):
            private[str((local/name).relative_to(ROOT))]={'sha256':c[key],'bytes':(local/name).stat().st_size}
    g=torch.load(folder/'gradient_audit.pt',weights_only=False,map_location='cpu')
    compressed((json.dumps({name:value.tolist() for name,value in g.items()},allow_nan=False)+'\n').encode(),out/'aggregate_gradient_audit.json.gz')
    trace=torch.load(folder/'closure_trace.pt',weights_only=False,map_location='cpu')
    trace_rows=[{'closure':i+1,'loss':x['loss'],'gradient_l2_norm':float(x['gradient'].norm()),'head_l2_norm':float(x['theta'].norm())} for i,x in enumerate(trace)]
    compressed((json.dumps(trace_rows,allow_nan=False)+'\n').encode(),out/'closure_trace.json.gz')
    summary={k:r[k] for k in ('status','accuracy_percent','correct_counts','comparisons','all_declared_tolerances_met',
                             'aggregations','communication','head_statistics','code','configuration')}
    summary.update(private_artifacts=private,execution=execution,verification=verification)
    write_json(out/'summary.json',summary)
    create_report(r,execution,verification,private)
    entries={str(p.relative_to(PUBLIC)):{'sha256':file_hash(p),'bytes':p.stat().st_size}
             for p in PUBLIC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    write_json(out/'manifest.json',{'files':entries,'checkpoint_policy':'complete checkpoints/features/labels/logs only in private _local on thanos'})
    verify()


def create_report(r,execution,verification,private):
    c=r['comparisons'];h=r['head_statistics'];bytes_=r['communication'];accuracy=r['accuracy_percent']
    lines=['# Refit della testa PathMNIST in simulazione federata','',
        f"**Federato: {accuracy['federated']:.6f}%; centralizzato: {accuracy['centralized']:.6f}%**, differenza {accuracy['federated']-accuracy['centralized']:+.6f} punti percentuali. Prima del refit: {accuracy['original']:.6f}%. Un solo seed42, checkpoint originale round50; nessuna SD tra seed.", '',
        '**Esito dei controlli:** loss e gradiente allo stesso punto del riferimento passano le tolleranze dichiarate. I punti terminali dei due L-BFGS troncati a100 iterazioni **non sono numericamente equivalenti entro tutte quelle tolleranze**. Questo mancato accordo non viene nascosto né corretto aumentando le tolleranze dopo il test.', '',
        '## Procedura e separazione dei dati','',
        'Fonte originale SHA256 fa56da01091debe9031a9acb2e3de0f34ce0f751d2c95d2d0d54c05f15697d1e; riferimento centralizzato SHA25631019b0e1ef8594de1f7da196a6ce4b7e84afdc58fb4ba3d42fcf14b7fd9d4b8. Pesi originali, nessun checkpoint tuned. Solo585 parametri fc.weight/fc.bias (64→9) ottimizzati. Encoder privati, decoder, corpo ResNet20V2 e tutti57 buffer BN bit-identici alla fonte; fusione x+D(E_i(x)), nessuna ricalibrazione o penalità.', '',
        'Dieci processi client separati, con cache/etichette delle sole proprie immagini training,89996 esempi complessivi. Nessuna matrice di feature pooled è costruita. Il server ottimizzatore invia solo la testa e riceve soltanto loss locale e585 gradienti Float32; media uniforme dei client, indipendente dalle loro numerosità. Ready/done sono stati di controllo. Per la verifica il client scrive metadati scalari/conteggi sul disco locale; l’assemblatore del report legge questi metadati, senza acquisire feature, etichette o logits grezzi. Fonte/dati preinstallati, simulazione su un host: nessuna pretesa di privacy crittografica.', '',
        'Stessa L-BFGS del centralizzato: LR1, max_iter100/max_eval125, history20, strong-Wolfe, tolerance_grad1e-7, tolerance_change1e-10, CE full-batch media uniforme, penalità0, Float32 senza AMP. Nessun tuning, nuovo candidato, sweep o selezione dopo il test. Tutte le feature sono estratte da modelli eval/no_grad con encoder proprietario.', '',
        f"Effettive {h['actual_iterations']} iterazioni, {h['function_evaluations']} valutazioni obiettivo/gradiente; il centralizzato ne aveva100 e105. CE federata {h['objective_history'][0]:.9f} → {h['final_uniform_client_training_ce']:.9f}; CE terminale centralizzata0.597145498. La testa è salvata e congelata prima di leggere il test. Dopo il test il checkpoint viene arricchito solo con RNG/contatori, senza cambiare pesi.", '',
        '## Loss e gradienti','',
        'Il vecchio optimizer centralizzato conserva il gradiente precedente all’ultimo aggiornamento: non è il gradiente finale. Il punto è ricostruito come theta_pre=theta_final−t*d, con piccolo roundoff Float32. Valutandolo presso tutti i client si confronta il gradiente aggregato con quello realmente archiviato dal centralizzato. La loss iniziale e quella alla testa centralizzata finale sono confrontate con i valori archiviati. Non si riesegue alcun refit centralizzato.', '',
        '| Controllo | Differenza assoluta | Tolleranza | Esito |','|---|---:|---|---|']
    for key,label in (('initial_loss','CE iniziale'),('central_final_loss_same_head','CE alla stessa testa centrale finale'),('central_pre_last_loss','CE al punto centrale precedente')):
        v=c[key];lines.append(f"| {label} | {v['abs_error']:.9e} | atol2e-6 + rtol2e-6 | passa |")
    g=c['central_pre_last_gradient'];lines.append(f"| Gradiente al punto precedente, max per-entry | {g['max_abs_error']:.9e} | atol1e-5 + rtol1e-4 | passa |")
    lines += [f"| CE ai due punti terminali | {c['final_optimized_loss']['abs_error']:.9e} | atol2e-4 + rtol1e-4 | non passa |",
              f"| Gradienti ai due punti terminali, max per-entry | {c['final_optimized_gradient']['max_abs_error']:.9e} | atol2e-4 + rtol1e-2 | non passa |", '',
              f"Errore L2 sul gradiente archiviato: {g['l2_error']:.9e}, relativo {g['relative_l2_error']:.9e}. Ai terminali, norma gradiente federato {c['final_optimized_gradient']['actual_l2_norm']:.9e}, riferimento valutato dalla media dei client {c['final_optimized_gradient']['reference_l2_norm']:.9e}; sono gradienti in punti diversi, non un controllo della stessa derivata.", '',
              '## Logits e accuratezza','',
              '| Split | Differenza max logits | RMS | Predizioni discordanti |','|---|---:|---:|---:|']
    for key,label in (('final_train_logits','Training'),('final_test_logits','Test')):
        v=c[key];lines.append(f"| {label} | {v['max_abs']:.6f} | {v['rms']:.6f} | {v['prediction_disagreements']}/{v['samples']} |")
    lines+=['', 'La tolleranza logits era0.02+0.002*abs(reference): non rispettata. Disaccordo test2.363510%, oltre lo0.1% dichiarato. Accuracy differisce0.130919pp, oltre la tolleranza0.05pp; risultati vicini in performance non implicano logits/predizioni identici.', '',
        '| Client | Originale corrette | Centrale corrette | Federato corrette | Centrale (%) | Federato (%) |','|---:|---:|---:|---:|---:|---:|']
    for v in r['clients']:
        n=v['correct_counts'];a=v['accuracy_percent'];lines.append(f"| {v['client_id']} | {n['original']} | {n['centralized']} | {n['federated']} | {a['centralized']:.6f} | {a['federated']:.6f} |")
    lines+=['',f"Totali: originale {r['correct_counts']['original']}, centrale {r['correct_counts']['centralized']}, federato {r['correct_counts']['federated']} su71800 predizioni. Media uniforme di tutte10 pipeline sui medesimi7180 test ufficiali; non71800 immagini indipendenti o ensemble. I conteggi originale e centralizzato riproducono esattamente gli archivi precedenti.", '',
        '## Interpretazione numerica','',
        'La media federata implementa lo stesso obiettivo/gradiente matematico sulle rappresentazioni congelate: il confronto con la derivata archiviata lo verifica direttamente nel punto disponibile. Float32 cambia l’ordine delle riduzioni rispetto al backward sulla matrice pooled del vecchio refit. Differenze iniziali piccole possono influire sulla ricerca di linea e sugli aggiornamenti di curvatura L-BFGS;107 valutazioni contro105 e storie CE diverse sono coerenti con tale sensibilità. Non si afferma uguaglianza bitwise o convergenza al medesimo ottimo: entrambi si fermano al budget100, con gradienti ben sopra1e-7.', '',
        'Il beneficio qualitativo del refit rispetto al39.62% originale è riprodotto senza trasferire le feature al server, ma la corrispondenza numerica terminale più stretta **non è dimostrata**. Non è un confronto tra baseline, una media di cinque seed o una modifica al protocollo del paper: resta una fase supervised aggiuntiva sulla sola testa. Nessun test ha guidato impostazioni o ripetizioni.', '',
        '## Aggregazioni e byte','',
        f"{r['aggregations']['optimization']} aggregazioni per ottimizzazione, incluse tutte le strong-Wolfe closure, più {r['aggregations']['verification']} aggregazioni diagnostiche train-only: totale {sum(r['aggregations'].values())}. Tutti10 client in ogni aggregazione. Ogni request2341 bytes (opcode+585 Float32); ogni response2345 bytes (opcode+loss+585 gradienti).", '',
        '| Categoria | Payload down | Payload up | Totale payload | Con framing Pipe |','|---|---:|---:|---:|---:|']
    for name,v in bytes_['categories'].items():lines.append(f"| {name} | {v['payload_down']} | {v['payload_up']} | {v['payload_total']} | {v['with_pipe_framing_bytes']} |")
    lines += ['',f"**Totale {bytes_['payload_total_bytes']:,} bytes applicativi; {bytes_['with_pipe_framing_total_bytes']:,} bytes con framing Pipe4 bytes/messaggio.** Solo ottimizzazione:5,014,020 bytes applicativi. Conteggi verificati da messaggi e107 closure. Inclusi broadcast finali per confronto e stati ready/done; esclusi dataset/body preinstallati, avvio processi OS, accessi/scritture dei file locali e overheadTCP/TLS non simulato. Non è costo comunicativo del training originale completo.", '',
        '## Tempi, checkpoint e verifiche','',
        f"GPU cuda:1, dieci client più server sulla stessa GPU. Sessione {r['wall_seconds']:.3f}s; processo {execution['process_wall_seconds']:.3f}s, incluse import/init. Preparazione client+feature {r['feature_setup_wall_seconds']:.3f}s, optimizer {r['optimization_seconds']:.3f}s, audit d’inferenza {r['inference_audit_seconds']:.3f}s. Picco GPU1 campionato ogni0.5s: {execution['gpu_memory_peak_mib']['1']}MiB, include contesti CUDA/processi; può perdere picchi più brevi. Picchi Torch e RSS per client in results.json. GPU0 resta451MiB con il processo preesistente; nessun processo altrui interrotto.", '',
        'Checkpoint completo: `_local/pathmnist_federated_head/seed-42/final.pt`. Fonte round0 in training-initial.pt e round50 in before.pt; final contiene classifier, decoder, tutti10 encoder/copie client,57 BN buffer originali,20 optimizer originali,scaler,RNG e indici invariati; aggiunge optimizer L-BFGS, server/client phase RNG,config/codice/contatori. Ogni client conserva anche final.pt e private_features.pt nella propria cartella. Nessuna feature/etichetta o checkpoint è pubblicato.', '',
        f"SHA256 finale `{r['checkpoint_sha256']['final.pt']}`. Verifica del checkpoint completo e tutti gli stati finiti passata; {verification['prior_campaign_unchanged_files']} artefatti della campagna precedente invariati. Suite: {verification['tests']['passed']} test passati in {verification['tests']['seconds']:.2f}s, inclusi7 nuovi test. Separare successo dei controlli di protocollo dalla mancata equivalenza numerica terminale.", '',
        'Codice/configurazione congelati nel commit `'+r['code']['base_commit']+'`; sorgenti identificati in artifacts/results.json.gz. Artefatti numerici lossless, gradienti aggregati e traccia CE in artifacts/; hash e percorsi privati in summary.json. Il manoscritto e gli altri esperimenti non sono stati modificati.', '',
        'Comando eseguito (exit0):','', '```bash',' '.join(execution['command']),'```','',
        'Verifica archivio:','', '```bash','/home/schroeder/miniconda3/envs/general_ml/bin/python -m research.pathmnist_federated_head.archive verify','```']
    text='\n'.join(lines)+'\n'
    # Keep the generated prose readable without changing numeric values or tolerances.
    for before,after in (('seed42','seed 42'),('round50','round 50'),('round0','round 0'),('a100','a 100'),
        ('Solo585','Solo 585'),('tutti57','tutti 57'),('SHA256310','SHA256 310'),('training,89996','training, 89.996'),
        ('e585','e 585'),('LR1,','LR 1,'),('max_iter100/max_eval125','max_iter 100/max_eval 125'),
        ('history20','history 20'),('penalità0','penalità 0'),('tolerance_grad1e-7','tolerance_grad 1e-7'),
        ('tolerance_change1e-10','tolerance_change 1e-10'),('aveva100 e105','aveva 100 e 105'),
        ('centralizzata0.','centralizzata 0.'),('atol2','atol 2'),('rtol2','rtol 2'),('atol1','atol 1'),('rtol1','rtol 1'),
        ('era0.','era 0.'),('test2.','test 2.'),('lo0.','lo 0.'),('differisce0.','differisce 0.'),('tolleranza0.','tolleranza 0.'),
        ('su71800','su 71.800'),('tutte10','tutte 10'),('medesimi7180','medesimi 7.180'),('non71800','non 71.800'),
        ('L-BFGS;107','L-BFGS; 107'),('contro105','contro 105'),('budget100','budget 100'),('sopra1e-7','sopra 1e-7'),
        ('al39.62','al 39.62'),('Tutti10','Tutti 10'),('request2341','request 2341'),('response2345','response 2345'),
        ('Pipe4','Pipe 4'),('ottimizzazione:5,','ottimizzazione: 5,'),('e107','e 107'),('overheadTCP','overhead TCP'),
        ('ogni0.5s','ogni 0.5 s'),('resta451','resta 451'),('tutti10','tutti 10'),('client,57','client, 57'),
        ('originali,20','originali, 20'),('scaler,RNG','scaler, RNG'),('inclusi7','inclusi 7'),('exit0','exit 0')):
        if before in ('e585','e107'):
            text=re.sub(r'(?<!\w)'+re.escape(before)+r'(?!\w)',after,text)
        else:
            text=text.replace(before,after)
    for digest in (r['configuration']['checkpoint_sha256'],r['configuration']['centralized_checkpoint_sha256'],r['checkpoint_sha256']['final.pt']):
        if digest not in text:raise AssertionError('Report formatting damaged a checkpoint identity')
    (PUBLIC/'FEDERATED_HEAD_REPORT.md').write_text(text);(PRIVATE/'report.md').write_text(text)


def verify():
    out=PUBLIC/'artifacts';manifest=json.loads((out/'manifest.json').read_text())
    for name,spec in manifest['files'].items():
        path=PUBLIC/name
        if file_hash(path)!=spec['sha256'] or path.stat().st_size!=spec['bytes']:raise ValueError('Changed archive '+name)
    r=json.loads(gzip.decompress((out/'results.json.gz').read_bytes()))
    if r!=json.loads((PRIVATE/'seed-42/results.json').read_text()):raise ValueError('Published/local results differ')
    audit();print(json.dumps({'protocol_and_archive_verified':True,'all_declared_tolerances_met':r['all_declared_tolerances_met']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('build','verify'))
    {'build':build,'verify':verify}[p.parse_args().mode]()
