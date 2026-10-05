"""Independent stdlib-only verification, numeric archive, report and handoff.

Never imports training, evaluates a model, reads images or deserializes a
checkpoint. Build only after all five definitive workers exit successfully.
"""
from __future__ import annotations

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
import zipfile

DOMAINS = ('MNIST','MNIST-M','SVHN','SynthDigits','USPS')
SEEDS = (42,43,44,45,46)
DEVICES = {seed: 'cuda:1' if seed % 2 == 0 else 'cuda:0' for seed in SEEDS}


def scientific_config(config):
    return {key:value for key,value in config.items() if key not in ('run_seeds','per_seed_device')}



def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def canon(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def finite(value):
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def close(a,b):
    if not math.isclose(a,b,rel_tol=0,abs_tol=1e-10):
        raise ValueError(f'Numeric inconsistency: {a} != {b}')


def audit(results,config,partition):
    expected_partition=partition.copy()
    expected_hash=expected_partition.pop('partition_sha256')
    if canon(expected_partition)!=expected_hash or expected_hash!=config['partition_sha256']:
        raise ValueError('Partition manifest/config identity differs')
    if sorted(results)!=list(SEEDS):
        raise ValueError('Exactly all five definitive seeds required')
    original_config=config.copy()
    original_config['run_seeds']=[42,43]
    original_config['per_seed_device']={'42':'cuda:1','43':'cuda:0'}
    metrics={}
    rounds=0
    costs={}
    for seed,result in results.items():
        if not finite(result) or result['status']!='completed' or result['completed_rounds']!=300:
            raise ValueError('Incomplete or non-finite definitive run')
        identity=result['identity']
        if identity['seed']!=seed or identity['device']!=DEVICES[seed] or identity['method']!='FusedSpaceFed':
            raise ValueError('Wrong seed/device/method')
        registered_config=original_config if seed in (42,43) else config
        if identity['config_sha256']!=canon(registered_config) or result['configuration']!=registered_config or identity['partition_sha256']!=expected_hash:
            raise ValueError('Run configuration/partition differs')
        if [row['round'] for row in result['history']]!=list(range(1,301)):
            raise ValueError('Round sequence is not exactly 1..300')
        totals={key:0 for key in ('warmup_steps','classification_steps','encoder_steps','decoder_steps','classifier_steps','warmup_samples','classification_samples')}
        for row in result['history']:
            if set(row['clients'])!=set(DOMAINS):
                raise ValueError('A domain did not participate')
            for client in row['clients'].values():
                if client['warmup_steps']!=24 or client['classification_steps']!=24 or client['encoder_steps']!=48:
                    raise ValueError('Wrong phase step counts')
                if client['decoder_steps']!=24 or client['classifier_steps']!=24 or client['warmup_samples']!=743 or client['classification_samples']!=743:
                    raise ValueError('Wrong local epoch/sample counts')
                if set(client['clipping'])!={'warmup_autoencoder','classification_autoencoder','classification_classifier'}:
                    raise ValueError('Wrong optimizer/frozen-phase flow')
                for stat in client['clipping'].values():
                    if stat['steps']!=24 or not 0<=stat['clipped_steps']<=24:
                        raise ValueError('Wrong gradient step counts')
                for key in totals:
                    totals[key]+=client[key]
        if len(result['evaluations'])!=1 or result['evaluations'][0]['round']!=300:
            raise ValueError('Test must occur only at the fixed final round')
        evaluation=result['evaluations'][0]
        if set(evaluation['domains'])!=set(DOMAINS):
            raise ValueError('Incomplete domain evaluation')
        for domain,metric in evaluation['domains'].items():
            if metric['total']!=partition['domains'][domain]['test']['count']:
                raise ValueError('Test example count differs')
            confusion=metric['confusion_matrix']
            if len(confusion)!=10 or any(len(row)!=10 for row in confusion):
                raise ValueError('Confusion shape differs')
            if any(type(value) is not int or value<0 for row in confusion for value in row):
                raise ValueError('Confusion counts must be non-negative integers')
            if sum(map(sum,confusion))!=metric['total'] or sum(confusion[i][i] for i in range(10))!=metric['correct']:
                raise ValueError('Confusion/count mismatch')
            if [sum(row) for row in confusion]!=partition['domains'][domain]['test']['labels']:
                raise ValueError('Confusion true-label counts differ from held-out split')
            if not 0<=metric['correct']<=metric['total']:
                raise ValueError('Invalid correct count')
            close(metric['accuracy_percent'],100*metric['correct']/metric['total'])
        close(evaluation['uniform_domain_accuracy_percent'],statistics.mean(m['accuracy_percent'] for m in evaluation['domains'].values()))
        close(evaluation['sample_weighted_accuracy_percent'],100*sum(m['correct'] for m in evaluation['domains'].values())/sum(m['total'] for m in evaluation['domains'].values()))
        if len(result['sessions'])!=1 or result['sessions'][0]['start_round']!=1 or result['sessions'][0]['end_round']!=300:
            raise ValueError('Expected one fresh session, no transfer/resume')
        close(result['total_session_wall_seconds'],result['sessions'][0]['wall_seconds'])
        if result['early_estimate']['round']!=5 or not result['early_estimate']['estimate_excludes_final_test']:
            raise ValueError('Initial timing estimate absent or scope changed')
        costs[str(seed)]=totals
        rounds+=len(result['history'])
    baseline=results[42]
    runtime_a=baseline['runtime'].copy(); runtime_a.pop('device')
    for result in results.values():
        if result['identity']['code']['source_sha256']!=baseline['identity']['code']['source_sha256']:
            raise ValueError('Scientific source bytes differ across seeds')
        runtime_b=result['runtime'].copy();runtime_b.pop('device')
        if runtime_a!=runtime_b:
            raise ValueError('Runtime differs beyond the assigned device')
        if result.get('costs')!=baseline.get('costs'):
            raise ValueError('Parameter/communication costs differ across seeds')
    for domain in DOMAINS:
        values=[results[seed]['evaluations'][0]['domains'][domain]['accuracy_percent'] for seed in SEEDS]
        metrics[domain]={'seeds':list(SEEDS),'values':values,'mean':statistics.mean(values),'std_ddof1':statistics.stdev(values),
                         'test_count_per_seed':partition['domains'][domain]['test']['count']}
    aggregate_metrics={}
    for name in ('uniform_domain_accuracy_percent','sample_weighted_accuracy_percent'):
        values=[results[seed]['evaluations'][0][name] for seed in SEEDS]
        aggregate_metrics[name]={'values':values,'mean':statistics.mean(values),'std_ddof1':statistics.stdev(values)}
    return {'status':'passed','rounds_verified':rounds,'domain_count_records_verified':25,
            'eval_round':300,'ddof':1,'trials':5,'domains':metrics,'aggregate_metrics':aggregate_metrics,
            'optimizer_and_sample_costs':costs,'checkpoint_deserialized':False,'model_evaluated':False,
            'common_scientific_source_sha256':baseline['identity']['code']['source_sha256'],'common_runtime':runtime_a,
            'run_code_commits':{str(seed):results[seed]['identity']['code']['commit'] for seed in SEEDS},
            'scientific_config_sha256':canon(scientific_config(config)),
            'config_sha256':canon(config),'partition_sha256':expected_hash}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def public_files(destination):
    excluded={'manifest.json','FEATURE_SHIFT_HANDOFF.zip'}
    return [path for path in sorted(destination.rglob('*')) if path.is_file()
            and path.name not in excluded and '__pycache__' not in path.parts
            and not any(part.startswith('.') for part in path.relative_to(destination).parts)]


def build(campaign_root,destination):
    pair=json.loads((campaign_root/'campaign.json').read_text())
    extension=json.loads((campaign_root/'campaign_extension.json').read_text())
    if pair['status']!='completed' or extension['status']!='completed':
        raise ValueError('Both campaign stages must be completed')
    start=datetime.fromisoformat(pair['started_utc']); end=datetime.fromisoformat(extension['ended_utc'])
    campaign={'status':'completed','runs':pair['runs']+extension['runs'],
              'started_utc':pair['started_utc'],'ended_utc':extension['ended_utc'],
              'elapsed_wall_seconds':(end-start).total_seconds(),
              'process_wall_seconds_sum':pair['process_wall_seconds_sum']+extension['process_wall_seconds_sum'],
              'maximum_training_processes':2,'maximum_per_gpu':1,'external_actions':'none'}
    if campaign['status']!='completed' or sorted(row['seed'] for row in campaign['runs'])!=list(SEEDS) or any(row['exit_code']!=0 for row in campaign['runs']):
        raise ValueError('All five processes must already have exited successfully')
    if (destination/'artifacts').exists() or (destination/'FEATURE_SHIFT_HANDOFF.zip').exists():
        raise FileExistsError('Refusing to overwrite a numeric archive or handoff')
    config=json.loads((destination/'config_five.json').read_text())
    partition=json.loads((destination/'partition_manifest.json').read_text())
    results={seed:json.loads((campaign_root/'runs'/f'seed-{seed}'/'results.json').read_text()) for seed in SEEDS}
    summary=audit(results,config,partition)
    original_config=json.loads((destination/'config.json').read_text())
    authorization=json.loads((destination/'five_run_authorization.json').read_text())
    if scientific_config(original_config)!=scientific_config(config):
        raise ValueError('Seed expansion changed scientific settings')
    if authorization['extended_config_sha256']!=canon(config) or authorization['original_config_sha256']!=canon(original_config) or authorization['scientific_config_sha256']!=summary['scientific_config_sha256']:
        raise ValueError('Expansion authorization/configuration hashes differ')
    if pair['config_sha256']!=canon(original_config) or extension['config_sha256']!=canon(config):
        raise ValueError('Original/extended campaign registry differs')
    if pair['partition_sha256']!=summary['partition_sha256'] or extension['partition_sha256']!=summary['partition_sha256']:
        raise ValueError('Supervisor/scientific partition differs')
    if extension['original_pair_sha256']!=sha(campaign_root/'campaign.json'):
        raise ValueError('Completed original campaign receipt changed')
    for stage in (pair,extension):
        if stage['maximum_training_processes']!=2 or stage['maximum_per_gpu']!=1 or stage['external_actions']!='none':
            raise ValueError('Execution policy changed')
        for row in stage['runs']:
            if summary['run_code_commits'][str(row['seed'])]!=stage['code_commit']:
                raise ValueError('Stage/worker code metadata differs')
    for name,identity in summary['common_scientific_source_sha256'].items():
        if sha(Path(name))!=identity:
            raise ValueError(f'Training source changed: {name}')
    records=sorted(campaign['runs'],key=lambda row:row['seed'])
    for i,row in enumerate(records):
        for second in records[i+1:]:
            if row['device']==second['device'] and max(datetime.fromisoformat(row['started_utc']),datetime.fromisoformat(second['started_utc']))<min(datetime.fromisoformat(row['ended_utc']),datetime.fromisoformat(second['ended_utc'])):
                raise ValueError('Overlapping workers on the same GPU')
    with (destination/'fedbn_table11.csv').open() as handle:
        references=list(csv.DictReader(handle))
    if len(references)!=15 or set(row['method'] for row in references)!={'FedBN','FedAvg','FedProx'}:
        raise ValueError('Published reference scope differs')
    artifacts=destination/'artifacts'
    artifacts.mkdir()
    archived={}
    for seed in SEEDS:
        source=campaign_root/'runs'/f'seed-{seed}'
        target=artifacts/f'seed-{seed}'
        target.mkdir()
        raw=(source/'results.json').read_bytes()
        with (target/'results.json.gz').open('wb') as handle:
            with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as compressed:
                compressed.write(raw)
        if gzip.decompress((target/'results.json.gz').read_bytes())!=raw:
            raise ValueError('Compression is not lossless')
        shutil.copyfile(source/'timings.jsonl',target/'timings.jsonl')
        timing_rows=[json.loads(line) for line in (target/'timings.jsonl').read_text().splitlines()]
        if timing_rows!=results[seed]['history']:
            raise ValueError('Timing history differs from saved results')
        archived[str(seed)]={'raw_results_sha256':sha(source/'results.json'),'timings_sha256':sha(source/'timings.jsonl'),
                             'private_checkpoint_sha256':sha(source/'checkpoint.pt'),'private_checkpoint_bytes':(source/'checkpoint.pt').stat().st_size}
    summary['result_origin']='ours'
    summary['uncertainty']='sample SD across five final run accuracies, ddof=1; fixed scientific data/config'
    summary['early_estimates']={str(seed):results[seed]['early_estimate'] for seed in SEEDS}
    summary['archived_source_files']=archived
    public_campaign={key:campaign[key] for key in ('status','started_utc','ended_utc','elapsed_wall_seconds','process_wall_seconds_sum',
                                                 'maximum_training_processes','maximum_per_gpu','external_actions')}
    public_campaign['run_code_commits']=summary['run_code_commits']
    public_campaign['scientific_config_sha256']=summary['scientific_config_sha256']
    public_campaign['partition_sha256']=summary['partition_sha256']
    public_campaign['stages']={name:{key:stage[key] for key in ('code_commit','config_sha256','started_utc','ended_utc','elapsed_wall_seconds','process_wall_seconds_sum','scheduler_sha256')}
                               for name,stage in [('original_pair',pair),('extension_three',extension)]}
    public_campaign['runs']=[{key:row[key] for key in ('seed','device','started_utc','ended_utc','exit_code','process_wall_seconds')}
                             for row in campaign['runs']]
    write(artifacts/'summary.json',summary)
    write(artifacts/'campaign_summary.json',public_campaign)
    comparison=[]
    reference_lookup={(row['method'],row['domain']):row for row in references}
    for domain in DOMAINS:
        record={'domain':domain,**{f'seed_{seed}_accuracy_percent':summary['domains'][domain]['values'][i] for i,seed in enumerate(SEEDS)},
                'ours_mean_accuracy_percent':summary['domains'][domain]['mean'],
                'ours_sample_sd_ddof1':summary['domains'][domain]['std_ddof1'],'ours_trials':5,'test_samples':summary['domains'][domain]['test_count_per_seed']}
        for method in ('FedBN','FedAvg','FedProx'):
            reference=reference_lookup[method,domain]
            record[method+'_reported_mean']=float(reference['mean_accuracy_percent'])
            record[method+'_reported_sd']=float(reference['published_sd_percent'])
            record[method+'_descriptive_delta_pp']=record['ours_mean_accuracy_percent']-float(reference['mean_accuracy_percent'])
        comparison.append(record)
    with (artifacts/'domain_comparison.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparison[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(comparison)
    rows=[]
    for record in comparison:
        rows.append([record['domain'],*[f"{record[f'seed_{seed}_accuracy_percent']:.6f}" for seed in SEEDS],
                     f"{record['ours_mean_accuracy_percent']:.6f} ± {record['ours_sample_sd_ddof1']:.6f}"])
    result_table=table(['Dominio',*[f'Seed {seed} (%)' for seed in SEEDS],'Nostri: media ± SD (%)'],rows)
    baseline_table=table(['Dominio','Nostri, n=5','FedBN pubblicato','FedAvg pubblicato','FedProx pubblicato'],[
        [record['domain'],f"{record['ours_mean_accuracy_percent']:.6f} ± {record['ours_sample_sd_ddof1']:.6f}",
         *[f"{record[method+'_reported_mean']:.2f} ± {record[method+'_reported_sd']:.2f}" for method in ('FedBN','FedAvg','FedProx')]] for record in comparison])
    times=[]
    for record in public_campaign['runs']:
        seed=record['seed'];result=results[seed]
        times.append([seed,record['device'],f"{record['process_wall_seconds']:.3f}",f"{result['total_session_wall_seconds']:.3f}",
                      f"{result['evaluations'][0]['seconds']:.3f}",f"{result['peak_cuda_allocated_mib']:.3f}",
                      f"{result['peak_cuda_reserved_mib']:.3f}",f"{result['peak_rss_mib']:.3f}"])
    time_table=table(['Seed','GPU','Processo osservato (s)','Sessione runner (s)','Test (s)','CUDA alloc. MiB','CUDA ris. MiB','RSS MiB'],times)
    estimate_table=table(['Seed','Round5: mediana round (s)','Tempo sessione a round5 (s)','Training restante stimato (s)'],[
        [seed,f"{results[seed]['early_estimate']['median_recent_round_seconds']:.6f}",
         f"{results[seed]['early_estimate']['elapsed_session_seconds']:.6f}",
         f"{results[seed]['early_estimate']['remaining_training_estimate_seconds']:.6f}"] for seed in SEEDS])
    data_table=table(['Dominio','Training','Test','Conteggi training, etichette 0–9'],[
        [domain,partition['domains'][domain]['train']['count'],partition['domains'][domain]['test']['count'],
         ', '.join(map(str,partition['domains'][domain]['train']['labels']))] for domain in DOMAINS])
    delta_table=table(['Dominio','Nostri − FedBN (pp)','Nostri − FedAvg (pp)','Nostri − FedProx (pp)'],[
        [record['domain'],*[f"{record[method+'_descriptive_delta_pp']:+.6f}" for method in ('FedBN','FedAvg','FedProx')]] for record in comparison])
    params=results[42]['costs']['parameters']
    from string import Template
    report=Template(Path(__file__).with_name('report_template.md').read_text()).substitute(
        result_table=result_table, baseline_table=baseline_table, delta_table=delta_table, data_table=data_table,
        time_table=time_table, estimate_table=estimate_table,
        total_test=sum(partition['domains'][domain]['test']['count'] for domain in DOMAINS),
        partition_hash=summary['partition_sha256'],scientific_config_hash=summary['scientific_config_sha256'],
        old_registry_hash=canon(original_config),new_registry_hash=canon(config),
        source_commit_pair=pair['code_commit'],source_commit_extension=extension['code_commit'],
        runtime=json.dumps(summary['common_runtime'],indent=2),
        uniform_mean=f"{summary['aggregate_metrics']['uniform_domain_accuracy_percent']['mean']:.6f}",
        uniform_sd=f"{summary['aggregate_metrics']['uniform_domain_accuracy_percent']['std_ddof1']:.6f}",
        weighted_mean=f"{summary['aggregate_metrics']['sample_weighted_accuracy_percent']['mean']:.6f}",
        weighted_sd=f"{summary['aggregate_metrics']['sample_weighted_accuracy_percent']['std_ddof1']:.6f}",
        wall_seconds=f"{campaign['elapsed_wall_seconds']:.3f}",wall_minutes=f"{campaign['elapsed_wall_seconds']/60:.3f}",
        start_utc=campaign['started_utc'],end_utc=campaign['ended_utc'],
        workers_seconds=f"{campaign['process_wall_seconds_sum']:.3f}",
        classifier_parameters=params['classifier'],encoder_parameters=params['private_encoder_per_client'],decoder_parameters=params['shared_decoder'],
        communication_round=results[42]['costs']['logical_communication_bytes_per_round'],
        communication_run=results[42]['costs']['logical_communication_bytes_total'])
    (destination/'FEATURE_SHIFT_REPORT.md').write_text(report)
    file_map={str(path.relative_to(destination)):{'bytes':path.stat().st_size,'sha256':sha(path)} for path in public_files(destination)}
    manifest={'schema':1,'status':'verified','result_origin':'ours','run_code_commits':summary['run_code_commits'],
              'config_sha256':summary['config_sha256'],'partition_sha256':summary['partition_sha256'],
              'seeds':list(SEEDS),'source_sha256':summary['common_scientific_source_sha256'],'files':file_map,
              'handoff_extra_files':{'fusedspacefed_core.py':{'sha256':sha('fusedspacefed_core.py'),'bytes':Path('fusedspacefed_core.py').stat().st_size}}}
    write(destination/'manifest.json',manifest)
    with zipfile.ZipFile(destination/'FEATURE_SHIFT_HANDOFF.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in public_files(destination)+[destination/'manifest.json']:
            archive.write(path,'research/feature_shift_digits/'+str(path.relative_to(destination)))
        archive.write('fusedspacefed_core.py','fusedspacefed_core.py')
    verification=verify_archive(destination)
    print(json.dumps({'status':'passed','summary':summary['domains'],'handoff':verification},indent=2))


def verify_archive(directory,payload_only=False):
    manifest=json.loads((directory/'manifest.json').read_text())
    if set(manifest['files'])!={str(path.relative_to(directory)) for path in public_files(directory)}:
        raise ValueError('Manifest file set incomplete')
    for name,item in manifest['files'].items():
        path=directory/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError(f'Archive file differs: {name}')
    results={seed:json.loads(gzip.decompress((directory/'artifacts'/f'seed-{seed}'/'results.json.gz').read_bytes())) for seed in SEEDS}
    config=json.loads((directory/'config_five.json').read_text())
    partition=json.loads((directory/'partition_manifest.json').read_text())
    expected=audit(results,config,partition)
    saved=json.loads((directory/'artifacts/summary.json').read_text())
    if any(saved.get(key)!=value for key,value in expected.items()):
        raise ValueError('Archived summary differs from reconstructed audit')
    for seed in SEEDS:
        target=directory/'artifacts'/f'seed-{seed}'
        source=saved['archived_source_files'][str(seed)]
        raw=gzip.decompress((target/'results.json.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=source['raw_results_sha256'] or sha(target/'timings.jsonl')!=source['timings_sha256']:
            raise ValueError('Raw result/timing checksum differs')
        if [json.loads(line) for line in (target/'timings.jsonl').read_text().splitlines()]!=results[seed]['history']:
            raise ValueError('Archived timing/result histories differ')
    handoff=directory/'FEATURE_SHIFT_HANDOFF.zip'
    if payload_only:
        for name,item in manifest['handoff_extra_files'].items():
            if sha(Path(name))!=item['sha256'] or Path(name).stat().st_size!=item['bytes']:
                raise ValueError('Extracted core payload differs')
        return {'status':'passed','public_payload_files':len(manifest['files']),'mode':'extracted_payload_only'}
    with zipfile.ZipFile(handoff) as archive:
        if archive.testzip() is not None:
            raise ValueError('Handoff ZIP CRC failure')
        expected_members={'research/feature_shift_digits/'+name for name in manifest['files']}|{'research/feature_shift_digits/manifest.json'}|set(manifest['handoff_extra_files'])
        if set(archive.namelist())!=expected_members:
            raise ValueError('Unexpected or missing ZIP members')
        if archive.read('research/feature_shift_digits/manifest.json')!=(directory/'manifest.json').read_bytes():
            raise ValueError('ZIP manifest differs')
        for name,item in manifest['files'].items():
            if hashlib.sha256(archive.read('research/feature_shift_digits/'+name)).hexdigest()!=item['sha256']:
                raise ValueError('ZIP payload hash differs')
        for name,item in manifest['handoff_extra_files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=item['sha256']:
                raise ValueError('ZIP extra payload differs')
    return {'status':'passed','public_payload_files':len(manifest['files']),'handoff_bytes':handoff.stat().st_size,'handoff_sha256':sha(handoff)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['build','verify'])
    parser.add_argument('--directory',type=Path,default=Path('research/feature_shift_digits'))
    parser.add_argument('--campaign',type=Path,default=Path('_local/feature_shift_digits'))
    parser.add_argument('--payload-only',action='store_true',help='Verify extracted files when the original ZIP is not beside the report')
    arguments=parser.parse_args()
    if arguments.mode=='build':
        build(arguments.campaign,arguments.directory)
    else:
        print(json.dumps(verify_archive(arguments.directory,payload_only=arguments.payload_only),indent=2))
