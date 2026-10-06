"""Numerical validation-only shortlist, extended confirmation and final freeze."""
import argparse
import json
from pathlib import Path
import sys
import statistics

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.pathmnist_calibrated.data import PUBLIC,PRIVATE,verify
from research.pathmnist_calibrated.prepare_plan import queue,job
from research.pathmnist_pathological.run import file_hash,write_json,assert_finite


def plan():return json.loads((PUBLIC/'search_plan.json').read_text())


def scores(stage,rounds):
    records=[]
    for path in sorted((PRIVATE/stage).glob(f'*/results-round-{rounds:03d}.json')):
        result=json.loads(path.read_text())
        if result['status']!='completed' or result['completed_rounds']!=rounds:raise ValueError('Incomplete run')
        assert_finite(result)
        for name,digest in result['code']['source_sha256'].items():
            if file_hash(ROOT/name)!=digest:raise ValueError('Sources changed')
        checkpoint=path.parent/f'round-{rounds:03d}.pt'
        if file_hash(checkpoint)!=result['checkpoint_sha256'][checkpoint.name]:raise ValueError('Checkpoint changed')
        point=[x for x in result['milestones'] if x['round']==rounds][-1]
        if point['split']!='training-holdout':raise ValueError('Test in calibration')
        for mode,value in point['evaluations'].items():
            metric=value['uniform_pipeline_accuracy_percent']
            if abs(metric-100*value['correct_total']/value['predictions_total'])>1e-10:raise ValueError('Count reconstruction failed')
            records.append({'candidate_id':result['configuration']['candidate_id'],'mode':mode,'seed':result['seed'],
                            'rounds':rounds,'accuracy_percent':metric,'result_path':str(path.relative_to(ROOT)),
                            'correct_total':value['correct_total'],'predictions_total':value['predictions_total'],
                            'result_sha256':file_hash(path),'checkpoint_sha256':file_hash(checkpoint)})
    return records


def require_campaign(path):
    receipt=json.loads(path.read_text())
    if receipt['status'] not in ('completed','completed_with_failed_candidates') or any(r['exit_code'] is None for r in receipt['runs']):
        raise ValueError('Training still running')
    return receipt


def shortlist():
    campaign=require_campaign(PRIVATE/'screening_campaign.json')
    rows=scores('screening',20);p=plan();partition=verify()
    best={}
    for row in rows:
        if row['candidate_id'] not in best or row['accuracy_percent']>best[row['candidate_id']]['accuracy_percent']:
            best[row['candidate_id']]=row
    ranked=sorted(best.values(),key=lambda r:(-r['accuracy_percent'],r['candidate_id']))
    chosen=[r['candidate_id'] for r in ranked[:3]]
    amendment=json.loads((PUBLIC/'execution_amendment.json').read_text()) if (PUBLIC/'execution_amendment.json').exists() else {}
    reference=amendment.get('mandatory_confirmation_reference','paper-reference')
    if reference not in chosen:chosen.append(reference)
    if (PUBLIC/'shortlist.json').exists():raise FileExistsError('Shortlist already frozen')
    write_json(PUBLIC/'shortlist.json',{'selected_candidates':chosen,'all_scores':rows,'candidate_ranking':ranked,
                                      'screening_campaign_sha256':file_hash(PRIVATE/'screening_campaign.json'),
                                      'failed_attempts':[r for r in campaign['runs'] if r['exit_code']!=0]})
    jobs=[job(c,seed,'confirmation',p['candidates'][c],partition['partition_sha256'],50) for c in chosen for seed in (142,143)]
    write_json(PUBLIC/'confirmation_queue.json',queue(jobs))
    print(json.dumps({'chosen':chosen,'ranking':ranked[:6]},indent=2))


def grouped(rows):
    groups={}
    for row in rows:groups.setdefault((row['candidate_id'],row['mode'],row['rounds']),[]).append(row)
    values=[]
    for (candidate,mode,rounds),members in groups.items():
        if sorted(x['seed'] for x in members)!=[142,143]:continue
        values.append({'candidate_id':candidate,'bn_mode':mode,'rounds':rounds,
                       'mean_validation_accuracy_percent':statistics.mean(x['accuracy_percent'] for x in members),
                       'sample_sd_ddof1':statistics.stdev(x['accuracy_percent'] for x in members),'scores':members})
    return sorted(values,key=lambda r:(-r['mean_validation_accuracy_percent'],r['rounds'],r['bn_mode']!='native',r['candidate_id']))


def extend():
    require_campaign(PRIVATE/'confirmation_campaign.json')
    rows=scores('confirmation',50);ranked=grouped(rows);chosen=[]
    for row in ranked:
        if row['candidate_id'] not in chosen:chosen.append(row['candidate_id'])
        if len(chosen)==2:break
    if len(chosen)!=2:raise ValueError('Not enough completed confirmations')
    if (PUBLIC/'extension.json').exists():raise FileExistsError('Extension already frozen')
    write_json(PUBLIC/'extension.json',{'selected_candidates':chosen,'ranking_round50':ranked})
    jobs=[];p=plan();partition=verify()
    for c in chosen:
        for seed in (142,143):
            j=job(c,seed,'confirmation',p['candidates'][c],partition['partition_sha256'],100)
            j['args'].append('--resume');jobs.append(j)
    write_json(PUBLIC/'extension_queue.json',queue(jobs))
    print(json.dumps({'extend':chosen,'ranking':ranked[:4]},indent=2))


def freeze():
    require_campaign(PRIVATE/'extension_campaign.json')
    rows=scores('confirmation',50)+scores('confirmation',100)
    ranked=grouped(rows)
    if not ranked:raise ValueError('No valid configuration')
    if (PUBLIC/'selection.json').exists():raise FileExistsError('Configuration already frozen')
    selected=ranked[0]
    value={**selected,'selected_candidate_id':selected['candidate_id'],
           'training_settings':plan()['candidates'][selected['candidate_id']],
           'final_seeds':[41,42,43,44,45],'ranking':ranked,
           'no_new_test_access_before_selection':True,'partition_sha256':verify()['partition_sha256'],
           'rule':plan()['selection_rule'],'plan_sha256':file_hash(PUBLIC/'search_plan.json')}
    write_json(PUBLIC/'selection.json',value)
    jobs=[]
    for seed in value['final_seeds']:
        c={'stage':'final','candidate_id':value['selected_candidate_id'],'settings':value['training_settings'],
           'seed':seed,'rounds':value['rounds'],'bn_mode':value['bn_mode'],
           'partition_sha256':value['partition_sha256'],'selection_sha256':file_hash(PUBLIC/'selection.json')}
        path=PUBLIC/'configs/final'/f'seed-{seed}.json';path.parent.mkdir(exist_ok=True);write_json(path,c)
        jobs.append({'name':f'final-seed-{seed}','script':'research/pathmnist_calibrated/runner.py',
                     'args':['--config',str(path.relative_to(ROOT))],
                     'output':f'_local/pathmnist_calibrated/final/seed-{seed}'})
    write_json(PUBLIC/'final_queue.json',queue(jobs))
    print(json.dumps({k:value[k] for k in ('selected_candidate_id','bn_mode','rounds','mean_validation_accuracy_percent','sample_sd_ddof1','training_settings')},indent=2))


def reduced_ranking(rows,candidate_order):
    order={name:i for i,name in enumerate(candidate_order)}
    return sorted(rows,key=lambda r:(-r['correct_total'],r['mode']!='native',order[r['candidate_id']]))


def freeze_screening():
    original=require_campaign(PRIVATE/'screening_campaign.json')
    retry=require_campaign(PRIVATE/'screening_retry_campaign.json')
    budget=json.loads((PUBLIC/'budget_reduction.json').read_text())
    if budget['final_seeds']!=[42] or budget['final_rounds']!=50:raise ValueError('Changed reduced budget')
    rows=scores('screening',20);ranked=reduced_ranking(rows,plan()['candidates'])
    if not ranked:raise ValueError('No completed finite candidate')
    if (PUBLIC/'selection.json').exists():raise FileExistsError('Selection already frozen')
    chosen=ranked[0]
    value={'selected_candidate_id':chosen['candidate_id'],'bn_mode':chosen['mode'],'rounds':50,
           'screening_seed':142,'selection_round':20,'validation_accuracy_percent':chosen['accuracy_percent'],
           'correct_total':chosen['correct_total'],'predictions_total':chosen['predictions_total'],
           'training_settings':plan()['candidates'][chosen['candidate_id']],'final_seeds':[42],
           'ranking':ranked,'no_new_test_access_before_selection':True,
           'partition_sha256':verify()['partition_sha256'],
           'budget_reduction_sha256':file_hash(PUBLIC/'budget_reduction.json'),
           'original_search_plan_sha256':file_hash(PUBLIC/'search_plan.json'),
           'original_campaign_sha256':file_hash(PRIVATE/'screening_campaign.json'),
           'retry_campaign_sha256':file_hash(PRIVATE/'screening_retry_campaign.json'),
           'rule':'greatest screening validation accuracy at seed142 round20; exact correct-count ties prefer native BN then original candidate insertion order',
           'confirmations':False,'extension':False,
           'failed_screening':[r for r in original['runs']+retry['runs'] if r['exit_code'] and (ROOT/r['output']).exists()]}
    write_json(PUBLIC/'selection.json',value)
    c={'stage':'final','candidate_id':value['selected_candidate_id'],'settings':value['training_settings'],
       'seed':42,'rounds':50,'bn_mode':value['bn_mode'],'partition_sha256':value['partition_sha256'],
       'selection_sha256':file_hash(PUBLIC/'selection.json')}
    path=PUBLIC/'configs/final/seed-42.json';path.parent.mkdir(parents=True,exist_ok=True);write_json(path,c)
    j={'name':'final-seed-42','script':'research/pathmnist_calibrated/runner.py',
       'args':['--config',str(path.relative_to(ROOT))],'output':'_local/pathmnist_calibrated/final/seed-42'}
    q=queue([j]);q['gpu_slots']={'cuda:1':1,'cuda:0':0};write_json(PUBLIC/'final_queue.json',q)
    print(json.dumps({k:value[k] for k in ('selected_candidate_id','bn_mode','screening_seed','selection_round','rounds','validation_accuracy_percent','training_settings')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('shortlist','extend','freeze','freeze-screening'))
    args=parser.parse_args()
    if (PUBLIC/'budget_reduction.json').exists() and args.stage!='freeze-screening':
        raise SystemExit('Confirmations/extensions superseded by user budget reduction; use freeze-screening')
    {'shortlist':shortlist,'extend':extend,'freeze':freeze,'freeze-screening':freeze_screening}[args.stage]()
