"""Freeze per-method LR using only the six predetermined validation runs."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from research.feature_shift_digits.data import DOMAINS,atomic_json,canonical_hash,file_hash


def select(directory):
    destination=directory/'selection.json'
    if destination.exists():
        raise FileExistsError('Selection is already frozen')
    queue=json.loads((directory/'validation_queue.json').read_text())
    plan=json.loads((directory/'search_plan.json').read_text());scores=[]
    receipt=json.loads(Path('_local/capacity_compute_control/validation_campaign.json').read_text())
    if receipt['status']!='completed' or len(receipt['runs'])!=6 or any(row['exit_code']!=0 for row in receipt['runs']):
        raise ValueError('All six calibration workers must have exited successfully')
    for job in queue['jobs']:
        path=Path(job['output'])/'results.json';result=json.loads(path.read_text())
        config=json.loads(Path(job['config']).read_text())
        if result['configuration']!=config or result['identity']['config_sha256']!=canonical_hash(config):
            raise ValueError('Validation run identity differs')
        if result['status']!='completed' or result['completed_rounds']!=60 or config['seed']!=142 or config['phase']!='validation':
            raise ValueError('Wrong validation run protocol')
        if len(result['evaluations'])!=1 or result['evaluations'][0]['round']!=60 or result['evaluations'][0]['split']!='validation':
            raise ValueError('Wrong validation evaluation')
        evaluation=result['evaluations'][0]
        if set(evaluation['domains'])!=set(DOMAINS):
            raise ValueError('Missing validation domain')
        values=[]
        for value in evaluation['domains'].values():
            if value['total']!=149 or sum(map(sum,value['confusion_matrix']))!=149 or sum(value['confusion_matrix'][i][i] for i in range(10))!=value['correct']:
                raise ValueError('Wrong validation counts')
            accuracy=100*value['correct']/149
            if not math.isclose(accuracy,value['accuracy_percent'],abs_tol=1e-10):
                raise ValueError('Validation accuracy differs from counts')
            values.append(accuracy)
        score=statistics.mean(values)
        if not math.isfinite(score) or not math.isclose(score,evaluation['uniform_domain_accuracy_percent'],abs_tol=1e-10):
            raise ValueError('Invalid validation mean')
        scores.append({'method':config['method'],'classifier_lr':config['training']['classifier_lr'],
                       'uniform_validation_accuracy_percent':score,'result_sha256':file_hash(path),
                       'configuration_sha256':canonical_hash(config),'directory':job['output']})
    chosen={}
    for method in ('FusedSpaceFed','FedAvg'):
        rows=[row for row in scores if row['method']==method]
        if sorted(row['classifier_lr'] for row in rows)!=plan['validation']['candidates_classifier_lr']:
            raise ValueError('Candidate set differs from search plan')
        chosen[method]=min(rows,key=lambda row:(-row['uniform_validation_accuracy_percent'],row['classifier_lr']))['classifier_lr']
    selection={'rule':'maximum fixed round60 uniform-domain validation accuracy; ties lower LR',
               'scores':scores,'selected_classifier_lr':chosen,'validation_seed':142,
               'search_plan_sha256':canonical_hash(plan),'validation_campaign_sha256':file_hash(Path('_local/capacity_compute_control/validation_campaign.json')),
               'no_test_access_for_selection':True,'final_seeds':[42,43,44]}
    atomic_json(destination,selection)
    print(json.dumps(selection,indent=2))
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=Path('research/capacity_compute_control'))
    select(parser.parse_args().directory)
