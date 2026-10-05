"""Pure synthetic five-run records for the independent final verifier."""
import copy
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('digits_independent_audit',Path(__file__).resolve().parents[1]/'audit_and_package.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    config=json.loads(Path('research/feature_shift_digits/config_five.json').read_text())
    partition={'domains':{domain:{'test':{'count':100,'labels':[10]*10},
                                 'train':{'count':743,'labels':[74]*9+[77]}} for domain in audit.DOMAINS},'kind':'synthetic_unit_only'}
    partition['partition_sha256']=audit.canon(partition)
    config['partition_sha256']=partition['partition_sha256']
    clients={domain:{'warmup_steps':24,'classification_steps':24,'encoder_steps':48,'decoder_steps':24,'classifier_steps':24,
                     'warmup_samples':743,'classification_samples':743,'loss':0.1,
                     'clipping':{name:{'steps':24,'clipped_steps':0,'sum_norm_before_clip':24.0,'max_norm_before_clip':1.0}
                                 for name in ('warmup_autoencoder','classification_autoencoder','classification_classifier')}}
             for domain in audit.DOMAINS}
    history=[{'round':r,'clients':copy.deepcopy(clients),'seconds':1.0} for r in range(1,301)]
    records={}
    for seed in audit.SEEDS:
        registered=copy.deepcopy(config)
        if seed in (42,43):
            registered['run_seeds']=[42,43];registered['per_seed_device']={'42':'cuda:1','43':'cuda:0'}
        diagonal=seed-36
        matrix=[[0]*10 for _ in range(10)]
        for label in range(10):
            matrix[label][label]=diagonal;matrix[label][(label+1)%10]=10-diagonal
        domains={domain:{'correct':diagonal*10,'total':100,'accuracy_percent':diagonal*10.0,
                         'confusion_matrix':copy.deepcopy(matrix),'loss':0.2} for domain in audit.DOMAINS}
        records[seed]={'status':'completed','completed_rounds':300,'configuration':registered,'history':copy.deepcopy(history),
                       'identity':{'seed':seed,'device':audit.DEVICES[seed],'method':'FusedSpaceFed','config_sha256':audit.canon(registered),
                                   'partition_sha256':partition['partition_sha256'],
                                   'code':{'commit':'original' if seed<44 else 'registry_only',
                                           'source_sha256':{'unchanged_scientific_source.py':'unit_hash'}}},
                       'runtime':{'device':audit.DEVICES[seed],'backend':'synthetic_only'},
                       'evaluations':[{'round':300,'domains':domains,'uniform_domain_accuracy_percent':diagonal*10.0,
                                       'sample_weighted_accuracy_percent':diagonal*10.0}],
                       'sessions':[{'start_round':1,'end_round':300,'wall_seconds':10.0}],
                       'total_session_wall_seconds':10.0,
                       'early_estimate':{'round':5,'estimate_excludes_final_test':True}}
    return records,config,partition


class IndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.results,self.config,self.partition=fixture()

    def test_five_values_mean_and_sample_sd_with_registry_only_change(self):
        result=audit.audit(self.results,self.config,self.partition)
        self.assertEqual(result['rounds_verified'],1500)
        self.assertEqual(result['trials'],5)
        self.assertEqual(result['domain_count_records_verified'],25)
        for metric in result['domains'].values():
            self.assertEqual(metric['seeds'],list(audit.SEEDS))
            self.assertEqual(metric['values'],[60.0,70.0,80.0,90.0,100.0])
            self.assertEqual(metric['mean'],80.0)
            self.assertAlmostEqual(metric['std_ddof1'],math.sqrt(250.0))
        self.assertEqual(result['optimizer_and_sample_costs']['42']['warmup_steps'],36000)
        self.assertEqual(result['optimizer_and_sample_costs']['42']['encoder_steps'],72000)

    def rejected(self,mutation):
        mutation()
        with self.assertRaises(ValueError):
            audit.audit(self.results,self.config,self.partition)

    def test_missing_seed_rejected(self):
        self.rejected(lambda:self.results.pop(46))

    def test_nonfinite_training_rejected(self):
        self.rejected(lambda:self.results[44]['history'][0]['clients']['MNIST'].__setitem__('loss',float('nan')))

    def test_changed_scientific_config_rejected(self):
        self.rejected(lambda:self.results[44]['configuration']['training'].__setitem__('classifier_lr',0.02))

    def test_changed_scientific_source_rejected(self):
        self.rejected(lambda:self.results[45]['identity']['code']['source_sha256'].__setitem__('unchanged_scientific_source.py','modified'))

    def test_wrong_device_rejected(self):
        self.rejected(lambda:self.results[46]['identity'].__setitem__('device','cuda:0'))

    def test_wrong_step_counts_rejected(self):
        self.rejected(lambda:self.results[46]['history'][200]['clients']['SVHN'].__setitem__('classification_steps',25))

    def test_missing_participation_rejected(self):
        self.rejected(lambda:self.results[42]['history'][1]['clients'].pop('USPS'))

    def test_wrong_round_sequence_rejected(self):
        self.rejected(lambda:self.results[42]['history'][3].__setitem__('round',5))

    def test_wrong_evaluation_round_rejected(self):
        self.rejected(lambda:self.results[42]['evaluations'][0].__setitem__('round',299))

    def test_additional_test_rejected(self):
        self.rejected(lambda:self.results[43]['evaluations'].append(copy.deepcopy(self.results[43]['evaluations'][0])))

    def test_wrong_test_count_rejected(self):
        self.rejected(lambda:self.results[46]['evaluations'][0]['domains']['MNIST'].__setitem__('total',99))

    def test_wrong_confusion_diagonal_rejected(self):
        self.rejected(lambda:self.results[46]['evaluations'][0]['domains']['MNIST']['confusion_matrix'][0].__setitem__(0,9))

    def test_wrong_domain_accuracy_rejected(self):
        self.rejected(lambda:self.results[43]['evaluations'][0]['domains']['MNIST'].__setitem__('accuracy_percent',71.0))

    def test_wrong_weighted_aggregate_rejected(self):
        self.rejected(lambda:self.results[43]['evaluations'][0].__setitem__('sample_weighted_accuracy_percent',70.1))

    def test_resumed_definitive_rejected(self):
        self.rejected(lambda:self.results[43]['sessions'].append({'start_round':100,'end_round':300,'wall_seconds':1.0}))

    def test_complete_synthetic_archive_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); destination=root/'public'; campaign=root/'private'
            destination.mkdir();campaign.mkdir()
            source_names=('fusedspacefed_core.py','research/feature_shift_digits/model.py',
                          'research/feature_shift_digits/data.py','research/feature_shift_digits/run_digits.py')
            for result in self.results.values():
                result['identity']['code']['source_sha256']={name:audit.sha(name) for name in source_names}
                result['costs']={'parameters':{'classifier':14219210,'private_encoder_per_client':72080,'shared_decoder':44995},
                                 'logical_communication_bytes_per_round':570793880,'logical_communication_bytes_total':171238164000}
                result.update(peak_cuda_allocated_mib=1.0,peak_cuda_reserved_mib=2.0,peak_rss_mib=3.0)
                result['evaluations'][0]['seconds']=1.0
                result['early_estimate'].update(median_recent_round_seconds=1.0,elapsed_session_seconds=5.0,
                                                remaining_training_estimate_seconds=295.0)
            original=copy.deepcopy(self.config)
            original.update(run_seeds=[42,43],per_seed_device={'42':'cuda:1','43':'cuda:0'})
            authorization={'original_config_sha256':audit.canon(original),'extended_config_sha256':audit.canon(self.config),
                           'scientific_config_sha256':audit.canon(audit.scientific_config(self.config))}
            for name,value in [('config.json',original),('config_five.json',self.config),('partition_manifest.json',self.partition),
                               ('five_run_authorization.json',authorization)]:
                audit.write(destination/name,value)
            shutil.copyfile('research/feature_shift_digits/fedbn_table11.csv',destination/'fedbn_table11.csv')
            records=[]
            for seed,result in self.results.items():
                folder=campaign/'runs'/f'seed-{seed}';folder.mkdir(parents=True)
                audit.write(folder/'results.json',result)
                (folder/'timings.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in result['history']))
                (folder/'checkpoint.pt').write_bytes(b'synthetic checkpoint; never deserialized')
                offset=0 if seed<44 else 10 if seed<46 else 20
                records.append({'seed':seed,'device':audit.DEVICES[seed],'exit_code':0,'process_wall_seconds':10.0,
                                'started_utc':f'2026-01-01T00:00:{offset:02d}+00:00',
                                'ended_utc':f'2026-01-01T00:00:{offset+10:02d}+00:00'})
            common={'status':'completed','maximum_training_processes':2,'maximum_per_gpu':1,'external_actions':'none',
                    'partition_sha256':self.partition['partition_sha256'],'scheduler_sha256':'synthetic_only'}
            pair={**common,'runs':records[:2],'code_commit':'original','config_sha256':audit.canon(original),
                  'started_utc':records[0]['started_utc'],'ended_utc':records[0]['ended_utc'],
                  'elapsed_wall_seconds':10.0,'process_wall_seconds_sum':20.0}
            audit.write(campaign/'campaign.json',pair)
            extension={**common,'runs':records[2:],'code_commit':'registry_only','config_sha256':audit.canon(self.config),
                       'started_utc':records[2]['started_utc'],'ended_utc':records[-1]['ended_utc'],
                       'elapsed_wall_seconds':20.0,'process_wall_seconds_sum':30.0,
                       'original_pair_sha256':audit.sha(campaign/'campaign.json')}
            audit.write(campaign/'campaign_extension.json',extension)
            with contextlib.redirect_stdout(io.StringIO()):
                audit.build(campaign,destination)
            self.assertEqual(audit.verify_archive(destination)['status'],'passed')
            self.assertEqual(audit.verify_archive(destination,payload_only=True)['mode'],'extracted_payload_only')
            with self.assertRaises(FileExistsError):
                audit.build(campaign,destination)
            (destination/'artifacts/summary.json').write_text('{}\n')
            with self.assertRaises(ValueError):
                audit.verify_archive(destination)


if __name__=='__main__':
    unittest.main()
