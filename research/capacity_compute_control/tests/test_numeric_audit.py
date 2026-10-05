"""Stdlib-only known-count tests for independent reporting, without models."""
import copy
import importlib.util
from pathlib import Path
import statistics
import unittest

spec=importlib.util.spec_from_file_location('control_report_audit',Path(__file__).resolve().parents[1]/'audit_results.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class NumericAuditTests(unittest.TestCase):
    def setUp(self):
        self.counts={domain:(i+1)*10 for i,domain in enumerate(audit.DOMAINS)}
        domains={}
        for domain,total in self.counts.items():
            matrix=[[0]*10 for _ in range(10)];matrix[0][0]=5;matrix[0][1]=total-5
            domains[domain]={'total':total,'correct':5,'accuracy_percent':500/total,'confusion_matrix':matrix}
        self.evaluation={'domains':domains,'uniform_domain_accuracy_percent':statistics.mean(m['accuracy_percent'] for m in domains.values()),
                         'sample_weighted_accuracy_percent':100*25/150}

    def test_unequal_domains_distinguish_uniform_and_sample_weighted(self):
        audit.check_metrics(self.evaluation,self.counts)
        self.assertNotEqual(self.evaluation['uniform_domain_accuracy_percent'],self.evaluation['sample_weighted_accuracy_percent'])
        self.evaluation['sample_weighted_accuracy_percent']=self.evaluation['uniform_domain_accuracy_percent']
        with self.assertRaises(ValueError): audit.check_metrics(self.evaluation,self.counts)

    def test_accuracy_not_supported_by_counts_rejected(self):
        self.evaluation['domains']['MNIST']['accuracy_percent']+=1
        with self.assertRaises(ValueError): audit.check_metrics(self.evaluation,self.counts)

    def test_negative_confusion_counts_rejected(self):
        self.evaluation['domains']['MNIST']['confusion_matrix'][1][1]=-1
        with self.assertRaises(ValueError): audit.check_metrics(self.evaluation,self.counts)

    def test_sample_sd_known_values(self):
        self.assertEqual(audit.stats([1,3,5]),{'values':[1,3,5],'mean':3,'sample_sd_ddof1':2.0})

    def test_pairs_preserve_all_seeds_including_losing_seed(self):
        records={method:{} for method in audit.METHODS}
        for method,values in [('FusedSpaceFed',[10,20,30]),('FedAvg',[9,22,25])]:
            for seed,value in zip(audit.SEEDS,values):
                records[method][seed]={'evaluations':[{'uniform_domain_accuracy_percent':value,'sample_weighted_accuracy_percent':value,
                    'domains':{d:{'accuracy_percent':value} for d in audit.DOMAINS}}]}
        result=audit.summary(records,{})
        difference=result['paired_fused_minus_fedavg']['uniform_domain_accuracy_percent']
        self.assertEqual(difference['values'],[1,-2,5]);self.assertAlmostEqual(difference['mean'],4/3)
        self.assertEqual(result['seeds'],[42,43,44])


if __name__=='__main__': unittest.main()
