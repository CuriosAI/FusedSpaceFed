from research.pathmnist_calibrated.selection_tools import reduced_ranking


def test_screening_exact_count_ties_use_native_then_original_order():
    rows=[{'candidate_id':'second','mode':'native','correct_total':4000,'accuracy_percent':44.5},
          {'candidate_id':'first','mode':'train-recalibrated','correct_total':4000,'accuracy_percent':44.50000000000001},
          {'candidate_id':'first','mode':'native','correct_total':4000,'accuracy_percent':44.49999999999999}]
    r=reduced_ranking(rows,['first','second'])
    assert [(x['candidate_id'],x['mode']) for x in r]==[('first','native'),('second','native'),('first','train-recalibrated')]


def test_greatest_validation_count_wins_regardless_of_bn_preference():
    rows=[{'candidate_id':'first','mode':'native','correct_total':4000},
          {'candidate_id':'second','mode':'train-recalibrated','correct_total':4001}]
    assert reduced_ranking(rows,['first','second'])[0]['candidate_id']=='second'
