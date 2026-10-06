"""Verify and archive the single, fixed head-only control; no image evaluation."""
import argparse
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from research.pathmnist_pathological.run import file_hash,write_json,assert_finite
from research.pathmnist_calibrated.archive import compressed,manifest
from research.pathmnist_head_only.run import check_complete,verify_counts,ORIGINAL_CHECKPOINT_SHA256

PUBLIC=ROOT/'research/pathmnist_head_only'
PRIVATE=ROOT/'_local/pathmnist_head_only'


def verify_archives():
    out=PUBLIC/'artifacts';m=json.loads((out/'manifest.json').read_text())
    files={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert files==set(m['files_sha256'])
    for name,digest in m['files_sha256'].items():assert file_hash(out/name)==digest
    print('Head-only archive hashes verified')


def archive():
    directory=PRIVATE/'seed-42';r=json.loads((directory/'results.json').read_text());assert_finite(r)
    c=json.loads((PRIVATE/'campaign.json').read_text())
    assert c['status']=='completed' and len(c['runs'])==1 and c['runs'][0]['exit_code']==0
    assert r['status']=='completed' and r['seed']==42 and r['source_round']==50 and r['test_evaluations']==2
    assert file_hash(directory/'before.pt')==ORIGINAL_CHECKPOINT_SHA256
    for name,digest in r['checkpoint_sha256'].items():assert file_hash(directory/name)==digest
    for name,digest in c['frozen_files'].items():assert file_hash(ROOT/name)==digest
    assert r['configuration']==json.loads((PUBLIC/'config.json').read_text())
    for name in ('metrics_before','metrics_after'):verify_counts(r[name])
    assert abs(r['delta_pp']-(r['metrics_after']['uniform_pipeline_accuracy_percent']-
                              r['metrics_before']['uniform_pipeline_accuracy_percent']))<1e-10
    original=torch.load(directory/'before.pt',map_location='cpu',weights_only=False)
    final=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
    check_complete(original,final)
    head_state=next(iter(final['head_only_refit']['head_optimizer']['state'].values()))
    assert head_state['n_iter']==r['head_statistics']['actual_iterations']<=100
    assert head_state['func_evals']==r['head_statistics']['function_evaluations']
    assert final['head_only_refit']['configuration']==r['configuration']
    initial=torch.load(directory/'training-initial.pt',map_location='cpu',weights_only=False)
    assert initial['seed']==42 and initial['round']==0
    assert r['checkpoint_sha256']['training-initial.pt']=='2e068afbd9caee88b7f41d7032f054f4918b28dea46b73d17f8701faabcc4a40'
    changed=subprocess.check_output(['git','diff','--name-only','2a3f69c05c315548cad71746a7b800613aa9ea5b'],cwd=ROOT,text=True).splitlines()
    assert all(name.startswith('research/pathmnist_head_only/') for name in changed)
    tests_log=PRIVATE/'logs/repository_tests.log'
    match=re.search(r'(\d+) passed in ([\d.]+)s',tests_log.read_text());assert match
    tests={'passed':int(match[1]),'seconds':float(match[2]),'command':
           '/home/schroeder/miniconda3/envs/general_ml/bin/python -m pytest -q tests research',
           'log_sha256':file_hash(tests_log)}
    out=PUBLIC/'artifacts';out.mkdir(parents=True,exist_ok=False)
    compressed(directory/'results.json',out/'results.json.gz')
    shutil.copyfile(PRIVATE/'campaign.json',out/'campaign.json')
    shutil.copyfile(tests_log,out/'tests.log')
    audit={'status':'passed','original_checkpoint_hash_matches':True,
           'only_shared_fc_weight_and_bias_changed':True,'all_57_original_BN_buffers_bit_identical':True,
           'all_ten_private_encoders_shared_decoder_optimizer_scaler_rng_and_partition_preserved':True,
           'all_client_classifiers_synchronized':True,'complete_final_checkpoint_loaded':True,
           'before_counts_match_original_archive':r['verification']['original_before_test_counts_reproduced'],
           'metrics_reconstructed_from_counts':True,'private_checkpoints':{
                str((directory/name).relative_to(ROOT)):{'sha256':digest,'bytes':(directory/name).stat().st_size}
                for name,digest in r['checkpoint_sha256'].items()},
           'tests':tests,'no_training_or_test_image_evaluation_in_archive':True}
    write_json(out/'verification.json',audit)
    summary={'seed':42,'source_round':50,'before_percent':r['metrics_before']['uniform_pipeline_accuracy_percent'],
             'after_percent':r['metrics_after']['uniform_pipeline_accuracy_percent'],'delta_pp':r['delta_pp'],
             'configuration':r['configuration'],'code':r['code'],'tests':tests,
             'session_seconds':r['session_seconds'],'head_refit_seconds':r['head_refit_seconds'],
             'checkpoint':str((directory/'final.pt').relative_to(ROOT)),
             'checkpoint_sha256':r['checkpoint_sha256']['final.pt'],'sample_sd_between_seeds':None}
    write_json(out/'summary.json',summary)
    before=r['metrics_before']['uniform_pipeline_accuracy_percent'];after=r['metrics_after']['uniform_pipeline_accuracy_percent']
    lines=['# PathMNIST: refit della sola testa sul checkpoint originale','',
      f'**Accuracy {before:.6f}% → {after:.6f}%: {r["delta_pp"]:+.6f} punti percentuali.** Seed 42, round 50 della run originale con iperparametri del paper. Metrica invariata: media uniforme di tutte le 10 pipeline sullo stesso test ufficiale di 7.180 immagini ciascuna. Prima 28.447/71.800, dopo 54.088/71.800 predizioni; il prima riproduce esattamente tutti i conteggi originali archiviati. Un singolo seed, nessuna SD fra seed.','',
      '## Intervento fissato prima dell’esecuzione','',
      f"Origine: `_local/pathmnist_pathological/seed-42/final.pt`, SHA256 `{ORIGINAL_CHECKPOINT_SHA256}`, codice training originale `{original['code']['base_commit']}`. SGD 0,01/Adam 0,001, warm-up 1/CE 3, batch 128, 50 round, full participation e aggregazione uniforme; optimizer persistenti, FP16 AMP della run originale. Si riutilizza il terminale senza ulteriori round o nuovi pesi tuned.",'',
      '**Fusione `x+D(E_i(x))` e tutte le BN originali invariati**, senza ricalibrazione o riduzione del decoder. Restano identici encoder privati, decoder, tutti i pesi del corpo classificatore e ogni buffer BN. Cambiano soltanto 585 parametri di `fc.weight` e `fc.bias` (64→9), sincronizzati nelle dieci copie client.','',
      'Feature estratte in eval/no_grad Float32 da tutte le 89.996 immagini training, ciascuna con l’encoder proprietario, senza backward sulla rappresentazione. Stessa funzione `head_calibration.refit` dell’esperimento precedente: CE media uniforme delle 10 loss locali full-batch; λ=0; L-BFGS LR 1, max_iter=100, history 20, strong-Wolfe, tolerance_grad1e-7, tolerance_change1e-10. Nessuna validation, sweep, selezione checkpoint o modifica dopo i test.','',
      f"Effettive {r['head_statistics']['actual_iterations']} iterazioni, {r['head_statistics']['function_evaluations']} valutazioni obiettivo/gradiente. CE training {r['head_statistics']['objective_history'][0]:.6f} → {r['head_statistics']['final_uniform_client_training_ce']:.6f}. Checkpoint finale salvato prima delle due valutazioni test prima/dopo; l’accuracy non ha guidato il refit.",'',
      '## Conteggi per pipeline','',
      '| Encoder | Corrette prima | Corrette dopo | Totale | Prima(%) | Dopo(%) |','|---:|---:|---:|---:|---:|---:|']
    for b,a in zip(r['metrics_before']['pipeline_metrics'],r['metrics_after']['pipeline_metrics']):
        assert b['client_id']==a['client_id']
        lines.append(f"| {b['client_id']} | {b['correct']} | {a['correct']} | {b['total']} | {b['accuracy_percent']:.6f} | {a['accuracy_percent']:.6f} |")
    lines+=['','Le 71.800 predizioni riutilizzano gli stessi 7.180 esempi con 10 encoder, non sono osservazioni indipendenti o un ensemble di logits. Tutte le pipeline sono incluse.','',
      '## Costo e checkpoint','',
      f"GPU cuda:1. Feature {r['feature_extraction_seconds']:.3f}s, refit {r['head_refit_seconds']:.3f}s, due test {r['test_seconds']:.3f}s; sessione {r['session_seconds']:.3f}s, processo incluse import/init {c['runs'][0]['process_wall_seconds']:.3f}s, controller {c['wall_seconds']:.3f}s. Training originale riutilizzato: {r['original_training_seconds']:.3f}s, distinto dal costo della nuova fase.",'',
      f"Picchi Torch allocato/riservato {r['peak_cuda_allocated_bytes']/2**20:.3f}/{r['peak_cuda_reserved_bytes']/2**20:.3f}MiB; RSS {r['peak_rss_kib']/1024:.3f}MiB. Nessun processo altrui interrotto, nessun pacchetto installato.",'',
      'Checkpoint su thanos in `/mnt/data/codex/FusedSpaceFed/_local/pathmnist_head_only/seed-42/`: `training-initial.pt` (originale round 0), `before.pt` (originale round 50) e **`final.pt`** (round 50 +refit). Quest’ultimo conserva classifier/decoder, tutti 10 encoder e copie client, buffer BN, 20 stati optimizer originali, scaler, generatori loader, worker/coordinator RNG, indici e configurazione/codice originali; aggiunge optimizer L-BFGS completo e RNG/configurazione della nuova fase sotto `head_only_refit`. Si usa la normale inferenza a guadagno 1.','',
      f"SHA256 finale: `{r['checkpoint_sha256']['final.pt']}`. Source/config hashes e hash/dimensioni degli altri checkpoint in artifacts/verification.json. Tutti gli stati caricati e finiti; controlli bit-identici passati per tutto eccetto la testa. Suite versionata: **{tests['passed']} test passati in {tests['seconds']:.2f}s**, inclusi 8 nuovi test dei vincoli head-only.",'',
      'Comando eseguito, exit 0 (stdout/stderr e log completi locali; receipt pubblicato):','',
      '```bash',*[' '.join(c['runs'][0]['command'])],'```','',
      f"Configurazione/sorgenti congelati nel commit `{r['code']['base_commit']}`; risultati e report in un commit separato. Artefatti numerici lossless in `artifacts/`, checkpoint/dati/log completi in `_local/`. Per verificare l’archivio: `/home/schroeder/miniconda3/envs/general_ml/bin/python research/pathmnist_head_only/archive.py verify`.",'',
      '## Interpretazione e limiti','',
      'A questo stato fisso, il refit della sola testa basta a migliorare l’accuracy: non sono necessarie ricalibrazione BN, attenuazione della fusione o pesi tuned per osservare questo esito. Il confronto isola l’intervento sulla testa nel seed 42; non ne stima la variabilità, non identifica la causa dei pesi terminali subottimali e non dimostra un vantaggio causale dell’encoder/fusione rispetto alle baseline. Non è una nuova media di cinque seed o replica del valore 50,94% del paper.','',
      'Il refit resta una fase supervised aggiuntiva rispetto al paper. La feature cache nel simulatore realizza un obiettivo equivalente alla media di gradienti locali full-batch; comunicazione/privacy della realizzazione distribuita non sono misurate. Test già noto, due valutazioni fissate prima/dopo, nessun altro tuning o esperimento. Manoscritto e risultati precedenti invariati; lavoro concluso.']
    text='\n'.join(lines)+'\n';(PUBLIC/'HEAD_ONLY_REPORT.md').write_text(text)
    (PRIVATE/'report.md').write_text(text);(ROOT/'_local/report_pathmnist_head_only.md').write_text(text)
    manifest(out);verify_archives()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('archive','verify'))
    a=p.parse_args();{'archive':archive,'verify':verify_archives}[a.mode]()
