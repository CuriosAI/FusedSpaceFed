# FusedSpaceFed Digits, calibrazione dedicata

Il piano e i criteri sono in [PROTOCOL.md](PROTOCOL.md) e `search_plan.json`.
Tutti i nuovi tentativi e checkpoint sono conservati separatamente in
`_local/feature_shift_digits_calibrated/`; il pilota e il controllo precedente
restano immutati. Questa directory raccoglie codice, configurazioni, verifiche
e, a conclusione, risultati numerici e rapporto.

Python usato: `/home/schroeder/miniconda3/envs/general_ml/bin/python`.
Non occorre installare dipendenze. Dalla radice del repository:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/controller.py \
  --queue research/feature_shift_digits_calibrated/screening_queue.json \
  --receipt _local/feature_shift_digits_calibrated/screening_campaign.json \
  --logs _local/feature_shift_digits_calibrated/logs/screening

/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/select_stage.py --stage confirmation
```

Dopo commit delle configurazioni di conferma, lo stesso controller usa
`confirmation_queue.json`, receipt `confirmation_campaign.json` e log nella
directory `logs/confirmation`. Poi `select_stage.py --stage final` congela una
sola configurazione e registra le cinque run da zero. Dopo commit, il controller
usa `final_queue.json`, receipt `final_campaign.json` e `logs/final`.

Il controller avvia al massimo due processi, uno per GPU, salva codice d'uscita
e tempi reali. Non crea le directory delle run prima del runner, non sovrascrive
artefatti e non interrompe processi esterni. La ripresa è esplicita, direttamente
con `runner.py --resume` e gli stessi argomenti/configurazione/device/commit.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/audit.py \
  --run _local/feature_shift_digits_calibrated/final/ID-CONFIGURAZIONE-seed-42
```

L'audit usa conteggi e confusion matrix già salvati: non rivaluta il test.
I riferimenti pubblicati rimangono nel CSV originale del pilota.
