# PathMNIST: refit della sola testa sul checkpoint originale

**Accuracy 39.619777% → 75.331476%: +35.711699 punti percentuali.** Seed 42, round 50 della run originale con iperparametri del paper. Metrica invariata: media uniforme di tutte le 10 pipeline sullo stesso test ufficiale di 7.180 immagini ciascuna. Prima 28.447/71.800, dopo 54.088/71.800 predizioni; il prima riproduce esattamente tutti i conteggi originali archiviati. Un singolo seed, nessuna SD fra seed.

## Intervento fissato prima dell’esecuzione

Origine: `_local/pathmnist_pathological/seed-42/final.pt`, SHA256 `fa56da01091debe9031a9acb2e3de0f34ce0f751d2c95d2d0d54c05f15697d1e`, codice training originale `4218d49f75269ca017cc3128e49c7df10d440220`. SGD 0,01/Adam 0,001, warm-up 1/CE 3, batch 128, 50 round, full participation e aggregazione uniforme; optimizer persistenti, FP16 AMP della run originale. Si riutilizza il terminale senza ulteriori round o nuovi pesi tuned.

**Fusione `x+D(E_i(x))` e tutte le BN originali invariati**, senza ricalibrazione o riduzione del decoder. Restano identici encoder privati, decoder, tutti i pesi del corpo classificatore e ogni buffer BN. Cambiano soltanto 585 parametri di `fc.weight` e `fc.bias` (64→9), sincronizzati nelle dieci copie client.

Feature estratte in eval/no_grad Float32 da tutte le 89.996 immagini training, ciascuna con l’encoder proprietario, senza backward sulla rappresentazione. Stessa funzione `head_calibration.refit` dell’esperimento precedente: CE media uniforme delle 10 loss locali full-batch; λ=0; L-BFGS LR 1, max_iter=100, history 20, strong-Wolfe, tolerance_grad1e-7, tolerance_change1e-10. Nessuna validation, sweep, selezione checkpoint o modifica dopo i test.

Effettive 100 iterazioni, 105 valutazioni obiettivo/gradiente. CE training 1.783756 → 0.597145. Checkpoint finale salvato prima delle due valutazioni test prima/dopo; l’accuracy non ha guidato il refit.

## Conteggi per pipeline

| Encoder | Corrette prima | Corrette dopo | Totale | Prima(%) | Dopo(%) |
|---:|---:|---:|---:|---:|---:|
| 0 | 2659 | 5437 | 7180 | 37.033426 | 75.724234 |
| 1 | 2725 | 5427 | 7180 | 37.952646 | 75.584958 |
| 2 | 2717 | 5432 | 7180 | 37.841226 | 75.654596 |
| 3 | 3066 | 5379 | 7180 | 42.701950 | 74.916435 |
| 4 | 3070 | 5384 | 7180 | 42.757660 | 74.986072 |
| 5 | 2943 | 5409 | 7180 | 40.988858 | 75.334262 |
| 6 | 3069 | 5381 | 7180 | 42.743733 | 74.944290 |
| 7 | 2862 | 5447 | 7180 | 39.860724 | 75.863510 |
| 8 | 2688 | 5415 | 7180 | 37.437326 | 75.417827 |
| 9 | 2648 | 5377 | 7180 | 36.880223 | 74.888579 |

Le 71.800 predizioni riutilizzano gli stessi 7.180 esempi con 10 encoder, non sono osservazioni indipendenti o un ensemble di logits. Tutte le pipeline sono incluse.

## Costo e checkpoint

GPU cuda:1. Feature 2.300s, refit 0.997s, due test 3.409s; sessione 9.341s, processo incluse import/init 15.005s, controller 16.039s. Training originale riutilizzato: 948.916s, distinto dal costo della nuova fase.

Picchi Torch allocato/riservato 1233.718/1294.000MiB; RSS 1764.484MiB. Nessun processo altrui interrotto, nessun pacchetto installato.

Checkpoint su thanos in `/mnt/data/codex/FusedSpaceFed/_local/pathmnist_head_only/seed-42/`: `training-initial.pt` (originale round 0), `before.pt` (originale round 50) e **`final.pt`** (round 50 +refit). Quest’ultimo conserva classifier/decoder, tutti 10 encoder e copie client, buffer BN, 20 stati optimizer originali, scaler, generatori loader, worker/coordinator RNG, indici e configurazione/codice originali; aggiunge optimizer L-BFGS completo e RNG/configurazione della nuova fase sotto `head_only_refit`. Si usa la normale inferenza a guadagno 1.

SHA256 finale: `31019b0e1ef8594de1f7da196a6ce4b7e84afdc58fb4ba3d42fcf14b7fd9d4b8`. Source/config hashes e hash/dimensioni degli altri checkpoint in artifacts/verification.json. Tutti gli stati caricati e finiti; controlli bit-identici passati per tutto eccetto la testa. Suite versionata: **339 test passati in 86.96s**, inclusi 8 nuovi test dei vincoli head-only.

Comando eseguito, exit 0 (stdout/stderr e log completi locali; receipt pubblicato):

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_head_only/run.py --config research/pathmnist_head_only/config.json --output _local/pathmnist_head_only/seed-42 --device cuda:1
```

Configurazione/sorgenti congelati nel commit `87b2b7759e1b435cacefec9cd2f2d81f7090c735`; risultati e report in un commit separato. Artefatti numerici lossless in `artifacts/`, checkpoint/dati/log completi in `_local/`. Per verificare l’archivio: `/home/schroeder/miniconda3/envs/general_ml/bin/python research/pathmnist_head_only/archive.py verify`.

## Interpretazione e limiti

A questo stato fisso, il refit della sola testa basta a migliorare l’accuracy: non sono necessarie ricalibrazione BN, attenuazione della fusione o pesi tuned per osservare questo esito. Il confronto isola l’intervento sulla testa nel seed 42; non ne stima la variabilità, non identifica la causa dei pesi terminali subottimali e non dimostra un vantaggio causale dell’encoder/fusione rispetto alle baseline. Non è una nuova media di cinque seed o replica del valore 50,94% del paper.

Il refit resta una fase supervised aggiuntiva rispetto al paper. La feature cache nel simulatore realizza un obiettivo equivalente alla media di gradienti locali full-batch; comunicazione/privacy della realizzazione distribuita non sono misurate. Test già noto, due valutazioni fissate prima/dopo, nessun altro tuning o esperimento. Manoscritto e risultati precedenti invariati; lavoro concluso.
