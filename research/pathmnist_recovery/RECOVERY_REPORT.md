# PathMNIST patologico: recupero esplorativo concluso

**Seed 42, round 50: 75.233983% sul test**, +24.293983 punti percentuali rispetto al 50,94% medio riportato nella [Tabella 4 del paper](../../paper/aistats_2027.tex). Ricerca fermata al primo superamento, il 6 ottobre 2026. È una singola run riutilizzata con una fase aggiuntiva: **nessuna SD fra seed**, nessuna replica certificata della media pubblicata. Manoscritto, altri esperimenti e tutti i tentativi precedenti restano invariati.

La soluzione mantiene encoder privati persistenti, decoder e classificatore condivisi, fusione additiva e le due fasi originali di training. Aggiunge una **variante esplicita**: guadagno positivo 0,1 nella fusione, ricalibrazione dei buffer BN su training e ottimizzazione del solo ultimo strato condiviso usando feature training fisse. Non va descritta come il FusedSpaceFed del paper a iperparametri invariati.

## Dati, partizione e metrica

Sorgente [MedMNIST 3.0.2 PathMNIST-64](https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1). Cache già verificata, PIL Resize 32×32 seguito da ToTensor, RGB Float32 senza augmentation per la configurazione finale. Training 89.996, test ufficiale 7.180. Dieci client patologici, due classi/client; ogni immagine training appartiene a un solo client, stessa partizione dati seed 42. Gli indici originali sono conservati nei checkpoint e nel manifesto congelato. Non vengono aggiunti dati o usata la validation ufficiale.

- SHA256 sorgente NPZ: `1e7fc200dd5aac79f39f0da26178be801ffac8e6d59c8533e51acebae953745a`.
- SHA256 partizione originale: `b70fb78ad84a7d7041c5d98938d208f8b23455c3254b8b949e32c58509828a80`.
- SHA256 manifesto holdout: `13d5cc86d039a8bf75b97ee6f6f94c6f177f52dd10829c66fd01150db7ceeb47`.
- Seed holdout: 20261006; 81.002 fit + 8.994 validation, disgiunti globalmente, holdout stratificato per client/classe. BN: 640 immagini fit/client, 6.400 totali, nessuna label.

| Client | Classi | Training completo | Fit | Validation |
|---:|---|---:|---:|---:|
| 0 | [0, 3] | 6589 | 5931 | 658 |
| 1 | [2, 7] | 9881 | 8893 | 988 |
| 2 | [4, 6] | 7946 | 7152 | 794 |
| 3 | [1, 5] | 10846 | 9762 | 1084 |
| 4 | [3, 8] | 9910 | 8920 | 990 |
| 5 | [0, 7] | 7822 | 7040 | 782 |
| 6 | [2, 4] | 9183 | 8265 | 918 |
| 7 | [1, 6] | 8697 | 7828 | 869 |
| 8 | [5, 8] | 12533 | 11280 | 1253 |
| 9 | [0, 3] | 6589 | 5931 | 658 |

Metrica primaria: ciascuno dei dieci encoder privati, con lo stesso decoder/classificatore condiviso, è valutato su **tutto** il test o holdout pooled. Si riporta la media uniforme delle dieci accuratezze, non local two-class accuracy, migliore encoder, ensemble di logits o migliore seed. Le 71.800 predizioni test riusano 7.180 immagini dieci volte: non sono 71.800 osservazioni indipendenti.

## Configurazione finale e fase aggiuntiva

La base completa seed 42 è stata allenata da zero su tutto il training nella precedente campagna, con C inizializzato seed 42 ed AE seed 1.000.042. Si riutilizza il suo terminale **nativo round 50**, senza ripresa da smoke o checkpoint intermedi. Impostazioni identiche al modello fit-only selezionato seed 142, round 50:

```json
{
  "classifier_lr": 0.1,
  "autoencoder_lr": 0.0001,
  "warmup_lr": 0.0001,
  "gradient_clip_norm": 5.0,
  "precision": "bf16",
  "schedule": "constant",
  "schedule_horizon": 100,
  "batch_size": 128,
  "warmup_epochs": 1,
  "local_epochs": 3,
  "aggregation": "uniform",
  "optimizer_reset": false,
  "classifier": "ResNet20V2",
  "autoencoder": "UNetSmallAE",
  "dz": 16,
  "classifier_optimizer": "SGD(momentum=0,weight_decay=0)",
  "autoencoder_optimizer": "Adam(betas=(0.9,0.999),eps=1e-8,weight_decay=0)"
}
```

Partecipano tutti i 10 client a ogni round. Warm-up: una epoca MSE encoder-only; classificazione: tre epoche CE, encoder/decoder/classificatore trainabili. SGD e Adam sono persistenti, senza reset; aggregazione uniforme solo degli stati condivisi. BF16 durante training senza GradScaler; ricalibrazione, feature, testa e test Float32. Schedule constant: il nominale orizzonte 100 non cambia i LR nei 50 round effettivi.

Rispetto agli iperparametri del [paper](../../paper/aistats_2027.tex), restano architettura, dati, 50 round, warm-up 1, CE 3 e batch 128; LR classifier cambia 0,01→0,1 e LR AE/warm-up 0,001→0,0001. Si aggiunge clipping norm 5. Il paper non specifica la precisione; la precedente implementazione di riferimento usava FP16 AMP, la base selezionata BF16. Si aggiungono inoltre le tre modifiche post-training descritte sotto: fusione attenuata, buffer BN ricalibrati e ottimizzazione della testa.

Dopo i 50 round, senza modificare encoder/decoder o pesi del corpo ResNet:

1. Usare `x + 0.1 D(E_i(x))`; α=0 è stato escluso dalle soluzioni. La convoluzione finale lineare del decoder consente una riscalatura esatta in copia, senza nuovi parametri.
2. Reset e ricalibrazione delle 19 BN condivise sulla miscela fit fissata, 50 batch 128, shuffle 20261006, cumulative averaging. Ogni immagine usa il suo encoder proprietario. Solo buffer, senza label/gradienti/test.
3. Congelare C body, BN, encoder e decoder; estrarre feature 64 da **tutti** gli 89.996 training attraverso l’encoder del client d’origine. Ottimizzare la testa condivisa 64→9, iniziando dai suoi pesi terminali, con L-BFGS LR 1, history 20, strong-Wolfe, max_iter=100, tolerance_grad1e-7 e tolerance_change1e-10. Penalità L2 selezionata λ=0; bias non penalizzato. Obiettivo: media uniforme delle 10 CE locali full-batch, poi eventuale 0,5λ||W_fc||².
4. Salvare il checkpoint completo della variante prima di caricare/evaluare il test. Un’unica valutazione della configurazione congelata; nessun test-adaptation, selezione checkpoint o riapertura dopo questo esito.

Sono state effettuate 100 iterazioni effettive della testa, 105 valutazioni dell’obiettivo/gradiente. CE uniforme training: 1.668712 → 0.457858. Queste misure provengono dal training e non hanno determinato early stopping sul test.

Architettura: ResNet20-v2 272,217 parametri, encoder 23,600, decoder 38,851; **334,668 parametri attivi/client**, 547,068 persistenti fra tutti i 10 client. La testa ha 585 parametri già inclusi nel ResNet, nessuna nuova capacità. La fase aggiuntiva cambia computazione e algoritmo: non si rivendica costo equivalente al protocollo originale.

La media di loss/gradienti della testa equivale matematicamente a una media di contributi locali full-batch. Il simulatore cachea centralmente feature prodotte dagli encoder proprietari. Non è una dimostrazione di privacy, né un’implementazione del traffico di rete distribuito; una realizzazione federata avrebbe aggregazioni/line-search aggiuntive oltre ai 50 round delle due fasi.

Motivazione: sulle feature fissate la CE della testa lineare è un problema convesso, quindi un refit condiviso economico verifica se il punto terminale ottenuto con aggiornamenti locali è subottimale. L-BFGS evita un nuovo sweep di learning rate della rete intera; il budget fisso e le penalità sono dichiarati prima della validation. Il guadagno positivo attenua il decoder preservando il percorso privato, e il suo valore è selezionato su validation. Nessuna conclusione di causalità sui componenti viene ricavata dal solo test finale.

## Selezione e tutti i tentativi

Il primo screening storico: 24 profili, 20 round, seed 142; il congelamento precedente selezionava c0.1/ae0.0001 ma ha prodotto 35,279944% test. Conservato integralmente in `../pathmnist_calibrated/`, senza riscriverne i report. La successiva autorizzazione ha riaperto la ricerca e revocato la deadline. La nuova ricerca ha proceduto per probabilità qualitativa/costo stimato: BN economiche, prolungamento dei checkpoint fit a 50, GN, guadagni positivi, infine refit della testa. Nessuna baseline è rieseguita.

| Tentativo test seed 42 | Variante | Accuratezza (%) | Sessione aggiuntiva(s) |
|---:|---|---:|---:|
| 1 | Originale, owner-cumulative BN da fit | 43.938719 | 3.510 |
| 2 | Originale, cross-layerwise BN da fit | 44.299443 | 4.923 |
| 3 | c0.1/ae0.0001, native diagnostica | 25.023677 | 3.131 |
| 4 | Originale, α=0,25 +BN da fit | 43.168524 | 3.544 |
| 5 | c0.1/ae0.0001, α=0,1 +BN da fit +refit fc | 75.233983 | 6.473 |

Il tentativo 3 era un controllo diagnostico dell’inferenza nativa, **non** la modalità favorita dalla validation; questa eccezione è dichiarata prima del test nel piano. Tentativi 1/2/4 trasferivano selezioni FP32 fit seed 142 a pesi FP16 completi seed 42; limite esplicito. Il tentativo 5 usa le stesse impostazioni di training/precisione/orizzonte tra fit e full.

Ricalibrazione BN: cinque modalità su sei checkpoint fit. Quattro erano i profili promettenti, più c0.01/ae0.0003 e il riferimento FP16 finito al round 18, fallito al 19: il suo controllo non è un risultato round 20. Tutti i 30 valori sono negli archivi normalization-validation. Quattro checkpoint sono ripresi esattamente 20→50 con optimizer, generatori e RNG; nessun nuovo seed o best-round.

| Fit seed 142, round 50 | Native (%) | BN da fit (%) |
|---|---:|---:|
| fp32-matched-reference | 27.069157 | 40.182344 |
| grid-c0.03-ae0.001 | 37.873026 | 48.475650 |
| grid-c0.1-ae0.0001 | 33.431176 | 44.456304 |
| grid-c0.1-ae0.001 | 40.942851 | 45.390260 |

Due profili GN da zero, 50 round, seed 142:

| Profilo | Validation nativa (%) | Sessione(s) |
|---|---:|---:|
| gn8-c0.03-ae0.0003-e3 | 33.476762 | 1838.379 |
| gn8-c0.1-ae0.0003-e1-flip90 | 29.177229 | 938.903 |

GN sostituisce 19 BN con GN8, stessi parametri affini; il profilo CE1 usa flip/rot90. Esiti negativi su validation; nessun test di questi profili. La ricerca dei guadagni 0/0,1/0,25/0,5/1/2 su quattro checkpoint fit a 50 include 42 modalità; α=0 è soltanto diagnostico. Il massimo positivo senza refit era 52,066934% sul riferimento FP32, α=0,25 con BN da fit.

Refit fc: **sei profili in parallelo 3+3 GPU**, cinque penalità 0/1e-4/1e-3/1e-2/0,1 ciascuno. Risultati completi, inclusi controlli identici senza refit:

| Profilo | Prima (%) | λ0 | λ1e-4 | λ1e-3 | λ1e-2 | λ0,1 |
|---|---:|---:|---:|---:|---:|---:|
| best-bn-g0.5 | 48.702468 | 72.211474 | 72.901935 | 72.913053 | 71.457638 | 69.175006 |
| best-bn-g1 | 48.475650 | 67.009117 | 67.566155 | 68.513453 | 66.825662 | 64.482989 |
| previous-g0.1 | 47.585057 | 80.205693 | 80.070047 | 78.593507 | 75.440294 | 71.558817 |
| reference-g0.1 | 51.392039 | 72.379364 | 72.460529 | 71.356460 | 69.063820 | 64.471870 |
| reference-g0.25 | 52.066934 | 66.846787 | 66.671114 | 66.406493 | 65.136758 | 61.974650 |
| reference-g1 | 40.182344 | 49.161663 | 49.306204 | 48.098732 | 46.187458 | 43.133200 |

Selezione congelata su un solo seed 142, risultato terminale 50 round: **80.205693%**,72.137/89.940. Si confrontano tutti i 30 refit e sei controlli; parità: controllo senza refit, ordine dei profili, poi penalità. Ranking integrale: `head_selection.json`; hash selezione `2b6c342de8f50a33d451dd0bbbd02fb94571a58b51fb7e4ad45d8d8065c88a3f`. Scelte λ/guadagno/profilo soltanto da validation; nessun seed finale alternativo. L’uso ripetuto dello stesso holdout può sovradattare il tuning.

## Risultato completo e interpretazione

| Encoder/pipeline | Corrette | Test immagini | Accuratezza (%) |
|---:|---:|---:|---:|
| 0 | 5395 | 7180 | 75.139276 |
| 1 | 5426 | 7180 | 75.571031 |
| 2 | 5426 | 7180 | 75.571031 |
| 3 | 5361 | 7180 | 74.665738 |
| 4 | 5401 | 7180 | 75.222841 |
| 5 | 5412 | 7180 | 75.376045 |
| 6 | 5416 | 7180 | 75.431755 |
| 7 | 5410 | 7180 | 75.348189 |
| 8 | 5381 | 7180 | 74.944290 |
| 9 | 5390 | 7180 | 75.069638 |

Totale 54.018/71.800 → **75.233983%**, identico alla media uniforme. Un singolo seed 42: nessuna SD fra seed inventata e nessuna sostituzione con la pipeline migliore. La differenza +24.293983pp dalla Tabella 4 è descrittiva; il 50,94% del paper è una media di cinque run senza incertezza in tabella e con partizioni/seed originali non completamente recuperati. Non è una dimostrazione di superiorità statistica o una replica esatta.

La validation controllata dello stesso checkpoint c0.1/ae0.0001, α=0,1 e stessa BN passa da 47,585057 a 80,205693% cambiando soltanto la testa. Questo documenta che i pesi dell’ultima testa terminale erano subottimali per quelle feature/obiettivo. Non dimostra quale causa di training li abbia prodotti, né il beneficio causale dell’encoder, warm-up o fusione rispetto a un controllo ablation. Non si dichiarano nuove misure Γ o vantaggi di costo sulle baseline.

Il test era già noto e cinque tentativi sono stati valutati nel recupero. Lo stop al primo superamento della soglia è adattivo: il risultato finale resta **esplorativo**. Le scelte puntuali della fase vincente sono congelate su validation prima del relativo test, ma la campagna nel suo insieme non è una conferma indipendente o test mai osservato. Nessuna ulteriore calibrazione, run, baseline, ablation o diagnostica parte dopo il successo.

## Tempi, memoria e impiego GPU

| Fase | Controller calendario(s) | Somma worker(s) | Processi |
|---|---:|---:|---:|
| attempt-01 | 10.038 | 9.003 | 1 |
| normalization-validation | 32.208 | 166.550 | 6 |
| attempt-02 | 11.041 | 10.003 | 1 |
| continuation-validation | 1169.467 | 4319.408 | 4 |
| groupnorm-validation | 1845.383 | 2788.521 | 2 |
| attempt-03 | 9.039 | 8.003 | 1 |
| fusion-validation | 40.143 | 130.235 | 4 |
| attempt-04 | 10.038 | 9.004 | 1 |
| head-validation | 42.230 | 229.644 | 6 |
| attempt-05 | 13.036 | 12.004 | 1 |

Unione degli intervalli delle campagne: 2763.747s, senza doppio conteggio delle sovrapposizioni. Somma controller 3182.623s; somma worker concorrenti 7682.375s. Quest’ultima non misura il tempo fisico di calcolo GPU né FLOPs. Dal primo lancio al successo: 5525.185s calendario, inclusi sviluppo, attese e interruzioni fra campagne. Lo screening precedente e i 20 round già eseguiti dei checkpoint ripresi sono costi storici aggiuntivi, non rimisurati come nuovi training.

Run completa riutilizzata: 1804.215s training/salvataggi, 1808.775s sessione, versione `3b9af9af88b74e50a2871eb6b1b96b18a245b424`. Non è stata ripresa: stati terminali compatibili riutilizzati per la fase aggiuntiva. 35.400 warm-up steps encoder-only + 106.200 joint steps per optimizer, Adam totale 141.600 e SGD 106.200; BF16 senza step saltati dal GradScaler.

Fase finale vincente: BN 0.521s, feature 1.887s, refit 0.968s, test 1.696s; sessione 6.473s. Processo incluse import/init: 12.004s; controller 13.036s. Sommare 1804 s storici al solo refit non equivale a misurare una nuova run end-to-end; i contributi sono riportati separatamente.

Picco Torch finale allocato/riservato: 1232.905/1408.000MiB; RSS 1767.047MiB. Training base: 1261.251/1450.000MiB. Picchi per worker delle altre fasi nei JSON. Non si sommano picchi non contemporanei come un picco GPU totale.

Host thanos.gasl.unich.it, due RTX 6000 Ada 48 GB, ambiente general_ml senza installazioni. Validation BN 6 worker 3+3, continuazioni 4 worker 2+2 e GN indipendenti, fusion probe 4 worker 2+2, refit 6 worker 3+3. GPU entrambe al 99% nella breve campagna refit; circa 5,5–6 GB usati/GPU in quel controllo. Processo esterno tesi_giovanni PID 4146 su GPU 0 lasciato intatto; nessun processo altrui interrotto. Test congelati sequenziali per poter fermare la ricerca alla soglia. Nessun nostro processo training resta attivo.

## Checkpoint, verifiche e riproduzione

Su thanos: `/mnt/data/codex/FusedSpaceFed/_local/pathmnist_recovery/attempt-05/`.

| File | Round | SHA256 |
|---|---:|---|
| initial.pt | 0 | `ba103a227f6f4b65c09ef31e301e6e545e4950371c85f9273442263bd8969e72` |
| precalibration.pt | 50 | `9b0b95d636f1b667c506dc358c65064d4eef71803ba493788ad9f5d382b7d9eb` |
| final.pt | 50 | `a9d055734f1272a2039bf3a545c480ec1a391b24de7df9c910ce3bd066528bcb` |

Checkpoint completi: C, D, tutti 10 encoder, copie client, 57 buffer BN, 20 dizionari degli optimizer originali, generatori loader e RNG CPU/Python/NumPy/CUDA, partizione/indici esatti, configurazione, codice e storia 50 round. SGD senza momentum ha stato vuoto previsto, BF16 non richiede scaler. `final.pt` aggiunge optimizer L-BFGS completo, RNG della fase e configurazione effettiva. Audit: tutti gli stati finiti e caricabili; D, encoder, C body, optimizer/RNG originali bit-identici alla base; C fc e BN sincronizzati in tutte le copie client.

**Attenzione operativa:** top-level `decoder` conserva i pesi allenati originali. Per l’inferenza effettiva usare `inference_decoder(checkpoint)` o moltiplicare D per `recovery.config.fusion_gain`. Non usare direttamente la vecchia valutazione α=1 sul nuovo checkpoint. `precalibration.pt` è lo stato originale completo prima della fase aggiuntiva; `initial.pt` è l’inizializzazione round0, non una nuova inizializzazione post-hoc.

331 test versionati passati (`pytest -q tests research`), inclusi 18 test di recupero: normalizzazione, gain esatto, warm-up/CE gradient flow, persistenza, ripresa GN+augmentation, obiettivo per client, refit head e stati completi. `verification.py` ricostruisce metriche da conteggi e verifica dati/config/partizioni/checkpoint senza training o altre valutazioni immagini.

Una raccolta indiscriminata `pytest -q` includeva test privati ignorati in `_local/`: 413 passati, 1 fallito perché un vecchio controllo queue richiede interprete senza import Torch/NumPy, incompatibile con la raccolta della suite scientifica. Non è stato modificato codice scientifico per quel vincolo: tutti 38 test privati queue passano in un processo unittest isolato. Esiti/log/hash in quality_checks.json; log dettagliato fallito conservato localmente.

Comando effettivamente eseguito, con stdout/stderr e exit 0 nel receipt:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_recovery/head_final.py \
  --config research/pathmnist_recovery/attempt-05.json \
  --output _local/pathmnist_recovery/attempt-05 --device cuda:1
```

Il runner rifiuta sovrascritture; per una futura replica occorre una nuova directory, senza crearla prima. Tutte le sorgenti/queue del comando sono congelate con hash nel receipt, codice esecuzione `704f285bf41ac7ca5a5bbbd07f8beb52b331d654`. Per verificare e rigenerare soltanto il report numerico:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/pathmnist_recovery/archive.py verify
/home/schroeder/miniconda3/envs/general_ml/bin/python research/pathmnist_recovery/report.py
```

Il report generator usa solo JSON pubblicati e cronologia Git; non legge checkpoint o immagini e non esegue esperimenti. Gli artefatti numerici gzip sono lossless, tutti i tentativi negativi inclusi. Manifesti SHA e provenance elencano percorsi/hash/dimensioni dei checkpoint privati. Log e checkpoint completi restano in `_local/pathmnist_recovery/`; base riutilizzata in `_local/pathmnist_calibrated/final/seed-42/`. Nessuna immagine, credenziale, review privata o checkpoint è versionato.

## Commit e limiti aperti

Base precedente conservata: `4bbc24ba12a98c498197bfe3d3187027f49395a3`. Commit del recupero fino al congelamento finale, tutti push su main:

- `912fabd101284431ec302cee01b4d137c21ec53d` — experiment: begin validation-led PathMNIST recovery
- `8d045d0ea7c350b59b573457966a60704b0c5ab6` — experiment: compare training-only BN calibration variants
- `b23f961e6cae86fbd618a7565432e218b451dede` — experiment: freeze next PathMNIST inference candidate on validation
- `cef54869721bf3b1b38295c1305a3cfc06cdf25a` — experiment: resume PathMNIST fit candidates to final horizon
- `da4bbc4b60e062b72e5b2ca7ef3cbf7d75d1c505` — experiment: add GroupNorm PathMNIST recovery profiles
- `7c50c0bb3ae4f8f80278ad4e5a25fb20ba7a39c9` — experiment: verify native PathMNIST inference and augmented resume
- `efd547273e43262e8861dede6b2b85157b9f6d2c` — experiment: validate decoder gains for PathMNIST recovery
- `e2190f723c0883ec7d02d3ee56cb0547fa06e310` — experiment: freeze positive PathMNIST fusion gain on validation
- `9e6c0bc87e10c63be45bae367982cd4579057139` — experiment: validate shared PathMNIST head calibration
- `704f285bf41ac7ca5a5bbbd07f8beb52b331d654` — experiment: freeze validated PathMNIST shared-head recovery

Il commit che aggiunge questo report e artifacts/attempt-05 archivia l’esito finale; identificabile con `git log -1 -- research/pathmnist_recovery/RECOVERY_REPORT.md`. La copia locale del report registra anche hash/esito del push finale dopo la creazione del commit. Nessun reset/rebase/forcepush.

Limiti aperti: un solo seed di selezione e finale; holdout ripetutamente consultato; test noto con stop adattivo; differenze di LR/clipping/precisione/fusione/BN e fase aggiuntiva rispetto al paper; comunicazione e privacy della fase head non implementate; nessun confronto controllato con baseline o isolamento causale dei componenti; partizioni/seed storici del 50,94% non interamente recuperati. Nessuna nuova SD, significatività o equivalenza di costo è affermata. Ablation e diagnostiche rimangono sospese; la ricerca termina al successo, senza modifica del manoscritto.
