# FusedSpaceFed Digits: calibrazione e cinque nuove run

Campagna completata: dieci screening, conferme a 300 round, una configurazione congelata e cinque nuove inizializzazioni sul training completo. Il pilota resta conservato. Nessuna baseline è stata rieseguita e il manoscritto non è stato modificato.

La media uniforme migliora dal 81.2670% del pilota al 85.8624% (+4.5954 punti). Tutti i domini migliorano rispetto al pilota. Le medie dei cinque nuovi seed restano inferiori a tutte le medie delle baseline pubblicate; su SVHN il divario da FedAvg è 2.7624 punti, con variabilità fra seed 2.8769. I risultati positivi di singoli seed non sostituiscono questa conclusione sulle medie.

## Risultati finali e competitività

Accuratezze percentuali al solo round 300. Media e deviazione standard campionaria (ddof=1) comprendono tutti i seed 42–46.

| Dominio | 42 | 43 | 44 | 45 | 46 | Media ± SD | FedAvg pubblicato | FedProx pubblicato | FedBN pubblicato |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 96.7929 | 96.9929 | 96.9143 | 97.3286 | 96.7071 | 96.9471 ± 0.2398 | 97.38 ± 0.05 | 97.30 ± 0.17 | 97.55 ± 0.11 |
| SVHN | 63.6318 | 69.1157 | 66.2806 | 70.9286 | 69.1812 | 67.8276 ± 2.8769 | 70.59 ± 0.51 | 71.55 ± 0.75 | 76.93 ± 0.25 |
| USPS | 96.6667 | 96.8817 | 96.1828 | 96.8280 | 96.7742 | 96.6667 ± 0.2819 | 96.91 ± 0.11 | 96.98 ± 0.19 | 97.69 ± 0.10 |
| SynthDigits | 85.9762 | 86.8567 | 85.6930 | 86.7513 | 86.3034 | 86.3161 ± 0.4964 | 86.66 ± 0.21 | 86.60 ± 0.18 | 87.46 ± 0.20 |
| MNIST-M | 80.2786 | 81.6714 | 81.5429 | 81.9571 | 82.3214 | 81.5543 ± 0.7733 | 82.44 ± 0.41 | 82.67 ± 0.75 | 83.57 ± 0.38 |

| Dominio | Δ FedAvg (pp) | Δ FedProx (pp) | Δ FedBN (pp) |
|---|---:|---:|---:|
| MNIST | -0.4329 | -0.3529 | -0.6029 |
| SVHN | -2.7624 | -3.7224 | -9.1024 |
| USPS | -0.2433 | -0.3133 | -1.0233 |
| SynthDigits | -0.3439 | -0.2839 | -1.1439 |
| MNIST-M | -0.8857 | -1.1157 | -2.0157 |

Rispetto alle medie pubblicate, i domini con media Fused almeno pari sono: FedAvg: nessuno; FedProx: nessuno; FedBN: nessuno. Questo è un confronto descrittivo, non una dimostrazione di significatività o una replica esatta.

Baseline trascritte e verificate nel CSV originale: [appendice FedBN, tabella 11, pagina PDF9/numero stampato21](https://michaelkamp.org/wp-content/uploads/2021/05/FedBN_appendix.pdf). Le loro SD sono pubblicate con ddof non specificato. Non si sono sostituite medie con i seed migliori.

| Dominio | Pilota conservato, media ± SD (%) | Δ nuovo−pilota (pp) |
|---|---:|---:|
| MNIST | 96.3229 ± 0.2133 | +0.6243 |
| SVHN | 58.8428 ± 1.9565 | +8.9848 |
| USPS | 95.6022 ± 0.2905 | +1.0645 |
| SynthDigits | 81.6599 ± 0.4789 | +4.6563 |
| MNIST-M | 73.9071 ± 1.0023 | +7.6471 |

Il confronto con il pilota usa gli stessi cinque seed, dati e protocollo. È riportato soltanto a campagna conclusa; non entra nel selettore della configurazione.

- Media uniforme fra domini: 85.8624 ± 0.8469%; seed: 84.6692, 86.3037, 85.3227, 86.7587, 86.2575.
- Media pesata per immagini test: 84.5147 ± 0.7453%; seed: 83.5888, 85.0646, 83.8830, 85.2972, 84.7399.

La tabella FedBN riporta ogni dominio. La media uniforme è una nostra sintesi; la media pesata è dominata da SynthDigits (97.791 dei 147.509 test) e non sostituisce i risultati per dominio.

## Calibrazione e congelamento

Piano registrato prima delle nuove run, SHA256 `ba564b57b239052ec1676827ac1c4ee43952449414eef63a5c19e26948319afc`. Partizione training-validation congelata: `3dfc26c3fda17f266fb2ffa9899f23b4a5b50cd006742bb84d5b215e412a6943`; 594 fit e 149 validation per ciascuno dei cinque domini. Nessuna immagine test usata per il ranking.

La ricerca precedente a 60 round favoriva LR/clip al bordo superiore della griglia. Sono stati registrati dieci screening a 120 round (seed142): L9 su LR CNN {0.02,0.05,0.1}, LR AE {0.0001,0.0003,0.001}, clip {2,5,10}, più il pilota. Le due migliori configurazioni, più entrambi i riferimenti deduplicati, sono state confermate da zero a 300 round sui seed142 e143. Il ranking finale usa la media delle due validation al round 300; parità per LR CNN/clip/LR AE crescenti. Nessuna scelta del checkpoint.

| Fase | Configurazione | Seed | LR CNN | LR AE | Clip | Validation uniforme (%) |
|---|---|---:|---:|---:|---:|---:|
| screening | pilot-reference | 142 | 0.01 | 0.0003 | 1.0 | 77.0470 |
| screening | expanded-0-0 | 142 | 0.02 | 0.0001 | 10.0 | 80.6711 |
| screening | previous-selected-reference | 142 | 0.02 | 0.0003 | 2.0 | 79.0604 |
| screening | expanded-0-2 | 142 | 0.02 | 0.001 | 5.0 | 81.6107 |
| screening | expanded-1-0 | 142 | 0.05 | 0.0001 | 2.0 | 81.2081 |
| screening | expanded-1-1 | 142 | 0.05 | 0.0003 | 5.0 | 81.7450 |
| screening | expanded-1-2 | 142 | 0.05 | 0.001 | 10.0 | 84.2953 |
| screening | expanded-2-0 | 142 | 0.1 | 0.0001 | 5.0 | 83.7584 |
| screening | expanded-2-1 | 142 | 0.1 | 0.0003 | 10.0 | 84.9664 |
| screening | expanded-2-2 | 142 | 0.1 | 0.001 | 2.0 | 83.4899 |
| confirmation | pilot-reference | 142 | 0.01 | 0.0003 | 1.0 | 78.7919 |
| confirmation | pilot-reference | 143 | 0.01 | 0.0003 | 1.0 | 79.3289 |
| confirmation | previous-selected-reference | 142 | 0.02 | 0.0003 | 2.0 | 78.5235 |
| confirmation | previous-selected-reference | 143 | 0.02 | 0.0003 | 2.0 | 78.9262 |
| confirmation | expanded-1-2 | 142 | 0.05 | 0.001 | 10.0 | 84.4295 |
| confirmation | expanded-1-2 | 143 | 0.05 | 0.001 | 10.0 | 83.7584 |
| confirmation | expanded-2-1 | 142 | 0.1 | 0.0003 | 10.0 | 85.1007 |
| confirmation | expanded-2-1 | 143 | 0.1 | 0.0003 | 10.0 | 85.2349 |

Congelata **expanded-2-1**: LR CNN=0.1, LR AE=0.0003, clip=10.0; media validation confermata=85.1678%. SHA256 selezione `f43e6cb2034792de61a416901388df75a778610783c43891bf7bfc69c7c83edc`. `selection.json` e le cinque configurazioni sono stati pubblicati prima delle nuove valutazioni test. Nessuna riapertura del tuning dopo il test.

## Configurazione finale, dati e verifiche

Stesso protocollo bilanciato D.2/tabella 8:743 training per dominio,300 round, tutti i 5 client, aggregazione uniforme, batch 32, un’epoca warm-up MSE encoder e un’epoca CE encoder/decoder/CNN. CNN benchmark invariata; UNetSmallAE, 3 canali/base 16/dz=64; encoder privato persistente, D/C condivisi (inclusi BN), fusione additiva. SGD CNN senza momentum/decadimento; Adam AE betas(0.9,0.999), epsilon1e-8, senza decadimento. Stati optimizer resettati per partecipazione, Adam continuo fra le fasi. Float32, determinismo, niente AMP/TF32. Una sola valutazione test finale, senza adattamento o ricalibrazione BN.

Dati `7a762ffb10da74e3f0dee9a6f519e6c4057e2a5b47995a0546ee5a871eb47b58`. Test: MNIST 14.000, SVHN 19.858, USPS 1.860, SynthDigits 97.791, MNIST-M 14.000. Tutte le confusion matrix 10×10, corrette/totali, medie, sequenze1–300, passi/budget, hash, sorgenti, exit code e sessioni sono verificati dall'audit senza una nuova valutazione del modello.

Provenienza riutilizzata dal pilota verificato: repository ufficiale [med-air/FedBN](https://github.com/med-air/FedBN), commit `2fa38adf627a8c8ba71c5fb515b1f2ba00aa8812`. Dati dal mirror collegato nel README degli autori, [Jemary/FedBN_Dataset](https://huggingface.co/datasets/Jemary/FedBN_Dataset), revisione `0b6cd64d780662b683a373ddb23aa25d1d968cf8`; ZIP SHA256 `6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221`. Questa provenienza non certifica identità con i byte usati nel 2021. Campionamento train_part0, preprocessing PIL e CNN sono documentati nei file/funzioni ufficiali `federated/fed_digits.py::prepare_data/train/test/communication`, `utils/data_preprocess.py::stratified_split/split`, e nei sorgenti del pilota conservato.

Verifica finale:271 test CPU superati, compresi tutti i test esistenti e i nuovi test mirati (log `verification_final_tests.log`). Prima della campagna passavano 260 test, poi 266 dopo gli strumenti di archivio; il coordinatore parallelo aggiunge 5 controlli. Si verificano separazione training/validation, protocollo e riferimenti, assenza di test in calibrazione, configurazioni non registrate e finali non congelate respinte, persistenza dei privati e ripresa esatta di stati/RNG, conteggi, tempi totali di ripresa, limiti degli slot, lettura dei codici d’uscita Linux e identità dei processi adottati. L’audit numerico controlla tutti i 23 risultati senza nuova inferenza.

## Tempi, memoria e calcolo

| Seed finale | GPU | Processo (s) | Sessione runner (s) | CUDA allocata/reservata (MiB) | RSS (MiB) |
|---:|---|---:|---:|---:|---:|
| 42 | cuda:1 | 1213.468 | 1206.967 | 275.292/350.000 | 1873.043 |
| 43 | cuda:0 | 1058.391 | 1052.526 | 275.292/350.000 | 1867.359 |
| 44 | cuda:1 | 1186.396 | 1180.797 | 275.292/350.000 | 1871.551 |
| 45 | cuda:0 | 1038.320 | 1032.688 | 275.292/350.000 | 1873.656 |
| 46 | cuda:1 | 1195.331 | 1189.199 | 275.292/350.000 | 1868.711 |

Parametri per client: CNN 14,219,210, E privato 72,080, D condiviso 44,995; totale attivo 14,336,285. Ogni run finale registra 36.000 passi warm-up e 36.000 CE (72.000 encoder,36.000 decoder e 36.000 CNN). Il warm-up è incluso nel calcolo.

- confirmation: 8 run; tempo controller 41.938 min; somma tempi processi 112.781 min; exit code tutti 0.
- final: 5 run; tempo controller 20.242 min; somma tempi processi 94.865 min; exit code tutti 0.
- screening: 10 run; tempo controller 26.263 min; somma tempi processi 52.453 min; exit code tutti 0.

Intervallo reale dall'inizio dello screening alla conclusione dei cinque seed: 93.598 min, compresi intervalli per selezione/commit. Somma tempi processi: 4.3350 ore, con limite superiore per i due worker adottati; somma delle durate effettive delle intere sessioni runner: 4.2427 ore. Somma FLOP convenzionali training: 8.053896 PFLOP. Per ogni finale: 551.898005 TFLOP (dense forward/backward più costo dichiarato optimizer/clipping). BN, attivazioni, pooling, loss, copie, controlli finiti, aggregazione e overhead kernel non sono compresi: non è energia misurata o conteggio di istruzioni GPU.

Nessuna interruzione/ripresa o errore numerico nei training archiviati. I tempi processo includono import/startup; quelli del runner includono verifica dati, addestramento, checkpoint e valutazione. Lo screening usa una run per GPU. Su successiva richiesta dell’utente è autorizzato un limite 3+3 per le sei conferme rimaste. Durante la verifica e sostituzione del solo coordinatore, i due training già attivi raggiungono naturalmente300 round; i quattro restanti sono avviati insieme,2 per GPU, senza ripetere le conferme concluse. Le cinque finali usano 3 processi su GPU 1 e 2 su GPU 0. Nessun segnale ai training o ai lavori esterni. GPU 0 resta condivisa con `tesi_giovanni`. Il vecchio coordinatore termina amministrativamente con codice 143 dopo l’uscita dei suoi figli (codici 0), non per un errore di training. Nei due worker adottati, l’intervallo UTC fino al rilevamento dell’uscita sovrastima la durata del processo di circa 100–110 secondi; è marcato come limite superiore. La durata reale della loro intera sessione runner, fino al checkpoint finale, è salvata separatamente e usata per interpretare i costi. Nessuna durata dell’ultima sessione è spacciata per l’intera run.

## Riproduzione e artefatti

Piano, shortlist, selezione, configurazioni ed esatti comandi sono versionati. `artifacts/` contiene tutti i risultati JSON compressi senza perdita, i timings JSONL, i receipt dei controller, `summary.json` e `final_accuracy.csv`. `artifact_manifest.json` contiene hash/byte e commit di training. Dataset, checkpoint e log completi restano privati; SHA dei checkpoint sono registrati, i checkpoint non sono pubblicati.

Comandi eseguiti dalla radice, Python general_ml; per ogni fase:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/controller.py \
  --queue research/feature_shift_digits_calibrated/FASE_queue.json \
  --receipt _local/feature_shift_digits_calibrated/FASE_campaign.json \
  --logs _local/feature_shift_digits_calibrated/logs/FASE
# Controller iniziale per screening e avvio conferme; fra le fasi:
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/select_stage.py --stage confirmation
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/select_stage.py --stage final
# Richiesta successiva: adozione conferme3+3, finali 3+2 (comandi completi nei receipt e README):
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/parallel_controller.py \
  --queue research/feature_shift_digits_calibrated/confirmation_queue.json \
  --receipt _local/feature_shift_digits_calibrated/confirmation_campaign.json \
  --logs _local/feature_shift_digits_calibrated/logs/confirmation \
  --gpu1-slots 3 --gpu0-slots 3 --adopt-stopped-scheduler 1200340
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/parallel_controller.py \
  --queue research/feature_shift_digits_calibrated/final_queue.json \
  --receipt _local/feature_shift_digits_calibrated/final_campaign.json \
  --logs _local/feature_shift_digits_calibrated/logs/final --gpu1-slots 3 --gpu0-slots 2
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits_calibrated/archive.py --verify
```

Commit scientifici della campagna: `60d9039f2f9efad64df18448445cdebb966d8a23`, `c6ec7572019b7a3299d77baec911410639efd8ec`, `e2d50ab710013197d4d6264c741db05664b415b1`, `e4dbfef42acf453eebeefbe487a726c48784e377`. Il commit finale di archivio è identificato dalla cronologia Git del manifesto.

## Limiti e interpretazione

Il pilota e i risultati del precedente controllo erano già noti: questa è una ricerca retrospettiva, con selezione numerica training-validation separata dal nuovo test. Una validation di 745 esempi riutilizzata e due seed di conferma non escludono sovradattamento della ricerca. La griglia L9 non esaurisce le interazioni; nessuna affermazione di ottimalità degli iperparametri.

Il confronto con gli avversari pubblicati è descrittivo: nessuna baseline comune rieseguita, seed/split originali e modalità esatta della statistica non certificati. Dati da mirror successivo collegato al repository degli autori; identificatori di riga sono disgiunti nei file disponibili, non ricostruiscono gli ID originali prima del resplit. MNIST-M deriva da MNIST, quindi i domini non sono sorgenti totalmente indipendenti. Fused ha encoder/decoder e computazione warm-up aggiuntivi; stesso classificatore non implica pari costo. Il BN condiviso di Fused differisce dal BN locale di FedBN. La nuova calibrazione dedicata ha un budget diverso da quello delle baseline pubblicate e del precedente controllo: queste run non sostituiscono il confronto di capacità/calcolo già archiviato, né ne estendono le conclusioni causali. Eccezioni, medie e tutti i seed rimangono visibili, senza selezione dopo il test.
