# PathMNIST — report finale, budget ridotto

**Run unica seed 42: accuratezza 35.279944%** a 50 round. Nessuna SD fra seed. Metrica: media uniforme di 10 pipeline private sul test ufficiale di 7.180 immagini ciascuna; 25331 corrette su 71.800 predizioni ricostruiscono il risultato. Non è il best encoder/seed/checkpoint.

Obiettivo prioritario dichiarato prima del congelamento: superare il 50,94% del paper. **Obiettivo non raggiunto**, con differenza -15.660056 punti percentuali. Questa soglia non ha guidato una selezione sul test o una riapertura della calibrazione. Le 71.800 predizioni riutilizzano gli stessi 7.180 esempi con dieci encoder: non sono 71.800 campioni indipendenti.

## Calibrazione e congelamento

Selezione su **un solo seed 142 e sul risultato a 20 round**: grid-c0.1-ae0.0001, modalità BN train-recalibrated, validation 48.771403%. 24 configurazioni previste, fallimenti e ripetizioni meccaniche conservati; tutte le metriche in CALIBRATION_REPORT.md e artifacts/calibration/. **Nessuna conferma su due seed e nessuna estensione a 100 round.** Regola di parità: native BN, poi ordine originale del candidato.

Run definitiva nuova su training completo 89.996 immagini e partizione dati 42 patologica congelata. Nessun nuovo dato, ottimizzazione/parallelismo o tuning dopo il congelamento; unica valutazione del test al termine, nessun arresto anticipato o selezione di checkpoint. L’orizzonte della schedule rimane quello della configurazione selezionata anche se i round finali sono 50.

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

## Confronto e differenze dal paper

Precedente run seed 42 con i default del paper:39,619777%; differenza descrittiva -4.339833pp. Tabella 4 riporta 50,94% medio, senza incertezza disponibile nella tabella: differenza descrittiva -15.660056pp. Non è una replica esatta o un confronto comune con baseline rieseguite. Una run non verifica la media pubblicata e non permette una SD fra seed o significatività statistica.

Restano encoder privati persistenti, decoder e classificatore condivisi, fusione additiva e le due loss originali; architettura ResNet20-v2/UNetSmallAE dz16 invariata. Iperparametri di LR/precisione/clipping/schedule/epoche locali differiscono dalla configurazione precedente soltanto come dichiarato nella configurazione congelata. Riferimento documentato: SGD LR 0,01, Adam LR 0,001, niente clipping/schedule, warm-up 1 e classificazione 3 epoche, 50 round. La precedente implementazione usa AMP FP16; il paper non specifica la precisione numerica.

La modalità selezionata train-recalibrated è una variante d’inferenza rispetto al paper: reset/re-stima dei soli buffer BN condivisi da 6.400 immagini fit fuse, senza label, gradienti, optimizer step o dati test. Encoder/decoder e pesi classifier sono invariati dalla ricalibrazione. Solo la modalità congelata è testata; native-final.pt e final.pt conservano entrambe le versioni.

## Costo e integrità

Run finale: 1804.215s training e salvataggi; 1808.775s sessione totale; 1815.354s calendario controller. GPU cuda:1, runner esistente. Picco alloc/res Torch 1261.251/1450.000MiB; RSS 1917.879MiB. Precisione/BN, clipping, loss e step nei risultati/timing. Nessuna ripresa del training definitivo; costo dello screening separato nel report calibrazione.

Checkpoint completi e caricabili su CPU verificati: initial.pt round 0, native-final.pt e final.pt round 50. Contengono classifier/decoder, 10 encoder privati, 57 buffer BN, 20 dizionari optimizer, scaler se richiesto dalla precisione, generatori loader e RNG CPU/Python/NumPy/CUDA, configurazione, versione del codice e indici esatti. SGD senza momentum ha stato vuoto previsto; BF16/FP32 non necessitano scaler. Inizializzazione verificata con C seed 42 ed AE seed 1.000.042. Tutti gli stati salvati sono finiti. Hash/dimensioni in checkpoint_verification.json.

Percorsi su thanos: _local/pathmnist_calibrated/final/seed-42/{initial.pt,native-final.pt,final.pt,latest.pt}; log in _local/pathmnist_calibrated/logs/final/. I dati/checkpoint completi restano privati; solo codice, configurazioni, indici, risultati numerici, hash, costi e report sono pubblicati. Test e verifiche allegati.

## Verifica dei conteggi e costo degli aggiornamenti

Le verifiche numeriche indipendenti sono passate: seed/configurazione/versione del codice coerenti, round 1–50 consecutivi, una sola valutazione reale del test al round 50, dieci pipeline con 7.180 esempi ciascuna e tutte le accuratezze ricostruibili dai conteggi. I tre checkpoint completi sono stati caricati e verificati su CPU. La suite già eseguita ha **313 test passati**, inclusi i sette test mirati di calibrazione e selezione.

| Pipeline / encoder privato | Corrette | Totale | Accuratezza (%) |
|---|---:|---:|---:|
| 0 | 2435 | 7180 | 33.913649 |
| 1 | 2650 | 7180 | 36.908078 |
| 2 | 2493 | 7180 | 34.721448 |
| 3 | 2526 | 7180 | 35.181058 |
| 4 | 2555 | 7180 | 35.584958 |
| 5 | 2488 | 7180 | 34.651811 |
| 6 | 2622 | 7180 | 36.518106 |
| 7 | 2513 | 7180 | 35.000000 |
| 8 | 2567 | 7180 | 35.752089 |
| 9 | 2482 | 7180 | 34.568245 |

Warm-up: **35,400** aggiornamenti encoder-only. Classificazione: **106,200** aggiornamenti di ciascun optimizer. Totali: **141,600** step Adam AE e **106,200** step SGD classifier. Sono conteggi della run completa, verificati per client e round contro il numero di batch previsto; BF16 non usa scaler e non introduce step saltati dal GradScaler. Tempi e memoria sono misurati, non stime.

L’esito negativo non dimostra un errore numerico: tutti i controlli di finitezza e integrità sono passati. La selezione breve sul seed 142 a 20 round non ha prodotto il miglioramento sperato nel singolo test finale del seed 42 a 50 round. Cambiano seed del modello, numerosità del training, durata e distribuzione rispetto al holdout: non sono state isolate le cause del divario, e non si attribuisce causalmente il risultato a una di queste differenze. Nessuna inferenza alternativa viene scelta o valutata dopo il test.

## Limiti e chiusura

Screening a 20 round con un seed può scegliere un profilo diverso dal migliore a 50; nessuna conferma indipendente. Validation ricavata dal training di uno stesso centro, test ufficiale da un altro centro; il test del precedente benchmark era già noto prima della ricerca, ma nessuna nuova accuratezza test è stata usata per scegliere. Una partizione fissa e un solo seed finale non stimano variabilità o ottimalità globale. Tutti i tentativi, compreso il riferimento FP16 numericamente fallito, sono conservati. Manoscritto e altre campagne invariati; nessuna baseline, ablation o diagnostica avviata. Il lavoro si ferma qui.
