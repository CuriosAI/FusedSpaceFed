# Calibrazione PathMNIST — budget ridotto

**Selezione su un solo seed (142), sul risultato terminale a 20 round. Nessuna conferma e nessuna estensione.** La disposizione dell’utente in BUDGET_REDUCTION.md sostituisce il budget iniziale; i file protetti e gli iperparametri dello screening restano invariati.

24 configurazioni pianificate; **23 completate a 20 round**. Il riferimento originale FP16 è fallito per loss non finita al round 19 (ultimo checkpoint completo 18) e non ha uno score valido a 20. Nessuna correzione silenziosa del candidato: il controllo FP32 con gli stessi LR/epoche è un candidato separato già previsto. I tentativi fermati all’import per collisione del nome select.py sono ripetuti con identici hash scientifici dopo la rinomina dell’helper; tutti i receipt/log e i checkpoint sono conservati.

Training fit 81.002 immagini + validation 8.994, disgiunte globalmente, stratificate per client/classe con holdout seed 20261006. Partizione dati originale 42 congelata, due classi/client; nessun dato test usato per calibrazione. Primaria: media uniforme di 10 pipeline sul pooled holdout, ricostruita dai conteggi. Il precedente test seed 42 del benchmark (39,619777%) era noto prima del task; non si presenta questo come un test mai osservato.

| Candidato | Native (%) | BN da fit (%) |
|---|---:|---:|
| paper-reference | fallito/non disponibile | fallito/non disponibile |
| fp32-matched-reference | 24.838781 | 34.392929 |
| grid-c0.003-ae0.0001 | 17.917501 | 19.202802 |
| grid-c0.003-ae0.0003 | 16.746720 | 17.816322 |
| grid-c0.003-ae0.001 | 11.517678 | 18.413387 |
| grid-c0.01-ae0.0001 | 24.124972 | 30.479208 |
| grid-c0.01-ae0.0003 | 25.683789 | 35.992884 |
| grid-c0.01-ae0.001 | 15.770514 | 26.007338 |
| grid-c0.03-ae0.0001 | 32.532800 | 44.386258 |
| grid-c0.03-ae0.0003 | 32.036913 | 43.331110 |
| grid-c0.03-ae0.001 | 36.941294 | 47.292640 |
| grid-c0.1-ae0.0001 | 41.720036 | 48.771403 |
| grid-c0.1-ae0.0003 | 42.651768 | 44.747610 |
| grid-c0.1-ae0.001 | 31.316433 | 47.379364 |
| clip-1 | 23.668001 | 36.625528 |
| clip-10 | 33.007561 | 46.849010 |
| cosine-c0.03 | 33.463420 | 43.702468 |
| local1-c0.03 | 29.142762 | 43.014232 |
| cosine-c0.1 | 40.601512 | 44.723149 |
| local1-c0.1 | 35.572604 | 41.301979 |
| warm-lr-1e-4 | 30.912831 | 35.324661 |
| bf16-no-clip | 25.220147 | 38.263287 |
| fp32-clip5 | 32.562820 | 39.556371 |
| local1-c0.3 | 33.328886 | 44.366244 |

## Congelamento

**grid-c0.1-ae0.0001, modalità train-recalibrated**, validation 48.771403%. Massimo fra candidati finiti e le modalità già dichiarate; parità esatta nei conteggi: BN nativa, poi ordine originale del candidato. Nessun best-round o best-seed. Configurazione unica congelata in selection.json prima della run finale.

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

Una sola run finale nuova: seed 42, training completo 89.996 immagini, stessa partizione patologica congelata, 50 round e unico test finale. L’orizzonte della schedule resta quello del candidato (100 se cosine), senza comprimerlo a 50. Nessuna riapertura del tuning dopo il test.

## Metodo, verifiche e costi

Architettura e capacità invariati: ResNet20-v2 con 9 logits, UNetSmallAE dz16; encoder privati persistenti, decoder/classifier condivisi, fusione additiva, MSE encoder-only nel warm-up e CE-only con aggiornamento di tutti i componenti. Aggregazione uniforme e piena partecipazione. SGD/Adam persistono senza reset. LR, epoche, clipping e schedule sono differenze esplicite rispetto alla configurazione precedente; il paper non specifica la precisione numerica. Il riferimento del codice preesistente usa AMP FP16.

La modalità train-recalibrated, se selezionata, è una variante rispetto al paper: ricalibra soltanto i buffer BatchNorm condivisi, con 6.400 immagini di fit fuse dai rispettivi encoder. Nessuna label, gradiente, optimizer step, dato test o adattamento degli encoder è usato. Native e train-recalibrated sono alternative d’inferenza della stessa configurazione allenata, selezionate esclusivamente sul holdout.

Test: parità esatta FP32 unclipped con il client originale, flusso delle due fasi, persistenza/ripresa degli optimizer e generatori, holdout disgiunto, BN senza modifica di pesi/RNG/input, scheduler senza reset Adam e parità della selezione su conteggi. Log allegati. Loss e stato modello/optimizer verificati finiti a ogni checkpoint delle run complete. Nessun pacchetto installato.

Primo lotto e tentativi meccanici: 896.137s calendario, 4876.291s somma worker. Ripetizioni degli avvii: 2536.270s calendario, 14113.208s somma worker. Entrambe le GPU, fino a 3 worker/GPU, nessun processo esterno interrotto. La somma worker concorrenti non è tempo fisico di calcolo GPU. Costi per profilo, RSS, alloc/res GPU, loss, norm/clipping e step tentati nei JSON/timings.

Tempo calendario dal primo avvio all’ultimo completamento, incluso l’intervallo fra i lotti: 3698.315s (61.639 minuti). Somma delle durate dei controller attivi: 3432.407s. Somma dei processi worker, inclusi i fallimenti: 18989.499s.

Checkpoint iniziali e terminali completi, optimizer/scaler/RNG, indici e log restano su thanos in _local/pathmnist_calibrated/screening/. I checkpoint nativi completi e gli indici fit rendono riproducibile la modalità BN alternativa; il finale memorizzerà i buffer effettivamente selezionati. Percorsi/hash nel summary.

Limiti: screening breve e un solo seed, nessuna conferma, selezione su holdout della distribuzione training e una sola partizione. Possibile sovrastima di validation e ranking diverso a 50 round. Una ricerca finita non prova ottimalità. Il finale è un singolo seed, senza SD fra seed. Manoscritto, baseline, ablation e diagnostiche non sono eseguiti/modificati.
