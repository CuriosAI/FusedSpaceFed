# Fase 2 — Ablation dei componenti su Digits

Cinque seed abbinati 42–46 per quattro metodi. Le cinque run FusedSpaceFed complete calibrate sono riusate; 15 nuove run da zero delle tre ablation, ciascuna per 300 round. Dati e classificatore immutati, training completo 743/client; test una volta al round 300 sui cinque domini e 147.509 esempi, nessun adattamento/checkpoint/seed selection.

| Variante | Intervento |
|---|---|
| full | Encoder privato persistente, decoder/classifier condivisi, x+D(Eᵢ(x)), 1 warm + 1 CE |
| no-warmup | 0 epoche warm-up, 1 CE; encoder rimane privato |
| shared-encoder | Encoder iniziale uguale per tutti; encoder aggiornati localmente e aggregati uniformemente insieme a decoder/classifier ogni round |
| decoder-only | Classifica D(Eᵢ(x)) senza aggiunta di x; warm-up e encoder privato mantenuti |

Stesse impostazioni già selezionate esclusivamente su validation da training (594 fit/149 validation/client) nella campagna calibrata precedente: LR C=0,1, LR AE=0,0003, clip=10; SGD senza momentum/decay per C, Adam standard per AE, reset a ogni partecipazione, Adam continuo warm→CE, batch 32, Float32, dz=64. Il template `training.warmup_epochs=1` identifica il metodo di riferimento; il campo esplicito `plan.variants.no-warmup.warmup_epochs=0` determina l’intervento nel runner. Nessun nuovo tuning: si tratta di un’ablation a iperparametri fissi selezionati su validation, non di varianti individualmente ottimizzate. Non si usano i test per cambiare le impostazioni.

| Variante | Uniforme: media ± SD (%) | Pesata: media ± SD (%) | Full−variante uniforme (pp) |
|---|---:|---:|---:|
| full | 85.862359 ± 0.846930 | 84.514708 ± 0.745273 | — |
| no-warmup | 85.714598 ± 0.789950 | 85.142330 ± 0.344917 | 0.147761 ± 0.530003 |
| shared-encoder | 86.178544 ± 0.468069 | 84.646089 ± 0.574453 | -0.316185 ± 0.577601 |
| decoder-only | 84.859627 ± 0.580810 | 83.608729 ± 0.486312 | 1.002732 ± 1.215783 |

SD campionaria tra tutti i cinque seed, ddof=1; differenze calcolate sui seed abbinati. Sono statistiche descrittive, non un test di significatività.

| Seed | Full | No warm-up | Encoder condiviso | Solo decoder |
|---:|---:|---:|---:|---:|
| 42 | 84.669219 | 85.022407 | 85.578351 | 84.808456 |
| 43 | 86.303678 | 85.449207 | 86.392795 | 84.976995 |
| 44 | 85.322698 | 84.996365 | 86.248399 | 85.341854 |
| 45 | 86.758720 | 86.437423 | 86.792319 | 83.896198 |
| 46 | 86.257479 | 86.667588 | 85.880856 | 85.274631 |

| Dominio | Full | No warm-up | Encoder condiviso | Solo decoder |
|---|---:|---:|---:|---:|
| MNIST | 96.947143 ± 0.239823 | 96.591429 ± 0.309847 | 96.918571 ± 0.285964 | 96.940000 ± 0.170129 |
| SVHN | 67.827576 ± 2.876878 | 65.764931 ± 3.586299 | 69.370531 ± 1.020393 | 66.710646 ± 1.910382 |
| USPS | 96.666667 ± 0.281938 | 96.612903 ± 0.409450 | 96.720430 ± 0.281938 | 96.225806 ± 0.728688 |
| SynthDigits | 86.316123 ± 0.496394 | 87.680870 ± 0.309636 | 86.183187 ± 0.595872 | 85.573110 ± 0.456493 |
| MNIST-M | 81.554286 ± 0.773321 | 81.922857 ± 3.261856 | 81.700000 ± 0.833774 | 78.848571 ± 0.843393 |

Valori per dominio e per seed, conteggi, confusioni, differenze abbinate e risultati esatti sono nel CSV/JSON. Tutte le eccezioni di segno rimangono visibili: il ranking osservato non viene corretto scegliendo checkpoint o seed.

## Contributi osservati

- Rispetto a `no-warmup`, il metodo completo ha media superiore di 0.147761 pp e supera la variante in 3/5 seed. La SD delle differenze abbinate è 0.530003 pp.
- Rispetto a `shared-encoder`, il metodo completo ha media inferiore di 0.316185 pp e supera la variante in 1/5 seed. La SD delle differenze abbinate è 0.577601 pp.
- Rispetto a `decoder-only`, il metodo completo ha media superiore di 1.002732 pp e supera la variante in 3/5 seed. La SD delle differenze abbinate è 1.215783 pp.

Il confronto dell’encoder condiviso verifica la persistenza privata in questo setting bilanciato di soli cinque domini: un risultato competitivo o superiore della variante condivisa limita la necessità empirica dell’encoder privato qui. Non viene trasferito ai setting label-skew del manoscritto. Un vantaggio della fusione additiva rispetto al solo decoder rimane condizionato alla configurazione e alla partizione; un effetto piccolo del warm-up va letto insieme al suo costo.

## Costo e lettura causale

Le architetture per client e i parametri attivi sono identici (14.336.285), salvo la condivisione dell’encoder sul server. Il numero di epoche CE rimane uno per client/round. La rimozione del warm-up riduce passi di encoder e computazione e cambia lo stato Adam iniziale della CE: è l’effetto complessivo di rimuovere questa fase, non un confronto a FLOPs identici. Non consumando lo shuffle dell’epoca warm-up, questa variante ha lo stesso seed/inizializzazione ma non le stesse permutazioni CE del metodo completo; tutti gli esempi restano coperti una volta in ogni epoca CE. Le altre due ablation mantengono il numero di fasi e la sequenza dei generatori. Il costo dell’addizione è escluso dalla convenzione FLOPs esistente; decoder-only ha lo stesso costo contabile ma un diverso percorso funzionale.

| Variante | FLOPs contabili/run | Dense FLOPs/run | Passi CE/run | Passi warm/run |
|---|---:|---:|---:|---:|
| full | 551898005448000 | 549229594368000 | 36000 | 36000 |
| no-warmup | 449968148424000 | 447341255424000 | 36000 | 0 |
| shared-encoder | 551898005448000 | 549229594368000 | 36000 | 36000 |
| decoder-only | 551898005448000 | 549229594368000 | 36000 | 36000 |

La convenzione conta forward/backward convolution/matmul (FMA=2) e clipping/optimizer semantici; esclude BN/ReLU/pool/loss, copie, aggregazione, overhead e qui la comunicazione extra dell’encoder condiviso. Non misura energia o istruzioni hardware. Parametri, budget e sorgenti del controllo FedAvg restano separati e invariati.

Fase nuova: 3235.428 s calendario, 16373.711 s somma processi, fino a 3 worker/GPU. Le cinque run full costano zero nuova esecuzione; i loro tempi originali sono registrati in summary.json. Nessun processo esterno interrotto, tutti i 15 exit 0, una sessione da zero per ogni run.

| Variante | Durata media nuova/sessione (s) | CUDA alloc massimo (MiB) | CUDA res massimo (MiB) |
|---|---:|---:|---:|
| full | 1132.436 | 275.292 | 350.000 |
| no-warmup | 895.012 | 275.292 | 350.000 |
| shared-encoder | 1185.401 | 275.292 | 350.000 |
| decoder-only | 1176.017 | 275.292 | 350.000 |

## Verifiche e riproduzione

Parità bit per bit degli aggiornamenti del percorso completo con DigitsClient precedente su due round sintetici; input al classificatore verificato; componenti aggiornati e warm-up assente esplicito; encoder persistenti/distinti o identici dopo aggregazione secondo variante; ripresa esatta di classifier, decoder, encoder e RNG; rifiuto di overwrite/configurazioni non registrate. Suite versionata completa e log allegati.

Audit indipendente di 20 run: seed/dati/configurazioni/sorgenti, 300 round consecutivi, una valutazione finale, cinque domini, conteggi/confusioni/accuratezze, sample SD, fasi/passaggi e costi. Checkpoint finali e RNG, log e risultati grezzi sono in `_local/digits_mechanism_diagnostics/phase2/`; SHA/byte in artifacts/private_checkpoints.json. Nessun dataset/checkpoint intero viene pubblicato. Manoscritto e risultati precedenti invariati.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/digits_mechanism_diagnostics/phase2/runner.py --config research/digits_mechanism_diagnostics/phase2/configs/no-warmup-seed-42.json --device cuda:1 --output _local/digits_mechanism_diagnostics/phase2/reproduction-no-warmup-seed-42
python research/digits_mechanism_diagnostics/phase2/archive.py verify
```

Limiti: un solo setting/partizione, cinque seed, configurazione comune scelta per il metodo completo. Una variante può rispondere diversamente al tuning; questo esperimento non ne stima il massimo raggiungibile. Non sono testate interazioni fra interventi combinati, altri dataset o adattamento finale. Le differenze sono condizionate al protocollo e non una prova universale di necessità dei componenti.
