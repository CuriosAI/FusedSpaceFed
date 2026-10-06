# Digits — decoder, componenti e gradienti

Le tre fasi sono concluse, in ordine, su Digits bilanciato già disponibile. Le run precedenti e il manoscritto sono invariati. Sono riusate le cinque run FusedSpaceFed calibrate e i loro checkpoint; soltanto le tre ablation sono riaddestrate, da zero con cinque seed abbinati (42–46), per 300 round. Tutti gli iperparametri sono quelli già scelti su validation da training, senza tuning delle varianti o selezione sul test.

| Fase | Evidenza e file |
|---|---|
| 1. Decoder/warm-up | [Report e 10 griglie visive](phase1/PHASE1_REPORT.md); 5 seed, 2 ancore, 160 esempi di training/client; replay locale su copie prima/dopo warm-up e CE. Output e gradienti finiti; warm-up modifica soltanto encoder. MSE media finale: warm-up −0,003578, poi CE +0,055155. Il decoder produce anche correzioni di colore/struttura e non va assunto una ricostruzione fedele. |
| 2. Componenti | [Report](phase2/PHASE2_REPORT.md), 15 nuove run e 5 full riusate, risultati per seed/dominio, conteggi e costi. Encoder condiviso competitivo/superiore; piccolo effetto medio del warm-up; circa 1 pp medio per l’aggiunta dell’input rispetto al solo decoder. |
| 3. Gradienti | [Report](phase3/PHASE3_REPORT.md), Gram verificabili, Γ originale/fusa, B, Φ, norme, dispersione del decoder, 5 batch/client e sensibilità BN a stato identico. Al finale riduzione di Γ grezza in 4/5 seed eval e 5/5 batch-stateless; nessuna riduzione universale all’inizializzazione. |

## Accuratezza e calcolo

Media ± SD campionaria tra tutti i cinque seed (`ddof=1`), senza best-seed. Primaria: media uniforme dei domini. La media pesata per campioni rimane distinta e cambia il ranking del warm-up.

| Variante | Uniforme (%) | Pesata (%) | FLOPs contabilizzati/run |
|---|---:|---:|---:|
| Full | 85,862359 ± 0,846930 | 84,514708 ± 0,745273 | 551.898.005.448.000 |
| Senza warm-up | 85,714598 ± 0,789950 | 85,142330 ± 0,344917 | 449.968.148.424.000 |
| Encoder condiviso | 86,178544 ± 0,468069 | 84,646089 ± 0,574453 | 551.898.005.448.000 |
| Solo decoder | 84,859627 ± 0,580810 | 83,608729 ± 0,486312 | 551.898.005.448.000 |

La variante senza warm-up risparmia circa il 18,469% di FLOPs secondo la convenzione dichiarata, mantiene un’epoca CE ma cambia anche lo stato Adam e lo shuffle CE. L’encoder condiviso ha gli stessi parametri attivi/client, diversa persistenza/condivisione e comunicazione extra. Il confronto è condizionato agli iperparametri del full, non il massimo ottenibile dalle varianti.

## Interpretazione dei gradienti

L’identità Γ fusa = Γ originale + B + Φ è ricostruita da Gram indipendenti in 20 casi medi e 100 casi per batch, residuo relativo massimo medio 1,684e−15. Al round 300 il rapporto Γf/Γo medio è 0,791656 ± 0,253408 in eval e 0,148367 ± 0,140864 in batch-stateless.

Questi cali grezzi sono accompagnati da variazioni di scala. In eval la dispersione normalizzata **aumenta** (0,794160 → 0,799588) e il coseno medio fra client **diminuisce** (0,090559 → 0,043966); in batch-stateless il cambiamento normalizzato è piccolo (0,796595 → 0,789012). Non si sostiene un forte miglioramento direzionale universale o una spiegazione causale dell’accuratezza. Il classifier è stato allenato su input fusi; il ramo originale è controfattuale, BN e scala incidono, e la sonda ha visto training. Il decoder aggiunge uno spazio condiviso distinto la cui dispersione è misurata esplicitamente.

I risultati non certificano la necessità dell’encoder privato su questo setting di cinque domini bilanciati e non si estrapolano ai benchmark label-skew del manoscritto. Non sono state provate interazioni fra componenti o iperparametri ottimali separati. Restano visibili tutte le eccezioni per seed/dominio.

## Costi e archivio

- Fase 1: 54,232 s calendario, 97,240 s somma processi, un worker/GPU.
- Fase 2: 3.235,428 s calendario (53,924 min), 16.373,711 s somma processi, fino a tre worker/GPU; 15 exit 0, nessuna ripresa/interruzione.
- Fase 3: 84,246 s calendario, 392,701 s somma processi, 3 worker su GPU 1 e 2 su GPU 0; 5 exit 0, nessun aggiornamento del modello.

I picchi di memoria e i tempi di ogni seed sono nei report/JSON; i picchi Torch escludono contesto, driver e altri processi. Nessun job esterno è stato interrotto. Dataset, checkpoint completi, RNG e log restano in `_local/digits_mechanism_diagnostics/`; gli hash delle ancore e dei checkpoint sono versionati. `probe.json` conserva indici e identificatori degli esempi, non il dataset.

Ogni fase ha configurazione, runner, receipt con i comandi effettivi, risultati lossless per seed, sintesi numerica, verifica e manifesto. Per verificare senza avviare training:

```bash
python research/digits_mechanism_diagnostics/phase1/archive.py verify
python research/digits_mechanism_diagnostics/phase2/archive.py verify
python research/digits_mechanism_diagnostics/phase3/archive.py verify
```

Suite versionata completa e test mirati: algebra/Gram, pesi, parametri/buffer/RNG immutati, flusso gradienti, input corretto, encoder privati/condivisi, parità con il driver originale e ripresa esatta. I log sono allegati alle fasi; nessun pacchetto installato o aggiornato.

## Commit e lavoro precedente

1. Fase 1: `3d898247ea70daf2329b49e804edf1af30b0d713` — `research: diagnose Digits decoder and warm-up`.
2. Fase 2: `5d1591b3981db70a6d4054bdf3f4658faf665de3` — `research: add five-seed Digits component ablations`.
3. Fase 3: il commit dedicato che versiona questo riepilogo e `phase3/`; identificabile con `git log -1 --format='%H %s' -- research/digits_mechanism_diagnostics/phase3/PHASE3_REPORT.md`.

Prima delle tre fasi è stato completato e pubblicato il controllo FedAvg capacità/FLOPs su cinque seed, commit `a393b4e6e4132f22a6c22516f7c4a1ebae84c5d2`: [report separato](../capacity_compute_control_five_seed/FIVE_SEED_REPORT.md). FedAvg ampliato 82,064993 ± 0,221497% vs Fused calibrato 85,862359 ± 0,846930%. Il loro tuning non era equivalente (FedAvg 9×60 round, Fused successivo 10×120 + 8×300); quel delta non è una prova causale a tuning equivalente. Il precedente controllo a sforzo comparabile rimane conservato. Nessuna baseline è stata rieseguita in queste tre fasi.
