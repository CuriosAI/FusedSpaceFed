# PathMNIST: run patologica FusedSpaceFed, seed 42

Run completata da zero il 6 ottobre 2026, su thanos, con **50 round** e tutti
i 10 client partecipanti. Il test ufficiale è stato valutato soltanto dopo
l'ultimo round, senza tuning, adattamento finale o selezione del checkpoint.

| Metrica finale, media uniforme delle 10 pipeline | Risultato |
|---|---:|
| Accuratezza | **39,619777%** |
| Balanced accuracy | 40,209407% |
| Macro F1 | 0,362904 |

Ogni pipeline combina il proprio encoder privato con classificatore e decoder
finali condivisi, ed è valutata sugli stessi **7.180** esempi ufficiali di test.
I conteggi sommano **28.447 corrette su 71.800 predizioni**: ricostruiscono
esattamente la metrica primaria. Le accuratezze per pipeline sono conservate
in `artifacts/results.json`. Una sola run non fornisce una SD fra seed.

## Configurazione e dati congelati

Impostazioni del paper: partizione patologica con due classi per client,
batch 128, warm-up dell'encoder di un'epoca e classificazione di tre epoche
per round; ResNet20-v2, UNetSmallAE, `dz=16`, SGD a 0,01 e Adam a 0,001.
Encoder privati e optimizer persistono; decoder e classificatore sono
aggregati uniformemente. Nessun clipping o scheduler. Il training CUDA AMP
è quello del client MedMNIST esistente; il paper non specifica la precisione.
`config.json` e `README.md` descrivono tutti i dettagli.

MedMNIST **3.0.2**, release ufficiale PathMNIST-64, MD5 verificato
`55aa9c1e0525abe5a6b9d8343a507616`; immagini ridimensionate a 32×32 mediante
PIL/torchvision Resize e ToTensor. Nessun esempio di validation è aggiunto al
training. I **89.996** esempi di training sono assegnati una sola volta,
con esattamente due classi per ciascun client: media 8.999,6, minimo 6.589,
massimo 12.533 immagini. Seed del modello e della partizione: **42**;
inizializzazione AE con seed `42 + 1.000.000`, come nel runner originale.

SHA256 del file con gli indici esatti:
`b70fb78ad84a7d7041c5d98938d208f8b23455c3254b8b949e32c58509828a80`.
Gli indici sono pubblicati senza immagini in `artifacts/partitions.json.gz`;
source URL, hash dei dati/cache e conteggi per client/classe sono nel manifesto.

## Checkpoint completi e verificati

Directory privata su thanos:
`/mnt/data/codex/FusedSpaceFed/_local/pathmnist_pathological/seed-42/`.

| File | Round | Dimensione |
|---|---:|---:|
| `initial.pt` | 0, prima di ogni aggiornamento | 18.182.596 byte |
| `final.pt` | 50, condivisi sincronizzati, prima del test | 23.390.414 byte |

Entrambi sono caricabili su CPU con `torch.load(..., weights_only=False)` e
contengono classificatore, decoder, **tutti i dieci encoder privati**, i
**57 buffer BatchNorm** del classificatore, 20 dizionari optimizer SGD/Adam,
10 stati GradScaler, generatori dei loader, stati RNG dei worker/coordinatore,
configurazione, seed, indici e identità del codice. Gli optimizer iniziali
hanno stato vuoto; SGD senza momentum conserva normalmente uno stato vuoto
anche alla fine. Il checkpoint `latest.pt` permette la ripresa dell'ultimo
round completo. Hash SHA256 dei checkpoint in `artifacts/verification.json`.

Verificati caricamento rigoroso dei modelli, dimensioni e finitezza su input
di training, stati/momenti degli optimizer, sincronizzazione finale,
disgiunzione/copertura della partizione e accuratezze dai conteggi.
**306 test passati**, inclusi quattro nuovi test sintetici per cache/sampler,
warm-up, aggregazione e ripresa esatta del round successivo. In sviluppo è
stato corretto un test che non esauriva entrambi i sampler; i tentativi sono
conservati localmente. Nessuna correzione del training e nessuna ripresa.

Il GradScaler nativo ha gestito overflow dei gradienti scalati: i contatori
Adam mostrano **36 aggiornamenti di classificazione saltati su 106.200**
previsti. Tutti gli aggiornamenti Adam di warm-up sono stati eseguiti.
Loss e stati salvati sono finiti; non sono stati cambiati gli iperparametri.
I contatori dei timing descrivono i mini-batch tentati, mentre i contatori
Adam effettivi sono conservati in `artifacts/optimizer_audit.json`.

## Durata, risorse e provenienza

Due worker, uno per GPU, con cinque client persistenti ciascuno; aggregazione
in ordine di client dopo il completamento di entrambi. La cache residente
mantiene preprocessing e campionamento del codice originale, verificati dai
test. Nessun lavoro preesistente è stato interrotto.

- Esecuzione completa: **963,66 s (16 min 3,66 s)**.
- Training e salvataggi fino al checkpoint finale: **948,92 s**.
- Valutazione finale parallela: **1,03 s**; media round senza salvataggio:
  **18,69 s**.
- Picco allocato PyTorch: **1,202 GiB per GPU**; massimo riservato,
  considerando anche il test: **1,350 GiB**. RSS massimo dei worker: circa
  **1,69 GiB** ciascuno. Questi valori non includono processi altrui o tutto
  il contesto CUDA.
- 35.400 mini-batch warm-up, 106.200 mini-batch classificazione;
  **17.999.200** esposizioni di immagini di training complessive.

Comando eseguito, dal repository, dopo la preparazione:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_pathological/run.py run
```

Codice base: `4218d49f75269ca017cc3128e49c7df10d440220`; il nuovo runner
e la configurazione sono identificati dai rispettivi SHA256 durante la run,
poi versionati nel commit dedicato. Provenienza completa, comando, codice
d'uscita 0, orari, versioni, timing e log sono in `artifacts/`.
Manoscritto, codice scientifico preesistente e altri esperimenti invariati.

## Limiti e utilizzo successivo

Il **39,62%** di questa run è inferiore al **50,94%** medio riportato nella
Tabella 4. Questo risultato non recupera né certifica le cinque run originali:
non erano disponibili i loro checkpoint, configurazioni effettive e indici.
Una singola run nella configurazione documentata non verifica quella media
né permette conclusioni statistiche sulle baseline, che non sono state
rieseguite. Nessuna impostazione è stata rivista dopo aver visto il test.

I checkpoint sono utilizzabili per diagnostiche decoder/warm-up e gradienti
a stato fissato, con tutti i dati necessari disponibili sul server. Non
contengono i confini intermedi di ogni fase nei round storici: tali misure
richiederanno un replay esplicitamente dichiarato da un checkpoint salvato.
Le diagnostiche non sono state avviate in questa attività.
