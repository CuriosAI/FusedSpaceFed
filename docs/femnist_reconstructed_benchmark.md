# Uso del benchmark FEMNIST ricostruito

Implementazione del profilo approvato, con la successiva riduzione proporzionale
delle sole classi insufficienti. [Protocollo e limiti di comparabilità](femnist_fedrep_protocol.md)
restano il riferimento scientifico: confronto fra cinque risultati nostri e
baseline pubblicate, senza replica esatta o riesecuzione delle baseline.
Nessuna delle cinque run definitive è stata avviata durante l'implementazione.

## File e compatibilità

- [femnist_reconstructed_data.py](../femnist_reconstructed_data.py): download
  riutilizzabile/riprendibile, quote intere, preparazione, audit e loader.
- [train_femnist_reconstructed.py](../train_femnist_reconstructed.py): MLP,
  client che riutilizza le due fasi del core, runner, checkpoint e reporting.
- [configs/femnist_reconstructed.json](../configs/femnist_reconstructed.json):
  tutte le scelte approvate e hash della partizione e dell'archivio.
- [tests/test_femnist_reconstructed.py](../tests/test_femnist_reconstructed.py):
  test sintetici del preparatore e del runner.
- [fusedspacefed_core.py](../fusedspacefed_core.py): due argomenti opzionali,
  `classifier` e `use_amp`, del solo `FusedSpaceFedClient`. I default rimangono
  ResNet20-v2 e AMP; le due fasi sono riutilizzate senza duplicare il training.
- `.gitignore` esclude `_local/`. Gli entry point FEMNIST per scrittore,
  MedMNIST, manoscritto e risultati precedenti rimangono invariati.

Si usa l'ambiente **general_ml** esistente, senza installazioni. I comandi
seguenti partono dalla radice del repository sulla macchina di lavoro; il
Python assoluto identifica esattamente quell'ambiente.

## Dati congelati

L'archivio NIST da 1,031,576,378 byte è già presente in
`_local/femnist_reconstructed/source/by_class.zip`. La ricevuta locale
`by_class.download.json` registra URL, dimensione e SHA-256. Il preparatore
verifica l'archivio esistente senza effettuare richieste di rete; `--download`
consente un download solo se manca. Un file `.part` è riutilizzato con HTTP
Range quando il server lo supporta. Non si estraggono altre cartelle del pool.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python femnist_reconstructed_data.py \
  --config configs/femnist_reconstructed.json \
  --root _local/femnist_reconstructed
```

Il comando è idempotente sulla partizione presente: verifica e riusa, senza
sovrascriverla. La prima preparazione avviene in una directory temporanea;
la directory `partition/` appare solo dopo l'audit riuscito. Ricette o dati
diversi su una destinazione già occupata producono un errore.

`partition/manifest.json` registra quote originali e assegnate per ogni
client/classe, fattori e rapporti interi, disponibilità, ID d'origine,
hash PNG, indice nell'array, seed, generatori, versioni e statistiche.
Gli array train/test separati sono uint8; il loader produce Float32/255,
1×28×28 e target int64, senza normalizzazione o augmentation aggiuntiva.
Il vecchio `partition_failure.json` rimane conservato e invariato.

| Insieme | Totale | Media/client | Min/client | Max/client |
|---|---:|---:|---:|---:|
| Richieste originali | 26967 | 179.780000 | 102 | 1050 |
| Assegnati prima dello split | 25339 | 168.926667 | 97 | 877 |
| Training | 22736 | 151.573333 | 87 | 789 |
| Test | 2603 | 17.353333 | 10 | 88 |

| Classe | Richiesti | Assegnati | Training | Test |
|---|---:|---:|---:|---:|
| a | 2828 | 2828 | 2543 | 285 |
| b | 2506 | 2506 | 2241 | 265 |
| c | 2474 | 2474 | 2221 | 253 |
| d | 2325 | 2325 | 2094 | 231 |
| e | 2558 | 2558 | 2300 | 258 |
| f | 2555 | 2493 | 2216 | 277 |
| g | 2578 | 2578 | 2326 | 252 |
| h | 2869 | 2869 | 2576 | 293 |
| i | 3078 | 2788 | 2493 | 295 |
| j | 3196 | 1920 | 1726 | 194 |

La regola cambia quote e proporzioni in 105 client. Sono verificati 150 client
con tre classi nel training, train/test non vuoti e 25,339 ID globalmente
unici. Il campionatore non impone questo totale: risulta dalle capacità e
dal maggior resto, senza nuovi sorteggi.

- SHA-256 archivio: `b387d65249b2d0ed429cf81967d4c40a9d01ca2f7bb3931c6a36d825bc22d411`.
- SHA-256 canonico partizione: `7a4c7614796a595d8a752aba5d7d275a18a2f6e542dfd338e8a381fcd5675734`.
- SHA-256 canonico configurazione: `809d987e704f495072fecd53bf5f08cb930e3685616417c9adab34d1c0d87101`.

## Run definitive predisposte, non eseguite

Comando esatto per **una** run, seed 41, GPU 1:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python train_femnist_reconstructed.py run \
  --config configs/femnist_reconstructed.json \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/runs/seed-41 \
  --seed 41 --device cuda:1
```

Le altre quattro run useranno seed 42–45 e directory distinte. Il runner non
avvia più seed automaticamente. `--device` è obbligatorio; `cuda` senza
indice è rifiutato, così come un fallback automatico a CPU se CUDA manca.
Il profilo prevede 200 round, 15 client/round, batch 10, un'epoca di warm-up,
cinque di classificazione, `d_z=64`, Float32 senza AMP/TF32 e aggregazione
uniforme. Gli optimizer sono nuovi a ogni partecipazione; Adam rimane lo
stesso fra le due fasi di quella partecipazione.

La valutazione del test avviene solo dopo l'aggregazione dei round **191–200**,
su tutti i client con condivisi correnti ed encoder persistenti, senza
adattamento. Si salvano corrette/totali e partecipazioni per ogni client,
accuratezza pesata per campioni e media uniforme fra client. Il valore per
seed è la media della finestra, senza ricerca del miglior checkpoint.
`summarize` accetta soltanto cinque run complete e compatibili, seed 41–45:
media dei cinque valori e deviazione standard campionaria `ddof=1`, in punti
percentuali. Non aggiunge righe al CSV delle baseline pubblicate.

## Checkpoint, ripresa e output

La directory di una nuova run deve essere assente. Un lock esclusivo impedisce
due processi sulla stessa directory. `--resume` è l'unico modo per riutilizzare
un output, e verifica configurazione, partizione, seed, modalità, device,
hash dei tre file di codice e versioni/runtime. Un commit di sola
documentazione è tracciato fra i commit di ripresa; codice differente blocca
la ripresa. La lettura di `checkpoint.pt` è limitata agli artefatti locali
della propria run.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python train_femnist_reconstructed.py run \
  --config configs/femnist_reconstructed.json \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/runs/seed-41 \
  --seed 41 --device cuda:1 --resume
```

I checkpoint sono salvati all'inizio e a fine round con flush/fsync e
sostituzione atomica. Contengono condivisi, encoder privati, encoder iniziale
per client non ancora attivi, partecipazioni, storico completo e stati
Python, NumPy legacy, PCG64 selezione, Torch CPU e Torch CUDA della GPU
selezionata. Il generatore dei batch è identificato dalla formula registrata
`seed*10000000 + (round-1)*10000 + client_index`; all'interno del round è
continuo fra le due fasi. Non servono optimizer persistenti al confine del
round, perché il protocollo li ricrea alla successiva partecipazione.

Un round interrotto viene rifatto dallo snapshot precedente, senza lasciare
aggiornamenti parziali nella run ripresa. `checkpoint.pt` è autorevole se
un'interruzione precede l'aggiornamento di `results.json`; la ripresa riconcilia
quest'ultimo. La checksum dello snapshot viene controllata quando corrisponde
al round del JSON. Si conserva un checkpoint corrente, invece di 200 copie.

- `results.json`: identificazione, commit e pulizia del tree, hash codice,
  configurazione, versioni, parametri, partecipazioni, loss per fase, passi,
  campioni processati, tempi, metriche della finestra e stato della run.
- `checkpoint.pt`: stato riprendibile, incluse componenti private e RNG.
- `timings.jsonl`: costi di serializzazione, hash e scrittura degli snapshot;
  operazioni di refresh separate dai round.

Le loss sono medie sui minibatch; il batch finale incompleto resta incluso.
I passi Adam dell'encoder comprendono entrambe le fasi, quelli del decoder e
del classificatore soltanto la classificazione. Sono registrati byte degli
optimizer locali, tempi per client/fase/aggregazione/test e volume teorico dei
due trasferimenti dei condivisi. I picchi RAM e CUDA si riferiscono al processo
della sessione; una ripresa non sostituisce i tempi già salvati dei round.
Valutazione e profilazione non consumano i generatori di training.

## Verifiche e smoke GPU eseguiti

Suite completa nell'ambiente esistente:

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/schroeder/miniconda3/envs/general_ml/bin/python -m pytest tests -q --durations=5
```

**29 test passati in 5.08 s**: 7 preesistenti e 22 nuovi. I nuovi test coprono
resti e parità, interi oltre la precisione Float64, capacità e classi assenti,
quote sufficienti invariate, ricostruzione identica e riuso della partizione,
hash/disgiunzione/dimensioni, gradienti delle due fasi, passi Adam continui,
persistenza e reset degli optimizer, aggregazione uniforme dei soli condivisi,
metriche, finestra inclusiva 191–200, lock/identità e ripresa. Su un caso
sintetico di tre round, ripresa e training ininterrotto producono gli stessi
encoder, condivisi, partecipanti, metriche e stati dei generatori verificati.
Un test colloca lo smoke dentro una finta finestra di valutazione e vieta
qualsiasi caricamento del test, inclusa la profilazione.

GPU ricontrollate immediatamente prima dello smoke: GPU 1, RTX 6000 Ada,
18 MiB usati e utilizzo 0%; GPU 0 lasciata al suo stato preesistente.
Il comando specifico è stato autorizzato fuori dal sandbox, che non accede
al driver. Python 3.12, Torch 2.5.1+cu124, Float32, AMP/TF32 disattivati,
algoritmi deterministici, quattro thread Torch, nessun worker del loader.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python train_femnist_reconstructed.py smoke \
  --config configs/femnist_reconstructed.json \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/smoke/gpu1-seed41-v1 \
  --seed 41 --device cuda:1 --rounds 3 --profile-training-inference
```

Lo smoke usa i valori di training definitivi ma si ferma a tre round, in una
directory separata. Esito: **completato**, nessuna accuratezza calcolata e
nessun test definitivo valutato. La profilazione forward usa soltanto il
primo batch di training dei 150 client (1,500 immagini), senza argmax o loss.

| Misura | Valore |
|---|---:|
| Tempo del processo, timer interno | 44.90 s |
| Calcolo round 1 / 2 / 3, senza checkpoint | 16.140 / 13.713 / 11.801 s |
| I/O snapshot round 1 / 2 / 3 | 0.050 / 0.066 / 0.093 s |
| Warm-up: passi / tempo | 825 / 5.164 s |
| Classificazione: passi / tempo | 4125 / 34.690 s |
| Encoder privati aggiornati almeno una volta | 41 |
| Picco CUDA allocato da Torch | 78.64 MiB |
| Picco CUDA riservato da Torch | 96.00 MiB |
| Picco RAM del processo, RSS | 1325.05 MiB |
| Forward proxy training, 150 batch | 0.241 s |
| Caricamento encoder nel proxy, 150 client | 0.139 s |

Il timer interno include avvio/import, audit, training, snapshot e proxy,
escludendo l'attesa di approvazione e gli ultimi istanti di stampa/chiusura.
La memoria CUDA riportata è quella dell'allocator Torch, non un picco NVML
dell'intero contesto/driver. Lo smoke è precedente al commit finale: il tree
parte da `939c584` e i risultati registrano i SHA-256 del codice effettivamente
eseguito, controllati prima del commit. Non si sono cambiati iperparametri
in base alle loss o alle accuratezze.

## Capacità e prima stima delle cinque run

Classificatore: **550,346** parametri; encoder privato: **71,792** per client;
decoder condiviso: **44,961**. La pipeline di un client ha **667,099** parametri,
quindi non ha la capacità totale della sola MLP di riferimento. Gli encoder
dei 150 client richiedono logicamente 43,075,200 byte (41.08 MiB) di soli
pesi Float32, oltre ai condivisi e all'overhead della simulazione.

Il volume teorico per round è 71,436,840 byte per upload+download di
classificatore e decoder dei 15 client; gli encoder non sono trasmessi.
Su 200 round sono circa 14.29 GB/run, senza metadati o overhead di rete.
La simulazione non misura una comunicazione distribuita reale.

La stima usa i tempi per passo delle due fasi, separatamente. Dal manifesto
`sum_i ceil(n_train[i]/10)=2339`; il campionamento uniforme di 15 su 150
client dà **46,780 passi di warm-up** e **233,900 passi di classificazione**
attesi per run. I tre round dello smoke hanno 825 passi warm-up, una
numerosità superiore alla media attesa: la semplice moltiplicazione dei
tempi di round non tiene conto di questa differenza.

Si aggiungono overhead medi di creazione/copia/aggregazione, I/O scalato
alla dimensione stimata del checkpoint con tutti i 150 encoder (circa
45.93 MB), e una stima di inferenza ottenuta dal proxy **training**. Il test
definitivo avrebbe 342 batch e 150 caricamenti encoder per valutazione:
il proxy fornisce solo una stima di costo, senza consultarne le predizioni.

| Costo previsto per una run | Stima |
|---|---:|
| Training e overhead di calcolo | 2379.85 s |
| Checkpoint/snapshot | 57.85 s |
| Dieci valutazioni, proxy dal training | 6.89 s |
| Totale indicativo | **2444.59 s ≈ 40.7 min** |
| Cinque run sequenziali sulla stessa GPU | **≈ 3.40 ore GPU** |

Calibrando separatamente sui tre round, l'intervallo indicativo è 36.5–46.3
min/run, cioè **3.04–3.86 ore per cinque run**. Non è un intervallo di
confidenza. Per pianificare si possono considerare circa quattro ore,
da rivedere durante le future run usando i tempi effettivamente salvati.

Limiti: soltanto tre round e una GPU libera, cache/kernel iniziali, scelta
casuale dei client, checkpoint e storico crescenti, contesa hardware/I/O,
proxy forward su training che non misura loader, copie e conteggi esatti del
test. La RAM dello smoke non è il picco finale con 150 encoder. La stima
esclude interruzioni, replay di round, trasferimenti di rete reali e parallelismo
fra più GPU; non è una misura del costo delle baseline pubblicate.

I dettagli macchina e gli artefatti di verifica restano in `_local/`; si
versionano soltanto codice, test, configurazione e questa documentazione.
