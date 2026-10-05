# Calibrazione FEMNIST con validation derivata dal training

Questa campagna seleziona soltanto iperparametri di ottimizzazione, mantenendo
architettura, dati e protocollo del benchmark ricostruito. Non è una replica
esatta di FedRep né un confronto controllato con baseline rieseguite.
Il manoscritto e gli esperimenti precedenti restano invariati.

## Decisioni fissate prima della ricerca

Il piano versionato è `configs/femnist_calibration_plan.json`.
La configurazione precedente, `configs/femnist_reconstructed.json`, resta
invariata ed è inclusa come riferimento: SGD lr=0.01, Adam lr=0.001 e
clipping L2=1.0. La scelta dei candidati si basa soltanto sui precedenti
tracciati di training, che mostrano frequente clipping e crescita dei
gradienti/loss di ricostruzione. Non usa accuratezze di test per progettare
o selezionare i tentativi.

| Candidato | LR classificatore | LR autoencoder | Clipping L2 |
|---|---:|---:|---:|
| 00-reference | 0.01 | 0.001 | 1.0 |
| 01-ae-3e-4 | 0.01 | 0.0003 | 1.0 |
| 02-ae-1e-4 | 0.01 | 0.0001 | 1.0 |
| 03-classifier-2e-2 | 0.02 | 0.0003 | 1.0 |
| 04-classifier-5e-3 | 0.005 | 0.0003 | 1.0 |
| 05-clip-2 | 0.01 | 0.0003 | 2.0 |

Momentum, decay, betas/epsilon, reset degli optimizer, warm-up 1,
classificazione 5, batch 10, 15 client/round, aggregazione uniforme,
precisione Float32, dz=64 e MLP 784→512→256→64→10 sono protetti.
Non si modifica il numero di epoche o la fusione additiva.

## Validation congelata

La partizione originale conserva SHA-256 canonico
`7a4c7614796a595d8a752aba5d7d275a18a2f6e542dfd338e8a381fcd5675734`.
Non si ricostruiscono client o dati e non si cambiano quote, preprocessing,
seed o split train/test. Si usa esclusivamente il training originale.

Per ogni client/classe si ordinano gli ID secondo SHA-256 della tupla canonica
`[20261004, client_id, label, origin_id]`, con ID come spareggio. Si riservano
`n // 5` esempi per validation; gli altri sono fit. L'ordine originale delle
righe viene conservato nelle due viste. Ogni classe deve restare non vuota
in entrambe: nessun fallback o nuovo sorteggio. Questo dà **18.374 fit e
4.362 validation**. Il manifesto locale conserva tutte le appartenenze,
statistiche, regola e hash dei file train e della vista.

`femnist_calibration_data.py` verifica il manifesto originale, l'unicità globale
degli ID e gli array di training. Non apre, mappa o calcola hash degli array
di test; `dataset(..., "test")` è rifiutato. Nessuna immagine viene riscritta.
La validation è identica per ogni candidato e seed e non consuma RNG di
training. Il manifesto dei dati originali resta byte per byte invariato.
SHA-256 canonico della vista fit/validation:
`23b0e2e40a822eed50c983b744fe5c1cf203d2f5db9f28ef57acc7dbaa5ad171`.

## Ricerca e selezione

1. Tutti i sei candidati: seed di calibrazione **141**, primi 80 round,
   punteggio medio sui round di validation **71–80**.
2. Promozione del riferimento e dei due migliori alternativi finiti;
   prosecuzione dei loro checkpoint fino a 200 round, punteggio **191–200**.
3. Gli stessi tre candidati da zero con seed **142**, 200 round,
   punteggio **191–200**. Nessuno stato del seed 141 è riusato nel seed 142.
4. Selezione mediante media delle due medie per seed dell'accuratezza validation
   pesata per campioni. La media uniforme client è registrata come secondaria.
   Nessuna scelta del checkpoint migliore o cambio di metrica; parità risolta
   per identificatore candidato crescente, con riferimento per primo.
5. Congelamento di un solo JSON e ricevuta di selezione prima di ogni nuovo test.

La validation usa condivisi correnti ed encoder privati persistenti di tutti
i 150 client, senza adattamento, con controllo degli output finiti e conteggi
corretti/totali. Le sue valutazioni non consumano i generatori di training.
I tentativi numericamente falliti sono conservati e resi ineleggibili;
un riferimento che non completa la conferma arresta la campagna.

Budget massimo: **1440 round di calibrazione**, non un nuovo sweep aperto.
Lo screening a 80 round limita il costo e può scartare configurazioni che
migliorerebbero più tardi: è un limite esplicito della ricerca. Il confronto
finale fra candidati promossi usa sempre l'orizzonte completo e due seed.
Il risultato può restare variabile: due seed e una sola validation non
certificano un ottimo globale o un guadagno sul test.

## Esecuzione e artefatti

Si usa `general_ml` esistente. Dalla radice, GPU 1 dopo controllo di disponibilità:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python calibrate_femnist_reconstructed.py search \
  --plan configs/femnist_calibration_plan.json \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/calibration-v1 --device cuda:1
```

La directory deve essere assente alla prima esecuzione; `--resume` continua la
stessa campagna. Un lock impedisce due controller sullo stesso output.
Ogni trial usa checkpoint e RNG del runner esistente, con identità che include
piano, config, codice, seed, device e hash della vista fit/validation.
Riprese di calibrazione e run definitive sono incompatibili.

Sono conservati piano, config di ogni candidato, split, tutti gli output dei
trial, checkpoint correnti, log stdout/stderr separati, exit code, durata di
ogni processo/sessione, picchi di memoria e snapshot JSON per tentativo.
La scelta dello screening è in `promotion.json`; la scelta finale è in
`selection.json`, accompagnata da `selected_config.json`.
Non si sovrascrivono campagne, configurazioni o selezioni differenti.

La campagna originale si fermava prima di nuovi trial se la GPU era occupata.
Dopo autorizzazione esplicita, la prosecuzione ha usato un wrapper operativo
privato che richiede almeno 2048 MiB liberi su GPU 1, anche in condivisione.
Il wrapper sostituisce in memoria soltanto il controllo di ammissione;
i cinque sorgenti congelati, i comandi dei trial e i checkpoint restano
invariati. Il solo controller inattivo di attesa è stato fermato dopo conferma;
nessun training o processo esterno è stato interrotto. Le due queue e i loro
costi sono conservati separatamente, senza sommare intervalli sovrapposti.
Non si installano pacchetti né si cambia ambiente.

## Selezione verificata e congelata

La ricerca è conclusa: sei screening, sei conferme e **1440 round unici**.
L'audit indipendente ha ricostruito conteggi, metriche, promozione e scelta,
verificato identità e checkpoint tramite hash senza caricare modelli o test.

| Candidato confermato | Validation pesata, seed 141 | Seed 142 | Media primaria | Media uniforme client |
|---|---:|---:|---:|---:|
| 00-reference | 78.25080238% | 61.03392939% | 69.64236589% | 69.56901851% |
| 05-clip-2 | 91.72397983% | 84.75469968% | 88.23933975% | 88.88827014% |
| **03-classifier-2e-2** | 91.35259055% | 86.18523613% | **88.76891334%** | 89.58338122% |

Ogni valore per seed è la media dei round 191–200. È stata selezionata la
configurazione **SGD lr=0.02, Adam lr=0.0003, clipping L2=1.0**. Non si è
scelto il seed migliore: il candidato clipping 2 ha un seed 141 superiore,
ma una media primaria sui due seed inferiore. La ricevuta è congelata alle
**2026-10-05 00:16:20 UTC**, prima delle cinque nuove run definitive.
SHA-256 canonico della configurazione selezionata:
`b8ed1193867310fc0f3b9102bf27a5600913696c04f4a412e87b9dd167f64e6b`.

`artifacts/femnist_calibration/` contiene tutti i risultati numerici di
screening e conferma, snapshot dei tentativi, timing, piano, config, ricevuta,
statistiche della vista e manifesto dei file. Dataset, ID degli esempi,
checkpoint e log completi restano locali. La ricerca ha usato il commit
`641e117bd40f3620527a4452f91ee8ca461692aa`; questo valore non è riscritto.

Dopo la ricerca è stato corretto soltanto il verificatore delle partecipazioni:
ogni valutazione deve coincidere con il cumulativo fino al proprio round,
non con il vettore finale del round 200. La suite CPU completa passa **196
test**. `verification_transition.json` vincola ricevuta e config immutate,
i cinque hash prima/dopo e l'AST del modulo fuori dalle sole due funzioni
di verifica. I quattro sorgenti di modello/dati/training sono identici;
aggiornamenti del modello e selezione non cambiano. Il fix è nel commit
`187152a6fe7547a1ff660343749f5c4b44245be0`.

## Cinque run dopo il congelamento

La configurazione selezionata è copiata e versionata come
`configs/femnist_reconstructed_calibrated.json`, con ricevuta in
`artifacts/femnist_calibration/selection.json`. Il runner accetta solamente
questa configurazione congelata oppure il precedente riferimento approvato.
La validazione protegge tutti i campi fuori dai tre iperparametri cercati.

Le nuove run definitive partono da zero su **tutto il training originale**,
seed **41–45**, 200 round. Il test locale originale è valutato soltanto nei
round **191–200**, senza adattamento, selezione di checkpoint o arresto
guidato dai risultati. Nessun checkpoint di calibrazione viene trasferito.
La configurazione resta congelata anche se l'esito è sfavorevole.

Il controller finale nativo sequenziale resta disponibile. Per questa campagna
l'utente ha autorizzato due run indipendenti in parallelo, una per GPU.
Lo scheduler operativo privato chiama direttamente il runner originale;
usa un solo writer del manifesto e conserva exit code, log, durata,
checkpoint e identità per ogni processo. Dopo il commit del congelamento:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python \
  _local/femnist_reconstructed/calibration-preparation/dual_gpu_final_controller.py
```

La mappa è fissata prima di ogni nuovo test: seed **41/43/45 su cuda:1**,
seed **42/44 su cuda:0**. Entrambe sono RTX 6000 Ada con 49140 MiB.
GPU 0 può ospitare altri lavori; il gate richiede 2048 MiB liberi e non
modifica alcun processo esterno. Massimo due worker nostri, uno per GPU.
La policy separata registra mappa, hardware, sorgenti e hash dello scheduler;
il device nominale del coordinatore è cuda:1 e non descrive tutti i worker.
Il supporto operativo supera 28 test sintetici; l'audit indipendente con
dispositivi espliciti ne supera 21. Non sono prove di prestazione reali.

Ogni worker esegue il comando seguente, variando soltanto seed/output e il
device secondo la mappa; le directory delle singole run sono create dal runner:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python train_femnist_reconstructed.py run \
  --config configs/femnist_reconstructed_calibrated.json \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/calibrated-v1/runs/seed-41 \
  --seed 41 --device cuda:1
```

`--resume` verifica run già complete e riprende quelle interrotte sulla stessa
directory, senza importare stati di calibrazione. Un errore delle run finali
arresta il controller e non riapre la ricerca. La verifica ricostruisce tutte
le accuratezze dai conteggi, controlla 200 round, dieci valutazioni, 150 client,
2603 esempi e assenza di valori non finiti, poi usa `summarize_runs`.

Le metriche per seed sono le medie dei dieci round; sintesi su cinque seed
con SD campionaria `ddof=1`, primaria pesata per campioni e secondaria media
uniforme client. Tutti i seed saranno riportati. Risultati nuovi e precedenti
restano separati; il CSV delle baseline pubblicate non è modificato.

I processi di calibrazione hanno impiegato complessivamente **15757.39 s
(4.38 h)**; il compute dei round unici è **15182.57 s**. Il calendario include
anche le attese GPU e le riprese del controller: queste grandezze hanno scope
diversi e non vanno sommate. La stima prima delle nuove run è **120–135 minuti
calendario** su due GPU, soggetta a variabilità dei client e contesa CPU/I/O;
il lavoro totale resta 1000 round. Tempo calendario, somma dei processi,
attese per device e memoria saranno riportati separatamente.
Non sono costi delle baseline pubblicate.
Il report conclusivo locale documenterà tentativi, scelta, risultati e costi;
gli artefatti numerici versionati saranno separati dall'archivio precedente.

Il test originale era già stato usato nella precedente campagna: le nuove
run non costituiscono un test su dati mai osservati nella storia del progetto.
La selezione corrente usa soltanto validation derivata dal training; il suo
vantaggio sulla validation non certifica un miglioramento sul test.
