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

La GPU è ricontrollata prima di ogni processo. Se occupata da altri processi,
la campagna si ferma conservando lo stato; nessun processo esterno è interrotto.
GPU 0 è esclusa finché ospita il lavoro di un altro utente. Non si installano
pacchetti, non si cambiano ambiente o limiti hardware.

## Cinque run dopo il congelamento

La configurazione selezionata sarà copiata e versionata come
`configs/femnist_reconstructed_calibrated.json`, con ricevuta in
`artifacts/femnist_calibration/selection.json`. Il runner accetta solamente
questa configurazione congelata oppure il precedente riferimento approvato.
La validazione protegge tutti i campi fuori dai tre iperparametri cercati.

Le nuove run definitive partono da zero su **tutto il training originale**,
seed **41–45**, 200 round. Il test locale originale è valutato soltanto nei
round **191–200**, senza adattamento, selezione di checkpoint o arresto
guidato dai risultati. Nessun checkpoint di calibrazione viene trasferito.
La configurazione resta congelata anche se l'esito è sfavorevole.

Il controller finale ricontrolla selezione congelata e hash del codice, usa
directory nuove e conserva exit code e durate di ogni processo:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python calibrate_femnist_reconstructed.py final \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/calibrated-v1 --device cuda:1
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

Costi attesi sulla sola GPU libera: circa 4–5 ore di calibrazione e
3,5–4 ore per le cinque run finali, soggetti a tempi effettivi, variabilità
dei client e contesa I/O. Non sono costi delle baseline pubblicate.
Il report conclusivo locale documenterà tentativi, scelta, risultati e costi;
gli artefatti numerici versionati saranno separati dall'archivio precedente.
