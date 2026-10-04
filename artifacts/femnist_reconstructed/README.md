# Cinque run FusedSpaceFed sul FEMNIST ricostruito

Campagna completata sulla macchina di lavoro thanos con l'ambiente `general_ml`
esistente: cinque processi sequenziali, seed 41–45, GPU `cuda:1`, 200 round
ciascuno, tutti con codice d'uscita 0. Nessuna ripresa o interruzione delle
run finali. I risultati sono nostri (`result_origin: ours`).

Il [protocollo](../../docs/femnist_fedrep_protocol.md) descrive provenienza e
discrepanze del confronto con i risultati pubblicati nella colonna FEMNIST
(150,3) della Tabella 1 di FedRep. Questa partizione è una nostra ricostruzione
con riduzione delle quote insufficienti; le baseline non sono state rieseguite.
Il [CSV originale](../../docs/femnist_fedrep_reference.csv) rimane invariato.

## Configurazione e correzione numerica

Codice eseguito: `cd433abf545647d64d1ec94db86d998de38e3401` (`fix: stabilize reconstructed FEMNIST optimizer updates`).
La [configurazione](../../configs/femnist_reconstructed.json) è comune ai cinque
seed; il manifesto include la copia completa e gli hash.

Il precedente tentativo del seed 41, codice `53302c9c02ee2f76b867860fc44cc9a636b6e99f`,
si era arrestato durante il round 53, prima di qualsiasi valutazione del test.
La diagnosi sui soli dati di training ha riprodotto logits non finiti sul
client `f_00076` dopo 14 update SGD; nell'ultimo update finito la norma del
gradiente SGD era circa `1.99e32`, mentre ingresso e ricostruzione erano
ancora finiti. Si applica ora clipping
della norma L2 dei gradienti a **1.0**, separatamente per ciascun optimizer,
prima dell'update, e si rifiutano output, gradienti o aggregati non finiti.
Si salvano norme e frequenze del clipping. Soglia fissata una volta sulla
diagnosi del training; nessuna scelta basata sul test e nessuno sweep.
Il replay corretto dello stesso round ha completato tutti i 15 client.
La suite completa, inclusi i test preesistenti, ha dato **35 passati in 7.36 s**.

Le run archiviate sono tutte ripartite dall'inizializzazione del proprio seed,
senza usare il checkpoint fallito, lo smoke o il replay. Il metodo conserva
encoder privato persistente, decoder e classificatore condivisi, fusione
additiva, warm-up del solo encoder e fase di classificazione con E/D/C.
MLP 784→512→256→64→10, UNetSmallAE con dz=64, batch 10, warm-up 1 epoca,
classificazione 5 epoche, 15 client/round, aggregazione uniforme. Adam 0.001
per E/D e SGD 0.01 con momentum 0.5 per C, reset a ogni partecipazione.
Float32, AMP/TF32 disattivati, runtime deterministico. Il clipping modifica
gli aggiornamenti; architetture, learning rate, dati e altre scelte restano
quelle approvate. Codice, configurazione e gli altri file preesistenti sono
rimasti identici durante tutte le run e fino all'audit prima dell'archiviazione.

## Risultati verificati

Per ogni seed si riporta la media delle dieci valutazioni dei round **191–200**.
Ogni valutazione comprende tutti i **150 client** e **2603 esempi** dei loro
test locali, con condivisi correnti ed encoder persistenti, senza adattamento.
La metrica primaria pesa i campioni; la seconda assegna ugual peso ai client.
Nessun miglior checkpoint, scelta del miglior seed, tuning sul test o arresto
anticipato. Valori in percentuale, deviazioni standard in punti percentuali.

| Seed | Pesata per campioni (%) | Uniforme fra client (%) | Durata processo (s) | Picco RSS (MiB) |
|---|---:|---:|---:|---:|
| 41 | 79.52746831 | 80.72938388 | 2447.090 | 1544.09 |
| 42 | 62.74683058 | 62.73142607 | 2567.137 | 1540.14 |
| 43 | 63.66500192 | 63.22923274 | 2619.215 | 1541.34 |
| 44 | 67.80253554 | 70.15349286 | 2596.344 | 1531.01 |
| 45 | 49.94237418 | 49.35874456 | 2700.235 | 1529.83 |
| Media | 64.73684211 | 65.24045602 | | |
| SD campionaria, ddof=1 | 10.63186682 | 11.47403402 | | |

Per ogni round: `weighted = 100 * sum(correct_i) / sum(total_i)` e
`uniform = mean_i(100 * correct_i / total_i)`. Per seed si fa la media
aritmetica dei dieci round; media e SD finali usano i cinque valori per seed.
Non si tratta di 50 campioni indipendenti: le dieci valutazioni riusano lo
stesso test congelato. La variabilità fra seed è ampia; tutti i cinque valori
restano inclusi. Stabilità numerica non implica convergenza uniforme.

`verification.json` registra un audit indipendente dei conteggi e delle
metriche (tolleranza assoluta 1e-10), identità, finestre, selezioni PCG64,
passi, partecipazioni, 150 encoder privati, checkpoint/RNG e finitezza di
1834 tensori del modello per seed. La partizione, gli array e l'archivio
sono stati verificati; anche quote e split sono stati ricostruiti senza
modificare i dati. L'audit non ha effettuato nuove valutazioni del modello.
Il sottocomando `summarize` esistente ha prodotto `summary.json`, controllato
contro le metriche ricostruite dai conteggi.

## Tempi e risorse

Somma delle durate dei cinque processi: **12930.021 s**
(3.5917 ore). Durata del sequenziatore,
incluse verifiche iniziali e fra run: **12930.956 s**.
Il precedente tentativo fallito (601.131 s) e la diagnosi/replay sono costi
separati e non entrano nel totale della campagna finale.

I timer esterni comprendono avvio/uscita e polling; i timer interni del runner
sono riportati separatamente nel manifesto. Ogni run ha una sola sessione.
I tempi per fase, client, aggregazione e valutazione sono contenuti nei tempi
di calcolo; i tempi checkpoint sono contenuti nei tempi I/O. Non sommare
timer annidati. `timings.jsonl` conserva tutti i 201 snapshot, incluso round 0.

Per ciascuna run: picchi CUDA Torch **78.6440 MiB allocati / 96.0000 MiB
riservati**, oltre al contesto/driver non misurato da questi contatori.
Picco RSS massimo osservato: **1544.0898 MiB**. GPU RTX 6000 Ada Generation;
Python 3.12.7, NumPy 2.0.2, Torch 2.5.1+cu124, CUDA 12.4, cuDNN 9.1.0.
La GPU è stata controllata libera prima di ciascuna run, senza interrompere
altri lavori. Nessun pacchetto o ambiente modificato.

Parametri: C 550346, E privato 71792/client, D condiviso 44961, pipeline
667099. Gli encoder dei 150 client occupano logicamente 43075200 byte Float32.
La capacità totale e i costi includono il percorso encoder-decoder, oltre
alla MLP. Comunicazione teorica dei condivisi: 71436840 byte/round,
14287368000 byte/run, senza overhead; non è traffico di rete misurato.
Passi effettivi, campioni processati, loss per minibatch, frequenze del
clipping, tempi e byte degli optimizer/snapshot sono in `verification.json`
e nei risultati originali compressi. Le loss di warm-up possono restare
elevate e il clipping interviene spesso: non si deduce qualità di
ricostruzione o ottimalità della soglia dalla sola finitezza.

## Comandi eseguiti e contenuto

Lo script locale `_local/femnist_reconstructed/logs/run_five_clipped.py`
ha eseguito questo comando per ogni seed 41, 42, 43, 44, 45, in ordine,
cambiando solo seed e directory; stdout/stderr separati con exit code salvati:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python train_femnist_reconstructed.py run \
  --config configs/femnist_reconstructed.json \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/runs/clipped-v1/seed-41 \
  --seed 41 --device cuda:1
```

Comando di sintesi eseguito sulle cinque directory originali:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python train_femnist_reconstructed.py summarize \
  --config configs/femnist_reconstructed.json \
  --runs _local/femnist_reconstructed/runs/clipped-v1/seed-41 \
         _local/femnist_reconstructed/runs/clipped-v1/seed-42 \
         _local/femnist_reconstructed/runs/clipped-v1/seed-43 \
         _local/femnist_reconstructed/runs/clipped-v1/seed-44 \
         _local/femnist_reconstructed/runs/clipped-v1/seed-45 \
  --output _local/femnist_reconstructed/logs/clipped-v1/summary.json
```

- `summary.json`: output originale del sottocomando, senza modifiche.
- `seed-41/` … `seed-45/`: `results.json.gz` lossless e `timings.jsonl`
  originali; il JSON contiene tutti i conteggi client/round della finestra.
- `verification.json`: esito, identità e costi derivati dall'audit.
- `manifest.json`: seed, commit eseguito, configurazione, hash di partizione,
  sorgente, file archiviati e JSON decompressi, tempi e memoria.

I SHA-256 dei file elencati si riferiscono ai byte, salvo gli hash canonici
di configurazione/partizione (JSON ordinato e compatto). Il manifesto non
include il proprio hash, per evitare autoreferenza. Gli hash dei checkpoint
locali servono alla tracciabilità: i checkpoint non sono nel pacchetto.
Compressione gzip senza nome file o timestamp, verificata byte per byte
dopo decompressione. I risultati e i log completi, i checkpoint e il tentativo
fallito rimangono in `_local/`; dati e immagini non sono pubblicati.

## Limiti del confronto

Non è una replica esatta di FedRep né una campagna comune con le baseline.
Split e seed originali degli autori non sono recuperabili dalle fonti
conservate; quote ridotte per f/i/j alterano numerosità e proporzioni in
105 client. Totali congelati: 25339 esempi, 22736 train e 2603 test, seed
dati 20261003; nessun riuso globale. Restano le discrepanze articolo/codice
documentate, differenze di protocollo/metrica e il costo aggiuntivo E/D.
Le baseline pubblicate non riportano incertezze nella colonna pertinente:
non si possono costruire confronti appaiati o attribuire loro la nostra SD.
Nessuna baseline, ablation o altro esperimento è stato aggiunto. Il
manoscritto, le figure e i risultati precedenti rimangono invariati.
