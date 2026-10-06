# Refit della testa PathMNIST in simulazione federata

**Federato: 75.462396%; centralizzato: 75.331476%**, differenza +0.130919 punti percentuali. Prima del refit: 39.619777%. Un solo seed 42, checkpoint originale round 50; nessuna SD tra seed.

**Esito dei controlli:** loss e gradiente allo stesso punto del riferimento passano le tolleranze dichiarate. I punti terminali dei due L-BFGS troncati a 100 iterazioni **non sono numericamente equivalenti entro tutte quelle tolleranze**. Questo mancato accordo non viene nascosto né corretto aumentando le tolleranze dopo il test.

## Procedura e separazione dei dati

Fonte originale SHA256 fa56da01091debe9031a9acb2e3de0f34ce0f751d2c95d2d0d54c05f15697d1e; riferimento centralizzato SHA256 31019b0e1ef8594de1f7da196a6ce4b7e84afdc58fb4ba3d42fcf14b7fd9d4b8. Pesi originali, nessun checkpoint tuned. Solo 585 parametri fc.weight/fc.bias (64→9) ottimizzati. Encoder privati, decoder, corpo ResNet20V2 e tutti 57 buffer BN bit-identici alla fonte; fusione x+D(E_i(x)), nessuna ricalibrazione o penalità.

Dieci processi client separati, con cache/etichette delle sole proprie immagini training, 89.996 esempi complessivi. Nessuna matrice di feature pooled è costruita. Il server ottimizzatore invia solo la testa e riceve soltanto loss locale e 585 gradienti Float32; media uniforme dei client, indipendente dalle loro numerosità. Ready/done sono stati di controllo. Per la verifica il client scrive metadati scalari/conteggi sul disco locale; l’assemblatore del report legge questi metadati, senza acquisire feature, etichette o logits grezzi. Fonte/dati preinstallati, simulazione su un host: nessuna pretesa di privacy crittografica.

Stessa L-BFGS del centralizzato: LR 1, max_iter 100/max_eval 125, history 20, strong-Wolfe, tolerance_grad 1e-7, tolerance_change 1e-10, CE full-batch media uniforme, penalità 0, Float32 senza AMP. Nessun tuning, nuovo candidato, sweep o selezione dopo il test. Tutte le feature sono estratte da modelli eval/no_grad con encoder proprietario.

Effettive 100 iterazioni, 107 valutazioni obiettivo/gradiente; il centralizzato ne aveva 100 e105. CE federata 1.783756256 → 0.595975816; CE terminale centralizzata 0.597145498. La testa è salvata e congelata prima di leggere il test. Dopo il test il checkpoint viene arricchito solo con RNG/contatori, senza cambiare pesi.

## Loss e gradienti

Il vecchio optimizer centralizzato conserva il gradiente precedente all’ultimo aggiornamento: non è il gradiente finale. Il punto è ricostruito come theta_pre=theta_final−t*d, con piccolo roundoff Float32. Valutandolo presso tutti i client si confronta il gradiente aggregato con quello realmente archiviato dal centralizzato. La loss iniziale e quella alla testa centralizzata finale sono confrontate con i valori archiviati. Non si riesegue alcun refit centralizzato.

| Controllo | Differenza assoluta | Tolleranza | Esito |
|---|---:|---|---|
| CE iniziale | 0.000000000e+00 | atol 2e-6 + rtol 2e-6 | passa |
| CE alla stessa testa centrale finale | 0.000000000e+00 | atol 2e-6 + rtol 2e-6 | passa |
| CE al punto centrale precedente | 0.000000000e+00 | atol 2e-6 + rtol 2e-6 | passa |
| Gradiente al punto precedente, max per-entry | 1.126318239e-08 | atol 1e-5 + rtol 1e-4 | passa |
| CE ai due punti terminali | 1.169681549e-03 | atol 2e-4 + rtol 1e-4 | non passa |
| Gradienti ai due punti terminali, max per-entry | 3.032219451e-03 | atol 2e-4 + rtol 1e-2 | non passa |

Errore L2 sul gradiente archiviato: 5.237660856e-08, relativo 6.080592766e-06. Ai terminali, norma gradiente federato 9.241770046e-03, riferimento valutato dalla media dei client 1.568172909e-02; sono gradienti in punti diversi, non un controllo della stessa derivata.

## Logits e accuratezza

| Split | Differenza max logits | RMS | Predizioni discordanti |
|---|---:|---:|---:|
| Training | 1.519005 | 0.335003 | 2060/89996 |
| Test | 1.483591 | 0.347266 | 1697/71800 |

La tolleranza logits era 0.02+0.002*abs(reference): non rispettata. Disaccordo test 2.363510%, oltre lo 0.1% dichiarato. Accuracy differisce 0.130919pp, oltre la tolleranza 0.05pp; risultati vicini in performance non implicano logits/predizioni identici.

| Client | Originale corrette | Centrale corrette | Federato corrette | Centrale (%) | Federato (%) |
|---:|---:|---:|---:|---:|---:|
| 0 | 2659 | 5437 | 5438 | 75.724234 | 75.738162 |
| 1 | 2725 | 5427 | 5463 | 75.584958 | 76.086351 |
| 2 | 2717 | 5432 | 5455 | 75.654596 | 75.974930 |
| 3 | 3066 | 5379 | 5376 | 74.916435 | 74.874652 |
| 4 | 3070 | 5384 | 5379 | 74.986072 | 74.916435 |
| 5 | 2943 | 5409 | 5401 | 75.334262 | 75.222841 |
| 6 | 3069 | 5381 | 5376 | 74.944290 | 74.874652 |
| 7 | 2862 | 5447 | 5466 | 75.863510 | 76.128134 |
| 8 | 2688 | 5415 | 5445 | 75.417827 | 75.835655 |
| 9 | 2648 | 5377 | 5383 | 74.888579 | 74.972145 |

Totali: originale 28447, centrale 54088, federato 54182 su 71.800 predizioni. Media uniforme di tutte 10 pipeline sui medesimi 7.180 test ufficiali; non 71.800 immagini indipendenti o ensemble. I conteggi originale e centralizzato riproducono esattamente gli archivi precedenti.

## Interpretazione numerica

La media federata implementa lo stesso obiettivo/gradiente matematico sulle rappresentazioni congelate: il confronto con la derivata archiviata lo verifica direttamente nel punto disponibile. Float32 cambia l’ordine delle riduzioni rispetto al backward sulla matrice pooled del vecchio refit. Differenze iniziali piccole possono influire sulla ricerca di linea e sugli aggiornamenti di curvatura L-BFGS; 107 valutazioni contro 105 e storie CE diverse sono coerenti con tale sensibilità. Non si afferma uguaglianza bitwise o convergenza al medesimo ottimo: entrambi si fermano al budget 100, con gradienti ben sopra 1e-7.

Il beneficio qualitativo del refit rispetto al 39.62% originale è riprodotto senza trasferire le feature al server, ma la corrispondenza numerica terminale più stretta **non è dimostrata**. Non è un confronto tra baseline, una media di cinque seed o una modifica al protocollo del paper: resta una fase supervised aggiuntiva sulla sola testa. Nessun test ha guidato impostazioni o ripetizioni.

## Aggregazioni e byte

107 aggregazioni per ottimizzazione, incluse tutte le strong-Wolfe closure, più 3 aggregazioni diagnostiche train-only: totale 110. Tutti 10 client in ogni aggregazione. Ogni request 2341 bytes (opcode+585 Float32); ogni response 2345 bytes (opcode+loss+585 gradienti).

| Categoria | Payload down | Payload up | Totale payload | Con framing Pipe |
|---|---:|---:|---:|---:|
| control | 0 | 10 | 10 | 50 |
| optimization | 2504870 | 2509150 | 5014020 | 5022580 |
| verification | 70230 | 70350 | 140580 | 140820 |
| inference_audit | 46810 | 10 | 46820 | 46900 |

**Totale 5,201,430 bytes applicativi; 5,210,350 bytes con framing Pipe 4 bytes/messaggio.** Solo ottimizzazione: 5,014,020 bytes applicativi. Conteggi verificati da messaggi e 107 closure. Inclusi broadcast finali per confronto e stati ready/done; esclusi dataset/body preinstallati, avvio processi OS, accessi/scritture dei file locali e overhead TCP/TLS non simulato. Non è costo comunicativo del training originale completo.

## Tempi, checkpoint e verifiche

GPU cuda:1, dieci client più server sulla stessa GPU. Sessione 13.819s; processo 18.965s, incluse import/init. Preparazione client+feature 7.605s, optimizer 1.845s, audit d’inferenza 2.995s. Picco GPU1 campionato ogni 0.5 s: 9077MiB, include contesti CUDA/processi; può perdere picchi più brevi. Picchi Torch e RSS per client in results.json. GPU0 resta 451MiB con il processo preesistente; nessun processo altrui interrotto.

Checkpoint completo: `_local/pathmnist_federated_head/seed-42/final.pt`. Fonte round 0 in training-initial.pt e round 50 in before.pt; final contiene classifier, decoder, tutti 10 encoder/copie client, 57 BN buffer originali, 20 optimizer originali,scaler, RNG e indici invariati; aggiunge optimizer L-BFGS, server/client phase RNG,config/codice/contatori. Ogni client conserva anche final.pt e private_features.pt nella propria cartella. Nessuna feature/etichetta o checkpoint è pubblicato.

SHA256 finale `2f704047f68034421f3d87e4d96d0d3b1450e790f02f64b7ece63ee168aa66d4`. Verifica del checkpoint completo e tutti gli stati finiti passata; 22 artefatti della campagna precedente invariati. Suite: 360 test passati in 87.53s, inclusi 7 nuovi test. Separare successo dei controlli di protocollo dalla mancata equivalenza numerica terminale.

Codice/configurazione congelati nel commit `8f1030c87d136f7c0cb2b60508b3ae3296ba1842`; sorgenti identificati in artifacts/results.json.gz. Artefatti numerici lossless, gradienti aggregati e traccia CE in artifacts/; hash e percorsi privati in summary.json. Il manoscritto e gli altri esperimenti non sono stati modificati.

Comando eseguito (exit 0):

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_federated_head/run.py --config research/pathmnist_federated_head/config.json --output _local/pathmnist_federated_head/seed-42 --device cuda:1
```

Verifica archivio:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -m research.pathmnist_federated_head.archive verify
```
