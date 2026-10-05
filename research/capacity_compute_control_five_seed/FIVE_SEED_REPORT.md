# Digits: controllo capacità/FLOPs su cinque seed

Sono stati aggiunti solo i seed FedAvg 45 e 46, da zero per 300 round, con esattamente la configurazione dei seed 42–44. Tutte le run precedenti sono conservate. Il confronto usa i cinque FusedSpaceFed calibrati già disponibili, senza nuovi tuning o run Fused.

Accuratezze in percentuale; media e SD campionaria tra seed (`ddof=1`). Metrica primaria: media uniforme tra i cinque domini; la metrica pesata per campioni è riportata separatamente.

| Seed | FedAvg uniforme | Fused uniforme | Δ Fused−FedAvg (pp) | FedAvg pesata | Fused pesata |
|---:|---:|---:|---:|---:|---:|
| 42 | 82.266969 | 84.669219 | +2.402250 | 80.199174 | 83.588798 |
| 43 | 82.149311 | 86.303678 | +4.154367 | 79.898854 | 85.064640 |
| 44 | 82.246221 | 85.322698 | +3.076477 | 79.682596 | 83.883017 |
| 45 | 81.775514 | 86.758720 | +4.983207 | 79.416849 | 85.297168 |
| 46 | 81.886952 | 86.257479 | +4.370527 | 79.202625 | 84.739914 |

| Metrica | FedAvg: media ± SD | Fused: media ± SD | Δ abbinato: media ± SD (pp) |
|---|---:|---:|---:|
| uniform_domain_accuracy_percent | 82.064993 ± 0.221497 | 85.862359 ± 0.846930 | 3.797365 ± 1.040218 |
| sample_weighted_accuracy_percent | 79.680020 ± 0.391976 | 84.514708 ± 0.745273 | 4.834688 ± 1.023123 |

| Dominio | FedAvg: media ± SD | Fused: media ± SD | Δ medio (pp) |
|---|---:|---:|---:|
| MNIST | 95.932857 ± 0.235747 | 96.947143 ± 0.239823 | +1.014286 |
| SVHN | 61.569141 ± 1.434203 | 67.827576 ± 2.876878 | +6.258435 |
| USPS | 95.935484 ± 0.253886 | 96.666667 ± 0.281938 | +0.731183 |
| SynthDigits | 81.308914 ± 0.540151 | 86.316123 ± 0.496394 | +5.007209 |
| MNIST-M | 75.578571 ± 0.640133 | 81.554286 ± 0.773321 | +5.975714 |

I valori di ogni dominio per ogni seed sono in `artifacts/accuracy.csv` e `artifacts/summary.json`; risultati, conteggi corrette/totali e matrici di confusione permettono la ricostruzione esatta.

## Protocollo e risorse

Stesso Digits bilanciato: MNIST, SVHN, USPS, SynthDigits, MNIST-M; 743 training/client, 5 client partecipanti a ogni round, aggregazione uniforme, batch 32, Float32 senza AMP/TF32, SGD senza momentum/weight decay e reset a ogni partecipazione. Test una sola volta al round 300, senza adattamento né selezione del checkpoint. Conteggi test: 14.000 / 19.858 / 1.860 / 97.791 / 14.000, totale 147.509. Seed abbinati 42–46; una sola partizione congelata.

FedAvg: CNN con larghezza del primo fully connected 2065 anziché 2048, 14.334.589 parametri. Fused: 14.336.285 parametri attivi/client (classifier 14.219.210 + encoder 72.080 + decoder 44.995), scarto FedAvg −0,011830%. Encoder privato persistente, decoder/classifier condivisi, fusione additiva, 1 epoca warm-up + 1 epoca classificazione. I campi AE/warm-up rimasti nel template JSON FedAvg sono inutilizzati dal suo percorso: non esegue un autoencoder.

FedAvg mantiene il budget cumulativo esistente, che include il warm-up Fused: copre prima tutti i 743 esempi e poi aggiunge minibatch CE con riporto intero del residuo. Per run: FedAvg 551.896.914.235.800 FLOPs contabilizzati, Fused 551.898.005.448.000; deficit 0,000197720%. Dense-only: 547.381.518.701.000 vs 549.229.594.368.000. FedAvg 63.000 passi CE e 1.956.535 esposizioni CE; Fused 36.000 warm-up + 36.000 CE e 1.114.500 esposizioni CE.

La convenzione conta convolution/matmul forward/backward misurati e clipping/optimizer semantici (FMA=2), esclude BN/ReLU/pool/loss, copie, aggregazione e overhead; non è misura di energia o istruzioni hardware. Matching approssimato della capacità e della computazione non rende equivalenti geometria, BN, personalizzazione o esposizioni supervisionate.

## Tuning: procedure diverse, nessun nuovo tuning

FedAvg conserva la selezione originaria: 9 candidati, 60 round ciascuno, seed 142; griglia LR C {0,005; 0,01; 0,02} × clip {0,5; 1; 2}. Validation derivata esclusivamente dal training: 594 fit e 149 validation/client, stratificata e congelata (seed 20261005). Selezione della media uniforme al round 60; LR 0,02 e clip 2. Il controllo originario Fused aveva lo stesso numero di tentativi, orizzonte e budget per round e resta intatto.

I cinque Fused qui confrontati provengono dalla campagna successiva: 10 screening di 120 round, seed 142, seguiti da 4 candidati × 2 seed (142/143) × 300 round di conferma. LR C {0,02; 0,05; 0,1}, LR AE {0,0001; 0,0003; 0,001}, clip {2; 5; 10}, L9 con riferimento pilota aggiuntivo; top-2 più riferimenti obbligatori. Stessa validation. Selezione della media a round 300 sui due seed di conferma: LR C 0,1, LR AE 0,0003, clip 10. Configurazione congelata prima delle cinque run complete; nessuna selezione sul test né riapertura del tuning.

Costo del tuning: FedAvg 794,146,760,411,175 FLOPs; campagna Fused successiva 5,294,405,825,388,000 FLOPs (esclusi i finali). **Lo sforzo di tuning non è equivalente**, e l'intervallo LR di FedAvg non include quello scelto per Fused. Il vantaggio descrittivo di questa tabella non dimostra che il solo meccanismo Fused causi la differenza a tuning equivalente.

Il controllo precedente a sforzo comparabile rimane in `research/capacity_compute_control/CAPACITY_COMPUTE_REPORT.md`: tre seed, Fused 82,719265 ± 0,762178% vs FedAvg 82,220834 ± 0,062803%, differenza media +0,498431 pp. Non viene sostituito dalla presente campagna.

## Esecuzione e verifiche

| Metodo | Seed | GPU | Sessione totale (s) | CUDA alloc/res (MiB) | RSS (MiB) |
|---|---:|---|---:|---:|---:|
| FedAvg | 42 | cuda:0 | 831.734 | 273.697/338.000 | 1651.062 |
| FedAvg | 43 | cuda:1 | 833.484 | 273.697/338.000 | 1655.957 |
| FedAvg | 44 | cuda:0 | 845.273 | 273.697/338.000 | 1654.965 |
| FedAvg | 45 | cuda:1 | 833.907 | 273.697/338.000 | 1647.656 |
| FedAvg | 46 | cuda:0 | 807.439 | 273.697/338.000 | 1652.391 |
| FusedSpaceFed | 42 | cuda:1 | 1206.967 | 275.292/350.000 | 1873.043 |
| FusedSpaceFed | 43 | cuda:0 | 1052.526 | 275.292/350.000 | 1867.359 |
| FusedSpaceFed | 44 | cuda:1 | 1180.797 | 275.292/350.000 | 1871.551 |
| FusedSpaceFed | 45 | cuda:0 | 1032.688 | 275.292/350.000 | 1873.656 |
| FusedSpaceFed | 46 | cuda:1 | 1189.199 | 275.292/350.000 | 1868.711 |

Nuovi seed 45/46: 840.283 s di calendario in parallelo, 1652.456 s somma processi. Entrambi exit 0, una sessione da round 1 a 300, nessuna interruzione/ripresa. GPU 0 condivisa con il processo preesistente autorizzato; nessun processo altrui segnalato/interrotto. Picchi Torch per processo, esclusi contesto/driver e altri job. Tempi delle run riusate sono originali e non nuovo costo.

283 test versionati passati (`pytest -q tests research`, 75,77 s), inclusi 12 dell’estensione. Parità byte del driver scientifico originale salvo registrazione dei due seed e metadati; configurazioni identiche salvo seed; parità esatta di due round sintetici (pesi, RNG, carry, metriche) e ripresa; medie, SD campionaria e abbinamento dei cinque seed. Una collisione del nome del nuovo file test è stata corretta senza modifiche al training. Un pytest senza perimetro includeva suite locali di archivio: il controllo che richiede un interprete senza Torch falliva nella suite combinata; i suoi 22 test sono passati eseguiti isolatamente, senza cambiarli. Log della verifica versionata incluso nel pacchetto. Audit: 10 run complete, 300 round consecutivi, un test finale, 5 domini/147.509 esempi, valori finiti, seed/config/data/source hash, conteggi/confusioni, timing e budget ricostruiti.

Provenienza: nuovo training al base commit `76a338ccd3115b57a084b830067d00c341b0a435` con estensione isolata non ancora committata al lancio, identificata dagli SHA256 nei risultati e pubblicata nel commit dedicato di questo pacchetto. Le sorgenti scientifiche originali e i vecchi manifesti non sono cambiati. I record conservano i commit originali per tutte le run riusate.

Hash congelati: partizione `7a762ffb10da74e3f0dee9a6f519e6c4057e2a5b47995a0546ee5a871eb47b58`; validation `3dfc26c3fda17f266fb2ffa9899f23b4a5b50cd006742bb84d5b215e412a6943`; profilo FLOPs `164f242a968689c6c1a819fd353317c0a858b8c316a80b8244da0ed8c53f6a12`. Fonti/versioni e preprocessing dettagliati sono conservati nel report originale `research/feature_shift_digits/FEATURE_SHIFT_REPORT.md`; non sono stati ricercati o modificati di nuovo.

## Riproduzione e archivio

Comandi effettivi e codici di uscita sono in `artifacts/campaign.json`. Per una nuova directory di output:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/capacity_compute_control_five_seed/runner.py --config research/capacity_compute_control_five_seed/configs/FedAvg-seed-45.json --partition _local/feature_shift_digits/prepared --output _local/capacity_compute_control_five_seed/FedAvg-seed-45-reproduction --device cuda:1
/home/schroeder/miniconda3/envs/general_ml/bin/python research/capacity_compute_control_five_seed/runner.py --config research/capacity_compute_control_five_seed/configs/FedAvg-seed-46.json --partition _local/feature_shift_digits/prepared --output _local/capacity_compute_control_five_seed/FedAvg-seed-46-reproduction --device cuda:0
python research/capacity_compute_control_five_seed/audit.py verify
```

Archivio pubblico: risultati lossless e timings dei due nuovi seed, sintesi dei dieci risultati, CSV, receipt, configurazioni, test, driver isolato e manifesto con hash dei file nuovi e riferimenti immutabili ai risultati precedenti. Dataset, checkpoint e log completi restano in `_local/`; i checkpoint non sono pubblicati. Manoscritto, baseline pubblicate e altri esperimenti invariati.

Limiti: un setting e una partizione, cinque seed condizionati a configurazioni selezionate, tuning diverso e test del setting già osservati nelle campagne precedenti. Nessuna varianza inventata, selezione del seed migliore o inferenza di significatività. Questo FedAvg ampliato/con budget aumentato è un nostro controllo e non il FedAvg pubblicato nella tabella FedBN.
