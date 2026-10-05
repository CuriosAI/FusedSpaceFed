# Controllo minimale di capacità e calcolo — Digits

In questo controllo FusedSpaceFed ottiene una media maggiore di FedAvg ampliato e con budget abbinato. La differenza primaria è **+0.498431 punti percentuali**. È un confronto diretto fra nostri metodi, distinto dal precedente confronto con baseline pubblicate. Non dimostra che i vantaggi in altri setting dipendano dai soli parametri o dalla sola computazione.

Il seed 42 favorisce FedAvg; 43 e 44 favoriscono FusedSpaceFed. Il delta medio è piccolo rispetto alla sua SD fra seed: tre ripetizioni non sostengono una conclusione causale robusta sull'origine dei vantaggi. Questo controllo indica un residuo vantaggio medio descrittivo con risorse abbinate, con eccezioni visibili; non certifica superiorità generale.

| Seed | Fused uniforme % | FedAvg uniforme % | Differenza pp | Fused pesata % | FedAvg pesata % |
| --- | --- | --- | --- | --- | --- |
| 42 | 81.839397 | 82.266969 | -0.427572 | 79.582941 | 80.199174 |
| 43 | 83.176185 | 82.149311 | +1.026874 | 81.176064 | 79.898854 |
| 44 | 83.142213 | 82.246221 | +0.895992 | 81.316394 | 79.682596 |

| Metrica | Fused media ± SD | FedAvg media ± SD | Delta abbinato media ± SD |
| --- | --- | --- | --- |
| uniform_domain_accuracy_percent | 82.719265 ± 0.762178 | 82.220834 ± 0.062803 | +0.498431 ± 0.804608 |
| sample_weighted_accuracy_percent | 80.691800 ± 0.962860 | 79.926875 ± 0.259427 | +0.764925 ± 1.209334 |

Media e SD campionaria fra tre seed prefissati, ddof=1. Primaria: media uniforme dei cinque domini, dichiarata prima della calibrazione. Conteggi/confusioni ricostruiscono ogni valore; nessun miglior seed o checkpoint. La media pesata è dominata dal test SynthDigits.

| Dominio | Fused media ± SD | FedAvg media ± SD |
| --- | --- | --- |
| MNIST | 96.388095 ± 0.224214 | 95.947619 ± 0.096451 |
| SVHN | 63.035888 ± 2.062086 | 62.302347 ± 1.385272 |
| USPS | 95.931900 ± 0.062081 | 95.860215 ± 0.268817 |
| SynthDigits | 82.440443 ± 0.916377 | 81.551131 ± 0.527525 |
| MNIST-M | 75.800000 ± 0.715249 | 75.442857 ± 0.568609 |

## Capacità e budget effettivo

Scenario scelto per costo e disponibilità: Digits bilanciato già congelato, 743 immagini/client, cinque client, 300 round, batch massimo 32, tutti i client e BN condivisi, test completo al solo round 300 senza adattamento. Stessa partizione, preprocessing, valutazione e seed 42–44 per entrambi.

Fused attivi/client: **14,336,285** = C 14,219,210 + E 72,080 + D 44,995. FedAvg: **14,334,589**, CNN con fc1 2065 anziché 2048, scarto **-0.011830%**. Fused memorizza globalmente C+D+5E = 14,624,605 parametri: capacità attiva abbinata non significa stesso totale memorizzato o stessa memoria.

| Metodo/seed | FLOPs contate TF | FLOPs dense TF | Warm-up passi | CE passi | Esposizioni CE |
| --- | --- | --- | --- | --- | --- |
| FusedSpaceFed/42 | 551.898005448 | 549.229594368 | 36000 | 36000 | 1114500 |
| FusedSpaceFed/43 | 551.898005448 | 549.229594368 | 36000 | 36000 | 1114500 |
| FusedSpaceFed/44 | 551.898005448 | 549.229594368 | 36000 | 36000 | 1114500 |
| FedAvg/42 | 551.896914236 | 547.381518701 | 0 | 63000 | 1956535 |
| FedAvg/43 | 551.896914236 | 547.381518701 | 0 | 63000 | 1956535 |
| FedAvg/44 | 551.896914236 | 547.381518701 | 0 | 63000 | 1956535 |

Scarto massimo del budget cumulativo FedAvg/Fused: **0.000197720%**. Il warm-up è incluso. FedAvg vede tutti gli esempi una volta per round, poi minibatch aggiuntivi; resti interi inferiori al costo di due esempi vengono riportati ai round successivi. Passi ed esposizioni sono reali, salvati per client/round. FedAvg qui è un controllo modificato, non la baseline pubblicata a una sola epoca.

Convenzione: [Torch 2.5.1 FlopCounterMode](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/flop_counter.py) sul forward/backward eseguito, FMA=2, comprese convoluzioni trasposte e gradienti attraverso D congelato. Aggiunta convenzione aritmetica SGD+norma/clipping=5P e Adam+norma/clipping=16P per passo. GPU/CPU producono le stesse firme dense. BN, attivazioni, pooling, loss, controlli finite, copie, aggregazione e overhead dei kernel esclusi: **budget abbinato nella metrica dichiarata, non misura esaustiva di istruzioni hardware, energia o tempo**. Le componenti scalari sono una stima semantica esplicita; conteggi dense e totali sono entrambi forniti.

## Calibrazione e congelamento

Train-only: 594 fit e 149 validation per dominio, seed dati 20261005, stratificazione deterministica per ID/quote; seed calibrazione 142. Nove candidati per metodo, 60 round e stesso budget contabile; LR C [.005,.01,.02], clip [.5,1,2], Fused LR AE [.0001,.0003,.001]. L9 bilanciato per Fused, griglia proiettata 3×3 per FedAvg, riferimento precedente incluso. Non è una ricerca esaustiva delle 27 interazioni Fused. L'utente ha esteso il piano prima di qualsiasi worker; il primo piano mai eseguito rimane conservato.

| Metodo | LR C | LR AE | Clip | Validation uniforme % |
| --- | --- | --- | --- | --- |
| FusedSpaceFed | 0.01 | 0.0003 | 1.0 | 71.946309 |
| FedAvg | 0.01 | — | 1.0 | 72.751678 |
| FusedSpaceFed | 0.005 | 0.0001 | 2.0 | 70.067114 |
| FedAvg | 0.005 | — | 2.0 | 72.751678 |
| FusedSpaceFed | 0.005 | 0.0003 | 0.5 | 55.302013 |
| FedAvg | 0.005 | — | 0.5 | 60.402685 |
| FusedSpaceFed | 0.005 | 0.001 | 1.0 | 63.758389 |
| FedAvg | 0.005 | — | 1.0 | 68.187919 |
| FusedSpaceFed | 0.01 | 0.0001 | 0.5 | 62.147651 |
| FedAvg | 0.01 | — | 0.5 | 68.187919 |
| FusedSpaceFed | 0.01 | 0.001 | 2.0 | 77.449664 |
| FedAvg | 0.01 | — | 2.0 | 75.973154 |
| FusedSpaceFed | 0.02 | 0.0001 | 1.0 | 76.107383 |
| FedAvg | 0.02 | — | 1.0 | 76.644295 |
| FusedSpaceFed | 0.02 | 0.0003 | 2.0 | 78.389262 |
| FedAvg | 0.02 | — | 2.0 | 78.120805 |
| FusedSpaceFed | 0.02 | 0.001 | 0.5 | 72.483221 |
| FedAvg | 0.02 | — | 0.5 | 72.483221 |

Impostazioni selezionate esclusivamente da validation al round 60:

```json
{
  "FedAvg": {
    "classifier_lr": 0.02,
    "gradient_clip_norm": 2.0
  },
  "FusedSpaceFed": {
    "autoencoder_lr": 0.0003,
    "classifier_lr": 0.02,
    "gradient_clip_norm": 2.0
  }
}
```

Parità: LR C più basso, poi clip più basso e LR AE più basso. Configurazioni definitive e selection SHA congelati e pubblicati prima dei nuovi test; nessuna riapertura del tuning. Altri optimizer, architettura e dati invariati. Reuse solo se impostazioni/dati/sorgenti combaciano: l'origine di ogni run è riportata. Nuove run da zero, nessun trasferimento dai checkpoint di calibrazione. I cinque vecchi test Fused erano già osservati: **controllo retrospettivo**, senza pretesa di test mai visto. La selezione usa soltanto la nuova validation da training.

## Tempi e verifiche

| Metodo/seed | Origine | GPU | Sessione s | Test s | CUDA alloc./ris. MiB | RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| FusedSpaceFed/42 | new | cuda:1 | 923.993 | 19.240 | 275.292/350.000 | 1874.508 |
| FedAvg/42 | new | cuda:0 | 831.734 | 11.800 | 273.697/338.000 | 1651.062 |
| FusedSpaceFed/43 | new | cuda:0 | 944.138 | 18.546 | 275.292/350.000 | 1876.266 |
| FedAvg/43 | new | cuda:1 | 833.484 | 11.948 | 273.697/338.000 | 1655.957 |
| FusedSpaceFed/44 | new | cuda:1 | 914.484 | 18.207 | 275.292/350.000 | 1874.074 |
| FedAvg/44 | new | cuda:0 | 845.273 | 12.486 | 273.697/338.000 | 1654.965 |

Calibrazione: 1410.977 s di calendario, 2816.308 s somma processi. Run definitive nuove: 2689.771 s di calendario, 5326.299 s somma processi. Le run riusate, se presenti, costano zero nuova esecuzione; i loro tempi originali sono separati nella tabella. Picchi allocator Torch per processo, esclusi contesto/driver/altri job. GPU 0 condivisa con autorizzazione, al massimo un worker per GPU, nessun processo altrui interrotto.

245 test CPU passati in 34,60 s prima del training, inclusi 12 del nuovo controllo; firma GPU sintetica verificata. Cinque ulteriori test stdlib della sintesi numerica passati dopo il training. Checkpoint/ripresa esatta per entrambi, encoder/RNG/carry, split disgiunto/riproducibile, arrotondamenti/budget e CLI dirette. Ulteriori prove sintetiche: selezione/tie/freeze e parità di due round Fused con il driver precedente. Audit indipendente dei 18 tentativi e sei finali: stati completi/exits, round, seed/hash, conteggi, confusioni, costi effettivi e medie/SD. Dataset, checkpoint e log integrali restano privati. Primo avvio fallito prima dei worker per collisione del nome select.py, corretto; primo push HTTP 504, retry HTTP/1.1 riuscito; nessuna modifica dei modelli per questi problemi operativi.

Partizione 7a762ffb10da74e3f0dee9a6f519e6c4057e2a5b47995a0546ee5a871eb47b58; validation 3dfc26c3fda17f266fb2ffa9899f23b4a5b50cd006742bb84d5b215e412a6943; profilo 164f242a968689c6c1a819fd353317c0a858b8c316a80b8244da0ed8c53f6a12; selection bd633c79a2520d5aef79c201c9d934a9b39acfa4363f34281985ef70ce3b50d3. Configurazioni, manifesti, tutti i tentativi numerici e timing sono in questa directory; raw/checkpoint/log in `_local/capacity_compute_control/` e vecchie run conservate. Comandi nel README. Manoscritto e altri esperimenti invariati.

## Limiti

Un solo feature-shift setting e tre seed, partizione fissa, screening corto con un seed di validation; interazioni e variabilità del tuning non stimate. Match congiunto di parametri e calcolo non ne separa gli effetti causali. Parametri privati, geometria del modello, BN e numero di esposizioni supervisionate differiscono. Il risultato non giustifica generalizzazioni ai setting label-skew del paper. Non sono inferite significatività o varianze mancanti; tutte le differenze e le eccezioni restano visibili.
