# FEATURE_SHIFT_REPORT — FusedSpaceFed sul Digits bilanciato di FedBN

## Esito e risultati per dominio

Le cinque run, seed **42–46**, sono concluse: ciascuna parte da zero, completa 300 round e termina con exit code 0. Sono stati eseguiti soltanto FusedSpaceFed e i suoi test di correttezza. Nessuna baseline è stata rieseguita. Il manoscritto e gli altri esperimenti rimangono invariati.

$result_table

Accuratezze in percentuale; SD campionaria fra cinque seed, **ddof=1**, in punti percentuali. Tutti i seed sono riportati. Nessun round, checkpoint o seed è scelto sul test.

$baseline_table

I valori delle baseline sono le medie e SD pubblicate nella [tabella 11 dell'appendice](https://michaelkamp.org/wp-content/uploads/2021/05/FedBN_appendix.pdf), PDF pagina 9, stampata 21. Il paper descrive cinque trial; il ddof della SD pubblicata non è specificato. La nostra SD usa esplicitamente ddof=1.

$delta_table

Le differenze sono descrittive: il confronto usa risultati pubblicati, senza riesecuzione comune delle baseline. Restano visibili tutti i domini, inclusi quelli sfavorevoli. Non vengono affermate significatività o superiorità in un confronto controllato.

La metrica primaria è l'accuratezza per dominio, ricostruita da corrette/totali. Le sintesi nostre sono: media uniforme dei domini **$uniform_mean%**, SD **$uniform_sd pp**; media pesata per esempi **$weighted_mean%**, SD **$weighted_sd pp**. La seconda è dominata dal test SynthDigits, molto più numeroso. Non si inventa una SD complessiva delle baseline dalle sole SD marginali.

## Setting e dati effettivi

Riferimento: appendice D.2, tabelle 3 e 8, con cinque client, uno per dominio, 743 training ciascuno, 300 round, una epoca locale di classificazione, batch 32 e CNN del benchmark. Il default corrente del codice `--iters=100` è superato dal valore esplicito 300 dell'appendice.

$data_table

Totali: **3.715 training**, **$total_test test**. Si usano direttamente `train_part0.pkl` e l'intero `test.pkl` distribuiti, senza campionamento aggiuntivo o nuova partizione. Il preprocessing degli autori combina gli originali train/test e applica uno split stratificato 80/20, `random_state=0`; `part_len=743.8` dà il primo blocco `[0:743]`. Non sono i test standard di torchvision.

Preprocessing conforme al codice: resize PIL bilineare 28×28 per SVHN/SynthDigits/USPS; MNIST/USPS grayscale su tre canali; MNIST-M già RGB 28×28. ToTensor e Normalize(mean=.5,std=.5) in [-1,1], senza augmentation. La cache mantiene gli stessi pixel uint8; normalizzazione Float32 al caricamento. I test sintetici confermano parità esatta delle cinque pipeline.

Gli identificatori sono dominio/membro ZIP/riga e rendono disgiunti i membri train/test usati. Gli ID originali precedenti al resplit non sono disponibili. MNIST-M deriva da MNIST: non si presume indipendenza delle origini fra domini. Uguali numerosità non implicano uguali istogrammi delle classi, in particolare per SVHN.

## Fonti, versioni e provenienza

- Repository ufficiale [med-air/FedBN](https://github.com/med-air/FedBN), commit `2fa38adf627a8c8ba71c5fb515b1f2ba00aa8812`. Il README identifica gli autori e il paper; non è un fork presunto ufficiale.
- Funzioni pertinenti: `utils/data_preprocess.py::stratified_split/split`, `utils/data_utils.py::DigitsDataset`, `nets/models.py::DigitModel`, `federated/fed_digits.py::prepare_data/train/test/communication`.
- Archivio `digit_dataset.zip` dal mirror HF `Jemary/FedBN_Dataset`, revisione `0b6cd64d780662b683a373ddb23aa25d1d968cf8`, 278.200.677 byte, SHA256 `6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221`, verificato contro LFS. Il README scambia i link dataset/modello: è stato scelto il file dati effettivo, senza scaricare modelli preaddestrati.
- Il mirror terzo è collegato dagli autori nel 2025, senza certificazione che i byte coincidano con la tabella del 2021. Hash dei PDF, versioni e hash dei file sorgente, membri ZIP, cache e identificatori sono conservati in `reference_sources.json` e `partition_manifest.json`.

Partizione canonica: `$partition_hash`. Configurazione scientifica canonica: `$scientific_config_hash`.

## Estensione da due a cinque run

L'utente ha ampliato la campagna mentre 42/43 erano in corso. Sono stati conservati senza riavvio e aggiunti 44/45/46 consecutivi, senza scegliere seed in base ai risultati. Cambiano solo i campi di registro `run_seeds` e `per_seed_device`; **ogni campo scientifico è identico** e i quattro sorgenti di training hanno gli stessi byte.

Il registro iniziale `config.json` ha hash `$old_registry_hash`; `config_five.json` ha hash `$new_registry_hash`. Non si riscrivono i JSON originali delle prime run. La ricevuta `five_run_authorization.json` vincola i due registri e la configurazione scientifica comune. La differenza di hash completo viene documentata, non nascosta.

Commit di esecuzione delle prime due: `$source_commit_pair`. Commit operativo dell'estensione: `$source_commit_extension`. I nuovi file estendono soltanto registrazione e orchestrazione; il runner scientifico originale non è modificato. Il controllore attende gli exit code della coppia iniziale, poi esegue 44/45 e infine 46, al massimo due processi e uno per GPU.

## Metodo e scelte prefissate

La CNN riproduce `DigitModel`: tre conv 3→64→64→128, BN/ReLU e due pool; FC 6272→2048→512→10, BN/ReLU prima dei logits. La tabella 3 ripete una riga conv; il codice eseguibile ne ha tre. Il test di equivalenza confronta stati e output con i pesi degli autori. **Tutti i parametri e buffer BN del classificatore sono condivisi**: FusedSpaceFed non acquisisce i BN privati di FedBN.

Encoder privato persistente per dominio; decoder condiviso UNetSmallAE, base 16 e dz=64; fusione `x + D(E_i(x))`. Warm-up: una epoca MSE aggiorna solo E attraverso D congelato. Classificazione: una epoca CE aggiorna E/D/C, senza termine MSE aggiuntivo. Aggregazione uniforme solo D/C, incluso BN; contatori BN interi copiati dal primo client, con stessi batch per tutti. Tutti i cinque client partecipano in ogni round.

SGD lr=.01 senza momentum/weight decay, coerente con FedBN e il default originario FusedSpaceFed. Adam lr=.0003, betas=(.9,.999),eps=1e-8,decay=0; clipping L2=1 per optimizer attivo. Struttura, AE LR e clipping riusano scelte precedenti su training/validation FEMNIST, senza sweep Digits. Gli optimizer vengono ricreati per partecipazione, Adam continuo fra le due fasi. Norma ausiliaria Float64 sicura; modello/gradienti Float32, niente AMP/TF32, determinismo, quattro thread e zero worker del loader. Ultimo batch 7 mantenuto.

L'epoca locale concordata è quella di classificazione. Il warm-up aggiunge un passaggio e risorse: non è un budget identico alle baseline. Nessun iperparametro è stato scelto o cambiato sul test Digits. Configurazioni integrali e decisioni sono nel pacchetto.

## Valutazione e runtime

Un solo test per run, al round **300 prefissato**, su tutti i test del proprio dominio: D/C correnti, E persistente, eval mode, senza adattamento o ricalibrazione BN. Nessun miglior checkpoint, early stopping o tuning dopo il test. Il codice originale verifica il test ogni round e salva l'ultimo modello; la frequenza è diversa e il round della statistica pubblicata non è esplicitamente recuperato.

Il runtime è comune, salvo device; nessun pacchetto o ambiente è stato creato/aggiornato:

```json
$runtime
```

## Tempi, stime iniziali e memoria

$estimate_table

Stime conservate dopo cinque round: mediana recente per round e training restante. Escludono test finale e checkpoint; non guidano cambiamenti scientifici. La prima previsione era 15–16 minuti per la coppia; dopo l'estensione era circa 45–50 minuti per la campagna intera.

$time_table

Calendario complessivo: **$wall_seconds s ($wall_minutes minuti)**, da $start_utc a $end_utc. Somma dei processi osservati: **$workers_seconds s**, diversa dal calendario per sovrapposizione. Polling di un secondo: può aggiungere fino a circa un secondo alle durate osservate. Sessione runner e test sono nidificati; non si sommano. Nessuna interruzione o ripresa. GPU 0 in condivisione autorizzata; nessun processo altrui interrotto.

I picchi CUDA sono dell'allocator Torch per worker, esclusi contesto/driver/job esterni; RSS per processo, senza somma contemporanea. Parametri: C=$classifier_parameters, E=$encoder_parameters per dominio, D=$decoder_parameters. Stessa CNN non significa stessa capacità/costo; E/D e warm-up aggiungono risorse. Comunicazione logica: $communication_round byte/round, $communication_run byte/run, senza misura di rete reale o overhead. Loss, passi optimizer, clipping, partecipazioni e timer di ogni client/round sono salvati.

## Verifiche

211 test CPU passati in 16,03 s, inclusi 196 preesistenti e 15 nuovi; ulteriori cinque test operativi dell'estensione passati in 3,72 s. Altri 17 test sintetici verificano l'audit indipendente, la costruzione completa del pacchetto, il rifiuto delle sovrascritture e il rilevamento di alterazioni. Comandi/log nei file di verifica. Test nuovi su preprocessing, parità CNN, dimensioni, gradienti/freeze, aggregazione BN/esclusione E, confusione/metriche, norme grandi e non finite, 743 righe/riuso/tamper/ID, checkpoint con stati e RNG identici, parallelismo/exits e overwrite. I test di sviluppo usano dati sintetici.

Audit indipendente con sola stdlib: cinque exit code 0 e status completed; **1.500 round totali**, tutti i client, 24 passi per fase/client e 743 esposizioni per fase. Una sola valutazione per seed al 300. Confusion matrix, corrette/totali e istogrammi ricostruiscono ogni accuratezza, media e SD ddof=1. Nessun nonfinito; hash/config/dati/source/runtime coerenti, registri e device distinti espliciti. L'audit non esegue modelli, usa GPU, apre immagini o deserializza checkpoint: questi ultimi sono verificati come byte.

Gzip dei risultati senza perdita; timing identici ai raw. Manifesto completo e verifica CRC/SHA di ogni membro del ZIP. Il CSV published rimane distinto dai risultati ours. Le baseline FEMNIST e tutti i precedenti risultati restano invariati.

## Limiti e consegna

Seed/statistic round originali mancanti, mirror 2025 non certificato 2021, runtime recente, BN condivisi contro FedBN locali, clipping/optimizer AE, warm-up e capacità/costi aggiuntivi limitano il confronto. La SD condiziona su questa partizione e configurazione, senza misurare variabilità dei dati o del tuning. I domini non hanno istogrammi identici e MNIST/MNIST-M hanno origini correlate. Non è una replica esatta o un confronto causale controllato; nessuna nuova campagna è aperta per colmare questi limiti.

Consegna in `research/feature_shift_digits/`: codice e test, i due registri equivalenti, protocollo/fonti/manifesto dati, baseline CSV, verifiche, `artifacts/summary.json`, `campaign_summary.json`, `domain_comparison.csv`, i cinque `results.json.gz` lossless con `timings.jsonl`, questo report, `manifest.json` e `FEATURE_SHIFT_HANDOFF.zip`. Dataset, immagini, checkpoint, log integrali, credenziali e review private rimangono esclusi. Tutti i raw sono conservati in `_local/feature_shift_digits/`.

Il ZIP include il core scientifico esatto usato; niente dataset/checkpoint. Per training servono repository Git e dati checksum-pinned del README; l'audit numerico non richiede GPU o librerie ML. Comandi di esecuzione e ripresa nel README. Verifica riproducibile:

```bash
python3 research/feature_shift_digits/audit_and_package.py verify --directory research/feature_shift_digits
```

I commit di implementazione ed estensione sono stati pubblicati prima delle rispettive nuove run. Il commit finale di risultati e il suo push sono riportati nella consegna in chat. Questo pacchetto documenta esclusivamente la campagna Digits; eventuali controlli successivi autorizzati saranno conservati separatamente.
