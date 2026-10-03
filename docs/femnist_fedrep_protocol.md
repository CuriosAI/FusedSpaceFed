# Confronto FEMNIST con i risultati pubblicati di FedRep

Questo documento identifica il riferimento di letteratura e il profilo per
**cinque sole run di FusedSpaceFed**. Le proposte del terzo compito sono state
approvate dall'utente nel quarto compito, il 3 ottobre 2026. Le baseline
rimangono risultati pubblicati. Dopo l'arresto per pool insufficiente (§10),
l'utente ha autorizzato la sola riduzione proporzionale delle quote delle
classi insufficienti (§11). Il benchmark implementa questa costruzione nostra,
distinta dal protocollo originale. Le cinque run definitive e il loro test
restano da eseguire; verifiche e smoke sono descritti nella
[guida operativa](femnist_reconstructed_benchmark.md).

## 1. Riferimento e punto di partenza

Il riferimento è **Tabella 1, ultima colonna, FEMNIST `(150,3)`**, nell'articolo
ICML 2021 di Collins et al.: pagina PDF 9, pagina PMLR 2097. La coppia significa
150 client e tre classi per client; il dataset complessivo ha dieci classi di
lettere. La colonna riporta accuratezze percentuali, compreso **78.56 per FedRep**.
Sono trascritti tutti e soli i suoi 15 valori in
[femnist_fedrep_reference.csv](femnist_fedrep_reference.csv). Le incertezze
mancano nella tabella e rimangono vuote nel CSV. [Articolo, Tabella 1][P-table]

Audit del 3 ottobre 2026. Commit FusedSpaceFed esaminato:
`7d3e7ce31702b85c9c3e92eac8b0c3de3bd9952e`; `main` pulito e allineato a
`origin/main` dopo fetch. L'ambiente `general_ml` già predisposto non viene
modificato.

Il benchmark esistente rimane distinto: client naturali per scrittore, 62
classi, ResNet20-v2, immagini ridimensionate a 32×32, fino a 3,400 scrittori,
340 partecipanti, 50 round e valutazione finale. Il codice concatena le
predizioni dei test locali, ottenendo un'accuratezza pesata per campioni.
Questo deriva da `FEMNISTWriterDataset`, `parse_args`, `run_method` ed
`evaluate_fused_by_writer` in [train_femnist.py](../train_femnist.py), dalla
sezione FEMNIST di [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) e dal benchmark
in [aistats_2027.tex](../paper/aistats_2027.tex). Nessuno di questi file o dei
risultati precedenti viene modificato.

## 2. Fonti e provenienza del codice

| Identificatore | Fonte e versione consultata | Riferimento pertinente |
|---|---|---|
| P | [Scheda PMLR][P-index] e [articolo ICML 2021][P-pdf] | §5.2, pp. 2095–2097; Tabella 1, p. 2097 |
| S | [Supplemento PMLR][S-pdf] | Appendice A.2, pagine PDF 1–2; Tabella 2, pagina 2 |
| A | [Preprint degli autori, arXiv v3 del 24 marzo 2023][A-pdf] | Appendice A.2, pagina 18: URL del codice originale; usato per la provenienza, non per sostituire la tabella ICML |
| C22 | [Commit storico `d82e62982e107bb3322541604f059c0ff28bd289`][C22-commit], 29 marzo 2022 | Copia della cronologia presso `rahulv0205/fedrep_experiments`; autore del commit associato da GitHub a `lgcollins` |
| C21 | [Commit storico `796429d99de93e539021d25ccc340367b2fcbc4d`][C21-commit], 25 agosto 2021 | Copia presso `LittleStory233/FedRep`; commit attribuito a `lgcollins`, confrontato con C22 |
| L | [LEAF][L-root], commit `09ec454a5675e32e1f0546b456b77857fdece018` del 19 dicembre 2021 | README FEMNIST e file di preprocessing; versione consultata, non identificata come quella usata nelle run pubblicate |
| N | [NIST Special Database 19][N-page], seconda edizione, settembre 2016 | Origine delle immagini PNG `by_class` usate dal generatore storico |

Gli autori indicano `https://github.com/lgcollins/FedRep` in A. Il supplemento
S cita anche `pliang279/LG-FedAvg`, che è il codice antecedente adattato dagli
autori, non un sostituto del loro repository. Il 3 ottobre 2026 il repository
originale e il relativo endpoint GitHub API restituiscono **404**.

C21 e C22 sono **copie storiche, non endpoint ufficiali attualmente verificati**.
Il README copiato che dichiara di essere ufficiale non basta a stabilire la
provenienza. Sono stati controllati commit, attribuzione GitHub dell'autore,
alberi e hash dei blob dei file scaricati. Il confronto fra le due copie
conferma alcuni dettagli comuni ma mostra anche un cambiamento nella
valutazione. Non è recuperato il commit esatto delle run della Tabella 1.
I dettagli ricavati dal codice sotto riportati hanno questa limitazione.

C22 contiene una licenza MIT; LEAF contiene BSD-2-Clause. Sono informazioni
sul codice consultato; non certificano la licenza di eventuali altri asset.
Documenti, codice di riferimento, manifesti con URL e SHA-256 e diff delle
versioni restano localmente in `_local/fedrep_protocol_reference/`, esclusa
da Git. Nessun codice di riferimento è stato eseguito.

SHA-256 dei PDF PMLR scaricati:

- Articolo: `455f34682f28ca4a399458e2d06003d21068f9f999257993b54143f8d9e5652e`.
- Supplemento: `42d522d892cfd3c0b70fa445dfa62e294da231b0b853e2d791290f9bc2a17359`.

## 3. Dettagli confermati dall'articolo e dal supplemento

| Aspetto | Dettaglio pubblicato | Fonte |
|---|---|---|
| Dati | FEMNIST limitato a dieci classi di lettere; distribuzione lognormale delle numerosità locali, seguendo Li et al. (2019), FedDANE | P, §5.2; S, A.2 |
| Partizione | 150 client, tre classi per client; media dichiarata 148 campioni/client, minimo 50 | P, Tabella 1 e §5.2; S, Tabella 2 |
| Modello | P descrive una MLP a due strati; S specifica due strati nascosti, senza dimensioni | P, §5.2; S, A.2 |
| Durata e partecipazione | 200 round; frazione 0.1 | P, §5.2; S, A.2 |
| FedRep locale | Dieci epoche della testa privata, poi cinque della rappresentazione condivisa per FEMNIST | P, §5.2; S, A.2 |
| Aggiornamenti delle altre baseline | Cinque epoche locali nel caso FEMNIST, con eccezioni proprie dei metodi | S, A.2 |
| Ottimizzazione di base | SGD, momentum 0.5, batch 10, learning rate 0.01; LR cercato in `{0.001,0.01,0.1}` | S, A.2 |
| Statistica | Media delle accuratezze locali degli utenti negli ultimi dieci round; intero processo ripetuto cinque volte e poi mediato | P, §5.2; S, A.2 |
| Righe `+FT` | Modello globale completamente addestrato, poi dieci epoche locali della sola testa prima del test | P, §5.2; S, A.2 |

La media 148 e il minimo 50 descrivono la partizione pubblicata: non sono
iperparametri da imporre scegliendo un seed favorevole. Le fonti non chiariscono
se tali conteggi includano il test. Le impostazioni speciali delle baseline
(per esempio LR di L2GD e coefficiente FedProx) rimangono proprietà dei
risultati `reported`; non diventano configurazioni da rieseguire.

La generalizzazione alle **cifre** e ai nuovi client della Figura 6 è un altro
esperimento. Non appartiene alla colonna qui trascritta.

## 4. Dettagli recuperati dal codice storico

### Dati, classi e split

[my_sample.py][C-data], funzioni `relabel_class`, `load_image` e `main`, usa
le cartelle PNG `raw_data/by_class/<classe>/train_<classe>`. Seleziona le
etichette FEMNIST 36–45, corrispondenti alle minuscole **a–j**, e le rimappa
a 0–9. Mescola i file e prende al massimo 4,000 immagini per classe. La
conversione è grayscale, thumbnail 28×28 con `Image.ANTIALIAS`, flatten a
784 valori e divisione per 255. I loader FEMNIST ricostruiscono il tensore
1×28×28: non applicano la normalizzazione MNIST `(0.1307,0.3081)`, che
appartiene a un altro ramo, né augmentation. [DatasetSplit][C-update-data]

Il generatore dichiara `NUM_USER=200`, `CLASS_PER_USER=3`; estrae
`LogNormal(4,1)+100`, dove 4 e 1 sono i parametri della normale nel logaritmo.
Il conteggio per classe è la parte intera del conteggio estratto diviso tre,
con minimo due. Le classi del client `i` sono `(i+j) mod 10`, per `j=0,1,2`.
I client sono sintetici, con identificatori `f_00000`, ecc.: non sono gli
scrittori naturali.

Dopo il mescolamento locale, il generatore separa `floor(0.9*N_i)` esempi di
training e il resto di test. Non produce validation. Il [README storico][C-readme]
indica invece una prima preparazione LEAF con `--sf 0.5 -k 50 -tf 0.8
-t sample`, seguita dal ricampionamento. Quel primo split 80/20 non dimostra
lo split effettivo delle run: `my_sample.py` rilegge i PNG grezzi e costruisce
un nuovo 90/10.

La distribuzione grezza è ancora collegata dal sito NIST; una richiesta
**HEAD**, senza scaricare il corpo del dataset, a
`https://s3.amazonaws.com/nist-srd/SD19/by_class.zip` ha restituito HTTP 200.
Il [downloader LEAF][L-download] indica questo archivio e `by_write.zip`.
Non sono stati recuperati i JSON originali dei 150 client della Tabella 1.
Un dataset torchvision EMNIST-Letters o i JSON naturali del nostro benchmark
non possono essere considerati automaticamente lo stesso dataset.

### Classificatore e ottimizzazione

[get_model][C-model-loader] costruisce per FEMNIST la classe
[MLP][C-net]: **784 → 512 → 256 → 64 → 10**, con bias e ReLU dopo ciascuno
dei tre strati nascosti. La MLP ha 550,346 parametri, calcolati dalle dimensioni
dei quattro layer. `dim_hidden=256` passato al costruttore non governa le
dimensioni interne. Non ci sono BatchNorm; il dropout dichiarato ha probabilità
zero e non è chiamato nel forward.

La testa FedRep è `layer_out`; gli altri tre layer sono condivisi. Il forward
restituisce **Softmax**, ma [LocalUpdate.train][C-update] passa l'output a
`CrossEntropyLoss`, che normalmente riceve logits: questa convenzione influenza
i gradienti e va distinta dal nostro uso della cross-entropy sui logits.

[scripts/FedRep_femnist.sh][C-script] conferma `num_classes=10`, `num_users=150`,
`epochs=200`, `frac=0.1`, `local_bs=10`, `local_ep=15`, `local_rep_ep=5`,
`lr=0.01`, ripetendo il comando cinque volte. Le 15 epoche sono 10 della
testa e 5 della rappresentazione. `LocalUpdate.train` ricrea SGD a ogni
partecipazione, con momentum 0.5, decay `1e-4` sui pesi e zero sui bias.
Il batch finale incompleto viene mantenuto. Il limite `local_updates=1000000`
di [args_parser][C-options] non è una prescrizione di un milione di passi:
è un tetto; i passi dipendono dalle epoche e dai campioni locali.

[main_fedrep.py][C-main] campiona 15 dei 150 client senza rimpiazzo a ogni
round e aggrega con pesi proporzionali ai **campioni di training** `lens[idx]`.
I parametri privati sono recuperati da `w_locals` alla partecipazione
successiva. Il learning rate passato rimane costante: il default `lr_decay`
non dimostra l'applicazione di uno scheduler.

### Valutazione, round e ripetizioni

[test_img_local_all][C-test] valuta i test dei singoli client e pesa le loro
accuratezze per il numero di **campioni di test**, ottenendo
`100 * totale_corrette / totale_test`. Il testo pubblicato dice media locale
fra tutti gli utenti senza esplicitare i pesi; non basta a certificare una
media uniforme dei client.

In C22, il test FedRep combina la rappresentazione globale corrente con la
testa privata più recente del client. In [C21][C21-test], invece, carica
l'intero ultimo stato locale, anche la rappresentazione: un client inattivo
può essere valutato con una rappresentazione precedente. Non è noto quale
procedura abbia prodotto la tabella.

Il ciclo di C22 esegue gli indici 0–199 per il training e accumula le dieci
valutazioni degli indici **190–199**. L'indice 200 è un passaggio aggiuntivo
con tutti i client per fine-tuning finale, escluso da quella media. In
numerazione dei round da 1, la finestra ordinaria è quindi **191–200**.
`test_freq=50` nello script non elimina le valutazioni degli ultimi dieci
round. Le righe `+FT` usano un endpoint distinto.

`--seed` è dichiarato con default 1, ma non viene usato da `main_fedrep.py`
per inizializzare i generatori. Il ciclo shell `RUN in 1 2 3 4 5` non passa
cinque seed; `my_sample.py` usa inoltre `random` e NumPy senza fissarli e
legge directory senza ordinamento. Sono confermate cinque ripetizioni,
**non i cinque seed originali**. Non risultano pubblicati in queste fonti
i cinque risultati grezzi o una deviazione standard della colonna.

## 5. Discrepanze e informazioni non recuperate

| Tema | Evidenza e conseguenza |
|---|---|
| Numero di client | Paper e comando: 150; generatore: 200. Il runner usa gli indici dei primi 150 client letti. Non è noto se gli autori modificarono il generatore o usarono un sottoinsieme. |
| Numerosità locali | S riporta media 148 e minimo 50. La ricetta C21/C22 assegna almeno 99 esempi totali, quindi almeno 89 di training con il suo split, se il pool basta. Non è documentato il raccordo con le statistiche pubblicate, né se il minimo 50 sia una soglia di selezione. |
| Split e percorsi | README: prima fase LEAF 80/20; generatore: nuovo 90/10. Il generatore scrive `train/mytrain.json` e `test/mytest.json`, mentre il runner legge directory `mytrain/` e `mytest/`: mancano istruzioni di raccordo complete. |
| Riutilizzo di immagini | Nel generatore, il cursore è azzerato quando `idx + conteggio < lunghezza`, cioè quando il blocco starebbe ancora nel pool. Questo permette di riusare gli stessi PNG in client diversi; split locali successivi possono sovrapporre training e test fra client. È una proprietà del codice esaminato, non la dimostrazione che i dati della tabella avessero tale problema. |
| Architettura | Descrizione a due strati nascosti in S; tre strati nascosti nelle due copie del codice. Le dimensioni effettivamente usate nelle run pubblicate non sono certificate. |
| Loss | Softmax seguito da CrossEntropyLoss nel codice; il nostro metodo usa cross-entropy sui logits. Usare gli stessi layer non elimina questa differenza. |
| Pesi delle medie | Codice: aggregazione pesata per training, accuratezza pesata per test. La prosa sperimentale non specifica i pesi dell'accuratezza. FusedSpaceFed nel paper aggrega uniformemente. |
| Stato usato al test | C21: intero ultimo modello locale; C22: componenti condivise correnti e componente privata persistente. La versione delle run originali manca. |
| Riproducibilità | Mancano manifesti delle immagini, mapping dei 150 client, split esatti, seed, commit delle run e statistiche grezze delle cinque ripetizioni. Il criterio/dataset usato per il tuning pubblicato non è specificato. |

Non è quindi possibile dichiarare recuperata una replica esatta del protocollo
eseguito. È possibile fissare una **ricostruzione documentata** e confrontare
le nostre misure con i valori pubblicati, rendendo esplicite le differenze.
Nessuna lacuna viene riempita automaticamente con i default del runner
FEMNIST attuale.

## 6. Profilo FusedSpaceFed approvato per la ricostruzione

Le seguenti sono **nostre decisioni di ricostruzione**, approvate dall'utente
nel quarto compito, anche quando riprendono un valore da una fonte. Non
certificano il protocollo originale. Un solo profilo vale per tutte le cinque
run previste. L'unica modifica successiva autorizzata riguarda l'allocazione
dei dati, per risolvere il problema storico in §10 senza riuso (§11).

| Voce | Decisione approvata e rapporto con il riferimento |
|---|---|
| Dati | In assenza dei JSON originali, ricostruire 150 client sintetici su a–j dai PNG NIST, con tre classi cicliche per client, pool massimo 4,000/classe e conteggi derivati dalla ricetta `LogNormal(4,1)+100`. Questa è una nuova partizione, non quella recuperata dagli autori. |
| Quote e unicità | Conservare richieste, seed e pool storici della nostra ricostruzione. Per le sole classi insufficienti ridurre proporzionalmente le quote con prodotti interi e maggiori resti, pari merito per ID crescente, come in §11. Allocare senza riuso degli identificatori d'origine. Fermarsi se una classe assegnata scompare da un client; nessun altro fallback. |
| Split | Mescolare per client e dividere 90/10 con arrotondamento come sopra, senza stratificazione aggiuntiva. Salvare manifesti e hash; training e test non vuoti e disgiunti anche fra client. |
| Validation | Nessuno split di validation per questo profilo a iperparametri prefissati. Non usare test o accuratezza negli ultimi dieci round per tuning, scelta dei seed o arresto anticipato. Un eventuale split di validation richiederebbe una decisione diversa prima delle run. |
| Preprocessing | Float32 1×28×28, grayscale e scala `[0,1]` secondo il generatore; preservare l'orientamento dei PNG. Nessun resize a 32×32, normalizzazione MNIST o augmentation. Verificare la conversione con le librerie già installate prima della preparazione definitiva. |
| Classificatore | MLP 784→512→256→64→10 con bias e ReLU, seguendo le dimensioni comuni a C21/C22. Restituire logits e usare cross-entropy standard per preservare la loss FusedSpaceFed. La discrepanza con il forward Softmax del codice storico rimane dichiarata. |
| Autoencoder | Conservare `UNetSmallAE`, base 16, `d_z=64`, encoder privato e decoder condiviso. 28 è divisibile per quattro: bottleneck 64×7×7, decoder di nuovo 28×28. `d_z=64` è una scelta nostra trasferita dal setup esistente, non un parametro FedRep verificato né selezionato sul test. |
| Fusione | `x + D(E_i(x))`, senza clipping, riscalamento o normalizzazione aggiuntiva. La ricostruzione ha output lineare come nel codice FusedSpaceFed attuale. |
| Round e partecipazione | 200 round; 15 client uniformemente campionati senza rimpiazzo a ogni round. Nessun passaggio finale aggiuntivo di training o adattamento al test. |
| Due fasi | Un'epoca MSE del solo encoder con decoder/classificatore congelati; poi cinque epoche di classificazione end-to-end con tutti e tre i componenti aggiornabili, senza loss di ricostruzione nella seconda fase. Il warm-up non equivale alle dieci epoche della testa FedRep. |
| Classificatore: optimizer | SGD LR 0.01, momentum 0.5, decay `1e-4` sui pesi e zero sui bias; batch 10, ultimo batch mantenuto, LR costante. Momentum e decay sono adattamenti espliciti rispetto all'attuale FusedSpaceFed. |
| Autoencoder: optimizer | Adam LR 0.001, betas `(0.9,0.999)`, epsilon `1e-8`, decay zero, usato nelle due fasi. Sono scelte FusedSpaceFed nostre, non impostazioni degli autori FedRep. |
| Inizializzazione | Per ogni run creare un solo template casuale di classificatore e uno di autoencoder, con inizializzazione dei layer PyTorch registrata. Usare copie dello stesso encoder iniziale per tutti i client e lo stesso decoder iniziale condiviso, come nel runner attuale; non reinizializzare l'encoder alle partecipazioni successive. Anche questa è una scelta nostra. |
| Persistenza | Conservare l'encoder di ogni client fra partecipazioni; sovrascrivere decoder e classificatore con gli stati condivisi correnti. Come nel runner attuale, ricreare gli optimizer a ogni partecipazione; mantenere lo stesso Adam fra le due fasi di quel round. Al test usare l'encoder persistente, o il suo stato iniziale se il client non ha ancora partecipato, registrando i conteggi delle partecipazioni. |
| Aggregazione | Conservare la **media uniforme** di decoder e classificatore definita nel paper FusedSpaceFed. Il codice FedRep pesa per campioni di training: è un limite di comparabilità. Non è approvata una variante pesata. |
| Precisione | Float32 senza AMP; il runner esistente usa AMP su CUDA. Registrare comunque precisione, versioni e impostazioni di determinismo effettive. |
| Seed | Un solo seed dati **20261003**, per una partizione congelata comune; seed delle cinque run **41,42,43,44,45**. Sono nostri seed, non quelli originali. Separare e registrare i generatori di dati/split, inizializzazione, selezione client e shuffle dei batch. |

L'adattamento richiede un classificatore selezionabile e un loader/runner
dedicato a questo setting: al punto di partenza `FusedSpaceFedClient`
costruiva direttamente ResNet20-v2 in
[fusedspacefed_core.py](../fusedspacefed_core.py), mentre `train_femnist.py`
fissa 62 classi. Il quarto compito aggiunge un preparatore e un runner dedicati,
riutilizzando le due fasi del core e preservandone i default: classificatore
opzionale e possibilità di disabilitare AMP sono i soli nuovi punti di
estensione. Gli entry point precedenti e il manoscritto restano invariati.

### Metrica primaria e reporting approvati

Per ogni seed `s`, dopo l'aggregazione dei round `t=191,...,200`, valutare
tutti i 150 client sui rispettivi test locali con decoder/classificatore
condivisi correnti ed encoder privato persistente, senza adattamento:

```text
A(s,t) = 100 * sum_i correct(s,t,i) / sum_i n_test(i)
A(s)   = mean_{t=191,...,200} A(s,t)
risultato ours = mean_{s in {41,42,43,44,45}} A(s)
```

La ponderazione per campioni segue il codice storico ed è la decisione
approvata, dato che il paper non esplicita i pesi. Salvare corrette/totali
e accuratezza per client e round, così la media uniforme dei client resta
calcolabile dagli stessi dati senza altre run. L'eventuale deviazione standard
nostra è quella campionaria dei **cinque valori `A(s)`**, con `ddof=1`, in
punti percentuali; i dieci round non sono dieci repliche indipendenti.

Le dieci valutazioni finali sono una finestra prefissata, non una ricerca del
miglior checkpoint. Eseguire la finestra soltanto dopo aver congelato tutte
le scelte; impedire alla valutazione di aggiornare parametri, optimizer o
generatori usati dal training. Non aggiungere una fase `+FT` a FusedSpaceFed:
rendere privata la testa cambierebbe il suo meccanismo. Nel confronto le righe
`+FT` restano identificate come risultati con adattamento della sola testa.

## 7. Capacità, costi e limiti del confronto

La stessa MLP non implica la stessa capacità totale: FusedSpaceFed aggiunge
encoder e decoder e un'epoca di warm-up. FedRep ha invece una testa privata
e un corpo condiviso, con un budget di aggiornamenti diverso. Le future run
devono registrare:

- Parametri e buffer del classificatore, dell'encoder e del decoder, distinguendo
  privati/condivisi; dimensione degli stati optimizer e dei checkpoint.
- Campioni realmente processati e passi optimizer, separati per warm-up e
  classificazione; epoche effettive per client e partecipazioni.
- Tempi reali di warm-up, classificazione, aggregazione, test e round; tempo
  totale delle cinque run. Nella simulazione seriale distinguere tempo totale
  e massimo dei tempi dei client, senza presentare quest'ultimo come tempo
  realmente misurato di un'esecuzione distribuita.
- Picchi di memoria GPU e RAM e stato privato conservato per i 150 client;
  hardware, precisione e versioni. Se si misura un costo di operazioni,
  specificare strumenti e cosa includa il conteggio.
- Byte uplink/downlink per classificatore **e decoder**, con dtype e buffer:
  per round il modello teorico è `2 * 15 * (P_C + P_D)` scalari. L'encoder
  resta locale; l'eventuale comunicazione di metadati va conteggiata a parte.

I risultati rimangono un confronto **`ours` contro `reported`**, con partizioni
e seed originali non recuperati, versione di valutazione incerta e adattamenti
espliciti di quote, proporzioni di classe, loss, aggregazione e budget. Non è una riesecuzione controllata
comune delle baseline. La sola nuova deviazione standard non permette test
statistici appaiati o incertezze sulle differenze con i valori pubblicati.
Non attribuire alle baseline tempi/costi misurati sulle nostre run. Non
rieseguire altre campagne per risolvere questi limiti.

## 8. Decisioni D1–D6 approvate nel quarto compito

| Decisione | Scelta approvata dall'utente |
|---|---|
| D1: disponibilità e partizione | Ricostruzione dichiarata a 150 client, seed dati 20261003 e allocazione disgiunta. Dopo il primo arresto (§10), è autorizzata soltanto la riduzione proporzionale con maggiori resti (§11). Non sono recuperati i JSON originali. |
| D2: classificatore e loss | MLP con le dimensioni del codice storico, che differiscono dalla descrizione del supplemento; logits con cross-entropy standard. |
| D3: aggregazione | Media uniforme del nostro paper; differenza dal riferimento dichiarata. |
| D4: iperparametri del nostro metodo | Warm-up 1, classificazione 5, `d_z=64`, Adam 0.001, SGD 0.01 con momentum/decay sopra dichiarati, inizializzazione comune, reset degli optimizer e Float32 senza AMP; nessun tuning sul test. |
| D5: valutazione | Stati condivisi correnti, test locali pesati per campioni, conteggi per client e media uniforme, finestra 191–200, nessun adattamento finale. |
| D6: repliche e presentazione | Seed 41–45 su una sola partizione congelata; confronto con risultati pubblicati con i limiti esplicitati. Esattamente cinque run FusedSpaceFed, senza baseline o campagne ulteriori. |

## 9. Verifiche svolte nel terzo compito (studio)

La trascrizione è stata controllata sulla pagina PDF originale, anche
visivamente, e contro il testo estratto della Tabella 1. Il CSV conserva ordine
e nomi dei metodi, precisione a due decimali, unità percentuale, provenienza
`reported` e campi di incertezza vuoti; non contiene un risultato FusedSpaceFed.
Sono stati esaminati il manoscritto attivo, le parti pertinenti del codice
FusedSpaceFed, le fonti pubblicate, le due copie storiche e il preprocessing
LEAF. Nessun dataset completo, ambiente o pacchetto è stato scaricato o
installato per l'esecuzione; nessun training è stato avviato. I soli file da
versionare sono questo documento e il CSV di riferimento.

## 10. Primo tentativo del quarto compito: arresto per insufficienza del pool

Il 3 ottobre 2026 il quarto compito è iniziato su `main` pulito e allineato
a `origin/main`, commit `939c584cbeb1db6cb37ba0314ca15ab0214d599a`, dopo
fetch. È stato scaricato, una sola volta, l'archivio NIST già identificato,
senza cambiare ambienti o pacchetti. L'archivio e la ricevuta verificabile
restano in `_local/femnist_reconstructed/source/`, esclusi da Git.

- URL: `https://s3.amazonaws.com/nist-srd/SD19/by_class.zip`.
- Dimensione: 1,031,576,378 byte.
- SHA-256: `b387d65249b2d0ed429cf81967d4c40a9d01ca2f7bb3931c6a36d825bc22d411`.

Il pool storico è `by_class/<hex>/train_<hex>/*.png`, come in §4. Il limite
4,000 è un massimo e non garantisce che ogni classe disponga di tanti PNG.
L'implementazione fissa separatamente `random.Random(20261003)` per il
mescolamento dei percorsi ordinati, `Generator(PCG64(20261003))` per i
conteggi lognormali e `random.Random(20261004)` per lo split locale. Sono
generatori espliciti della nostra ricostruzione, non seed originali recuperati.
La verifica sugli identificatori effettivi dell'archivio ha dato:

| Lettera | PNG distinti nella cartella storica | Pool dopo il limite 4,000 | Richiesti dalla ricetta | Mancanti |
|---|---:|---:|---:|---:|
| a | 11196 | 4000 | 2828 | 0 |
| b | 5551 | 4000 | 2506 | 0 |
| c | 2792 | 2792 | 2474 | 0 |
| d | 11421 | 4000 | 2325 | 0 |
| e | 28299 | 4000 | 2558 | 0 |
| f | 2493 | 2493 | 2555 | 62 |
| g | 3839 | 3839 | 2578 | 0 |
| h | 9713 | 4000 | 2869 | 0 |
| i | 2788 | 2788 | 3078 | 290 |
| j | 1920 | 1920 | 3196 | 1276 |

La prima guardia interviene al client `f_00108` (indice 108): per `j` la
richiesta cumulativa arriva a 1964, oltre i 1920 disponibili. Il controllo
indipendente delle richieste di tutti i 150 client conferma insufficienze
anche per `f` e `i`. La ricetta richiederebbe 26,967 esempi complessivi;
questo è un conteggio richiesto, **non una partizione preparata**.

Quel tentativo si è fermato prima di convertire o salvare array train/test
e manifesti di una partizione. Il rapporto macchina è conservato soltanto in
`_local/femnist_reconstructed/partition_failure.json`.

In quel tentativo non sono stati cambiati seed, lognormale, classi, numerosità o split, né
riusati PNG o aggiunti esempi da altre cartelle. Il download resta
riutilizzabile. Non sono stati avviati smoke test GPU, valutazioni del test,
baseline o run definitive. L'implementazione era rimasta incompleta e non
pubblicata. L'autorizzazione successiva dell'utente risolve esclusivamente
l'allocazione mediante la regola esplicita in §11; questo resoconto è
conservato come traccia dell'arresto iniziale.

Verifiche del lavoro già scritto prima dell'arresto: `py_compile` dei due
moduli dedicati e del core riuscito; suite esistente
`tests/test_core.py`: **7 passed in 3.71s**, su CPU nell'ambiente `general_ml`;
`git diff --check` riuscito. La guardia di esaurimento è stata riprodotta
indipendentemente sull'archivio. Non sono completati i test mirati del nuovo
runner, la verifica di ripresa e lo smoke GPU in quel tentativo. Nessun commit
o push era stato eseguito. La successiva implementazione e le sue verifiche
sono descritte nella guida operativa.

## 11. Allocazione proporzionale autorizzata e partizione congelata

Si mantengono archivio NIST, cartelle `train_<hex>`, a–j, 150 client, tre
classi cicliche, seed 20261003 e tutti i 26,967 esempi **richiesti** dai
sorteggi originali. Non si risorteggia. Per la classe `c`, siano `k[i,c]` le
richieste originali, `D[c]` la loro somma, `N[c]` il numero di identificatori
del pool dopo il limite 4,000 e `M[c]=min(D[c],N[c])`.

Se `D[c] <= N[c]`, le quote restano `k[i,c]`. Altrimenti:

1. Quota iniziale `a[i,c] = (k[i,c] * M[c]) // D[c]`.
2. Distribuire `M[c] - sum_i a[i,c]` esempi ai resti
   `(k[i,c] * M[c]) % D[c]` maggiori.
3. Risolvere le parità per identificatore del client crescente.

Tutti i prodotti, divisioni e resti dell'allocazione usano interi Python,
senza arrotondamento floating point. I client senza quella classe hanno
richiesta e assegnazione zero. I cursori avanzano e non si azzerano. La
funzione `proportional_quotas` in
[femnist_reconstructed_data.py](../femnist_reconstructed_data.py) implementa
la regola; `plan_partition` conserva separatamente quote richieste e assegnate.

| Classe ridotta | Richiesta D | Pool N | Assegnazione M | Rapporto intero di riduzione |
|---|---:|---:|---:|---|
| f | 2555 | 2493 | 2493 | 2493/2555 |
| i | 3078 | 2788 | 2788 | 2788/3078 |
| j | 3196 | 1920 | 1920 | 1920/3196 |

Le altre sette classi mantengono tutte le quote. Le quote di **105 client**
cambiano: anche numerosità e proporzioni di classe cambiano nei client
coinvolti. Ogni client conserva le tre classi prima dello split e nel training;
lo split locale resta un mescolamento non stratificato, seguito da
`floor(0.9*N_i)` training e resto test. Non si garantiscono tre classi in ogni
test locale. Tutti i test sono non vuoti.

| Conteggio per client | Totale | Media | Minimo | Massimo |
|---|---:|---:|---:|---:|
| Richieste originali | 26967 | 179.780000 | 102 | 1050 |
| Assegnati prima dello split | 25339 | 168.926667 | 97 | 877 |
| Training | 22736 | 151.573333 | 87 | 789 |
| Test | 2603 | 17.353333 | 10 | 88 |

Il totale 25,339 è stato verificato sul risultato della regola e non imposto
al campionatore. Non sono stati ricercati media 148 o minimo 50 della
pubblicazione. Il manifesto contiene disponibilità, fattori e rapporti di
riduzione, sorteggi originali, richieste e assegnazioni di ogni client/classe,
identificatori d'origine, hash PNG e array, seed, versioni e statistiche.

- Partizione canonica SHA-256:
  `7a4c7614796a595d8a752aba5d7d275a18a2f6e542dfd338e8a381fcd5675734`.
- Configurazione canonica SHA-256:
  `809d987e704f495072fecd53bf5f08cb930e3685616417c9adab34d1c0d87101`.
- Artefatti: `_local/femnist_reconstructed/partition/`, esclusi da Git.
- Rapporto di fallimento precedente conservato, SHA-256:
  `260f7cef5adb56c039f7d36e0a9f1fd3931f56eaee919487d4fa2a6722337ea4`.

L'audit controlla i 25,339 identificatori unici, disgiunzione globale dei due
split, tre classi di training per client, quote invariate nelle classi
sufficienti e riuso della stessa partizione senza sovrascritture. Questi sono
dati e regole della **nostra ricostruzione adattata alla capacità disponibile**,
non una replica esatta del benchmark FedRep o una riesecuzione comune delle
baseline.

[P-index]: https://proceedings.mlr.press/v139/collins21a.html
[P-pdf]: https://proceedings.mlr.press/v139/collins21a/collins21a.pdf
[P-table]: https://proceedings.mlr.press/v139/collins21a/collins21a.pdf#page=9
[S-pdf]: https://proceedings.mlr.press/v139/collins21a/collins21a-supp.pdf#page=1
[A-pdf]: https://arxiv.org/pdf/2102.07078v3#page=18
[C22-commit]: https://github.com/rahulv0205/fedrep_experiments/commit/d82e62982e107bb3322541604f059c0ff28bd289
[C21-commit]: https://github.com/LittleStory233/FedRep/commit/796429d99de93e539021d25ccc340367b2fcbc4d
[C-data]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/my_sample.py#L14
[C-readme]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/README.md#L15
[C-update-data]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/models/Update.py#L24
[C-model-loader]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/utils/train_utils.py#L94
[C-net]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/models/Nets.py#L16
[C-update]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/models/Update.py#L530
[C-script]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/scripts/FedRep_femnist.sh#L3
[C-options]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/utils/options.py#L7
[C-main]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/main_fedrep.py#L124
[C-test]: https://github.com/rahulv0205/fedrep_experiments/blob/d82e62982e107bb3322541604f059c0ff28bd289/models/test.py#L102
[C21-test]: https://github.com/LittleStory233/FedRep/blob/796429d99de93e539021d25ccc340367b2fcbc4d/models/test.py#L103
[L-root]: https://github.com/TalwalkarLab/leaf/tree/09ec454a5675e32e1f0546b456b77857fdece018
[L-download]: https://github.com/TalwalkarLab/leaf/blob/09ec454a5675e32e1f0546b456b77857fdece018/data/femnist/preprocess/get_data.sh#L6
[N-page]: https://www.nist.gov/srd/nist-special-database-19
