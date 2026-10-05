# Digits: nuova calibrazione di FusedSpaceFed

Questa campagna è distinta dal pilota immutato in `research/feature_shift_digits/`
e dal controllo di capacità/calcolo in `research/capacity_compute_control/`.
Si esegue soltanto FusedSpaceFed. Nessun cambiamento al manoscritto o alle baseline
pubblicate; i cinque nuovi seed definitivi sono 42–46, tutti da zero.

## Protocollo invariato

Stessi dati bilanciati degli autori: `train_part0`, 743 immagini per MNIST,
MNIST-M, SVHN, SynthDigits e USPS. Stesso preprocessing e CNN FedBN,
UNetSmallAE a tre canali, base16, dz64; encoder privato persistente,
decoder/classificatore condivisi, somma input+ricostruzione. Si condividono anche
i buffer BN; non si adotta il BN locale di FedBN. Ogni round coinvolge tutti
i cinque client, con aggregazione uniforme. Un'epoca di warm-up MSE aggiorna
solo l'encoder, poi un'epoca CE aggiorna encoder, decoder e classificatore.
Batch32, SGD senza momentum/decadimento per CNN, Adam per AE, stati optimizer
reinizializzati per partecipazione (Adam continuo fra le due fasi), Float32,
nessun AMP/TF32. Nessuna variazione di architettura, fusione, epoche, batch o dati.
Si calibrano esclusivamente LR del classificatore, LR AE e clipping L2.

Fonti e loro versioni restano quelle verificate nel pilota:
[FedBN ufficiale](https://github.com/med-air/FedBN), commit
`2fa38adf627a8c8ba71c5fb515b1f2ba00aa8812`,
[appendice](https://michaelkamp.org/wp-content/uploads/2021/05/FedBN_appendix.pdf)
D.2, tabella8 (300 round) e tabella11 (accuratezze pubblicate).
I dettagli di download/mirror, SHA e discrepanze sono conservati nel rapporto
del pilota e nei suoi manifesti, che non vengono riscritti.

## Ricerca registrata prima delle nuove run

Si riusa la partizione stratificata derivata esclusivamente dai 743 training:
594 fit e149 validation per dominio, hash
`3dfc26c3fda17f266fb2ffa9899f23b4a5b50cd006742bb84d5b215e412a6943`.
Lo split non viene risorteggiato. Il criterio è la media uniforme delle cinque
accuratezze validation; con149 esempi in ciascun dominio coincide con la metrica
pesata. La precedente ricerca a60 round aveva favorito LR/clip al bordo
superiore: questo dato di validation motiva l'estensione della griglia.

1. Dieci run da zero, seed142, 120 round, una sola validation finale.
   Nove combinazioni L9 di LR CNN {0.02,0.05,0.1}, LR AE
   {0.0001,0.0003,0.001}, clip {2,5,10}, più il riferimento pilota
   (0.01,0.0003,1). La configurazione precedentemente selezionata
   (0.02,0.0003,2) è compresa nelle nove.
2. Le due configurazioni meglio classificate, più entrambi i riferimenti,
   deduplicati, sono addestrate da zero a300 round su seed142 e143.
   La vincitrice massimizza la media fra questi due valori di validation
   al round300. Non si sceglie il checkpoint migliore. Parità risolte per
   LR CNN, clip, LR AE crescenti, quindi ID.
3. Si congela `selection.json` e le cinque configurazioni prima di accedere
   ai nuovi test. Cinque inizializzazioni nuove, seed42–46, training completo
   743/client, 300 round, una sola valutazione test al round300, senza
   adattamento, ricalibrazione BN, arresto anticipato o scelta del miglior seed.

Budget massimo: dieci screening, otto conferme, cinque definitive,
5100 round. Una run propria per GPU; GPU0 condivisa secondo autorizzazione
dell'utente, GPU1 disponibile. Il controller controlla la VRAM e non invia
segnali ad altri processi. I tentativi sono in una nuova directory privata;
nessuna sovrascrittura automatica. Checkpoint comprendono E privati, D/C,
RNG Python/NumPy/Torch/CUDA e generatori di ogni loader. La ripresa richiede
identità di seed, device, configurazione, partizione, commit e sorgenti.

## Interpretazione e limiti

È una calibrazione successiva a risultati test già osservati, non una
validazione prospettica priva di conoscenza del pilota. Le decisioni numeriche
e il ranking usano soltanto la validation; i vecchi test non entrano nel
selettore. Una piccola validation riutilizzata può sovradattare la selezione;
L9 non copre tutte le interazioni, e due seed di conferma non garantiscono un
ottimo globale. Dopo il congelamento non si riapre la ricerca.

Il confronto con FedBN/FedAvg/FedProx usa soltanto i valori pubblicati della
tabella11, conservati nel CSV originale. Non è una riesecuzione comune:
identità dei vecchi split/seed e modalità esatta delle statistiche non sono
certificate, il mirror dati è successivo al paper, Fused aggiunge AE e warm-up.
Accuratezze nostre sono ricostruite dai conteggi, tutti i cinque seed sono
riportati e le deviazioni fra seed sono campionarie, ddof=1. Le differenze
rispetto a medie pubblicate sono descrittive, senza test di significatività.
Parametri, passi, tempi, picchi memoria e conteggio FLOP convenzionale incluse
le due fasi sono registrati; i FLOP escludono BN/attivazioni/loss/copie e non
equivalgono a istruzioni hardware o energia. Hash dei byte test sono controlli
d'integrità, non valutazioni di un modello durante la calibrazione.

## Aggiornamento di sola orchestrazione richiesto dall'utente

Dopo lo screening e le prime due conferme, l'utente ha richiesto maggiore
concorrenza: le sei conferme rimaste eseguono3 processi su ciascuna GPU; le
cinque definitive eseguono3 processi su GPU1 e2 su GPU0. Il piano scientifico,
split, griglia, criterio, seed, epoche e round restano quelli registrati.
`execution_amendment.json` registra questa modifica organizzativa senza
riscrivere il piano originale. I due training già attivi continuano da dove
sono: il nuovo coordinatore adotta i PID e legge lo stato d'uscita Linux dei
figli del vecchio coordinatore sospeso. Solo il vecchio coordinatore viene
ritirato dopo l'uscita dei suoi figli; nessun segnale ai training o ai lavori
esterni. Tempi e commit sono attribuiti separatamente ai singoli worker.
