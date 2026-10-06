# PathMNIST: ricerca di recupero del 50,94%

Il 6 ottobre 2026 l’utente ha riaperto la ricerca dopo il risultato finale
35,279944%, revocando la deadline e chiedendo di fermarsi appena una soluzione
supera il 50,94%. I tentativi precedenti, i dati e il manoscritto restano
invariati. Il nuovo lavoro usa `general_ml`, senza installare pacchetti.

Ordine iniziale, basato sul rapporto fra probabilità qualitativa di successo
e costo stimato; non si tratta di probabilità numeriche misurate:

| Priorità | Soluzione | Motivo | Costo stimato |
|---|---|---|---|
| 1 | BN condivisa ricalibrata su training nel checkpoint originale seed 42, round 50 | Il controllo FP32 con LR/epoche originali migliora da 24,838781 a 34,392929% su validation a 20 round; il checkpoint originale completo è disponibile | 1–3 minuti, nessun training |
| 2 | Ricalibrazione BN che rappresenti gli encoder privati sugli input globali, e statistiche coerenti con la modalità eval | Durante il test ogni encoder vede tutte le classi; la calibrazione precedente usa solo le classi locali del suo client | Pochi minuti di validation sui checkpoint fit già disponibili |
| 3 | Prolungare su validation i profili promettenti fino all’orizzonte finale di 50 round | Il ranking a 20 round non ha previsto l’esito a 50; riutilizzare i checkpoint evita 20 round per tentativo | Circa 20–45 minuti per candidato, GPU in parallelo |
| 4 | Normalizzazione indipendente dalla composizione dei batch, e regularizzazione/ottimizzazione mirate | La BN sotto due classi per client è una possibile fonte di incoerenza; cause non ancora isolate | Nuovi training di validation, circa 30–60 minuti per run |
| 5 | Varianti dichiarate delle due fasi, della loss di ricostruzione e della fusione | Solo se gli interventi più economici non bastano; mantenere riconoscibile il meccanismo FusedSpaceFed | Budget da aggiornare sulla base dei costi misurati |

La prima soluzione mantiene encoder privati persistenti, decoder e pesi del
classificatore condivisi, fusione additiva, training originale e optimizer.
Modifica esclusivamente i buffer BN condivisi con 640 immagini di fit per
client, senza label, gradienti, aggiornamenti di pesi o immagini test. È una
variante d’inferenza rispetto al paper. La scelta della modalità è sostenuta
dal controllo di validation già disponibile; il trasferimento dal controllo
FP32 a 20 round al checkpoint FP16 a 50 round è un limite esplicito.

Ogni passo successivo sarà dichiarato prima delle sue valutazioni. Gli
iperparametri saranno scelti su validation derivata dal training, evitando
di validare su immagini già usate per allenare il checkpoint. Nessuna
baseline viene rieseguita. Architettura e dati restano quelli concordati
finché una variante non è esplicitamente documentata.

Il test ufficiale è già stato osservato. La richiesta di cercare fino al
superamento della soglia introduce uno stop adattivo su un benchmark noto:
un eventuale successo sarà un risultato esplorativo di un singolo seed,
non una conferma indipendente della media del paper. Tutti i tentativi e
le valutazioni, anche negativi, saranno conservati. Nessuna media sarà
sostituita dal migliore encoder o seed. Il manoscritto non sarà modificato.

## Primo esito e secondo passo congelato

Il primo tentativo ha ottenuto **43,938719%**, quindi non supera la soglia.
La calibrazione/inferenza ha richiesto 3,510 secondi di sessione, oltre al
training originale già eseguito; costi e receipt del processo sono archiviati.
I checkpoint completi prima/dopo la calibrazione sono conservati in
`_local/pathmnist_recovery/attempt-01/`.

Il passo 2 confronta cinque modalità: native, owner-cumulative,
owner-layerwise, cross-cumulative e cross-layerwise. La calibrazione usa
sempre lo stesso pool di 6.400 esempi fit unici. Owner applica a ogni immagine
l’encoder del suo client d’origine; cross assegna gli stessi esempi agli
encoder con una permutazione fissa, indipendente dal client e dalla label,
640 esempi per encoder. I buffer risultanti restano interamente condivisi.

Cumulative replica la media progressiva di statistiche in training mode.
Layerwise misura media e varianza globale degli input di ciascuna BN,
seguendo l’ordine del grafo: i livelli precedenti usano già le statistiche
finali in eval. Cambiano solo i buffer, senza pesi, label o optimizer step.
Sono scelte nostre, non un protocollo recuperato dagli autori.

Le cinque modalità vengono confrontate su sei checkpoint fit-only seed 142:
il riferimento FP32, i profili c0.1/ae0.0001, c0.1/ae0.001, c0.03/ae0.001,
c0.01/ae0.0003 (round 20), e il checkpoint finito round 18 del riferimento
FP16 fallito al round 19. Il round 18 è un controllo separato, non un risultato
a 20 round. Sei worker indipendenti, tre per GPU, valutano soltanto le 8.994
immagini di validation. Per un successivo trasferimento al checkpoint
originale a 50 round, la modalità sarà scelta usando il riferimento FP32
con gli stessi LR/epoche, con parità a favore di native e poi dell’ordine
predefinito. Il riferimento FP16 a 18 round servirà come controllo di
trasferimento, senza sostituire il criterio dichiarato.

## Passo 3: orizzonte coerente e normalizzazione durante il training

Il secondo tentativo, cross-layerwise scelto sul riferimento di validation,
ha ottenuto **44,299443%** sul test: non supera la soglia. Il piccolo guadagno
di validation delle nuove calibrazioni non giustifica altre varianti BN
simili. La ricerca passa al training, senza cambiare le partizioni.

Quattro checkpoint fit-only seed 142 vengono copiati in nuove directory e
ripresi esattamente dal round 20 al round 50: riferimento FP32, c0.1/ae0.0001,
c0.1/ae0.001 e c0.03/ae0.001. Si conservano stati privati, optimizer, generatori
dei loader e RNG. I sorgenti/iperparametri precedenti sono identici; i round
20 e i loro risultati restano archiviati. Selezione al terminale round 50,
mai sul migliore round. Due worker per GPU lasciano un posto per GPU a due
nuovi training indipendenti.

Due varianti FusedSpaceFed-GN iniziano da zero sul fit, seed 142, 50 round.
Sostituiscono soltanto le 19 BN del ResNet20-v2 con GroupNorm a 8 gruppi,
stessi canali e parametri affini; i pesi del classificatore restano condivisi.
Questo elimina le statistiche accumulate dei batch: la normalizzazione è
per immagine ed è la stessa durante training e inferenza. È una modifica
di architettura/normalizzazione rispetto al paper, da riportare come tale.
Il motivo è un’ipotesi sulla coerenza dei gradienti fra client a due classi,
non una causa già dimostrata del risultato negativo. Fonti tecniche:
[Wu e He, Group Normalization, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html),
[documentazione ufficiale PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.GroupNorm.html).

Profili GN: (1) SGD LR 0,03, Adam/warm-up LR 0,0003, CE 3, nessuna augmentation;
(2) SGD LR 0,1, Adam LR 0,0003, warm-up LR 0,0001, CE 1, flip e rotazioni di
90 gradi solo durante il training. Entrambi mantengono warm-up 1, batch 128,
BF16, clipping 5, SGD senza momentum/decay e Adam persistente, dz16, encoder
privati, decoder/classificatore condivisi e fusione additiva. L’augmentation
è una nostra ipotesi di invarianza dell’orientamento dei tessuti e non viene
applicata al test. Nessun nuovo esempio o sorgente dati.

Si seleziona la configurazione/modo con la migliore accuratezza uniforme su
validation a 50 round. Per GN è ammessa soltanto inferenza nativa, senza
ricalibrazione o adattamento al test. La successiva prova usa seed 42 e
training completo: riutilizza pesi finali compatibili se già disponibili,
altrimenti parte da zero. Il confronto con il valore pubblicato rimane
esplorativo e ogni modifica al protocollo viene identificata.

Per non aspettare inutilmente le prove GN più lunghe, la selezione è organizzata
in due onde prima di leggere i risultati a 50 round: prima il massimo di
validation dei quattro profili esistenti completati; poi, se il suo test non
supera la soglia, il massimo dei due profili GN. Dentro ciascuna onda vale il
massimo dei conteggi corretti al round 50, con parità a favore di native e poi
dell’ordine dichiarato. Un training finale dell’onda 1 può sovrapporsi alle
validation GN. Le valutazioni test restano sequenziali, con configurazione
congelata per ciascun tentativo. Se un tentativo supera il 50,94%, non si
avviano altri candidati; eventuali lavori nostri ancora attivi vengono
chiusi conservando l’ultimo checkpoint completo e indicando lo stop richiesto
dall’utente. I processi di altri utenti non vengono interrotti.

Le quattro continuazioni hanno ottenuto un massimo di validation 48,475650%,
inferiore alla soglia cercata; il nuovo budget prioritizza quindi i profili
GN prima di investire in un ulteriore training finale simile ai precedenti.
È congelato anche un controllo diagnostico quasi gratuito dell’inferenza
nativa sui pesi della run completa c0.1/ae0.0001 seed 42: erano stati testati
soltanto con BN ricalibrata (35,279944%). Questo controllo mantiene tutti i
pesi e gli iperparametri e ripristina l’inferenza del paper. **La modalità
nativa non è quella preferita dalla validation**: il suo test è una diagnosi
esplorativa del protocollo d’inferenza sul benchmark già noto, e non verrà
presentato come una modalità selezionata su validation indipendente.

Il controllo nativo ha ottenuto 25,023677%, quindi non recupera la soglia.
Le due varianti GN hanno ottenuto 29,177229% (CE1 + flip/rot90) e 33,476762%
(CE3), soltanto su validation; non vengono valutate sul test.

Il prossimo passo rapido confronta guadagni di fusione α = 0, 0.1, 0.25,
0.5, 1 e 2, con formula `x + α D(E_i(x))`, sui checkpoint fit-only a 50
round. La linearità della convoluzione finale del decoder permette di
applicare esattamente il guadagno, in una copia, senza modificare encoder
o classificatore. Le BN vengono eventualmente ricalibrate solo sul fit.
α=0 è un controllo diagnostico che esclude il percorso privato: **non è
ammesso come soluzione FusedSpaceFed o come criterio di stop sul test**.
I guadagni positivi sono varianti esplicite rispetto alla fusione α=1 del
paper. La selezione usa soltanto validation; nessun nuovo test è letto.

## Passo 4 congelato: fusione positiva attenuata

Le quattro sonde a 50 round sono concluse. Il riferimento FP32 con LR/epoche
originali raggiunge **52,066934%** su validation con **α=0,25 e BN
owner-cumulative da fit** (46.829/89.940 predizioni), contro 40,182344% con
α=1 e la stessa modalità BN. La sonda migliore degli altri profili positivi
raggiunge 48,702468%; α=0 rimane escluso. Il massimo di conteggi sul
riferimento compatibile, con parità a favore di α=1, poi native e ordine dei
guadagni, determina `attempt-04.json` prima del suo test.

Per il primo trasferimento si riutilizza il checkpoint originale completo
seed42, round50, allenato FP16 con gli stessi LR/epoche. È un limite:
il modello di validation usa FP32, fit e seed142; non sono traiettorie
identiche. Non si dichiara una nuova run da zero con questa variante.
La prova costa secondi; se fallisce, una run FP32 da zero con la stessa
configurazione selezionata resta un passo successivo possibile.

La formula d'inferenza è `x + 0.25 D(E_i(x))`: tutti gli encoder privati,
i pesi e gli optimizer allenati sono preservati, ma la fusione e i buffer
BN differiscono dal protocollo del paper. Il checkpoint finale conserva
il decoder **allenato originale**, non un decoder riscalato con momenti
Adam incompatibili; il coefficiente è in `recovery.config.fusion_gain`.
Per riprodurre l'inferenza si usa `inference.inference_decoder(checkpoint)`.
`precalibration.pt` conserva lo stato precedente anche per una ripresa
esatta del training originale. L'architettura e il numero di parametri
sono invariati. La ricalibrazione usa sempre i 6.400 esempi fit fissati,
senza label o dati test. Questa è una variante d'inferenza selezionata su
validation, con stop esplorativo su un test già noto.

Il trasferimento α=0,25 ha ottenuto **43,168524%** sul test e non supera la
soglia. Il prossimo controllo rapido usa sei worker (3+3 GPU) sui checkpoint
fit-only a 50 round: riferimento FP32 con α=0,1/0,25/1, c0.03/ae0.001 con
α=0,5/1, c0.1/ae0.0001 con α=0,1. Ogni profilo usa BN da fit e confronta
cinque penalità L2 del solo ultimo strato condiviso: 0, 1e-4, 1e-3, 1e-2,
0,1. Si conserva anche il controllo senza refit. Questa **fase aggiuntiva**
non appartiene al paper: corpo del classificatore, encoder privati e decoder
sono fissi; la testa 64→9 condivisa ottimizza la media delle CE dei client,
pesati uniformemente, con L-BFGS (massimo 100 iterazioni). Il termine L2 è
0,5 λ ||W_fc||², senza penalità sul bias. Tutte le feature training sono
prodotte dall'encoder del client d'origine; non si assegnano immagini fit
a encoder di altri client. I gradienti dell'obiettivo equivalgono alla
media di gradienti locali full-batch. Il simulatore conserva feature locali
in memoria: non si afferma un protocollo di privacy/deployment equivalente
al metodo originale, né un costo equivalente alle sue sole due fasi.

Si sceglie il massimo dei conteggi di validation terminale sui profili
registrati; in parità si preferisce il controllo senza refit, poi l'ordine
dei profili e delle penalità. Non si usano test o adattamento al test per
questa selezione. I nuovi optimizer della testa e ogni tentativo rimangono
archiviati. La ricerca di training aggiuntivo viene rimandata fino all'esito
di questo controllo più economico.
