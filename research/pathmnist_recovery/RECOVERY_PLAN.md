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
