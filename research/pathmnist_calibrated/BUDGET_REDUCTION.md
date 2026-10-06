# Riduzione del budget — disposizione dell'utente

La disposizione successiva dell'utente sostituisce la parte relativa a
conferme, estensioni e campagna finale del piano originale. I file protetti
dallo screening (`search_plan.json`, dati, partizione e codice di training)
restano immutati; questo documento e `budget_reduction.json` sono separati.

- Soltanto i **24 candidati già registrati**, seed142, **20 round**.
- Nessuna conferma sui seed142/143, nessuna estensione a100 round,
  nessun candidato/architettura/analisi aggiuntiva.
- Selezione diretta della maggiore accuratezza di validation dello
  screening, fra le due modalità dichiarate (`native`, `train-recalibrated`).
  In caso di parità: BN nativa, poi ordine di inserimento del candidato
  nel piano originale. Candidati con errore numerico non hanno uno score
  valido e non sono sostituiti silenziosamente con altre impostazioni.
- Un solo profilo congelato e **una run finale nuova seed42**, training
  completo89.996 immagini e stessa partizione dati42, **50 round**,
  unico test al termine. Gli iperparametri del candidato, incluso l'orizzonte
  cosine100 se applicabile, restano quelli dello screening.
- Runner esistente, una GPU per la run finale; nessuna nuova implementazione
  di parallelismo/ottimizzazione. Entrambe le GPU per i tentativi di screening.
- Un commit per calibrazione/configurazione congelata e push, poi un commit
  per risultato finale/report e push. Manoscritto e altri esperimenti invariati.

Il primo lotto è già terminato senza interventi sui processi. Il riferimento
FP16 ha una loss non finita al round19, ultimo checkpoint completo18;
l'esito resta conservato e dichiarato. Diciotto tentativi successivi si sono
fermati prima di creare output per una collisione fra `select.py` e il modulo
standard Python. L'helper è stato rinominato `selection_tools.py`; i tentativi
con errore d'avvio sono ripetuti con **le stesse configurazioni e gli stessi
hash scientifici**, senza nuovi candidati, mantenendo tutti i log e receipt.

Il report dichiarerà: selezione su un seed e a20 round, assenza di conferme
o estensione, singola run finale senza SD fra seed, costi misurati e differenze
di precisione/clipping/LR/schedule/epoche o BN rispetto al paper. Le ablation
e diagnostiche rimangono sospese. Non si riapre il tuning dopo il test.
