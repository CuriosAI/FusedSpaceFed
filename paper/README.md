# Sorgenti e riproduzione del manoscritto

Il sorgente principale è `aistats_2027.tex`, nella versione AISTATS scelta
dall'autore. Alla loro importazione, i sei file erano copie byte per byte dei
file omonimi nell'archivio originale `Federated_FGD.zip`. La revisione
scientifica del 4 ottobre 2026 modifica manoscritto e bibliografia;
stili e immagini storiche rimangono invariati. Anonimizzazione e limite di
pagine sono rinviati.

| File | Ruolo |
|---|---|
| `aistats_2027.tex` | Manoscritto principale |
| `references.bib` | Bibliografia |
| `aistats2027.sty` | Stile AISTATS richiesto dal manoscritto |
| `fancyhdr.sty` | Dipendenza locale richiesta dallo stile AISTATS |
| `accuracy_FSF.jpg` | Figura dell'accuratezza |
| `gamma_FSF.jpg` | Figura della dispersione dei gradienti |

Le figure già presenti alla radice del repository sono conservate nella loro
posizione. Il PDF Springer fornito dall'autore rimane un riferimento; gli altri
manoscritti dello ZIP rimangono fra gli originali locali.

## Provenienza e punto di partenza

Importazione del 3 ottobre 2026 su `main`, dopo un pull fast-forward che ha
confermato l'allineamento con `origin/main`.

- Commit precedente all'importazione: `25fed731ff1aba2d7cf06da775db477b03f2af85`.
- SHA-256 di `Federated_FGD.zip`:
  `b19204bc6c797b30ec1ec50da2be4f26b25ee1f626e5bd7d1435c19bfd52c5a1`.
- SHA-256 del sorgente originale `aistats_2027.tex`:
  `64a2d707d624f593b7eb29c089b0e3185b372ed52bed0d6768a8b1ed1a032377`.

Gli input originali, gli hash completi, l'estrazione con controllo dei percorsi
e il diff ricontrollato sono conservati in `_local/`, esclusa da Git in questo
clone. Il confronto con `machine_learning.tex` conferma la conservazione di
algoritmo, proposizione, prova, corollario, celle delle cinque tabelle, file e
didascalie delle due figure, 22 etichette e 32 chiamate bibliografiche attive.
Questo controllo riguarda il contenuto dei sorgenti.

## Riproduzione delle nuove tabelle e figure

Dalla radice del repository, con l'ambiente `general_ml` già disponibile:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python scripts/build_paper_artifacts.py
/home/schroeder/miniconda3/envs/general_ml/bin/python scripts/build_paper_artifacts.py --check
```

Il generatore produce nove tabelle in `paper/generated/`, tre grafici in PDF
e SVG in `paper/figures/` e CSV/JSON con provenienza e hash in
`artifacts/paper_revision/`. Il controllo richiede uguaglianza byte per byte,
verifica gli hash dell'archivio FEMNIST e ricostruisce le statistiche dai
conteggi corretti/totali di tutti i client. Non esegue training, valutazione
di modelli, tuning o download. Le istruzioni dettagliate sono in
`artifacts/paper_revision/README.md`.

I 144 valori storici sono recuperati dall'oggetto Git immutabile
`fc067b423115a2b5e0c51bae5a025b86aeaf37d1`: sono aggregati del manoscritto,
non file originali per seed. Le varianze mediche mancanti restano mancanti;
la dispersione FEMNIST naturale è conservata come riportata, senza certificarne
una convenzione non verificabile. Le cinque run ricostruite hanno invece
conteggi completi, tutti i seed e SD campionaria con `ddof=1`.
I riferimenti FedRep sono in una tabella distinta, dal CSV originale invariato.
Le immagini JPG rimangono in appendice: mancano i dati numerici necessari a
rigenerarle fedelmente.

## Compilazione della revisione

Servono pdfLaTeX, BibTeX e i pacchetti LaTeX dichiarati nel preambolo; lo stile
bibliografico `apalike.bst` è fornito dalla distribuzione TeX. Dalla radice del
repository, eseguire nell'ordine, proseguendo solo se ciascun comando riesce:

```bash
cd paper
mkdir -p build
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build aistats_2027.tex
bibtex build/aistats_2027
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build aistats_2027.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build aistats_2027.tex
```

Il PDF, se la compilazione termina correttamente, è
`build/aistats_2027.pdf`. Log e temporanei rimangono in `build/`, ignorata da Git.

La revisione compila con i pacchetti già disponibili, senza installazioni.
L'algoritmo usa `float` ed elenchi LaTeX, evitando le tre dipendenze mancanti
all'importazione. I quattro passaggi terminano con codice 0 e citazioni e
riferimenti risolti. Rimane un avviso tipografico di 5,12 pt nel blocco iniziale
dello stile, oltre agli avvisi di spaziatura underfull.

## Verifica storica dell'importazione

Il 3 ottobre 2026 sono state verificate l'identità dei sei file importati
rispetto allo ZIP, l'esistenza dei file locali referenziati e la presenza in
`references.bib` di tutte le chiavi bibliografiche citate.

La distribuzione già disponibile è TeX Live 2023/Debian, con pdfTeX
3.141592653-2.6-1.40.25 e BibTeX 0.99d. Il controllo dei pacchetti dichiarati
ha rilevato tre dipendenze di sistema mancanti: `algorithm.sty`,
`algorithmicx.sty` e `algpseudocode.sty`.

La prova con il primo comando pdfLaTeX riportato sopra è terminata con codice
1, fermandosi su `File algorithm.sty not found`; il PDF non è stato prodotto.
La verifica iniziale e i suoi log furono conservati localmente;
`build/aistats_2027.log` contiene ora la compilazione aggiornata.
All'importazione, BibTeX e i passaggi successivi non erano stati eseguiti.

Questa è la verifica storica dell'importazione, che conservò i file originali
senza correzioni. La successiva revisione supera il blocco di compilazione
senza modificare lo stile o installare pacchetti.
