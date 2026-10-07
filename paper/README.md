# Manoscritto: versione sottoposta e sorgenti precedenti

La **versione effettivamente sottoposta ad AISTATS 2027 (submission 2248)** è
archiviata in [submitted/aistats2027-2248](submitted/aistats2027-2248/), con PDF
inviato, ZIP originale e sorgenti LaTeX invariati. È il riferimento per le
revisioni successive. La cartella dei sorgenti è presente anche nel progetto
Overleaf con il nome `aistats-inviato_20261007`.

Il resto di questo documento descrive l'importazione precedente del 3 ottobre
2026 e il sorgente `aistats_2027.tex`, conservato come versione precedente.

## Sorgenti precedenti

Il sorgente principale è `aistats_2027.tex`, nella versione AISTATS scelta
dall'autore. I sei file importati sono copie byte per byte dei file omonimi
contenuti nell'archivio originale `Federated_FGD.zip`.

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

## Compilazione

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

## Verifica dell'importazione

Il 3 ottobre 2026 sono state verificate l'identità dei sei file importati
rispetto allo ZIP, l'esistenza dei file locali referenziati e la presenza in
`references.bib` di tutte le chiavi bibliografiche citate.

La distribuzione già disponibile è TeX Live 2023/Debian, con pdfTeX
3.141592653-2.6-1.40.25 e BibTeX 0.99d. Il controllo dei pacchetti dichiarati
ha rilevato tre dipendenze di sistema mancanti: `algorithm.sty`,
`algorithmicx.sty` e `algpseudocode.sty`.

La prova con il primo comando pdfLaTeX riportato sopra è terminata con codice
1, fermandosi su `File algorithm.sty not found`; il PDF non è stato prodotto.
I log sono conservati localmente in `build/aistats_2027.log` e
`build/pdflatex-import.stdout.log`. BibTeX e i passaggi successivi non sono
stati eseguiti.

La preparazione delle dipendenze mancanti e l'eventuale correzione di ulteriori
errori di compilazione sono rinviate a un incarico successivo. L'importazione
conserva tutti i file originali senza correzioni al sorgente o allo stile.
