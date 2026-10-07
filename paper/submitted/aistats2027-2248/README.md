# Versione sottoposta ad AISTATS 2027 — submission 2248

Questa cartella conserva la versione effettivamente sottoposta, fornita
dall'autore e archiviata nel repository l'8 ottobre 2026.

- [PDF inviato](2248_FusedSpaceFed_Enhanced_Fe.pdf): copia originale, senza ricompilazione.
- [ZIP originale esportato da Overleaf](federated_fgd.zip).
- [Sorgenti LaTeX estratti](source/): tutti i 34 file dello ZIP, con contenuto invariato.

La cartella dei sorgenti è presente anche nel progetto Overleaf con il nome
`aistats-inviato_20261007`. Lo ZIP originale
è l'esportazione fornita dall'autore per questa versione sottoposta.

Le modifiche manuali dell'autore sono preservate. Per le revisioni successive,
questa cartella costituisce il riferimento della submission; i sorgenti
precedenti rimangono disponibili in `paper/` e nella cronologia Git.

## Sorgente principale

Il file da compilare per la versione senza colori di revisione è
`source/aistats_2027_mau2_clean.tex`. Questo richiama
`source/aistats_2027_mau2.tex`, che contiene il manoscritto.

Con una distribuzione TeX completa, dalla radice del repository:

```bash
cd paper/submitted/aistats2027-2248/source
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error aistats_2027_mau2_clean.tex
bibtex aistats_2027_mau2_clean
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error aistats_2027_mau2_clean.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error aistats_2027_mau2_clean.tex
```

Il PDF di riferimento rimane quello originale inviato, conservato nella
cartella superiore. L'archiviazione non modifica i sorgenti e non certifica
l'identità di una futura ricompilazione con il PDF sottoposto.

## Integrità dei file originali

SHA-256:

```text
federated_fgd.zip
13c59cba21291ab4e0c571261d6d74112899ad039a5dba69348e736fef74b168

2248_FusedSpaceFed_Enhanced_Fe.pdf
8908f6de698a3de13b7314cc09a033cae5e16e6835cf0b1892d598af8ab4f3db
```

[SHA-256 di tutti i file originali](SHA256SUMS) permette di verificare anche
ogni sorgente estratto.

