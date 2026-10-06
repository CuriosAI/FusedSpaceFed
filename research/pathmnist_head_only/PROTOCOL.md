# PathMNIST: refit della sola testa sul checkpoint originale

Controllo unico richiesto dall'utente dopo il recupero con tre modifiche.
Si usa **soltanto** il checkpoint originale paper-settings seed42, round50:
`_local/pathmnist_pathological/seed-42/final.pt`, SHA256
`fa56da01091debe9031a9acb2e3de0f34ce0f751d2c95d2d0d54c05f15697d1e`.
SGD0,01, Adam0,001, warm-up1/CE3, batch128,50 round, optimizer persistenti;
la run originale usa FP16 AMP. Non si usano pesi tuned o checkpoint derivati.

Mantiene `x+D(E_i(x))`, tutti gli encoder privati, decoder condiviso e
**tutti i pesi e buffer BN originali** del corpo ResNet20-v2. Nessuna
ricalibrazione. Feature64 estratte in eval/no_grad Float32 dalle89.996
immagini training, ciascuna con l'encoder del suo client d'origine.

Si ottimizzano soltanto i585 parametri fc.weight/fc.bias della testa64→9,
inizializzati dai pesi terminali, con la stessa funzione `refit` dell'ultimo
esperimento: L-BFGS LR1, max_iter100, history20, strong-Wolfe,
tolerance_grad1e-7, tolerance_change1e-10, penalità zero. CE full-batch
media uniforme fra10 client, ognuno con il proprio numero di campioni.
Feature detached; nessun backward su encoder/decoder/corpo/BN.

Nessun tuning, validation, selezione checkpoint o stop per accuracy.
Si salva il checkpoint completo post-refit **prima** di leggere i nuovi
valori test. Prima/dopo: medesimo test7.180 immagini, ogni encoder con
gli stessi stati condivisi, media uniforme di tutte le10 accuratezze.
Le71.800 predizioni riusano gli stessi7.180 esempi; non sono indipendenti.
Il confronto prima deve riprodurre i conteggi originali archiviati.

Solo Cfc cambia; anche i buffer BN/counter devono restare bit-identici.
Tutti gli stati optimizer/scaler/RNG/partizione della run originale sono
conservati, con optimizer L-BFGS e RNG del refit separati. Il checkpoint
resta utilizzabile con la normale inferenza a guadagno1. I checkpoint
`before.pt`, `training-initial.pt`, `final.pt` e i log restano in `_local/`.
Risultati numerici, hash, comando/exit status e breve report sono pubblicati.

Questo isola l'effetto del refit **per questa singola run**. Rimane una
fase aggiuntiva al metodo del paper, con feature cached nel simulatore e
un obiettivo equivalente alla media di gradienti locali full-batch;
privacy/traffico di rete non sono misurati. Non certifica una media di
cinque seed, superiorità statistica o il beneficio causale degli altri
componenti. Manoscritto e precedenti esperimenti invariati.
