# Performance objective before freezing the configuration

The user prioritised a final accuracy above the paper's PathMNIST value of
**50.94%** while retaining the short execution budget. This instruction was
received while the last predefined screening candidates were still running,
before configuration selection or any new test evaluation.

The active procedure maximises validation accuracy among the 24 original
profiles and their two already declared inference modes at screening seed
142, round 20. It then trains one fresh seed-42 model on the complete training
partition for 50 rounds, retaining all selected hyperparameters and the
schedule horizon. There are no added candidates, confirmations or extensions.

The threshold is a performance objective, not a criterion for selecting a
test checkpoint or reopening calibration. The test score cannot be
guaranteed from validation on a different split. Whether the frozen run
exceeds 50.94% will be reported explicitly, including an unsuccessful outcome.
Only one final seed is run, so no between-seed SD or statistical superiority
can be claimed. The manuscript's value is not changed.
