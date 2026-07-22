# Current State

- Goal: retrain DPA-KT full model for 200 epochs with 5-fold CV on every
  dataset, then materialize the results into two notebooks (ENG + VN) and
  extract figures from the sweep.
- Stage: sweep running as `bgp_f82556a75001E4mvHUPacYNb9T` (19/35 folds
  done as of 2026-07-22 01:38 UTC). Figure watcher running as
  `bgp_f878359f0001FeJ8AlPpUdSlfA`, polls for `sweep_manifest.json` and
  will auto-generate figures + embed them in both notebooks when done.
- Current task: wait for sweep + figure watcher.
- Next action: poll sweep progress; once `sweep_manifest.json` appears
  the watcher will write `code/notebooks/figures-200/*.png` and update
  `DPA_KT_200_epochs_{ENG,VN}.ipynb`.