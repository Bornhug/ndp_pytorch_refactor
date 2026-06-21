# nanoTabPFN_version10_original02 R^2 Tables

- Source: `nanoTabPFN_version10_original02/results.txt`

- Per-dataset standard errors are recomputed from 5-fold CV fold scores using `nanoTabPFN_version10_original02/checkpoints_run01/model_final.pt` under the same `max_features_eval=32`, `new_instances_eval=200`, `n_splits=5`, `random_state=0` setup.

## R^2

| Dataset | R^2 |
|---|---:|
| abalone | 0.2872 +/- 0.0768 |
| airfoil_self_noise | 0.4735 +/- 0.0324 |
| auction_verification | 0.5817 +/- 0.0290 |
| boston | 0.7911 +/- 0.0326 |
| cars | 0.8860 +/- 0.0212 |
| concrete_compressive_strength | 0.6479 +/- 0.0687 |
| cpu_activity | 0.9502 +/- 0.0126 |
| energy_efficiency | 0.8514 +/- 0.0229 |
| grid_stability | 0.6930 +/- 0.0403 |
| kin8nm | 0.3746 +/- 0.0323 |
| Moneyball | 0.8982 +/- 0.0181 |
| pumadyn32nh | -0.0073 +/- 0.0098 |
| QSAR_fish_toxicity | 0.4425 +/- 0.0831 |
| quake | -0.1564 +/- 0.0488 |
| sensory | 0.0270 +/- 0.0389 |
| socmob | 0.6813 +/- 0.0560 |
| space_ga | 0.4647 +/- 0.0533 |
| student_performance | 0.1673 +/- 0.0364 |

## Dataset Metadata

| Dataset | OpenML Dataset ID | Original Samples | Features | Categorical Features |
|---|---:|---:|---:|---:|
| abalone | 42726 | 4177 | 8 | 1 |
| airfoil_self_noise | 44957 | 1503 | 5 | 0 |
| auction_verification | 44958 | 2043 | 7 | 2 |
| boston | 531 | 506 | 13 | 2 |
| cars | 44994 | 804 | 17 | 0 |
| concrete_compressive_strength | 44959 | 1030 | 8 | 0 |
| cpu_activity | 44978 | 8192 | 21 | 0 |
| energy_efficiency | 44960 | 768 | 8 | 0 |
| grid_stability | 44973 | 10000 | 12 | 0 |
| kin8nm | 44980 | 8192 | 8 | 0 |
| Moneyball | 41021 | 1232 | 14 | 6 |
| pumadyn32nh | 44981 | 8192 | 32 | 0 |
| QSAR_fish_toxicity | 44970 | 908 | 6 | 0 |
| quake | 550 | 2178 | 3 | 0 |
| sensory | 546 | 576 | 11 | 11 |
| socmob | 541 | 1156 | 5 | 4 |
| space_ga | 507 | 3107 | 6 | 0 |
| student_performance | 44967 | 649 | 30 | 17 |
