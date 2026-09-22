# Per-competition scoreboard — is one agnostic model the best everywhere?

Every engine below is scored on the same 21 exact-kickoff tournament holdouts and
the same 98 slates, under both competition rubrics. `empirical` is the engine the
NCR incumbent is built on, so the agnostic-vs-competition-specific comparison is
`E4` against it. The Six Nations champion is absent because it only has two folds
of 13-target coverage; that comparison lives in `allrugby_sixnations/LEDGER.md`.

`mae` is raw predicted-vs-actual fantasy points; `mae_calibrated` applies a
leave-one-fold-out affine recalibration, so neither is in-sample.

## Pooled over all 98 slates

### `six_nations` rubric

| engine                           | mae    | mae_calibrated | spearman | mean_capture | slates |
| -------------------------------- | ------ | -------------- | -------- | ------------ | ------ |
| C10 (phase 2)                    | 6.0932 | 6.1944         | 0.6311   | 0.773        | 98     |
| C9 (phase 2)                     | 6.0997 | 6.1907         | 0.6314   | 0.7755       | 98     |
| E4 (phase 2)                     | 6.1075 | 6.1707         | 0.6339   | 0.775        | 98     |
| empirical t3                     | 6.1587 | 6.2309         | 0.6279   | 0.7696       | 98     |
| C5 (phase 1)                     | 6.1771 | 6.1936         | 0.6299   | 0.7709       | 98     |
| empirical (NCR incumbent engine) | 6.1972 | 6.2262         | 0.6248   | 0.7725       | 98     |
| P3 frozen                        | 6.2767 | 6.2478         | 0.6287   | 0.7689       | 98     |
| stratum mean                     | 6.3568 | 6.5216         | 0.5916   | 0.7215       | 98     |
| v4 GBDT                          | 6.6093 | 6.5067         | 0.6127   | 0.7487       | 98     |

### `ncr` rubric

| engine                           | mae    | mae_calibrated | spearman | mean_capture | slates |
| -------------------------------- | ------ | -------------- | -------- | ------------ | ------ |
| E4 (phase 2)                     | 6.0254 | 6.0611         | 0.5434   | 0.73         | 98     |
| empirical t3                     | 6.0467 | 6.1096         | 0.5367   | 0.7218       | 98     |
| empirical (NCR incumbent engine) | 6.0608 | 6.0869         | 0.5343   | 0.7279       | 98     |
| C5 (phase 1)                     | 6.0669 | 6.0677         | 0.5415   | 0.7278       | 98     |
| C10 (phase 2)                    | 6.0737 | 6.066          | 0.5434   | 0.7294       | 98     |
| C9 (phase 2)                     | 6.0737 | 6.066          | 0.5434   | 0.7294       | 98     |
| P3 frozen                        | 6.1727 | 6.1305         | 0.5367   | 0.7187       | 98     |
| stratum mean                     | 6.2701 | 6.3929         | 0.489    | 0.6674       | 98     |
| v4 GBDT                          | 6.5178 | 6.3806         | 0.5182   | 0.7023       | 98     |

## Per tournament family

### `six_nations` rubric — mae

| tournament            | P3 frozen | C5 (phase 1) | E4 (phase 2) | C9 (phase 2) | C10 (phase 2) | empirical (NCR incumbent engine) | empirical t3 | v4 GBDT | stratum mean | best          |
| --------------------- | --------- | ------------ | ------------ | ------------ | ------------- | -------------------------------- | ------------ | ------- | ------------ | ------------- |
| autumn_internationals | 6.3523    | 6.2777       | 6.2125       | 6.1939       | 6.2056        | 6.3053                           | 6.2729       | 6.5637  | 6.5923       | C9 (phase 2)  |
| british_irish_lions   | 5.9024    | 5.8891       | 5.8677       | 5.8729       | 5.8478        | 5.9073                           | 5.9588       | 6.0138  | 6.0169       | C10 (phase 2) |
| nations_championship  | 6.472     | 6.3746       | 6.4199       | 6.4052       | 6.439         | 6.4095                           | 6.4503       | 6.8153  | 6.9285       | C5 (phase 1)  |
| pacific_nations_cup   | 7.7907    | 7.6664       | 7.6739       | 7.693        | 7.7116        | 7.7177                           | 7.642        | 8.0581  | 7.9149       | empirical t3  |
| rugby_championship    | 6.0824    | 5.8261       | 5.7082       | 5.6914       | 5.6751        | 5.8315                           | 5.7258       | 6.6675  | 5.9629       | C10 (phase 2) |
| rugby_world_cup       | 6.4833    | 6.2748       | 6.2643       | 6.273        | 6.2659        | 6.3101                           | 6.4152       | 7.1474  | 6.6317       | E4 (phase 2)  |
| six_nations           | 5.7698    | 5.7462       | 5.6393       | 5.6397       | 5.638         | 5.7272                           | 5.675        | 6.0299  | 5.9056       | C10 (phase 2) |
| summer_internationals | 6.5316    | 6.5336       | 6.4758       | 6.4467       | 6.427         | 6.6022                           | 6.5587       | 6.6678  | 6.5764       | C10 (phase 2) |

### `six_nations` rubric — spearman

| tournament            | P3 frozen | C5 (phase 1) | E4 (phase 2) | C9 (phase 2) | C10 (phase 2) | empirical (NCR incumbent engine) | empirical t3 | v4 GBDT | stratum mean | best         |
| --------------------- | --------- | ------------ | ------------ | ------------ | ------------- | -------------------------------- | ------------ | ------- | ------------ | ------------ |
| autumn_internationals | 0.6174    | 0.6277       | 0.6321       | 0.6282       | 0.6249        | 0.62                             | 0.627        | 0.603   | 0.5834       | E4 (phase 2) |
| british_irish_lions   | 0.5973    | 0.5912       | 0.5925       | 0.586        | 0.5889        | 0.5889                           | 0.5858       | 0.5992  | 0.5466       | v4 GBDT      |
| nations_championship  | 0.619     | 0.6243       | 0.6286       | 0.6321       | 0.6285        | 0.6287                           | 0.6285       | 0.6026  | 0.5521       | C9 (phase 2) |
| pacific_nations_cup   | 0.6249    | 0.6248       | 0.6296       | 0.6281       | 0.6257        | 0.6185                           | 0.6281       | 0.6078  | 0.6101       | E4 (phase 2) |
| rugby_championship    | 0.633     | 0.6342       | 0.6369       | 0.6339       | 0.6341        | 0.6256                           | 0.6356       | 0.6166  | 0.5809       | E4 (phase 2) |
| rugby_world_cup       | 0.6359    | 0.6402       | 0.636        | 0.6326       | 0.6349        | 0.6323                           | 0.6215       | 0.6081  | 0.5787       | C5 (phase 1) |
| six_nations           | 0.6734    | 0.6736       | 0.6795       | 0.6791       | 0.6782        | 0.6733                           | 0.6764       | 0.6512  | 0.6402       | E4 (phase 2) |
| summer_internationals | 0.5724    | 0.5715       | 0.5796       | 0.5759       | 0.5762        | 0.565                            | 0.5643       | 0.5652  | 0.5571       | E4 (phase 2) |

### `six_nations` rubric — mean_capture

| tournament            | P3 frozen | C5 (phase 1) | E4 (phase 2) | C9 (phase 2) | C10 (phase 2) | empirical (NCR incumbent engine) | empirical t3 | v4 GBDT | stratum mean | best          |
| --------------------- | --------- | ------------ | ------------ | ------------ | ------------- | -------------------------------- | ------------ | ------- | ------------ | ------------- |
| autumn_internationals | 0.7382    | 0.7488       | 0.7568       | 0.7613       | 0.7534        | 0.7517                           | 0.7351       | 0.6866  | 0.6958       | C9 (phase 2)  |
| british_irish_lions   | 0.7959    | 0.796        | 0.7923       | 0.7892       | 0.7907        | 0.7893                           | 0.7959       | 0.7977  | 0.7519       | v4 GBDT       |
| nations_championship  | 0.7132    | 0.7232       | 0.7263       | 0.7223       | 0.7306        | 0.7275                           | 0.7001       | 0.6786  | 0.5689       | C10 (phase 2) |
| pacific_nations_cup   | 0.7866    | 0.782        | 0.7806       | 0.7772       | 0.776         | 0.7817                           | 0.7957       | 0.7631  | 0.733        | empirical t3  |
| rugby_championship    | 0.7909    | 0.7976       | 0.7944       | 0.7987       | 0.7967        | 0.7971                           | 0.799        | 0.7799  | 0.74         | empirical t3  |
| rugby_world_cup       | 0.7847    | 0.7988       | 0.798        | 0.7968       | 0.7968        | 0.7887                           | 0.7675       | 0.724   | 0.7188       | C5 (phase 1)  |
| six_nations           | 0.8066    | 0.8065       | 0.8099       | 0.8108       | 0.8078        | 0.8043                           | 0.803        | 0.7884  | 0.761        | C9 (phase 2)  |
| summer_internationals | 0.6813    | 0.6723       | 0.693        | 0.6911       | 0.6865        | 0.6916                           | 0.6888       | 0.6788  | 0.6558       | E4 (phase 2)  |

### `ncr` rubric — mae

| tournament            | P3 frozen | C5 (phase 1) | E4 (phase 2) | C9 (phase 2) | C10 (phase 2) | empirical (NCR incumbent engine) | empirical t3 | v4 GBDT | stratum mean | best                             |
| --------------------- | --------- | ------------ | ------------ | ------------ | ------------- | -------------------------------- | ------------ | ------- | ------------ | -------------------------------- |
| autumn_internationals | 6.0849    | 5.9993       | 5.9703       | 6.0209       | 6.0209        | 6.0056                           | 6.0388       | 6.3144  | 6.3913       | E4 (phase 2)                     |
| british_irish_lions   | 5.8607    | 5.8504       | 5.8573       | 5.9037       | 5.9037        | 5.8512                           | 5.9049       | 5.9436  | 6.0034       | C5 (phase 1)                     |
| nations_championship  | 6.3767    | 6.2783       | 6.3091       | 6.3128       | 6.3128        | 6.2925                           | 6.3463       | 6.7583  | 6.8128       | C5 (phase 1)                     |
| pacific_nations_cup   | 7.3849    | 7.2933       | 7.3229       | 7.3202       | 7.3202        | 7.2994                           | 7.2868       | 7.6518  | 7.5437       | empirical t3                     |
| rugby_championship    | 6.0142    | 5.7535       | 5.654        | 5.714        | 5.714         | 5.7053                           | 5.6251       | 6.6799  | 5.9088       | empirical t3                     |
| rugby_world_cup       | 6.4298    | 6.2026       | 6.1893       | 6.2226       | 6.2226        | 6.163                            | 6.2523       | 7.1314  | 6.5562       | empirical (NCR incumbent engine) |
| six_nations           | 5.8191    | 5.7857       | 5.7207       | 5.7693       | 5.7693        | 5.7718                           | 5.7304       | 6.0609  | 5.9226       | E4 (phase 2)                     |
| summer_internationals | 6.3327    | 6.3186       | 6.2995       | 6.3704       | 6.3704        | 6.3761                           | 6.3678       | 6.447   | 6.4583       | E4 (phase 2)                     |

### `ncr` rubric — spearman

| tournament            | P3 frozen | C5 (phase 1) | E4 (phase 2) | C9 (phase 2) | C10 (phase 2) | empirical (NCR incumbent engine) | empirical t3 | v4 GBDT | stratum mean | best                             |
| --------------------- | --------- | ------------ | ------------ | ------------ | ------------- | -------------------------------- | ------------ | ------- | ------------ | -------------------------------- |
| autumn_internationals | 0.5163    | 0.5213       | 0.5322       | 0.5322       | 0.5322        | 0.5214                           | 0.5267       | 0.4993  | 0.4816       | C9 (phase 2)                     |
| british_irish_lions   | 0.5055    | 0.4985       | 0.4956       | 0.4952       | 0.4952        | 0.4898                           | 0.4918       | 0.5039  | 0.4451       | P3 frozen                        |
| nations_championship  | 0.5511    | 0.5624       | 0.5648       | 0.566        | 0.566         | 0.5674                           | 0.5636       | 0.5271  | 0.48         | empirical (NCR incumbent engine) |
| pacific_nations_cup   | 0.5666    | 0.5741       | 0.5707       | 0.5726       | 0.5726        | 0.5615                           | 0.5651       | 0.5522  | 0.5436       | C5 (phase 1)                     |
| rugby_championship    | 0.5402    | 0.5457       | 0.5474       | 0.5473       | 0.5473        | 0.5355                           | 0.5463       | 0.521   | 0.4775       | E4 (phase 2)                     |
| rugby_world_cup       | 0.5435    | 0.5583       | 0.5553       | 0.5565       | 0.5565        | 0.5539                           | 0.545        | 0.5068  | 0.4704       | C5 (phase 1)                     |
| six_nations           | 0.582     | 0.5842       | 0.5864       | 0.5866       | 0.5866        | 0.5824                           | 0.5827       | 0.5572  | 0.5322       | C10 (phase 2)                    |
| summer_internationals | 0.466     | 0.4708       | 0.4749       | 0.4735       | 0.4735        | 0.4557                           | 0.4555       | 0.4587  | 0.4443       | E4 (phase 2)                     |

### `ncr` rubric — mean_capture

| tournament            | P3 frozen | C5 (phase 1) | E4 (phase 2) | C9 (phase 2) | C10 (phase 2) | empirical (NCR incumbent engine) | empirical t3 | v4 GBDT | stratum mean | best                             |
| --------------------- | --------- | ------------ | ------------ | ------------ | ------------- | -------------------------------- | ------------ | ------- | ------------ | -------------------------------- |
| autumn_internationals | 0.6713    | 0.7025       | 0.7169       | 0.716        | 0.716         | 0.7094                           | 0.6869       | 0.6599  | 0.645        | E4 (phase 2)                     |
| british_irish_lions   | 0.7413    | 0.7358       | 0.7387       | 0.7386       | 0.7386        | 0.7387                           | 0.7361       | 0.7431  | 0.6928       | v4 GBDT                          |
| nations_championship  | 0.6737    | 0.696        | 0.7145       | 0.7096       | 0.7096        | 0.7021                           | 0.6961       | 0.6317  | 0.5278       | E4 (phase 2)                     |
| pacific_nations_cup   | 0.7577    | 0.7652       | 0.7535       | 0.7559       | 0.7559        | 0.768                            | 0.7587       | 0.7493  | 0.7056       | empirical (NCR incumbent engine) |
| rugby_championship    | 0.7421    | 0.7451       | 0.7501       | 0.7486       | 0.7486        | 0.7471                           | 0.7542       | 0.7299  | 0.6825       | empirical t3                     |
| rugby_world_cup       | 0.7281    | 0.7596       | 0.7492       | 0.7504       | 0.7504        | 0.7533                           | 0.7321       | 0.6641  | 0.6508       | C5 (phase 1)                     |
| six_nations           | 0.7642    | 0.7678       | 0.7651       | 0.7648       | 0.7648        | 0.762                            | 0.7593       | 0.7471  | 0.7079       | C5 (phase 1)                     |
| summer_internationals | 0.6209    | 0.6272       | 0.6378       | 0.6357       | 0.6357        | 0.6302                           | 0.6194       | 0.6142  | 0.5981       | E4 (phase 2)                     |

## Paired-by-slate bootstraps (candidate minus baseline, 90% CI)

| baseline                         | metric       | rubric      | mean     | p05      | p95      | excludes_0 |
| -------------------------------- | ------------ | ----------- | -------- | -------- | -------- | ---------- |
| empirical (NCR incumbent engine) | mae          | ncr         | -0.03542 | -0.0616  | -0.00895 | True       |
| empirical (NCR incumbent engine) | mae          | six_nations | -0.08973 | -0.11999 | -0.06008 | True       |
| empirical (NCR incumbent engine) | spearman     | ncr         | 0.00905  | 0.00522  | 0.01272  | True       |
| empirical (NCR incumbent engine) | spearman     | six_nations | 0.00911  | 0.00578  | 0.01226  | True       |
| empirical (NCR incumbent engine) | mean_capture | ncr         | 0.00213  | -0.0015  | 0.00543  | False      |
| empirical (NCR incumbent engine) | mean_capture | six_nations | 0.00251  | -0.00094 | 0.00602  | False      |
| C5 (phase 1)                     | mae          | ncr         | -0.04153 | -0.06155 | -0.0214  | True       |
| C5 (phase 1)                     | mae          | six_nations | -0.06953 | -0.09194 | -0.04865 | True       |
| C5 (phase 1)                     | spearman     | ncr         | 0.00191  | -8e-05   | 0.00376  | False      |
| C5 (phase 1)                     | spearman     | six_nations | 0.00406  | 0.0022   | 0.00588  | True       |
| C5 (phase 1)                     | mean_capture | ncr         | 0.0022   | -0.00138 | 0.00565  | False      |
| C5 (phase 1)                     | mean_capture | six_nations | 0.00413  | -0.00032 | 0.00961  | False      |
| P3 frozen                        | mae          | ncr         | -0.1473  | -0.1928  | -0.1078  | True       |
| P3 frozen                        | mae          | six_nations | -0.16921 | -0.21195 | -0.12991 | True       |
| P3 frozen                        | spearman     | ncr         | 0.00663  | 0.00327  | 0.00982  | True       |
| P3 frozen                        | spearman     | six_nations | 0.00524  | 0.00253  | 0.00795  | True       |
| P3 frozen                        | mean_capture | ncr         | 0.01129  | 0.00601  | 0.017    | True       |
| P3 frozen                        | mean_capture | six_nations | 0.00607  | 0.00221  | 0.01037  | True       |
| empirical t3                     | mae          | ncr         | -0.02139 | -0.04673 | 0.00402  | False      |
| empirical t3                     | mae          | six_nations | -0.05121 | -0.07932 | -0.02516 | True       |
| empirical t3                     | spearman     | ncr         | 0.00664  | 0.00324  | 0.01015  | True       |
| empirical t3                     | spearman     | six_nations | 0.00606  | 0.00328  | 0.00887  | True       |
| empirical t3                     | mean_capture | ncr         | 0.00824  | 0.00329  | 0.01329  | True       |
| empirical t3                     | mean_capture | six_nations | 0.00535  | 0.00093  | 0.00976  | True       |
| C9 (phase 2)                     | mae          | ncr         | -0.04839 | -0.05558 | -0.04169 | True       |
| C9 (phase 2)                     | mae          | six_nations | 0.00779  | -0.00603 | 0.02155  | False      |
| C9 (phase 2)                     | spearman     | ncr         | -3e-05   | -0.00051 | 0.00047  | False      |
| C9 (phase 2)                     | spearman     | six_nations | 0.00257  | 0.00125  | 0.00389  | True       |
| C9 (phase 2)                     | mean_capture | ncr         | 0.00063  | -0.00083 | 0.00202  | False      |
| C9 (phase 2)                     | mean_capture | six_nations | -0.00054 | -0.00281 | 0.00175  | False      |

