# Data QA report (ARCADE syntax subset)

Pairs checked: 1500

| Check | Count |
|---|---|
| missing image | 0 |
| missing mask | 0 |
| unexpected size | 0 |
| unexpected mode | 0 |
| color image | 0 |
| gray stored as rgb | 549 |
| non binary mask | 0 |
| mask size mismatch | 0 |
| empty mask | 0 |
| constant image | 0 |
| exact duplicate groups | 5 |
| near duplicate pairs (r >= 0.97) | 12 |
| near duplicate pairs across splits | 1 |

| Split | Images in JSON | Images on disk | Missing | Extra | Unannotated | Annotations |
|---|---|---|---|---|---|---|
| train | 1000 | 1000 | 0 | 0 | 0 | 4976 |
| val | 200 | 200 | 0 | 0 | 0 | 1168 |
| test | 300 | 300 | 0 | 0 | 0 | 1672 |

| Split | Images | Mean intensity (mean, sd) | Vessel fraction (median) | Skeleton radius px (median, p95) | Black border fraction (mean) |
|---|---|---|---|---|---|
| test | 300 | 124.0, 16.2 | 0.0333 | 4.24, 9.00 | 0.0141 |
| train | 1000 | 113.2, 22.0 | 0.0316 | 5.00, 8.94 | 0.0251 |
| val | 200 | 134.6, 13.0 | 0.0331 | 4.47, 9.06 | 0.0137 |

Exact duplicates (pixel-identical images):

* train_20, train_214 (train, Dice between the two masks 0.936)
* train_168, train_793 (train, Dice between the two masks 0.744)
* train_337, train_972 (train, Dice between the two masks 0.904)
* train_478, train_887 (train, Dice between the two masks 0.931)
* train_501, train_612 (train, Dice between the two masks 0.898)

Near duplicates:

* train_20 and train_214: r = 1.000
* train_168 and train_793: r = 1.000
* train_337 and train_972: r = 1.000
* train_478 and train_887: r = 1.000
* train_501 and train_612: r = 1.000
* train_585 and train_782: r = 0.999
* train_38 and train_208: r = 0.998
* train_576 and train_926: r = 0.985
* train_853 and train_871: r = 0.976
* val_116 and val_151: r = 0.976
* train_720 and val_151: r = 0.971 (across splits)
* test_219 and test_299: r = 0.970
