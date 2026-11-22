# Benchmark: known molecules

## 1. Voltage: xTB vs measured (V vs SHE, pH 0)

| molecule | measured | xTB | error | source | notes |
|---|---|---|---|---|---|
| p-benzoquinone | 0.699 | 0.699 | +0.000 | table (calibration point) | takes up 1 H2 at most (no further reversible hydrogenated form) |
| o-benzoquinone | 0.792 | 0.767 | -0.025 | textbook | takes up 1 H2 at most (no further reversible hydrogenated form) |
| methyl-p-benzoquinone | 0.645 | 0.632 | -0.013 | textbook | oxidised form tautomerises to C=C1C=C(O)C=CC1=O (0.16 eV lower); takes up 1 H2 at most (no further reversible hydrogenated form) |
| 1,4-naphthoquinone | 0.470 | 0.326 | -0.144 | textbook | takes up 1 H2 at most (no further reversible hydrogenated form) |
| anthraquinone-2-sulfonic acid (AQS) | 0.187 | 0.167 | -0.020 | textbook | solubility assumed 1 M: sulfonic/phosphonic acid groups are ionised in water, where the ESOL estimate does not apply (AQDS measured > 1 M); takes up 1 H2 at most (no further reversible hydrogenated form) |
| anthraquinone-2,7-disulfonic acid (AQDS) | 0.213 | 0.319 | +0.106 | Er2015 | solubility assumed 1 M: sulfonic/phosphonic acid groups are ionised in water, where the ESOL estimate does not apply (AQDS measured > 1 M); takes up 1 H2 at most (no further reversible hydrogenated form) |

Mean absolute error 0.061 V, mean signed error -0.019 V over 5 molecules (calibration point excluded). Stated model band: +/-0.3 V.

## 2. Ranking among the 74 molecules of the search library

Against H2: the molecule as the positive side of a cell with a hydrogen electrode (negatives score 0 here by design). In pairs: the best two-molecule pair it forms with any library molecule, against the best pair of each library molecule.

| molecule | usable | E0 (xTB) | against H2: score | beats % | in pairs: best score | beats % |
|---|---|---|---|---|---|---|
| AQDS (negative, Huskinson et al. 2014) | yes | 0.319 | 17.1 | 94 | 18.4 | 90 |
| AQS (negative, Yang et al. 2014) | yes | 0.167 | 0.0 | 0 | 23.9 | 100 |
| BQDS, oxidised Tiron (positive, Yang et al. 2014) | yes | 1.165 | 62.5 | 97 | 11.7 | 77 |
| 2,6-DHAQ (negative, Lin et al. 2015) | yes | -0.048 | 0.0 | 0 | 0.1 | 13 |
| alloxazine (negative, Lin et al. 2016) | yes | -0.019 | 0.0 | 0 | 0.8 | 19 |
| p-benzoquinone | yes | 0.699 | 8.0 | 68 | 1.4 | 23 |

## 3. The known pair AQS | BQDS (Yang et al. 2014) against the library's pairs

Voltage 1.00 V, 26.7 Wh/L, score 13.4; better than 98 % of the 416 library pairs.
