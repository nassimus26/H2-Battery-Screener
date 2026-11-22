# First-principles method tested on known molecules

Measured values are used only here, to test. See h2screener/abinitio.py for the method.

## GFN2-xTB (2026-11-23 05:22, 0 s)

- gas reactions, V: mean abs error 0.620, RMS 0.724, max 2.134, mean signed -0.309 (n = 33)
- gas reactions, kJ/mol: mean abs error 478.986, RMS 602.240, max 1405.774, mean signed -405.948 (n = 33)
- formation test, kJ/mol: mean abs error 105.884, RMS 136.732, max 276.150, mean signed -44.806 (n = 22)
- formation train, kJ/mol: mean abs error 112.467, RMS 154.016, max 411.851, mean signed -55.514 (n = 23)
- aqueous voltage, V: mean abs error 0.463, RMS 0.466, max 0.584, mean signed +0.463 (n = 6)

| oxidation of | electrons | dG measured | dG computed | error kJ/mol | error V |
|---|---|---|---|---|---|
| hydrogen | 2 | -228.6 | -320.7 | -92.1 | -0.477 |
| oxygen | 4 | 0.0 | 0.0 | +0.0 | +0.000 |
| hydrogen peroxide | 2 | -123.0 | 61.0 | +184.0 | +0.954 |
| carbon monoxide | 2 | -257.2 | -580.6 | -323.4 | -1.676 |
| methane | 8 | -801.1 | -1220.6 | -419.5 | -0.544 |
| ethane | 14 | -1442.6 | -2189.3 | -746.7 | -0.553 |
| ethylene | 12 | -1314.4 | -2014.5 | -700.1 | -0.605 |
| acetylene | 10 | -1226.6 | -1843.8 | -617.2 | -0.640 |
| propane | 20 | -2074.2 | -3147.6 | -1073.4 | -0.556 |
| propene | 18 | -1931.8 | -2953.8 | -1022.0 | -0.588 |
| cyclopropane | 18 | -1973.5 | -2950.9 | -977.4 | -0.563 |
| butane | 26 | -2703.6 | -4109.4 | -1405.8 | -0.560 |
| isobutane | 26 | -2699.8 | -4104.1 | -1404.3 | -0.560 |
| methanol | 6 | -689.3 | -981.0 | -291.7 | -0.504 |
| formaldehyde | 4 | -520.5 | -771.5 | -251.0 | -0.650 |
| formic acid | 2 | -272.0 | -393.8 | -121.8 | -0.631 |
| ethanol | 12 | -1306.1 | -1932.3 | -626.2 | -0.541 |
| dimethyl ether | 12 | -1362.0 | -1947.1 | -585.1 | -0.505 |
| acetaldehyde | 10 | -1113.0 | -1686.2 | -573.2 | -0.594 |
| acetic acid | 8 | -871.5 | -1319.8 | -448.3 | -0.581 |
| ethylene oxide | 10 | -1232.9 | -1722.3 | -489.4 | -0.507 |
| ammonia | 3 | -326.5 | -366.2 | -39.7 | -0.137 |
| hydrazine | 4 | -616.6 | -534.7 | +81.9 | +0.212 |
| hydrogen cyanide | 5 | -633.4 | -871.8 | -238.4 | -0.494 |
| methylamine | 9 | -998.0 | -1332.3 | -334.3 | -0.385 |
| nitric oxide | 2 | -86.6 | -12.3 | +74.3 | +0.385 |
| nitrogen dioxide | 4 | -51.3 | 292.6 | +343.9 | +0.891 |
| nitrous oxide | 2 | -103.7 | 308.2 | +411.9 | +2.134 |
| hydrogen sulfide | 6 | -495.3 | -815.5 | -320.2 | -0.553 |
| sulfur trioxide | 2 | 71.0 | 180.2 | +109.2 | +0.566 |
| carbonyl sulfide | 6 | -525.3 | -932.0 | -406.7 | -0.702 |
| carbon disulfide | 12 | -1061.7 | -1835.0 | -773.3 | -0.668 |
| chloromethane | 6 | -659.9 | -980.0 | -320.1 | -0.553 |

| quinone | measured E V | computed E V | error V | source |
|---|---|---|---|---|
| p-benzoquinone | 0.699 | 1.153 | +0.454 | standard potential table |
| o-benzoquinone | 0.792 | 1.221 | +0.429 | Clark 1960 |
| methyl-p-benzoquinone | 0.645 | 1.079 | +0.434 | Clark 1960 |
| 1,4-naphthoquinone | 0.470 | 0.882 | +0.412 | Clark 1960 |
| AQS | 0.187 | 0.651 | +0.464 | Clark 1960 |
| AQDS | 0.213 | 0.797 | +0.584 | Er et al. 2015 |

## WB97X-V/DEF2-TZVP//GFN2-xTB (2026-11-23 05:22, 0 s)

- gas reactions, V: mean abs error 0.097, RMS 0.106, max 0.178, mean signed +0.089 (n = 33)
- gas reactions, kJ/mol: mean abs error 87.768, RMS 114.043, max 270.417, mean signed +86.032 (n = 33)
- formation test, kJ/mol: mean abs error 13.743, RMS 17.896, max 36.114, mean signed +9.058 (n = 22)
- formation train, kJ/mol: mean abs error 9.003, RMS 13.455, max 36.116, mean signed -0.097 (n = 23)

| oxidation of | electrons | dG measured | dG computed | error kJ/mol | error V |
|---|---|---|---|---|---|
| hydrogen | 2 | -228.6 | -200.7 | +27.9 | +0.145 |
| oxygen | 4 | 0.0 | 0.0 | +0.0 | +0.000 |
| hydrogen peroxide | 2 | -123.0 | -131.2 | -8.2 | -0.043 |
| carbon monoxide | 2 | -257.2 | -250.0 | +7.2 | +0.037 |
| methane | 8 | -801.1 | -715.4 | +85.7 | +0.111 |
| ethane | 14 | -1442.6 | -1293.5 | +149.1 | +0.110 |
| ethylene | 12 | -1314.4 | -1203.1 | +111.3 | +0.096 |
| acetylene | 10 | -1226.6 | -1162.0 | +64.6 | +0.067 |
| propane | 20 | -2074.2 | -1862.0 | +212.2 | +0.110 |
| propene | 18 | -1931.8 | -1757.7 | +174.1 | +0.100 |
| cyclopropane | 18 | -1973.5 | -1786.3 | +187.2 | +0.108 |
| butane | 26 | -2703.6 | -2433.5 | +270.1 | +0.108 |
| isobutane | 26 | -2699.8 | -2429.4 | +270.4 | +0.108 |
| methanol | 6 | -689.3 | -615.4 | +73.9 | +0.128 |
| formaldehyde | 4 | -520.5 | -474.2 | +46.3 | +0.120 |
| formic acid | 2 | -272.0 | -237.7 | +34.3 | +0.178 |
| ethanol | 12 | -1306.1 | -1173.8 | +132.3 | +0.114 |
| dimethyl ether | 12 | -1362.0 | -1216.8 | +145.2 | +0.125 |
| acetaldehyde | 10 | -1113.0 | -1005.1 | +107.9 | +0.112 |
| acetic acid | 8 | -871.5 | -773.9 | +97.6 | +0.126 |
| ethylene oxide | 10 | -1232.9 | -1119.0 | +113.9 | +0.118 |
| ammonia | 3 | -326.5 | -285.5 | +41.0 | +0.142 |
| hydrazine | 4 | -616.6 | -560.5 | +56.1 | +0.145 |
| hydrogen cyanide | 5 | -633.4 | -596.3 | +37.1 | +0.077 |
| methylamine | 9 | -998.0 | -890.9 | +107.1 | +0.123 |
| nitric oxide | 2 | -86.6 | -87.4 | -0.8 | -0.004 |
| nitrogen dioxide | 4 | -51.3 | -58.2 | -6.9 | -0.018 |
| nitrous oxide | 2 | -103.7 | -113.7 | -10.0 | -0.052 |
| hydrogen sulfide | 6 | -495.3 | -414.5 | +80.8 | +0.140 |
| sulfur trioxide | 2 | 71.0 | 68.2 | -2.8 | -0.015 |
| carbonyl sulfide | 6 | -525.3 | -466.4 | +58.9 | +0.102 |
| carbon disulfide | 12 | -1061.7 | -952.3 | +109.4 | +0.094 |
| chloromethane | 6 | -659.9 | -593.8 | +66.1 | +0.114 |

| quinone | measured E V | computed E V | error V | source |
|---|---|---|---|---|
| p-benzoquinone | 0.699 | - | - | standard potential table |
| o-benzoquinone | 0.792 | - | - | Clark 1960 |
| methyl-p-benzoquinone | 0.645 | - | - | Clark 1960 |
| 1,4-naphthoquinone | 0.470 | - | - | Clark 1960 |
| AQS | 0.187 | - | - | Clark 1960 |
| AQDS | 0.213 | - | - | Er et al. 2015 |

## CCSD(T)/cc-pvtz//GFN2-xTB (2026-11-23 05:22, 0 s)

- gas reactions, V: mean abs error 0.057, RMS 0.067, max 0.162, mean signed +0.035 (n = 33)
- gas reactions, kJ/mol: mean abs error 43.280, RMS 55.290, max 144.530, mean signed +38.491 (n = 33)
- formation test, kJ/mol: mean abs error 10.529, RMS 20.254, max 74.078, mean signed +5.102 (n = 22)
- formation train, kJ/mol: mean abs error 9.514, RMS 14.320, max 42.880, mean signed +4.174 (n = 23)

| oxidation of | electrons | dG measured | dG computed | error kJ/mol | error V |
|---|---|---|---|---|---|
| hydrogen | 2 | -228.6 | -215.5 | +13.1 | +0.068 |
| oxygen | 4 | 0.0 | 0.0 | +0.0 | +0.000 |
| hydrogen peroxide | 2 | -123.0 | -131.5 | -8.5 | -0.044 |
| carbon monoxide | 2 | -257.2 | -240.9 | +16.3 | +0.084 |
| methane | 8 | -801.1 | -765.8 | +35.3 | +0.046 |
| ethane | 14 | -1442.6 | -1381.5 | +61.1 | +0.045 |
| ethylene | 12 | -1314.4 | -1264.4 | +50.0 | +0.043 |
| acetylene | 10 | -1226.6 | -1193.4 | +33.2 | +0.034 |
| propane | 20 | -2074.2 | -1986.7 | +87.5 | +0.045 |
| propene | 18 | -1931.8 | -1857.3 | +74.5 | +0.043 |
| cyclopropane | 18 | -1973.5 | -1900.9 | +72.6 | +0.042 |
| butane | 26 | -2703.6 | -2594.8 | +108.8 | +0.043 |
| isobutane | 26 | -2699.8 | -2589.8 | +110.0 | +0.044 |
| methanol | 6 | -689.3 | -659.6 | +29.7 | +0.051 |
| formaldehyde | 4 | -520.5 | -496.8 | +23.7 | +0.061 |
| formic acid | 2 | -272.0 | -256.7 | +15.3 | +0.079 |
| ethanol | 12 | -1306.1 | -1254.7 | +51.4 | +0.044 |
| dimethyl ether | 12 | -1362.0 | -1302.7 | +59.3 | +0.051 |
| acetaldehyde | 10 | -1113.0 | -1068.1 | +44.9 | +0.046 |
| acetic acid | 8 | -871.5 | -832.8 | +38.7 | +0.050 |
| ethylene oxide | 10 | -1232.9 | -1192.1 | +40.8 | +0.042 |
| ammonia | 3 | -326.5 | -317.8 | +8.7 | +0.030 |
| hydrazine | 4 | -616.6 | -618.1 | -1.5 | -0.004 |
| hydrogen cyanide | 5 | -633.4 | -612.3 | +21.1 | +0.044 |
| methylamine | 9 | -998.0 | -964.2 | +33.8 | +0.039 |
| nitric oxide | 2 | -86.6 | -88.3 | -1.7 | -0.009 |
| nitrogen dioxide | 4 | -51.3 | -67.8 | -16.5 | -0.043 |
| nitrous oxide | 2 | -103.7 | -123.3 | -19.6 | -0.102 |
| hydrogen sulfide | 6 | -495.3 | -415.0 | +80.3 | +0.139 |
| sulfur trioxide | 2 | 71.0 | 39.8 | -31.2 | -0.162 |
| carbonyl sulfide | 6 | -525.3 | -452.9 | +72.4 | +0.125 |
| carbon disulfide | 12 | -1061.7 | -917.2 | +144.5 | +0.125 |
| chloromethane | 6 | -659.9 | -637.5 | +22.4 | +0.039 |

| quinone | measured E V | computed E V | error V | source |
|---|---|---|---|---|
