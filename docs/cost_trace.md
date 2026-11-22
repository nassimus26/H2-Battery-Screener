# Why the build cost is so high: one pair, line by line

Pair (best in the library at 09:04 on 2026-11-22):
negative `Nc1ncc2c(n1)=NCNN=2` (E = -0.105 V, 1.69 M, 150 g/mol),
positive `NCc1nnnc2c1=NCCN=2` (E = +0.700 V, 2.29 M, 164 g/mol), cell voltage 0.805 V.
Model: `options.pair_configuration` → `options._optimise`, mass-production prices.

## Car setting (220 kW for 0.38 h = 84 kWh), reaction speed at the middle of its range

| step | number | where it comes from |
|---|---|---|
| reaction speed (exchange current), both electrodes | 1 mA/cm² | middle (log-centre) of 0.001–100 mA/cm², `estimate.positive_exchange_current`, `pairs.pair_configuration` |
| highest charge voltage allowed | 1.195 V | 0.805 V + 2 × 0.195 V: the negative (−0.105 V) is only 0.195 V above the hydrogen-gas limit (−0.3 V, `pairs.HER_MARGIN`); the smaller of the two electrode margins, doubled (`options.pair_configuration`) |
| current density chosen | **23 mA/cm²** | the highest current whose charge voltage stays under 1.195 V. It is *not* the cost optimum: the water-window limit binds (recharge = 23 min = the full 0.38 h, i.e. the fastest current allowed) |
| discharge voltage at that current | 0.479 V | 0.805 V − 2 × 0.16 V activation loss (Butler–Volmer, 1 mA/cm² on each side) − 0.004 V ohmic |
| power per m² of cells | 110 W/m² (11 mW/cm²) | 0.023 A/cm² × 0.479 V |
| m² of cells | **2,001 m²** | 220,000 W / 110 W/m² |
| stack | $300,187 = **$3,591/kWh** | 2,001 m² × $150/m² (liquid cell, mass production, `stackcost.py`) |
| pumps, pipes, controls | $67,330 = $805/kWh | 10 kW flow-battery prices (Noack 2016) scaled by (220/10)^0.6 |
| chemicals | $29,866 = $357/kWh | 12 kg of molecules per kWh at 0.479 V × about $25–36/kg (price from synthesis difficulty) |
| power electronics | $21,450 = $257/kWh | 220 kW × $97.5/kW |
| tanks | $5,809 = $70/kWh | |
| **total** | **$5,079/kWh** | vs Li-ion pack $70/kWh → **+$5,009** |

Round-trip efficiency at that point: 0.41.

## What changes it

| setting | i (mA/cm²) | m² | stack $/kWh | chemicals $/kWh | total $/kWh |
|---|---|---|---|---|---|
| car, middle speed (1 mA/cm²) | 23 | 2,001 | 3,591 | 357 | **5,079** |
| car, fast measured speed 10 mA/cm² | 158 | 283 | 507 | 347 | **1,985** |
| car, very fast 100 mA/cm² | 318 | 112 | 201 | 277 | **1,600** |
| stationary 100 kW × 6 h, middle speed | 19 | 1,043 | 261 | 344 | **722** |
| stationary, 10 mA/cm² | 33 | 458 | 115 | 256 | **482** |
| stationary, 100 mA/cm² | 78 | 170 | 43 | 228 | **380** |

## Steps where the model looks too pessimistic (or wrong)

1. **Reaction-speed range.** It is centred at 1 mA/cm² for every new molecule. Known fast organic
   couples (quinones such as AQDS, viologens, TEMPO) have rate constants around 10⁻² cm/s. At 1 M that is
   an exchange current of the order of 1 A/cm². So the middle of the range is about 100–1,000 times slower than what
   working organic flow batteries show. This one number makes the stack 2,000 m² in a car.
2. **The charge-voltage limit sets the current, not the cost.** The negative electrode is only 0.195 V away
   from making hydrogen gas (the −0.3 V margin tolerated on carbon is an assumption). At slow
   kinetics this caps the current at 23 mA/cm². The limit is real physics, but the margins are assumptions.
3. **Car vs stationary.** A car needs 2.6 kW per kWh, so the stack, pumps and electronics are sized by power.
   A flow cell's power parts are expensive per kW, and a car makes them dominate. Stationary (0.17 kW per kWh) is 7 times cheaper per kWh.
4. **Unequal comparison.** The total includes power electronics ($257/kWh in the car case) and pumps.
   The Li-ion figure ($70/kWh, stationary pack) has neither: no inverter and, in the car case, it is not the
   BEV pack price ($99). Pumps and pipes use small-series 2016 prices with no mass-production discount.
5. **Efficiency target ignored.** The sidebar's "round-trip efficiency it must keep" (0.75) is not
   applied in `options._optimise` (only in `rank.py`). This pair is sized at 0.41–0.44 efficiency.
6. **Chemicals stay expensive even with fast kinetics** (stationary: $228/kWh of $380). The molecules are priced
   at $25–36/kg from their synthesis difficulty, against $3–5/kg for bulk quinones. With small heavy
   molecules and 0.8 V, 8–12 kg are needed per kWh. Cheaper chemistry needs a higher voltage, more H per
   molecule, or simpler molecules.
7. Slightly *optimistic*: the amount of chemicals assumes 100 % of the molecules are cycled. The
   efficiency model uses only 10–90 % state of charge, so about 25 % more would be needed.

The p10–p90 range of the build cost over the reaction-speed range is now shown next to the
"vs Li-ion" figure (`Config.capex_band`).
