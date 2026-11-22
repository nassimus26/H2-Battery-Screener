# How the simulator works

This document explains the methods behind the hydrogen battery simulator: how
candidates are found, how each one is turned into a sized battery, how the AI
search proposes new molecules, where every price comes from, and what the
results can and cannot tell you.

## 1. Principles

- **From laws and data, not from lists of known batteries.** Every chemistry
  the data allows is generated and pushed through the same physics. Published
  hydrogen batteries are used only to check the model, never as candidates or
  for fitting.
- **Every number says where it comes from.** Each input is tagged `database`,
  `physics`, `measured`, `assumption` or `unknown` (`h2screener/provenance.py`).
  Engineering ranges live in `targets.toml`, not in the code.
- **Unknowns are ranges, not guesses.** When no law or measurement fixes an
  input (for example how fast a new molecule reacts at an electrode), the model
  samples a wide range by Monte Carlo and reports p10 / p50 / p90, plus which
  input dominates the spread.
- **Always compared with lithium-ion.** Every result is set against current car
  packs (Tesla 4680/2170, BYD Blade; prices from BloombergNEF 2025) in
  `data/reference_batteries.toml`.

## 2. Overview

```
 potential table ─┐                          ┌─ AI molecule search (SELFIES + surrogate + xTB)
 (half-reactions) │                          │
                  ▼                          ▼
            1. generate couples      molecules vs H2  /  pairs of molecules
                  │                          │
                  └──────────┬───────────────┘
                             ▼
            2. voltage (Nernst, real-gas fugacity)
            3. stability, safety, chemical closure
            4. kinetics → efficiency vs current (Monte Carlo)
            5. sizing for the job (power × hours): current density with the
               lowest lifetime cost, every pressure / storage option
            6. costs (stack, catalyst, tank, molecules, tanks & pumps, electronics)
            7. ranking (weighted geometric mean) and comparison with Li-ion
                             ▼
            app: tables, charts, "Why it costs this much", interesting candidates
```

## 3. Hydrogen couples from the potential table

**Generation** (`discover.py`). Every half-reaction of the standard electrode
potential table is paired with every hydrogen electrode it can work against:
H⁺/H₂ in acid and H₂O/H₂,OH⁻ in alkaline. A Materials Project generator
(`A + x H₂ → B + y H₂O` and metal hydrides, including never-made phases) is
written but needs an API key and has not been run yet.

**Voltage** (`reactions.py`, `realgas.py`). Standard potential E⁰ plus the Nernst
term, with the real fugacity of hydrogen at the storage pressure from CoolProp
equations of state. The same equations give gas density and tell whether a
reactant condenses (chlorine is liquid above about 7 bar).

**Stability and safety** (`rank.py`, `safety.py`, `materials.py`).
- Water is oxidised above 1.229 V plus a margin; such couples are penalised.
- Radioactive elements are excluded.
- Hazards come from PubChem GHS statements; element supply risk from the
  Herfindahl–Hirschman index of production (pymatgen data).
- Burst energy of the tank (isentropic expansion, in kg TNT), choked-flow leak
  rate through a 1 mm² hole, and time to reach 25 % of the flammability limit in
  a room.

**Chemical closure and maintenance** (`maintenance.py`). Using the same table
(Frost-diagram rules): can hydrogen over-reduce the product, can a species
disproportionate, does the positive side make oxygen? Hydrogen loss through the
tank wall (Sieverts' law with measured 316L steel permeation fits) and seals
gives the refill interval.

## 4. Efficiency and power: the cell model

`performance.py` computes the cell voltage at current density *i* and state of
charge:

```
V_discharge = OCV − η_act,pos − η_act,neg − i·ASR − η_mass-transport
OCV      Nernst (gas fugacity, concentration of dissolved species)
η_act    Butler–Volmer, α = ½:  (2RT/F)·asinh(i / 2i₀)
ASR      electrolyte gap / conductivity + separator + contacts
η_mt     −(RT/nF)·ln(1 − i/i_lim),  i_lim from Fick's law (Stokes–Einstein
         diffusion for ions, gas diffusion for gas electrodes)
```

Inputs no law fixes are sampled from the ranges in `targets.toml [estimates]`.
The most important one is the **exchange current i₀** of an untested electrode:
10⁻⁶ to 10⁻¹ A/cm². This range dominates almost every new couple and molecule:
a slow reaction means low current per m², so a large and expensive stack. One
lab measurement of a promising candidate would replace the range with a number.

## 5. Sizing a battery for the job

`design.py` (table couples) and `options.py` (molecules, pairs and every storage
option) size a complete system for the power and duration chosen in the sidebar.

- **Current density is chosen for the lowest lifetime cost.** A bigger stack
  runs at lower current: more efficient, but more expensive. The scan keeps the
  current density that minimises the levelised cost of storage while still
  meeting the efficiency target.
- **Cell area** = power / (current density × discharge voltage).
- **Hydrogen storage**: swing volume between the "empty" and maximum pressures
  (real gas), vessel wall from the hoop stress, or alternatives: piston
  (constant pressure), metal hydride (van 't Hoff plateau, ML-HydPARK data),
  hydride as the electrode itself, or refuelling with hydrogen made off board.
- **Lifetime cost (LCOS)** = (build cost + present value of operation, hydrogen
  make-up and charging electricity) / present value of the energy delivered.

## 6. Molecules that store hydrogen

`molecules.py` evaluates any molecule, known or new:

1. **Hydrogen uptake.** Every way of adding H₂ to two heteroatoms of its
   conjugated system is generated; only closed-shell, valence-correct partners
   are kept. Up to three H₂ in a row are tried (multi-step molecules).
2. **Voltage.** Each form is relaxed with **GFN2-xTB** in implicit water (ALPB):
   `E = −(E[MH₂] − E[M] − E[H₂]) / 2F + offset`. The offset is fixed by one
   measured couple, p-benzoquinone / hydroquinone at 0.6992 V. Stated
   uncertainty: ±0.3 V.
3. **Tautomers.** If a lower-energy tautomer of the reduced form puts a hydrogen
   onto a carbon (keto–enol type), the couple is marked not reversible. If it
   only moves protons between N and O atoms (fast in water), the stable form is
   used and the couple stays reversible. Sulfonic and phosphonic acid groups are
   set aside while the hydrogenated partner's bonds are rebuilt.
4. **Descriptors.** Solubility (ESOL; 1 mol/L assumed for sulfonic and
   phosphonic acids, which are ionised in water and outside ESOL's range),
   synthetic difficulty (SA score, 1–10),
   Michael-addition sites, aldehydes, PubChem novelty.
5. **Stability alerts.** A molecule is unusable if it carries a group that
   breaks down in water or on its own:
   - the decomposition alerts of the published Brenk set (not the ones that
     describe the redox cores we want, such as quinones);
   - cumulated double bonds (C=C=C) and strained fused small rings;
   - groups that hydrolyse: aminals / hemiaminals outside a ring (N–C–N,
     N–C–O on a chain carbon), imines on a nitrogen outside a ring, N–H imines,
     N–N single bonds (hydrazines) and C-nitroso groups. These are written to
     match aromatic-flagged atoms, which the Brenk patterns miss.

A molecule can then be built **against hydrogen gas** (molecule on the positive
side, hydrogen electrode on the negative) or paired with a second molecule.

## 7. Two-molecule batteries (no gas)

`pairs.py`. Hydrogen moves between two molecules as protons through the
membrane and electrons through the wire; it is never a gas. The animation in
the app's "Two-molecule batteries" tab shows the flow, discharging or charging.

```
negative:  N-H₂ → N + 2H⁺ + 2e⁻        positive:  P + 2H⁺ + 2e⁻ → P-H₂
voltage    V = E(P) − E(N)              (the common xTB offset cancels)
energy     Wh/L = V · 2F / 3600 / (1/c_neg + 1/c_pos)   (both tanks counted)
```

- Both potentials must stay inside the water window (no hydrogen or oxygen
  evolution on carbon electrodes, with a safety headroom).
- Solubility is capped at 4 mol/L.
- Multi-step molecules can move 1–3 H₂ per molecule; a single molecule with two
  well-separated steps can even serve both sides (a "symmetric" battery).
- **Variety.** The pair list shows each molecule in at most two pairs, so one
  strong molecule cannot fill the list with copies of itself.

## 8. The AI molecule search

`ai_search.py` is an **active-learning loop in the style of Bayesian
optimisation**. It is not a language model.

1. **Generate.** Children of the current best molecules by **SELFIES** mutation
   and crossover. Every SELFIES string decodes to a valid molecule, so the
   search walks freely through chemical space, mostly through molecules never
   made before.
2. **Filter.** Size limit (heavy atoms), neutral closed shell, at least one
   site that can take up H₂, no instability alert.
3. **Predict.** A **surrogate model** (extra-trees ensemble of 200 trees on
   2048-bit Morgan fingerprints plus size, solubility and difficulty)
   predicts each child's voltage. The spread between the trees is its
   uncertainty.
4. **Choose.** An **upper confidence bound** (prediction + κ × uncertainty; κ is
   the "Exploration" slider) picks a batch that mixes children predicted to be
   good with children the model is unsure about.
5. **Compute.** GFN2-xTB gives the true voltages of the batch. The surrogate is
   retrained on them and the loop repeats.

**What "good" means.**
- *Molecule mode:* energy per litre of electrolyte (voltage × solubility),
  zero outside 0.2 V to the oxygen limit, lowered for molecules that are
  expensive to make and for Michael-addition sites.
- *Pair mode:* the best pair score a molecule can make with the library. To
  explore different combinations, parents are taken from the molecules of
  several *different* top pairs (each molecule in one pair at most), then from
  other good partners.

Every computed molecule is kept in `cache/molecule_search.csv`; each search run
(settings, progress, results) is saved in `cache/runs/` and can be restored or
deleted (to `cache/runs/trash/`) from the app.

## 9. Where every price comes from

The user chooses only the **production scale** ("Built as": mass production or
small series). Everything else is estimated by the model
(`stackcost.py`, `synthesis.py`, `materials.py`).

| Part | How it is priced | Source |
|---|---|---|
| Cell stack, cell with a hydrogen-gas electrode | per m² of cell: 240 USD/m² (170–340) mass production, 640 USD/m² small series; platinum excluded | US DOE Program Record 26002 (2026): 275 kW truck fuel cell, 52.5 m², $155/kW at 50,000/yr, $282/kW below 1,000/yr, stack ≈ 60 % of system, catalyst > half of stack |
| Cell stack, two-liquid cell | 150 USD/m² (90–250) mass production; 3,260 USD/m² small series | mass: membrane (DOE ionomer and ePTFE prices), two carbon electrodes (DOE GDL price), carbon-plastic plate (Noack et al.), ×1.5 for frames, seals and assembly (an assumption); small series: the built 10 kW stack of Noack et al. 2016 |
| Platinum catalyst | loading × area × platinum price | pymatgen element prices |
| Power electronics | 97.5 USD/kW (70–150) | NREL 2023 storage cost benchmark, central inverter |
| Hydrogen tank | steel mass from the hoop stress × steel price × fabrication factor | element prices; `targets.toml` |
| Molecules | USD/kg from synthetic difficulty, with a ×3 range (below) | fitted to bulk-priced chemicals |
| Electrolyte tanks, pumps, pipes, heat exchangers, sensors | itemised prices of a built 10 kW / 120 kWh flow battery, scaled to the system with cost ∝ size^0.6 (tanks by litres, the rest by power) | Noack et al. 2016 (Tables 1–2); Darling et al. 2014 for the scaling rule |
| Hydrogen fill and make-up | 6 USD/kg | `targets.toml` |

**Molecule price.** `log10(USD/kg) = a + b × SA score`, fitted by least squares
to seven chemicals with known bulk prices: hydroquinone, p-benzoquinone,
anthraquinone, AQDS (at-scale estimate of Dieterich et al. 2018), methyl
viologen, 4-hydroxy-TEMPO and riboflavin. Easy molecules come out at
2–10 USD/kg, a difficulty of 6 at about 270 USD/kg. The range is wide on
purpose: published work finds the SA score tracks price only loosely
(correlation ≈ 0.3 on log price). Raw-element prices are **not** used as a floor
for organic molecules: they price purified carbon, hydrogen, nitrogen and oxygen
(hydroquinone would come out at 17 USD/kg; it sells at about 3).

**Explanations.** Each detail view has "Where the build cost goes" (bar chart,
with the Li-ion pack price drawn in) and "Why it costs this much"
(`explain.py`): the biggest parts, the arithmetic of the stack
(area = power / (current density × voltage)), where each price comes from and its
range, the molecule's estimated price, and which setting would change the
result.

## 10. Ranking

`rank.py` scores each candidate as a **weighted geometric mean of
desirabilities** between 0 and 1: cost, efficiency, charge speed, safety,
supply risk, mass, volume, rechargeability and chemical closure. The weights
are the "Ranking weights" sliders; only their ratios matter. A zero in any term
(radioactive, never meets the efficiency target, …) gives a zero score, but the
candidate stays in the table with its reason. "Candidates to size in detail"
sets how many of the top-ranked couples get the full sizing of section 5.

## 11. Exploring a candidate

- **Interesting candidates** (first tab): add any couple, molecule or pair from
  its detail window. All of them are recalculated every time a sidebar setting
  changes and shown side by side (build cost, lifetime cost, Wh/kg, Wh/L,
  efficiency, best pressure).
- **How one setting changes them:** a sweep of one setting (pressure, power,
  hours, efficiency target, lifetime, electricity price, production scale) with
  one line per candidate and a star at the current value.
- **Built as:** switches every price between mass production and small series.

## 12. Validation

- Mn–H₂ (Chen et al., Nature Energy 3 (2018) 428), about 1.3 V measured: model
  open-circuit voltage 1.22 V.
- H₂–Br₂ (Cho et al., J. Electrochem. Soc. 159 (2012) A1806), 1.4 W/cm² peak
  measured: model p90 0.72 W/cm². The generic exchange-current range is too
  pessimistic for fast halogen couples; per-couple kinetics are the main open
  input.
- The molecule price fit reproduces every anchor within ×4 and all but one
  within its ×3 range (methyl viologen sells at about 3 USD/kg; the fit gives 9).
- **Known flow-battery molecules** (`scripts/benchmark_known.py`, full output in
  [benchmark_known.md](benchmark_known.md)):
  - xTB voltages against measured values at pH 0: mean absolute error 0.061 V
    over five molecules (o-benzoquinone, methyl-p-benzoquinone,
    1,4-naphthoquinone, AQS, AQDS; largest error −0.144 V for naphthoquinone),
    well inside the stated ±0.3 V.
  - The published AQS | BQDS pair (Yang et al. 2014) beats 98 % of the pairs in
    the search library; AQS is the best negative and AQDS is in the top 10 %.
  - Alkaline molecules (2,6-DHAQ, alloxazine) rank low, as expected: the model
    covers acid cells only.
  - The benchmark first found three blind spots, now fixed: sulfonic-acid
    molecules failed outright (the bond rebuild crashed on hypervalent sulfur);
    fast proton shifts between N and O atoms were wrongly called "not
    reversible" (alloxazine, flavins); ESOL gave ionised molecules a solubility
    1000 times too low (AQDS: 1 mM instead of > 1 M).
- `tests/` (51 tests) checks the physics, the cost arithmetic, the stability
  alerts, the pair logic and the explanations.

## 13. Limits: what the results cannot tell you yet

- **Reaction speed of new couples and molecules is unknown.** It is sampled over
  a wide, mostly slow range, and it usually decides the stack size and cost.
- **xTB voltages carry about ±0.3 V.** A molecule that looks outstanding should
  be checked with a higher level of theory (DFT) before any lab work.
- **The molecule price is a difficulty-based estimate** good to about ×3, not a
  synthesis route. A process-chemistry estimate (route, steps, yields, precursor
  prices) should replace it for any molecule that gets that far.
- **Lifetime is not predicted.** No law gives it; it is reported as unknown.
- **Tanks and pumps use small-series prices** (no mass-production discount
  found in the literature), and the electricity the pumps use is not counted.
- **Stability alerts are pattern-based.** They catch known unstable groups, not
  every decomposition path.
- **Acid cells only.** Molecules built for alkaline batteries (DHAQ,
  alloxazine with ferrocyanide) are judged at pH 0, where they look poor.
- **Solubility of ionised molecules is assumed** (1 mol/L for sulfonic and
  phosphonic acids), because ESOL does not apply to them.
- The Materials Project generator, Pourbaix stability and machine-learned
  kinetics are planned, not done (see README, "Next").

## 14. References

- Ertl & Schuffenhauer, *J. Cheminform.* 1 (2009) 8 — SA score.
- Brenk et al., *ChemMedChem* 3 (2008) 435 — structural alerts.
- Delaney, *J. Chem. Inf. Comput. Sci.* 44 (2004) 1000 — ESOL solubility.
- Bannwarth, Ehlert & Grimme, *J. Chem. Theory Comput.* 15 (2019) 1652 — GFN2-xTB.
- Krenn et al., *Mach. Learn.: Sci. Technol.* 1 (2020) 045024 — SELFIES.
- Noack et al., *Energies* 9 (2016) 627 — techno-economic model of a 10 kW / 120 kWh flow battery.
- US DOE Program Record 26002 (2026) — heavy-duty fuel-cell system cost, 2024 status (Strategic Analysis, Inc.).
- NREL, 2023 commercial PV-plus-storage cost benchmark — inverter price.
- Darling, Gallagher, Kowalski, Ha & Brushett, *Energy Environ. Sci.* 7 (2014) 3459 — flow-battery cost pathways, equipment scaling.
- Dieterich et al., *Transl. Mater. Res.* (2018), doi:10.1088/2053-1613/aacb0e — cost of AQDS.
- CoPriNet, *Digital Discovery* (2023), doi:10.1039/D2DD00071G — compound price prediction; SA score vs price.
- PNNL, Energy Storage Grand Challenge Cost and Performance Assessment (2020) — flow-battery cost structure.
- Witman et al., ML-HydPARK, doi:10.5281/zenodo.10680097 — metal hydride data.
- Chen et al., *Nature Energy* 3 (2018) 428; Cho et al., *J. Electrochem. Soc.* 159 (2012) A1806 — validation cells.
