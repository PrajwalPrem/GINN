# GINN: Geodesic-Informed Neural Network

Code for orbit fitting with a geodesic-informed neural network (GINN). 

The code is designed for tests of Schwarzschild and alternative metric models, focusing on S2-star astrometry and radial velocity modelling. 

## Overview

The project models the motion of a star (e.g. S2) using:
- astrometric measurements (right ascension and declination),
- line-of-sight velocity data,
- relativistic orbital dynamics,
- and a neural network correction to model deviations from the Keplerian approximation.

The code supports multiple spacetime metrics, including:
- Schwarzschild
- Hairy Horndeski
- Yukawa
- Schwarzschild-de Sitter
- Reissner–Nordström
- Bardeen
- JNW
- GHS dilaton

## Repository structure

```text
GINN/
├── data/
│   ├── tab_gillessen_pos.csv
│   └── tab_gillessen_vr.csv
├── ginn/
│   ├── __init__.py
│   ├── analysis.py
│   ├── core.py
│   ├── higher_order.py
│   ├── physics.py
│   └── plotting.py
├── notebooks/
│   ├── 00_validation_and_checks.ipynb
│   ├── 01_schwarzschild.ipynb
│   ├── 02_hairy_horndeski.ipynb
│   ├── 03_yukawa.ipynb
│   ├── 04_sds.ipynb
│   ├── 05_reissner_nordstrom.ipynb
│   ├── 06_jnw.ipynb
│   ├── 07_bardeen.ipynb
│   └── compare_orbits.ipynb
└── README.md
