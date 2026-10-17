# structure-preserving contact neural operators

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%25-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%25-orange.svg)](https://pytorch.org/)

standard neural ODEs and HNNs are cool for conservative physics, but they completely break down the second you introduce friction, viscosity, or non-holonomic drag. they accumulate error and blow up. 

this repo is a pytorch framework that fixes that by using contact geometry and softplus-constrained dissipation. basically, it enforces thermodynamic consistency and keeps long-term trajectories stable without exploding.

---

## what it fixes
* standard models assume lossless flows and violate thermodynamic bounds ($\Delta S < 0$)
* error piles up super fast over long rollout horizons
* they totally fail and output NaNs when you test them out-of-distribution

## how it works
instead of standard symplectic spaces, we build the integration step on contact manifolds. coupled with softplus dissipation layers, it naturally bounds energy loss and stops the model from drifting.

---

## repo structure

```text
├── core/
│   ├── contact_layer.py        # contact geometry integrator & softplus dissipation
│   ├── rollout.py              # multi-step unrolled trajectory rollouts
│   └── data_utils.py           # n-dimensional canonical coordinate mapping
├── models/
│   ├── baseline_net.py         # standard unconstrained neural ODE baseline
│   └── hnn_baseline.py         # conservative hamiltonian neural network baseline
├── data/
│   ├── dataset.py              # pendulum & chaplygin sleigh trajectory generators
│   └── benchmark_fluid.py      # viscous fluid wave damping dataset generators
├── training/
│   └── train.py                # thermodynamic loss function & execution steps
├── evaluation/
│   ├── evaluate_energy.py      # long-term autoregressive energy conservation checks
│   └── sensitivity.py          # step-size (dt) and horizon scaling benchmarks
└── assets/                     # validation plots & comparison pdfs
