# tbm — Transferable Belief Model

Pure Python, zero-dependency implementation of Smets' Transferable Belief Model (TBM) — belief function algebra over arbitrary frames of discernment.

## What it is

The TBM is a formal framework for reasoning under uncertainty that generalises both Bayesian probability and Dempster-Shafer theory. The key distinction from Bayesian methods is the **open-world assumption**: mass may be assigned to the empty set `m(∅) > 0`, representing the possibility that none of the hypotheses in the current frame is correct. This mass is preserved (not normalised away) under Smets' conjunctive combination rule, making the TBM suitable for inference problems where the frame may be incomplete.

This library provides the core belief function algebra. It is designed to be used as a backend for higher-level inference frameworks — it has no opinions about what the focal elements represent.

## Install

```bash
pip install git+https://github.com/billyziege/tbm.git
```

Or, using `uv`:

```bash
uv add git+https://github.com/billyziege/tbm.git
```

Requires Python ≥ 3.10. No runtime dependencies.

## Quick start

```python
from tbm import MassFunction, smets_combination, pignistic_transform

# A frame of discernment with two hypotheses: A and B.
# m1 assigns 0.7 to {A}, 0.1 to {A, B}, 0.2 to ∅ (conflict mass).
m1 = MassFunction({
    frozenset({"A"}):    0.7,
    frozenset({"A","B"}): 0.1,
    frozenset():          0.2,   # empty set = m(∅)
})

m2 = MassFunction({
    frozenset({"B"}):    0.6,
    frozenset({"A","B"}): 0.4,
})

# Smets' unnormalized conjunctive combination (open-world).
combined = smets_combination([m1, m2])

# Convert to a probability distribution for decision-making.
probs = pignistic_transform(combined)   # dict[frozenset, float]
```

## API

### `MassFunction`

```python
MassFunction(masses: dict[frozenset, float])
```

Sparse basic belief assignment over a frame of discernment. Keys are focal elements (frozensets of arbitrary hashable identifiers); `frozenset()` is the empty set. Values must be non-negative and sum to 1.0.

Provides `__getitem__`, `__repr__`, and equality comparison. Masses below 1e-12 are pruned on construction.

### `smets_combination`

```python
smets_combination(mass_functions: list[MassFunction]) -> MassFunction
```

Applies Smets' unnormalized conjunctive combination rule:

```
m(A) = Σ_{B ∩ C = A} m1(B) · m2(C)   for all A ⊆ Θ ∪ {∅}
```

`m(∅)` accumulates conflict mass and is **not** normalised away. The result sums to 1.0 including `m(∅)`.

### `pignistic_transform`

```python
pignistic_transform(mass_function: MassFunction) -> dict[frozenset, float]
```

Converts a mass function to a probability distribution (BetP) for decision-making by redistributing each focal element's mass equally among its singletons. The empty set is handled according to Smets' convention: its mass is redistributed uniformly across all singletons in the frame.

### `extend_frame`, `focal_elements`

Frame utility functions:

```python
extend_frame(mass_function, new_element) -> MassFunction
focal_elements(mass_function) -> list[frozenset]
```

`extend_frame` adds a new hypothesis to an existing mass function (useful when the frame grows during inference). `focal_elements` returns all focal elements with non-zero mass.

## Repository structure

```
tbm/
├── src/tbm/
│   ├── __init__.py          # public API
│   ├── mass_function.py     # MassFunction class
│   ├── combination.py       # smets_combination
│   ├── transforms.py        # pignistic_transform
│   └── frame.py             # extend_frame, focal_elements
├── tests/
│   ├── conftest.py
│   └── test_tbm.py          # 42 tests
├── docs/contracts/
│   └── initial.md           # design contract
└── pyproject.toml
```

## Development

```bash
uv sync --dev
uv run pytest
```

## Background

The TBM was developed by Philippe Smets as a subjective interpretation of belief functions. The key references are:

- Smets, P. (1990). The combination of evidence in the transferable belief model. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(5), 447–458.
- Smets, P., & Kennes, R. (1994). The transferable belief model. *Artificial Intelligence*, 66(2), 191–234.

This library is developed as the belief-function backend for a Lorentzian Mixture Model (LMM) framework for semantic concept structure characterization in token spaces produced by large language models. See [`lmm-for-llm`](https://github.com/billyziege/lmm_for_llm) for the full framework.
