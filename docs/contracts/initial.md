# TBM Initial Contract

**Contract ID:** TBM initial  
**Repository:** tbm  
**Module:** `src/tbm/`  
**Dependencies:** none  
**Status:** Pending implementation

---

## Purpose

Provide a pure-Python belief function algebra library implementing the Transferable Belief Model (TBM) of Smets. The library must be fully general — it has no knowledge of any specific application domain (not token space, not Cauchy distributions). It operates entirely on `MassFunction` objects: sparse mappings from focal elements (frozensets of arbitrary hashable identifiers) to mass values.

---

## Key Design Constraints

- **Zero runtime dependencies.** No NumPy, SciPy, or any third-party package.
- **Open world.** `frozenset()` (the empty set) is a valid focal element. `m(∅) > 0` represents the belief that the true state is outside the frame of discernment.
- **Smets' unnormalized combination.** `m(∅)` is preserved through combination — it is never normalized away.
- `MassFunction` must enforce: all masses ≥ 0; all masses sum to 1 (including `m(∅)`); `m` is only defined on focal elements with positive mass (sparse).

---

## Public API

### `MassFunction` (class — `mass_function.py`)

```python
class MassFunction:
    def __init__(self, masses: dict[frozenset, float]) -> None:
        """
        Parameters
        ----------
        masses : dict mapping frozenset → float
            Keys are focal elements (frozenset of hashable identifiers).
            frozenset() is the empty set (open-world mass).
            Values must be non-negative and sum to 1.0 (within tolerance).
        """

    def __getitem__(self, focal_element: frozenset) -> float:
        """Return mass for focal_element; 0.0 if not a focal element."""

    def empty_set_mass(self) -> float:
        """Return m(∅)."""

    def focal_elements(self) -> list[frozenset]:
        """Return all focal elements with positive mass, including ∅ if present."""

    def committed_mass(self) -> float:
        """Return 1 - m(∅): total mass committed to the frame."""
```

### `smets_combination` (`combination.py`)

```python
def smets_combination(mass_functions: list[MassFunction]) -> MassFunction:
    """
    Apply Smets' unnormalized conjunctive combination rule to a list of
    MassFunctions from independent sources.

    m(A) = Σ_{B ∩ C = A} m1(B) * m2(C)   for all A ⊆ Θ ∪ {∅}

    m(∅) accumulates conflict mass and is NOT normalized away.
    Result sums to 1 (including m(∅)).

    Parameters
    ----------
    mass_functions : list of MassFunction, length ≥ 1
        Sources assumed to be independent.

    Returns
    -------
    MassFunction
        Combined belief assignment.
    """
```

### `pignistic_transform` (`transforms.py`)

```python
def pignistic_transform(
    mass_fn: MassFunction,
    frame: frozenset,
) -> dict[Any, float]:
    """
    Compute the pignistic probability BetP for each singleton in frame.

    BetP({θ}) = Σ_{A ∋ θ, A ≠ ∅} m(A) / |A|  /  (1 - m(∅))

    m(∅) is excluded from the transform (open-world normalisation).
    Raises ValueError if committed_mass == 0 (all mass on ∅).

    Parameters
    ----------
    mass_fn : MassFunction
    frame : frozenset
        The full set of singleton elements (the frame of discernment).

    Returns
    -------
    dict mapping element → float
        Pignistic probabilities; sums to 1.0 over all elements in frame.
    """
```

### `extend_frame` (`frame.py`)

```python
def extend_frame(
    mass_fn: MassFunction,
    new_element: Any,
    theta: frozenset,
) -> MassFunction:
    """
    Add new_element to the frame of discernment.

    Mass currently on the full set Θ is redistributed to Θ ∪ {new_element}.
    All other focal elements are unchanged. m(∅) is unchanged.

    Parameters
    ----------
    mass_fn : MassFunction
    new_element : hashable
        The new element being added.
    theta : frozenset
        The current full frame (before extension).

    Returns
    -------
    MassFunction
        Updated belief assignment over the extended frame.
    """
```

### `focal_elements` (`frame.py`)

```python
def focal_elements(mass_fn: MassFunction) -> list[frozenset]:
    """Return all focal elements with positive mass (alias for MassFunction.focal_elements)."""
```

---

## Invariants and Edge Cases

- Combining a single `MassFunction` with `smets_combination` returns an equal `MassFunction`.
- Combining two `MassFunction`s where all focal elements are disjoint yields a result with `m(∅) = 1.0` (total conflict).
- `pignistic_transform` on a `MassFunction` with `m(∅) = 1.0` must raise `ValueError`.
- Masses below a small tolerance (1e-12) may be treated as zero and pruned from the focal element list.
- `MassFunction` construction must raise `ValueError` if masses do not sum to 1.0 within tolerance or if any mass is negative.
