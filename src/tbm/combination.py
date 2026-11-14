"""Smets' unnormalized (open-world) combination rule for MassFunctions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mass_function import MassFunction


def smets_combination(mass_functions: list["MassFunction"]) -> "MassFunction":
    """Apply Smets' unnormalized conjunctive combination rule.

    m(A) = Σ_{B ∩ C = A} m1(B) · m2(C)   for all A ⊆ Θ ∪ {∅}

    m(∅) accumulates conflict mass and is NOT normalized away.  The result
    sums to 1.0 (including m(∅)).

    Parameters
    ----------
    mass_functions : list of MassFunction, length ≥ 1
        Sources assumed to be independent.

    Returns
    -------
    MassFunction
        Combined belief assignment.
    """
    from .mass_function import MassFunction  # local import to avoid circular

    if not mass_functions:
        raise ValueError("mass_functions must contain at least one MassFunction.")

    # Start with the first mass function's internal dict.
    accumulator: dict[frozenset, float] = dict(
        zip(mass_functions[0].focal_elements(),
            (mass_functions[0][fe] for fe in mass_functions[0].focal_elements()))
    )

    for mf in mass_functions[1:]:
        new_acc: dict[frozenset, float] = {}
        for fe_a, v_a in accumulator.items():
            if v_a == 0.0:
                continue
            for fe_b in mf.focal_elements():
                v_b = mf[fe_b]
                if v_b == 0.0:
                    continue
                intersection = fe_a & fe_b
                new_acc[intersection] = new_acc.get(intersection, 0.0) + v_a * v_b
        accumulator = new_acc

    return MassFunction(accumulator)
