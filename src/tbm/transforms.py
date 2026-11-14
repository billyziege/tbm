"""Pignistic transform: convert a MassFunction to a probability distribution for decision-making."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mass_function import MassFunction


def pignistic_transform(
    mass_fn: "MassFunction",
    frame: frozenset,
) -> dict[Any, float]:
    """Compute the pignistic probability BetP for each singleton in *frame*.

    BetP({θ}) = Σ_{A ∋ θ, A ≠ ∅} m(A) / |A|  /  (1 − m(∅))

    m(∅) is excluded from the transform (open-world normalisation).

    Parameters
    ----------
    mass_fn : MassFunction
    frame : frozenset
        The full set of singleton elements (the frame of discernment).

    Returns
    -------
    dict mapping element → float
        Pignistic probabilities; sums to 1.0 over all elements in *frame*.

    Raises
    ------
    ValueError
        If committed_mass == 0 (all mass on ∅).
    """
    committed = mass_fn.committed_mass()
    if committed == 0.0:
        raise ValueError(
            "pignistic_transform is undefined when all mass is on ∅ (committed_mass == 0)."
        )

    betp: dict[Any, float] = {elem: 0.0 for elem in frame}

    for fe in mass_fn.focal_elements():
        if len(fe) == 0:
            # Skip the empty-set focal element
            continue
        m_fe = mass_fn[fe]
        if m_fe == 0.0:
            continue
        share = m_fe / len(fe)
        for elem in fe:
            if elem in betp:
                betp[elem] += share

    # Normalise by (1 - m(∅))
    betp = {elem: v / committed for elem, v in betp.items()}

    return betp
