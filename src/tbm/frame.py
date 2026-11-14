"""Frame-of-discernment utilities: extend_frame, focal_elements, empty-set mass extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mass_function import MassFunction


def extend_frame(
    mass_fn: "MassFunction",
    new_element: Any,
    theta: frozenset,
) -> "MassFunction":
    """Add *new_element* to the frame of discernment.

    Mass currently on the full set Θ is redistributed to Θ ∪ {new_element}.
    All other focal elements are unchanged.  m(∅) is unchanged.

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
    from .mass_function import MassFunction  # local import to avoid circular

    new_theta = theta | frozenset({new_element})
    new_masses: dict[frozenset, float] = {}

    for fe in mass_fn.focal_elements():
        m = mass_fn[fe]
        if m == 0.0:
            continue
        if fe == theta:
            # Redistribute to the extended full set
            target = new_theta
        else:
            target = fe
        new_masses[target] = new_masses.get(target, 0.0) + m

    return MassFunction(new_masses)


def focal_elements(mass_fn: "MassFunction") -> list[frozenset]:
    """Return all focal elements with positive mass (alias for MassFunction.focal_elements)."""
    return mass_fn.focal_elements()
