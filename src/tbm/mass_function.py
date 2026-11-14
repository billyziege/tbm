"""MassFunction: sparse representation of a basic belief assignment over a frame of discernment."""

from __future__ import annotations

from typing import Any


_TOLERANCE = 1e-6
_PRUNE = 1e-12


class MassFunction:
    """Sparse basic belief assignment over a frame of discernment.

    Parameters
    ----------
    masses : dict[frozenset, float]
        Keys are focal elements (frozensets of arbitrary hashable identifiers).
        ``frozenset()`` is the empty set (open-world mass m(∅)).
        Values must be non-negative and sum to 1.0 within *_TOLERANCE*.
    """

    def __init__(self, masses: dict[frozenset, float]) -> None:
        for fe, v in masses.items():
            if v < 0.0:
                raise ValueError(
                    f"Mass must be non-negative; got {v!r} for focal element {fe!r}."
                )
        total = sum(masses.values())
        if abs(total - 1.0) > _TOLERANCE:
            raise ValueError(
                f"Masses must sum to 1.0 within tolerance {_TOLERANCE}; got {total!r}."
            )
        # Store only entries with significant mass (prune near-zeros).
        self._masses: dict[frozenset, float] = {
            fe: v for fe, v in masses.items() if v > _PRUNE
        }
        if not self._masses and abs(total - 1.0) <= _TOLERANCE:
            # All masses were below the prune threshold but the sum was valid —
            # this can only happen with degenerate input (sum of tiny values ≈ 1).
            # Keep the original dict rather than lose all information.
            self._masses = dict(masses)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, focal_element: frozenset) -> float:
        """Return mass for *focal_element*; 0.0 if not a focal element."""
        return self._masses.get(focal_element, 0.0)

    def empty_set_mass(self) -> float:
        """Return m(∅)."""
        return self._masses.get(frozenset(), 0.0)

    def focal_elements(self) -> list[frozenset]:
        """Return all focal elements with positive mass, including ∅ if present."""
        return list(self._masses.keys())

    def committed_mass(self) -> float:
        """Return 1 − m(∅): total mass committed to the frame."""
        return 1.0 - self.empty_set_mass()
