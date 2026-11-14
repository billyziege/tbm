"""Tests for the TBM initial contract."""

import math
import pytest

from tbm import MassFunction, smets_combination, pignistic_transform, extend_frame, focal_elements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# MassFunction construction and validation
# ---------------------------------------------------------------------------

class TestMassFunctionConstruction:
    def test_valid_single_focal_element(self):
        mf = MassFunction({frozenset({"a"}): 1.0})
        assert approx_eq(mf[frozenset({"a"})], 1.0)

    def test_valid_two_focal_elements(self):
        mf = MassFunction({frozenset({"a"}): 0.6, frozenset({"b"}): 0.4})
        assert approx_eq(mf[frozenset({"a"})], 0.6)
        assert approx_eq(mf[frozenset({"b"})], 0.4)

    def test_valid_with_empty_set(self):
        mf = MassFunction({frozenset(): 0.3, frozenset({"a"}): 0.7})
        assert approx_eq(mf[frozenset()], 0.3)
        assert approx_eq(mf[frozenset({"a"})], 0.7)

    def test_all_mass_on_empty_set(self):
        mf = MassFunction({frozenset(): 1.0})
        assert approx_eq(mf[frozenset()], 1.0)

    def test_raises_if_sum_exceeds_tolerance(self):
        with pytest.raises(ValueError):
            MassFunction({frozenset({"a"}): 0.6, frozenset({"b"}): 0.5})

    def test_raises_if_sum_below_tolerance(self):
        with pytest.raises(ValueError):
            MassFunction({frozenset({"a"}): 0.3, frozenset({"b"}): 0.3})

    def test_raises_if_negative_mass(self):
        with pytest.raises(ValueError):
            MassFunction({frozenset({"a"}): 1.2, frozenset({"b"}): -0.2})

    def test_raises_if_exactly_negative(self):
        with pytest.raises(ValueError):
            MassFunction({frozenset({"a"}): -0.1, frozenset({"b"}): 1.1})

    def test_sum_tolerance_exactly_at_limit(self):
        # Sum differs by just over 1e-6 → should raise
        with pytest.raises(ValueError):
            MassFunction({frozenset({"a"}): 1.0 + 2e-6})

    def test_sum_tolerance_within_limit(self):
        # Sum differs by less than 1e-6 → should be fine
        mf = MassFunction({frozenset({"a"}): 1.0 + 5e-7})
        assert mf[frozenset({"a"})] > 0

    def test_empty_dict_raises(self):
        # Empty dict sums to 0.0, not 1.0
        with pytest.raises(ValueError):
            MassFunction({})

    def test_getitem_missing_returns_zero(self):
        mf = MassFunction({frozenset({"a"}): 1.0})
        assert mf[frozenset({"b"})] == 0.0
        assert mf[frozenset()] == 0.0


# ---------------------------------------------------------------------------
# MassFunction methods
# ---------------------------------------------------------------------------

class TestMassFunctionMethods:
    def test_empty_set_mass_present(self):
        mf = MassFunction({frozenset(): 0.4, frozenset({"x"}): 0.6})
        assert approx_eq(mf.empty_set_mass(), 0.4)

    def test_empty_set_mass_absent(self):
        mf = MassFunction({frozenset({"x"}): 1.0})
        assert approx_eq(mf.empty_set_mass(), 0.0)

    def test_focal_elements_returns_all_positive(self):
        mf = MassFunction({frozenset({"a"}): 0.5, frozenset({"b"}): 0.5})
        fes = mf.focal_elements()
        assert frozenset({"a"}) in fes
        assert frozenset({"b"}) in fes
        assert len(fes) == 2

    def test_focal_elements_includes_empty_set(self):
        mf = MassFunction({frozenset(): 0.1, frozenset({"a"}): 0.9})
        fes = mf.focal_elements()
        assert frozenset() in fes

    def test_committed_mass(self):
        mf = MassFunction({frozenset(): 0.3, frozenset({"a"}): 0.7})
        assert approx_eq(mf.committed_mass(), 0.7)

    def test_committed_mass_no_empty_set(self):
        mf = MassFunction({frozenset({"a"}): 1.0})
        assert approx_eq(mf.committed_mass(), 1.0)

    def test_committed_mass_all_on_empty_set(self):
        mf = MassFunction({frozenset(): 1.0})
        assert approx_eq(mf.committed_mass(), 0.0)


# ---------------------------------------------------------------------------
# smets_combination
# ---------------------------------------------------------------------------

class TestSmetsComibination:
    def test_single_function_identity(self):
        mf = MassFunction({frozenset({"a"}): 0.7, frozenset({"b"}): 0.3})
        result = smets_combination([mf])
        assert approx_eq(result[frozenset({"a"})], 0.7)
        assert approx_eq(result[frozenset({"b"})], 0.3)
        assert approx_eq(result[frozenset()], 0.0)

    def test_two_identical_categorical(self):
        # m1 = m2: fully on {a}
        mf = MassFunction({frozenset({"a"}): 1.0})
        result = smets_combination([mf, mf])
        assert approx_eq(result[frozenset({"a"})], 1.0)
        assert approx_eq(result[frozenset()], 0.0)

    def test_conflict_accumulates_on_empty_set(self):
        # Fully disjoint focal elements → total conflict
        m1 = MassFunction({frozenset({"a"}): 1.0})
        m2 = MassFunction({frozenset({"b"}): 1.0})
        result = smets_combination([m1, m2])
        assert approx_eq(result[frozenset()], 1.0)
        assert approx_eq(result[frozenset({"a"})], 0.0)
        assert approx_eq(result[frozenset({"b"})], 0.0)

    def test_partial_conflict(self):
        # m1: 0.8 on {a}, 0.2 on {a,b}
        # m2: 0.6 on {a}, 0.4 on {b}
        m1 = MassFunction({frozenset({"a"}): 0.8, frozenset({"a", "b"}): 0.2})
        m2 = MassFunction({frozenset({"a"}): 0.6, frozenset({"b"}): 0.4})
        result = smets_combination([m1, m2])
        # {a}∩{a}={a}: 0.8*0.6 = 0.48
        # {a}∩{b}=∅: 0.8*0.4 = 0.32 → conflict
        # {a,b}∩{a}={a}: 0.2*0.6 = 0.12
        # {a,b}∩{b}={b}: 0.2*0.4 = 0.08
        assert approx_eq(result[frozenset({"a"})], 0.48 + 0.12)  # 0.60
        assert approx_eq(result[frozenset({"b"})], 0.08)
        assert approx_eq(result[frozenset()], 0.32)

    def test_result_sums_to_one(self):
        m1 = MassFunction({frozenset({"a"}): 0.5, frozenset({"b"}): 0.3, frozenset({"a", "b"}): 0.2})
        m2 = MassFunction({frozenset({"a"}): 0.4, frozenset({"c"}): 0.6})
        result = smets_combination([m1, m2])
        total = sum(result[fe] for fe in result.focal_elements())
        assert approx_eq(total, 1.0)

    def test_empty_set_mass_preserved_unnormalized(self):
        # m1 has m(∅)=0.1; combined with m2 that has no conflict
        m1 = MassFunction({frozenset(): 0.1, frozenset({"a"}): 0.9})
        m2 = MassFunction({frozenset({"a"}): 1.0})
        result = smets_combination([m1, m2])
        # ∅∩{a}=∅: 0.1*1.0=0.1
        # {a}∩{a}={a}: 0.9*1.0=0.9
        assert approx_eq(result[frozenset()], 0.1)
        assert approx_eq(result[frozenset({"a"})], 0.9)

    def test_three_functions(self):
        m1 = MassFunction({frozenset({"a"}): 0.5, frozenset({"a", "b"}): 0.5})
        m2 = MassFunction({frozenset({"a"}): 0.5, frozenset({"a", "b"}): 0.5})
        m3 = MassFunction({frozenset({"a"}): 0.5, frozenset({"a", "b"}): 0.5})
        result = smets_combination([m1, m2, m3])
        # Everything intersects to subsets of {a,b}, so no conflict
        total = sum(result[fe] for fe in result.focal_elements())
        assert approx_eq(total, 1.0)
        assert approx_eq(result[frozenset()], 0.0)


# ---------------------------------------------------------------------------
# pignistic_transform
# ---------------------------------------------------------------------------

class TestPignisticTransform:
    def test_categorical_mass(self):
        mf = MassFunction({frozenset({"a"}): 1.0})
        frame = frozenset({"a", "b"})
        betp = pignistic_transform(mf, frame)
        assert approx_eq(betp["a"], 1.0)
        assert approx_eq(betp["b"], 0.0)

    def test_uniform_mass_on_full_frame(self):
        mf = MassFunction({frozenset({"a", "b"}): 1.0})
        frame = frozenset({"a", "b"})
        betp = pignistic_transform(mf, frame)
        assert approx_eq(betp["a"], 0.5)
        assert approx_eq(betp["b"], 0.5)

    def test_sums_to_one(self):
        mf = MassFunction({frozenset({"a"}): 0.3, frozenset({"a", "b"}): 0.4, frozenset({"b"}): 0.3})
        frame = frozenset({"a", "b"})
        betp = pignistic_transform(mf, frame)
        assert approx_eq(sum(betp.values()), 1.0)

    def test_normalises_by_one_minus_empty_set(self):
        # m(∅)=0.2, m({a})=0.8 → BetP({a})=0.8/0.8=1.0
        mf = MassFunction({frozenset(): 0.2, frozenset({"a"}): 0.8})
        frame = frozenset({"a", "b"})
        betp = pignistic_transform(mf, frame)
        assert approx_eq(betp["a"], 1.0)
        assert approx_eq(betp["b"], 0.0)

    def test_raises_when_all_mass_on_empty_set(self):
        mf = MassFunction({frozenset(): 1.0})
        frame = frozenset({"a", "b"})
        with pytest.raises(ValueError):
            pignistic_transform(mf, frame)

    def test_three_elements(self):
        mf = MassFunction({frozenset({"a", "b", "c"}): 1.0})
        frame = frozenset({"a", "b", "c"})
        betp = pignistic_transform(mf, frame)
        for elem in ("a", "b", "c"):
            assert approx_eq(betp[elem], 1 / 3)

    def test_mixed_masses(self):
        # m({a})=0.6, m({a,b})=0.4; frame={a,b}
        # BetP({a}) = (0.6/1 + 0.4/2) / 1.0 = 0.6 + 0.2 = 0.8
        # BetP({b}) = (0.4/2) / 1.0 = 0.2
        mf = MassFunction({frozenset({"a"}): 0.6, frozenset({"a", "b"}): 0.4})
        frame = frozenset({"a", "b"})
        betp = pignistic_transform(mf, frame)
        assert approx_eq(betp["a"], 0.8)
        assert approx_eq(betp["b"], 0.2)

    def test_all_elements_in_result(self):
        mf = MassFunction({frozenset({"a"}): 1.0})
        frame = frozenset({"a", "b", "c"})
        betp = pignistic_transform(mf, frame)
        for elem in ("a", "b", "c"):
            assert elem in betp


# ---------------------------------------------------------------------------
# extend_frame
# ---------------------------------------------------------------------------

class TestExtendFrame:
    def test_extends_frame(self):
        theta = frozenset({"a", "b"})
        mf = MassFunction({theta: 0.5, frozenset({"a"}): 0.5})
        new_theta = theta | frozenset({"c"})
        result = extend_frame(mf, "c", theta)
        # Mass on old theta moves to new theta
        assert approx_eq(result[new_theta], 0.5)
        assert approx_eq(result[frozenset({"a"})], 0.5)

    def test_non_full_set_unchanged(self):
        theta = frozenset({"a", "b"})
        mf = MassFunction({frozenset({"a"}): 0.3, frozenset({"a", "b"}): 0.7})
        result = extend_frame(mf, "c", theta)
        # {a} unchanged; {a,b}=theta → {a,b,c}
        assert approx_eq(result[frozenset({"a"})], 0.3)
        assert approx_eq(result[frozenset({"a", "b", "c"})], 0.7)

    def test_empty_set_mass_unchanged(self):
        theta = frozenset({"a", "b"})
        mf = MassFunction({frozenset(): 0.1, frozenset({"a", "b"}): 0.9})
        result = extend_frame(mf, "c", theta)
        assert approx_eq(result[frozenset()], 0.1)
        assert approx_eq(result[frozenset({"a", "b", "c"})], 0.9)

    def test_no_mass_on_full_set(self):
        theta = frozenset({"a", "b"})
        mf = MassFunction({frozenset({"a"}): 0.6, frozenset({"b"}): 0.4})
        result = extend_frame(mf, "c", theta)
        # Nothing on theta → nothing on new theta
        assert approx_eq(result[frozenset({"a", "b", "c"})], 0.0)
        assert approx_eq(result[frozenset({"a"})], 0.6)
        assert approx_eq(result[frozenset({"b"})], 0.4)

    def test_result_sums_to_one(self):
        theta = frozenset({"a", "b"})
        mf = MassFunction({frozenset(): 0.1, frozenset({"a"}): 0.3, frozenset({"a", "b"}): 0.6})
        result = extend_frame(mf, "c", theta)
        total = sum(result[fe] for fe in result.focal_elements())
        assert approx_eq(total, 1.0)


# ---------------------------------------------------------------------------
# focal_elements function (alias)
# ---------------------------------------------------------------------------

class TestFocalElementsFunction:
    def test_alias_matches_method(self):
        mf = MassFunction({frozenset({"a"}): 0.5, frozenset({"b"}): 0.5})
        assert set(focal_elements(mf)) == set(mf.focal_elements())

    def test_includes_empty_set(self):
        mf = MassFunction({frozenset(): 0.2, frozenset({"a"}): 0.8})
        fes = focal_elements(mf)
        assert frozenset() in fes

    def test_only_positive_mass(self):
        mf = MassFunction({frozenset({"a"}): 1.0})
        fes = focal_elements(mf)
        assert frozenset({"b"}) not in fes
        assert frozenset() not in fes
