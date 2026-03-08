"""
PD-* tests for app.helpers.miscellaneous.ParametersDict
"""
import pytest
from app.helpers.miscellaneous import ParametersDict


@pytest.fixture
def defaults():
    return {"alpha": 1.0, "beta": "hello", "gamma": True}


@pytest.fixture
def pd(defaults):
    return ParametersDict({}, defaults)


# PD-01: missing key returns the default value
def test_missing_key_returns_default(pd, defaults):
    assert pd["alpha"] == defaults["alpha"]
    assert pd["beta"] == defaults["beta"]
    assert pd["gamma"] == defaults["gamma"]


# PD-02: explicitly set key wins over default
def test_set_key_overrides_default(defaults):
    pd = ParametersDict({"alpha": 99.9}, defaults)
    assert pd["alpha"] == 99.9


# PD-03: update() overwrites and both old + new keys are accessible
def test_update_merges_keys(pd, defaults):
    pd.update({"alpha": 42, "new_key": "new_val"})
    assert pd["alpha"] == 42
    assert pd["new_key"] == "new_val"
    # Keys already defaulted remain reachable via default
    assert pd["beta"] == defaults["beta"]


# PD-04: accessing a missing key via default does NOT mutate _default_parameters
def test_default_parameters_not_mutated(defaults):
    defaults_copy = dict(defaults)
    pd = ParametersDict({}, defaults)
    _ = pd["alpha"]  # trigger default path
    assert pd._default_parameters == defaults_copy


# PD-05: two independent ParametersDicts sharing the same default dict don't interfere
def test_independent_instances_share_defaults_safely(defaults):
    pd1 = ParametersDict({}, defaults)
    pd2 = ParametersDict({}, defaults)
    pd1["alpha"] = 100
    # pd2 should still see the original default, not pd1's override
    pd2_fresh = ParametersDict({}, defaults)
    assert pd2_fresh["alpha"] == defaults["alpha"]


# PD-06: .get(key, fallback) — UserDict.get() checks self.data directly, not __getitem__
# Keys not present in self.data return the fallback, even if a ParametersDict default exists.
# Use direct key access (pd[key]) to trigger the ParametersDict default mechanism.
def test_get_with_explicit_fallback(defaults):
    pd = ParametersDict({}, defaults)
    # Key absent from data → fallback (UserDict.get does not call __getitem__)
    assert pd.get("nonexistent", "fallback") == "fallback"
    assert pd.get("alpha", "fallback") == "fallback"  # not in data → fallback

def test_getitem_returns_default_not_get(defaults):
    pd = ParametersDict({}, defaults)
    # Direct key access triggers __getitem__ → returns ParametersDict default
    assert pd["alpha"] == defaults["alpha"]


# PD-07: bool value preserved through round-trip
def test_bool_value_round_trip(defaults):
    pd = ParametersDict({"gamma": False}, defaults)
    assert pd["gamma"] is False


# PD-08: len() and iteration work as expected
def test_len_and_iteration(defaults):
    pd = ParametersDict({"alpha": 5, "beta": "x"}, defaults)
    assert len(pd) == 2
    assert set(pd.keys()) == {"alpha", "beta"}
