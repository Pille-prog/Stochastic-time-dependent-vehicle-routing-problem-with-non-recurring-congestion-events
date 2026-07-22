"""Scaffold wiring checks — superseded by the real suites as tickets 04+ land."""

from hypothesis import given
from hypothesis import strategies as st

import stdvrp


def test_package_importable() -> None:
    assert stdvrp.__version__


@given(st.lists(st.integers()))
def test_hypothesis_is_wired(xs: list[int]) -> None:
    assert sorted(sorted(xs)) == sorted(xs)
