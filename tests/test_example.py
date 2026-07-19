"""Example test file."""

import pytest


def test_example():
    """Example test case."""
    assert 1 + 1 == 2


def test_import_src():
    """Test that src package can be imported."""
    import src
    assert hasattr(src, '__version__')
