# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def model1d():
    """Load 1D model once for all tests."""
    DATA_FILE = Path(__file__).parent.parent / "data" / "BHPTNRSur1dq1e4.h5"
    if not DATA_FILE.exists():
        pytest.skip(f"Data file {DATA_FILE} not found, skipping tests.")
    from bhptnrsurrogate import BHPTNRSur1dq1e4
    return BHPTNRSur1dq1e4

@pytest.fixture(scope="session")
def model2d():
    """Load 2D model once for all tests."""
    DATA_FILE = Path(__file__).parent.parent / "data" / "BHPTNRSur2dq1e3.h5"
    if not DATA_FILE.exists():
        pytest.skip(f"Data file {DATA_FILE} not found, skipping tests.")
    from bhptnrsurrogate import BHPTNRSur2dq1e3
    return BHPTNRSur2dq1e3