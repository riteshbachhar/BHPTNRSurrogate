"""Integration tests for end-to-end waveform generation.

These tests require the HDF5 data files in data/ and are skipped if missing.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# --------------------------------------------------------------------------
# BHPTNRSur1dq1e4
# --------------------------------------------------------------------------
class TestBHPTNRSur1dq1e4:
    def test_basic_waveform(self, model1d):
        t, h = model1d.generate_surrogate(q=10)
        assert isinstance(t, np.ndarray)
        assert isinstance(h, dict)
        assert len(t) > 0
        # all keys are (l,m) tuples with positive and negative m
        for mode in h:
            assert len(mode) == 2
            assert np.iscomplexobj(h[mode])
            assert len(h[mode]) == len(t)

    def test_single_mode_selection(self, model1d):
        t, h = model1d.generate_surrogate(q=10, modes=[(2, 2)], neg_modes=False)
        assert list(h.keys()) == [(2, 2)]

    def test_calibrated_false(self, model1d):
        t, h = model1d.generate_surrogate(q=10, modes=[(2, 2)], calibrated=False, neg_modes=False)
        assert (2, 2) in h
        assert len(h[(2, 2)]) == len(t)

    def test_neg_modes(self, model1d):
        t, h = model1d.generate_surrogate(q=10, modes=[(2, 2), (3, 3)], neg_modes=True)
        assert (2, -2) in h
        assert (3, -3) in h
        # symmetry: h(l,-m) = (-1)^l * conj(h(l,m))
        assert_allclose(h[(2, -2)], np.conj(h[(2, 2)]), rtol=1e-12)
        assert_allclose(h[(3, -3)], -np.conj(h[(3, 3)]), rtol=1e-12)

    def test_lmax_filtering(self, model1d):
        t, h = model1d.generate_surrogate(q=10, lmax=3, neg_modes=False)
        for (l, m) in h:
            assert l <= 3

    def test_high_mass_ratio(self, model1d):
        t, h = model1d.generate_surrogate(q=1000, modes=[(2, 2)], neg_modes=False)
        assert (2, 2) in h
        assert len(h[(2, 2)]) > 0

    def test_spin_warning_printed(self, model1d, capsys):
        model1d.generate_surrogate(q=10, spin1=0.5, modes=[(2, 2)], neg_modes=False)
        assert "warning" in capsys.readouterr().out


# --------------------------------------------------------------------------
# BHPTNRSur2dq1e3
# --------------------------------------------------------------------------
class TestBHPTNRSur2dq1e3:
    def test_basic_waveform_positive_spin(self, model2d):
        t, h = model2d.generate_surrogate(q=10, spin1=0.5)
        assert isinstance(t, np.ndarray)
        assert isinstance(h, dict)
        assert len(t) > 0
        for mode in h:
            assert np.iscomplexobj(h[mode])
            assert len(h[mode]) == len(t)

    def test_basic_waveform_negative_spin(self, model2d):
        t, h = model2d.generate_surrogate(q=10, spin1=-0.5)
        assert isinstance(t, np.ndarray)
        assert isinstance(h, dict)
        assert len(t) > 0

    def test_zero_spin(self, model2d):
        t, h = model2d.generate_surrogate(q=10, spin1=0.0, modes=[(2, 2)], neg_modes=False)
        assert (2, 2) in h

    def test_single_mode_selection(self, model2d):
        t, h = model2d.generate_surrogate(q=10, spin1=0.3, modes=[(2, 2)], neg_modes=False)
        assert list(h.keys()) == [(2, 2)]

    def test_calibrated_false(self, model2d):
        t, h = model2d.generate_surrogate(
            q=10, spin1=0.3, modes=[(2, 2)], calibrated=False, neg_modes=False
        )
        assert (2, 2) in h

    def test_neg_modes(self, model2d):
        t, h = model2d.generate_surrogate(q=10, spin1=0.3, modes=[(2, 2), (3, 3)], neg_modes=True)
        assert (2, -2) in h
        assert (3, -3) in h

    def test_high_mass_ratio(self, model2d):
        t, h = model2d.generate_surrogate(q=500, spin1=0.2, modes=[(2, 2)], neg_modes=False)
        assert (2, 2) in h
        assert len(h[(2, 2)]) > 0
