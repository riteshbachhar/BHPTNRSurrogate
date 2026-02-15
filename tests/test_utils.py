"""Tests for bhptnrsurrogate.common_utils.utils"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from bhptnrsurrogate.common_utils.utils import (
    chars_to_string,
    amp_ph_to_comp,
    re_im_to_comp,
    coorbital_to_inertial,
    phase_rotation,
    sum_modes,
    generate_negative_m_mode,
)


@pytest.fixture
def sample_waveform_dict():
    t = np.linspace(-1000, 0, 500)
    h22 = np.exp(-((t + 500) ** 2) / 5000) * np.exp(-1j * 0.1 * t)
    h33 = 0.3 * np.exp(-((t + 500) ** 2) / 5000) * np.exp(-1j * 0.15 * t)
    return {(2, 2): h22, (3, 3): h33}


# --------------------------------------------------------------------------
# chars_to_string
# --------------------------------------------------------------------------
class TestCharsToString:
    def test_basic_conversion(self):
        chars = [72, 101, 108, 108, 111]  # "Hello"
        assert chars_to_string(chars) == "Hello"

    def test_empty(self):
        assert chars_to_string([]) == ""


# --------------------------------------------------------------------------
# amp_ph_to_comp
# --------------------------------------------------------------------------
class TestAmpPhToComp:
    def test_unit_amplitude_zero_phase(self):
        result = amp_ph_to_comp(1.0, 0.0)
        assert_allclose(result, 1.0 + 0j)

    def test_unit_amplitude_pi_phase(self):
        result = amp_ph_to_comp(1.0, np.pi)
        assert_allclose(result, -1.0 + 0j, atol=1e-15)

    def test_zero_amplitude(self):
        result = amp_ph_to_comp(0.0, 1.5)
        assert_allclose(result, 0.0 + 0j)

    def test_array_inputs(self):
        amp = np.array([1.0, 2.0, 0.5])
        phase = np.array([0.0, np.pi / 2, np.pi])
        result = amp_ph_to_comp(amp, phase)
        expected = amp * np.exp(1j * phase)
        assert_allclose(result, expected)


# --------------------------------------------------------------------------
# re_im_to_comp
# --------------------------------------------------------------------------
class TestReImToComp:
    def test_basic(self):
        result = re_im_to_comp(3.0, 4.0)
        assert result == 3.0 + 4j

    def test_arrays(self):
        re = np.array([1.0, 2.0])
        im = np.array([3.0, 4.0])
        result = re_im_to_comp(re, im)
        assert_allclose(result, np.array([1 + 3j, 2 + 4j]))


# --------------------------------------------------------------------------
# coorbital_to_inertial
# --------------------------------------------------------------------------
class TestCoorbitalToInertial:
    def test_22_mode_unchanged(self, sample_waveform_dict):
        h_coorb = {(2, 2): sample_waveform_dict[(2, 2)]}
        h_inert = coorbital_to_inertial(h_coorb)
        assert_allclose(h_inert[(2, 2)], h_coorb[(2, 2)])

    def test_higher_mode_rotated(self):
        t = np.linspace(-100, 0, 200)
        # (2,2) with known phase
        h22 = np.exp(-1j * 0.1 * t)
        # constant coorbital HM
        h33_coorb = np.ones_like(t, dtype=complex)
        h_coorb = {(2, 2): h22, (3, 3): h33_coorb}
        h_inert = coorbital_to_inertial(h_coorb)

        # orbital phase = unwrap(angle(h22)) / 2
        orbital_phase = np.unwrap(np.angle(h22)) / 2
        expected_33 = h33_coorb * np.exp(1j * 3 * orbital_phase)
        assert_allclose(h_inert[(3, 3)], expected_33)


# --------------------------------------------------------------------------
# phase_rotation
# --------------------------------------------------------------------------
class TestPhaseRotation:
    def test_zero_rotation_is_identity(self, sample_waveform_dict):
        h_rot = phase_rotation(sample_waveform_dict, 0.0)
        for mode in sample_waveform_dict:
            assert_allclose(h_rot[mode], sample_waveform_dict[mode])

    def test_known_rotation(self):
        h = {(2, 2): np.array([1.0 + 0j]), (3, 3): np.array([1.0 + 0j])}
        delta = np.pi / 4
        h_rot = phase_rotation(h, delta)
        # (2,2): exp(i*2*pi/4) = exp(i*pi/2) = i
        assert_allclose(h_rot[(2, 2)], np.array([1j]), atol=1e-15)
        # (3,3): exp(i*3*pi/4)
        expected = np.exp(1j * 3 * np.pi / 4)
        assert_allclose(h_rot[(3, 3)], np.array([expected]), atol=1e-15)


# --------------------------------------------------------------------------
# sum_modes
# --------------------------------------------------------------------------
class TestSumModes:
    def test_sum_two_modes(self):
        h = {
            (2, 2): np.array([1.0 + 2j, 3.0 + 4j]),
            (3, 3): np.array([0.5 + 0.5j, 1.0 + 1.0j]),
        }
        result = sum_modes(h)
        expected = np.array([1.5 + 2.5j, 4.0 + 5.0j])
        assert_allclose(result, expected)


# --------------------------------------------------------------------------
# generate_negative_m_mode
# --------------------------------------------------------------------------
class TestGenerateNegativeMMode:
    def test_symmetry_relation(self):
        h = {
            (2, 2): np.array([1.0 + 2j, 3.0 + 4j]),
            (3, 3): np.array([0.5 + 0.5j, 1.0 - 1.0j]),
        }
        h_all = generate_negative_m_mode(h)

        # positive modes preserved
        assert_allclose(h_all[(2, 2)], h[(2, 2)])
        assert_allclose(h_all[(3, 3)], h[(3, 3)])

        # h(l,-m) = (-1)^l * conj(h(l,m))
        assert_allclose(h_all[(2, -2)], (-1) ** 2 * np.conj(h[(2, 2)]))
        assert_allclose(h_all[(3, -3)], (-1) ** 3 * np.conj(h[(3, 3)]))

    def test_m_zero_raises(self):
        h = {(2, 0): np.array([1.0 + 0j])}
        with pytest.raises(ValueError, match="nonnegative"):
            generate_negative_m_mode(h)

    def test_negative_m_raises(self):
        h = {(2, -2): np.array([1.0 + 0j])}
        with pytest.raises(ValueError, match="nonnegative"):
            generate_negative_m_mode(h)
