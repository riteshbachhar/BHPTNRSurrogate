"""Tests for bhptnrsurrogate.common_utils.check_inputs"""

import pytest
from bhptnrsurrogate.common_utils.check_inputs import (
    check_input_modes,
    check_domain_of_validity,
    check_extrinsic_params,
    check_user_inputs,
)


# --------------------------------------------------------------------------
# check_input_modes
# --------------------------------------------------------------------------
class TestCheckInputModes:
    def test_valid_subset(self):
        available = [(2, 2), (3, 3), (4, 4)]
        requested = [(2, 2), (3, 3)]
        check_input_modes(requested, available)  # should not raise

    def test_exact_match(self):
        modes = [(2, 2), (3, 3)]
        check_input_modes(modes, modes)  # should not raise

    def test_invalid_mode_raises(self):
        available = [(2, 2), (3, 3)]
        requested = [(2, 2), (5, 5)]
        with pytest.raises(ValueError):
            check_input_modes(requested, available)

    def test_single_valid_mode(self):
        available = [(2, 2), (3, 3), (4, 4)]
        check_input_modes([(2, 2)], available)

    def test_empty_request_passes(self):
        available = [(2, 2), (3, 3)]
        check_input_modes([], available)


# --------------------------------------------------------------------------
# check_domain_of_validity
# --------------------------------------------------------------------------
class TestCheckDomainOfValidity:
    def test_in_domain_scalar_no_warning(self, capsys):
        # bounds: [[lower], [upper]]
        check_domain_of_validity(1.0, [[0.0], [2.0]])
        assert "warning" not in capsys.readouterr().out

    def test_out_of_domain_scalar_prints_warning(self, capsys):
        check_domain_of_validity(3.0, [[0.0], [2.0]])
        assert "warning" in capsys.readouterr().out

    def test_at_lower_bound_no_warning(self, capsys):
        check_domain_of_validity(0.0, [[0.0], [2.0]])
        assert "warning" not in capsys.readouterr().out

    def test_at_upper_bound_no_warning(self, capsys):
        check_domain_of_validity(2.0, [[0.0], [2.0]])
        assert "warning" not in capsys.readouterr().out

    def test_below_lower_bound_prints_warning(self, capsys):
        check_domain_of_validity(-0.1, [[0.0], [2.0]])
        assert "warning" in capsys.readouterr().out

    def test_2d_in_domain(self, capsys):
        check_domain_of_validity([1.0, 0.5], [[0.0, -1.0], [2.0, 1.0]])
        assert "warning" not in capsys.readouterr().out

    def test_2d_one_param_out(self, capsys):
        check_domain_of_validity([1.0, 2.0], [[0.0, -1.0], [2.0, 1.0]])
        assert "warning" in capsys.readouterr().out


# --------------------------------------------------------------------------
# check_extrinsic_params
# --------------------------------------------------------------------------
class TestCheckExtrinsicParams:
    def test_all_none_passes(self):
        check_extrinsic_params(None, None, None, None, False)

    def test_mass_and_distance_set_passes(self):
        check_extrinsic_params(50.0, 100.0, None, None, False)

    def test_all_extrinsic_set_passes(self):
        check_extrinsic_params(50.0, 100.0, 0.5, 0.3, False)

    def test_only_mtot_raises(self):
        with pytest.raises(ValueError, match="M_tot and dist_mpc"):
            check_extrinsic_params(50.0, None, None, None, False)

    def test_only_dist_raises(self):
        with pytest.raises(ValueError, match="M_tot and dist_mpc"):
            check_extrinsic_params(None, 100.0, None, None, False)

    def test_only_orb_phase_raises(self):
        with pytest.raises(ValueError, match="orb_phase and inclination"):
            check_extrinsic_params(None, None, 0.5, None, False)

    def test_only_inclination_raises(self):
        with pytest.raises(ValueError, match="orb_phase and inclination"):
            check_extrinsic_params(None, None, None, 0.3, False)

    def test_mode_sum_all_none_raises(self):
        with pytest.raises(ValueError, match="should NOT be None"):
            check_extrinsic_params(None, None, None, None, True)

    def test_mode_sum_all_set_passes(self):
        check_extrinsic_params(50.0, 100.0, 0.5, 0.3, True)


# --------------------------------------------------------------------------
# check_user_inputs (integration of all three checks)
# --------------------------------------------------------------------------
class TestCheckUserInputs:
    def test_valid_inputs(self):
        check_user_inputs(
            X_in=1.0,
            X_bounds=[[0.0], [2.0]],
            modes_requested=[(2, 2)],
            modes_available=[(2, 2), (3, 3)],
            M_tot=None,
            dist_mpc=None,
            orb_phase=None,
            inclination=None,
            mode_sum=False,
        )

    def test_invalid_modes_propagates(self):
        with pytest.raises(ValueError):
            check_user_inputs(
                X_in=1.0,
                X_bounds=[[0.0], [2.0]],
                modes_requested=[(5, 5)],
                modes_available=[(2, 2)],
                M_tot=None,
                dist_mpc=None,
                orb_phase=None,
                inclination=None,
                mode_sum=False,
            )

    def test_invalid_extrinsic_propagates(self):
        with pytest.raises(ValueError):
            check_user_inputs(
                X_in=1.0,
                X_bounds=[[0.0], [2.0]],
                modes_requested=[(2, 2)],
                modes_available=[(2, 2)],
                M_tot=50.0,
                dist_mpc=None,
                orb_phase=None,
                inclination=None,
                mode_sum=False,
            )
