# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Numerical regression tests, not an independent validation of modal physics."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import eval_legendre, spherical_jn, spherical_yn

from openstb.simulator.distortion.solid_sphere import SolidSphereFormFunction
from openstb.simulator.environment.invariant import InvariantEnvironment


def make_plugin(**overrides):
    parameters = dict(
        radius_m=0.25,
        solid_density_kg_m3=7700.0,
        compressional_speed_m_s=5960.0,
        shear_speed_m_s=3240.0,
        fluid_density_kg_m3=1000.0,
    )
    return SolidSphereFormFunction(**(parameters | overrides))


@pytest.fixture(scope="module")
def reference():
    """Load only historical modal functions, without plotting or running main.

    Extracting the existing functions avoids a Matplotlib test dependency and
    does not duplicate the equations in the test. This is a regression oracle,
    not an independently derived theoretical solution.
    """
    path = Path(__file__).resolve().parents[2] / (
        "Experiments/scattering_models/solid_sphere/free_field/"
        "SolidSphereEchoAnalytic.py"
    )
    names = {
        "hankel1_spherical", "hankel1_sph_deriv", "modes_solid",
        "compute_modal_sum", "scale_far_field_form_function",
        "compute_hickling_form_function",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = dict(
        np=np, spherical_jn=spherical_jn, spherical_yn=spherical_yn,
        eval_legendre=eval_legendre,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["compute_hickling_form_function"]


@pytest.mark.parametrize("sound_speed", [1410.0, 1480.0])
def test_complex_modal_response_against_standalone(reference, sound_speed):
    ka = np.r_[0, 1e-9, np.linspace(0.1, 40, 401), 24.5, 25.5]
    frequency = ka * sound_speed / (2 * np.pi * 0.25)
    expected, _ = reference(
        frequency, sound_speed, 0.25, 5960, 3240, 1000, 7700, 1, 80
    )
    actual = make_plugin().form_function_hickling(ka, sound_speed)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(abs(actual), abs(expected), rtol=1e-10, atol=1e-11)
    nonzero = abs(expected) > 1e-8
    wrapped_error = np.angle(actual[nonzero] * np.conjugate(expected[nonzero]))
    np.testing.assert_allclose(wrapped_error, 0, atol=1e-10)


@pytest.mark.parametrize("shape", [(1, 7, 1), (3, 7, 1), (1, 7, 4), (2, 7, 3)])
@pytest.mark.parametrize("scale", [0.0, 1.0, 2.5])
def test_apply_conjugation_shape_mask_and_input(shape, scale):
    plugin = make_plugin(scale=scale)
    f = np.array([-14000, -500, 0, 1e-7, 500, 14000, 22000.0])
    rng = np.random.default_rng(27)
    spectrum = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    original = spectrum.copy()
    environment = InvariantEnvironment(salinity=0, sound_speed=1410, temperature=10)
    tt = SimpleNamespace(tx_position=np.zeros(3))
    ff = plugin.form_function_hickling(2 * np.pi * abs(f) * 0.25 / 1410, 1410)
    expected = spectrum * np.where(f > 0, scale * np.conjugate(ff), 0)[None, :, None]
    for carrier in (0, 5250):
        actual = plugin.apply(0, f, spectrum, carrier, environment, (500, 10000), tt)
        assert actual.shape == shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(spectrum, original)
    if scale:
        # A nonreal response makes this test sensitive to omitted conjugation.
        assert np.any(abs(ff.imag) > 1e-3)
        assert np.any(abs(expected[:, -2:, :]) > 0)


def test_apply_reads_environment_at_ping_and_tx_position():
    plugin = make_plugin()
    position = np.array([1, 2, 3.0])
    calls = []

    def sound_speed(time, pos):
        calls.append((time, pos))
        return np.array([1480.0, 1480.0])

    f = np.array([14000.0])
    actual = plugin.apply(
        7, f, np.ones((1, 1, 1)), 5250,
        SimpleNamespace(sound_speed=sound_speed), (1000, 20000),
        SimpleNamespace(tx_position=position),
    )
    expected = np.conjugate(plugin.form_function_hickling(2 * np.pi * f * .25 / 1480, 1480))
    np.testing.assert_allclose(actual[0, :, 0], expected)
    assert len(calls) == 1 and calls[0][0] == 7
    np.testing.assert_array_equal(calls[0][1], position)


def test_zero_threshold_and_empty_grid():
    plugin = make_plugin()
    np.testing.assert_array_equal(plugin.form_function_hickling([0, 1e-9, 1e-8], 1410), 0)
    assert plugin.form_function_hickling([], 1410).shape == (0,)
    tiny = plugin.form_function_hickling([2e-8, 1e-5, 1e-3], 1410)
    assert np.isfinite(tiny).all()


@pytest.mark.parametrize("name", [
    "radius_m", "solid_density_kg_m3", "compressional_speed_m_s",
    "shear_speed_m_s", "fluid_density_kg_m3", "ka_eps",
])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_positive_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        make_plugin(**{name: value})


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_invalid_modal_count(value):
    with pytest.raises(ValueError, match="n_terms"):
        make_plugin(n_terms=value)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_invalid_scale(value):
    with pytest.raises(ValueError, match="scale"):
        make_plugin(scale=value)


@pytest.mark.parametrize("ka", [[-1], [np.nan], [np.inf], [[1, 2]], 1.0])
def test_invalid_ka(ka):
    with pytest.raises(ValueError, match="ka"):
        make_plugin().form_function_hickling(ka, 1410)


@pytest.mark.parametrize("speed", [0, -1, np.nan, np.inf])
def test_invalid_modal_sound_speed(speed):
    with pytest.raises(ValueError, match="sound_speed"):
        make_plugin().form_function_hickling([1], speed)


@pytest.mark.parametrize("speeds", [[], [0], [-1], [np.nan], [np.inf], [1410, 1480]])
def test_invalid_environment_speed(speeds):
    with pytest.raises(ValueError, match="sound speed"):
        make_plugin().apply(
            0, [1000], np.ones((1, 1, 1)), 0,
            SimpleNamespace(sound_speed=lambda time, pos: speeds), (0, 2000),
            SimpleNamespace(tx_position=np.zeros(3)),
        )


@pytest.mark.parametrize("frequency,spectrum", [
    ([[1]], np.ones((1, 1, 1))),
    ([np.nan], np.ones((1, 1, 1))),
    ([np.inf], np.ones((1, 1, 1))),
    ([1], np.ones(1)),
    ([1], np.ones((1, 2, 1))),
])
def test_invalid_apply_arrays(frequency, spectrum):
    with pytest.raises(ValueError):
        make_plugin().apply(0, frequency, spectrum, 0, None, (0, 2), None)


def test_modal_truncation_on_sampled_reference_band():
    """Only checks this sampled Armco iron band, not arbitrary convergence."""
    ka = np.r_[np.linspace(10, 40, 601), 24.5, 25.5]
    responses = [make_plugin(n_terms=n).form_function_hickling(ka, 1410) for n in (60, 80, 100)]
    for actual in responses:
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, responses[1], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_dump_contents_and_once_per_instance(tmp_path, enabled):
    path = tmp_path / "diagnostic" / "solid_sphere_ff_debug.npz"
    plugin = make_plugin(debug_dump=enabled, debug_dump_path=path, scale=2)
    f = np.array([-1000., 0., 500., 14000.])
    spectrum = np.full((2, 4, 3), 1 + 2j)
    env = InvariantEnvironment(salinity=0, sound_speed=1410, temperature=10)
    tt = SimpleNamespace(tx_position=np.zeros(3))
    result = plugin.apply(0, f, spectrum, 5250, env, (500, 10000), tt)
    assert path.exists() == enabled
    if not enabled:
        return
    original_bytes = path.read_bytes()
    with np.load(path, allow_pickle=False) as dump:
        assert dump["target_model"].item() == "solid_sphere"
        for name in ("f_hz", "ka", "ff_hickling_complex", "ff_openstb_complex",
                     "response_complex", "S_in_complex", "S_out_complex"):
            assert dump[name].shape == f.shape
        np.testing.assert_array_equal(dump["f_hz"], f)
        np.testing.assert_allclose(
            dump["ff_openstb_complex"], np.conjugate(dump["ff_hickling_complex"])
        )
        np.testing.assert_allclose(dump["S_out_complex"], result[0, :, 0])
        np.testing.assert_allclose(
            dump["response_complex"],
            np.where(f > 0, 2 * dump["ff_openstb_complex"], 0),
        )
        assert dump["sound_speed_m_s"] == 1410
        assert dump["baseband_frequency_hz"] == 5250
    plugin.apply(1, f, spectrum * 3, 0, env, (500, 10000), tt)
    assert path.read_bytes() == original_bytes


def test_dump_failure_warns_without_changing_response(tmp_path):
    path = tmp_path / "directory.npz"
    path.mkdir()
    plugin = make_plugin(debug_dump=True, debug_dump_path=path)
    env = InvariantEnvironment(salinity=0, sound_speed=1410, temperature=10)
    with pytest.warns(UserWarning, match="dump failed"):
        result = plugin.apply(
            0, [14000], np.ones((1, 1, 1)), 0, env, (0, 20000),
            SimpleNamespace(tx_position=np.zeros(3)),
        )
    assert np.isfinite(result).all()
    assert not plugin._debug_dumped


def test_optional_validation_band_and_dump(tmp_path):
    ka = np.array([0., 5., 10., 24.5, 40., 45.])
    f = ka * 1410 / (2 * np.pi * .25)
    env = InvariantEnvironment(salinity=0, sound_speed=1410, temperature=10)
    tt = SimpleNamespace(tx_position=np.zeros(3))
    spectrum = np.ones((1, len(f), 1), dtype=complex)
    path = tmp_path / "band.npz"
    restricted = make_plugin(
        validation_ka_bounds=(10, 40), debug_dump=True, debug_dump_path=path
    )
    result = restricted.apply(0, f, spectrum, 0, env, (0, 50000), tt)
    general = make_plugin().apply(0, f, spectrum, 0, env, (0, 50000), tt)
    actual_ka = 2 * np.pi * abs(f) * .25 / 1410
    active = (f > 0) & (actual_ka >= 10) & (actual_ka <= 40)
    np.testing.assert_allclose(result, general * active[None, :, None])
    assert abs(general[0, -1, 0]) > 0  # No intrinsic 40-ka cutoff.
    with np.load(path, allow_pickle=False) as dump:
        np.testing.assert_array_equal(dump["validation_ka_bounds"], [10, 40])
        np.testing.assert_array_equal(dump["response_mask"], active)
        np.testing.assert_allclose(dump["S_out_complex"], result[0, :, 0])


@pytest.mark.parametrize("bounds", [(40, 10), (10, 10), (-1, 40), (10,), (10, np.inf)])
def test_invalid_validation_band(bounds):
    with pytest.raises(ValueError, match="validation_ka_bounds"):
        make_plugin(validation_ka_bounds=bounds)
