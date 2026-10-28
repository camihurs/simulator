# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Compare the experiment's Hickling source with the analytic standalone."""

import ast
from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest


def test_hickling_factory_matches_standalone():
    root = Path(__file__).resolve().parents[2]
    experiment = root / "Experiments/simple_points_study"
    original = runpy.run_path(str(experiment / "shared_params.py"))["SIM_PARAMS"]
    original_copy = deepcopy(original)
    params = deepcopy(original)
    params["signal"]["mode"] = "hickling"
    params["environment"]["sound_speed_ms"] = 1410.0
    factory = runpy.run_path(str(experiment / "signal_factory.py"))
    signal, f0, carrier = factory["build_signal_from_params"](params)

    # Read the actual standalone builder, not a retyped source equation. Load
    # only configuration assignments and build_source to avoid plotting imports.
    path = root / (
        "Experiments/scattering_models/solid_sphere/free_field/"
        "SolidSphereEchoAnalytic.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        or (isinstance(node, ast.FunctionDef) and node.name == "build_source")
    ]
    reference = {"np": np, "Path": Path, "__file__": str(path)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), reference)
    reference.update(
        SOURCE_CASE="hickling_fig16_max",
        SOUND_SPEED_M_S=params["environment"]["sound_speed_ms"],
        SPHERE_RADIUS_M=params["solid_sphere"]["radius_m"],
    )
    source = reference["build_source"]()
    assert f0 == pytest.approx(source["f0_hz"], rel=1e-14)
    assert signal.duration == pytest.approx(source["duration_s"], rel=1e-14)
    assert signal.n_cycles == source["n_cycles"] == 5
    assert carrier == source["baseband_frequency_hz"] == 0
    assert 10 * f0 == pytest.approx(source["sample_rate_hz"])

    t = source["time_s"]
    actual = signal.sample(t, carrier)
    expected = source["signal"]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    active = (t >= 0) & (t < signal.duration)
    np.testing.assert_allclose(abs(actual[active]), 1, atol=1e-14)
    np.testing.assert_array_equal(actual[~active], 0)
    phase_error = np.angle(actual[active] * np.conjugate(expected[active]))
    np.testing.assert_allclose(phase_error, 0, atol=1e-13)

    # Explicit endpoints: active at t=0, inactive at and beyond T.
    boundaries = np.array([-1 / source["sample_rate_hz"], 0,
                           signal.duration / 2, signal.duration,
                           signal.duration + 1 / source["sample_rate_hz"]])
    samples = signal.sample(boundaries, carrier)
    np.testing.assert_array_equal(samples[[0, 3, 4]], 0)
    np.testing.assert_allclose(samples[1], -1, atol=1e-13)
    np.testing.assert_allclose(samples[2], 1, atol=1e-13)

    # Also compare the complex FFT: magnitude alone cannot detect phase errors.
    np.testing.assert_allclose(
        np.fft.fft(actual, 16384), np.fft.fft(expected, 16384),
        rtol=1e-11, atol=1e-11,
    )
    assert original == original_copy
    print(f"f0={f0:.12f} Hz; duration={signal.duration:.12g} s")
    print(f"max complex sample error={np.max(abs(actual - expected)):.3e}")
    print(f"max wrapped phase error={np.max(abs(phase_error)):.3e} rad")
