"""Presentation/dump tests; no standalone imports or simulation execution."""

from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.mark.parametrize("model", ["rigid", "solid"])
@pytest.mark.parametrize("style", ["screen", "paper"])
def test_dump_views_and_styles(monkeypatch, tmp_path, model, style):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(__file__).resolve().parents[1] / "Experiments/simple_points_study"
    monkeypatch.syspath_prepend(str(directory))
    module = runpy.run_path(str(directory / "debug_plots.py"))
    functions = module["plot_form_function_from_dump"]
    settings = deepcopy(module["SIM_PARAMS"])
    settings["debug"].update(target_model=model, plot_style=style)
    monkeypatch.setitem(functions.__globals__, "SIM_PARAMS", settings)
    path = tmp_path / "dump.npz"
    ka = np.linspace(.1, 50, 100)
    ff = np.exp(-1j * ka) * (1 + .1 * np.sin(ka))
    if model == "solid":
        np.savez(path, target_model="solid_sphere", ka=ka, f_hz=ka,
                 ff_hickling_complex=ff, ff_openstb_complex=np.conj(ff),
                 validation_ka_bounds=[10, 40])
        settings["debug"]["solid_plugin_dump_path"] = str(path)
    else:
        np.savez(path, ka=ka, ff_complex=ff, theta_sample_rad=np.pi)
        settings["debug"]["plugin_dump_path"] = str(path)
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")
    try:
        functions()
        assert len(plt.get_fignums()) == 2
        for number in plt.get_fignums():
            for ax in plt.figure(number).axes:
                assert bool(ax.get_title()) == (style == "screen")
        rate = module["simulation_sample_rate"](np.arange(21993) / 219920.300364381,
                                                 21992.0300364381)
        assert rate == 10 * 21992.0300364381
    finally:
        plt.close("all")


@pytest.mark.parametrize("model", ["rigid", "solid"])
@pytest.mark.parametrize("source_kind", ["sine", "lfm"])
def test_incident_frequency_order_phase_and_exact_source_record(monkeypatch, model, source_kind):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openstb.simulator.system.signal import LFMChirp, SinusoidBurst

    directory = Path(__file__).resolve().parents[1] / "Experiments/simple_points_study"
    monkeypatch.syspath_prepend(str(directory))
    module = runpy.run_path(str(directory / "debug_plots.py"))
    settings = deepcopy(module["SIM_PARAMS"])
    settings["debug"].update(target_model=model, plot_style="screen", incident_phase_min_relative=1e-3)
    plot = module["plot_incident_spectrum"]
    monkeypatch.setitem(plot.__globals__, "SIM_PARAMS", settings)
    monkeypatch.setattr(plt, "show", lambda: None)
    if source_kind == "lfm":
        signal = LFMChirp(500, 10000, .02, 190, True, {"name": "hann", "parameters": {}})
        fs, carrier = 30000., 5250.
    else:
        signal = SinusoidBurst(14000, 2)
        fs, carrier = 140000., 0.
    sample_time = np.arange(int(.1*fs)) / fs
    time, source = module["reconstruct_simulation_source"](signal, fs, sample_time, carrier)
    guard = int(np.ceil(signal.duration*1.1*fs))
    assert len(time) == len(sample_time)+guard
    np.testing.assert_array_equal(source, signal.sample(time, carrier))
    if source_kind == "lfm":
        # Sampling only Ns changes the current OpenSTB RMS scaling. Make sure
        # plots use the full-record source and only then crop to Ns.
        assert not np.allclose(source[:len(sample_time)], signal.sample(sample_time, carrier))
    else:
        np.testing.assert_array_equal(source[:len(sample_time)], signal.sample(sample_time, carrier))
    try:
        plot(source, fs, carrier)
        fig = plt.gcf()
        x = fig.axes[0].lines[0].get_xdata()
        y = fig.axes[0].lines[0].get_ydata()
        assert np.all(np.diff(x) > 0)
        nfft = max(settings["debug"]["incident_fft_points"], len(source))
        coefficients = np.fft.fft(source if model == "solid" or carrier else source.real, nfft)/fs
        if model == "rigid" and not carrier:
            coefficients *= 2
        f = np.fft.fftfreq(nfft, 1/fs)+carrier
        order = np.argsort(f[f >= 0])
        expected = abs(coefficients[f >= 0][order])
        np.testing.assert_allclose(y, expected)
        phase = fig.axes[1].lines[0].get_ydata()
        assert np.isnan(phase[y < y.max()*1e-3]).all()
        neighboring = np.isfinite(phase[1:]) & np.isfinite(phase[:-1])
        assert np.all(abs(np.diff(phase)[neighboring]) <= np.pi)
        assert np.isfinite(phase).any()
    finally:
        plt.close("all")
