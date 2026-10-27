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
