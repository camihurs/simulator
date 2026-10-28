"""Optional standalone form-function comparison; no simulator execution."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from openstb.simulator.distortion.solid_sphere import SolidSphereFormFunction


@pytest.fixture
def standalone():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(__file__).resolve().parents[1] / (
        "Experiments/scattering_models/solid_sphere/free_field/SolidSphereEchoAnalytic.py"
    )
    module = runpy.run_path(str(path))
    yield module
    plt.close("all")


def test_comparison_uses_dump_parameters_and_detects_phase_error(tmp_path, standalone):
    # Different c, radius and material than standalone defaults.
    plugin = SolidSphereFormFunction(.3, 2700, 6420, 3040, 1020, n_terms=60)
    ka = np.linspace(.1, 30, 120)
    f = ka * 1480 / (2*np.pi*.3)
    ff = plugin.form_function_hickling(ka, 1480)
    path = tmp_path / "solid.npz"
    data = dict(
        target_model="solid_sphere", response_geometry="far_field_backscatter_approximation",
        ka=ka, f_hz=f, ff_hickling_complex=ff, ff_openstb_complex=np.conj(ff),
        sound_speed_m_s=1480, radius_m=.3, solid_density_kg_m3=2700,
        compressional_speed_m_s=6420, shear_speed_m_s=3040,
        fluid_density_kg_m3=1020, n_terms=60, ka_eps=1e-8,
    )
    np.savez(path, **data)
    result = standalone["compare_openstb_dump"](path)
    assert all(value["complex_relative_l2"] < 1e-10 for value in result.values())
    data["ff_openstb_complex"] *= np.exp(.2j)
    np.savez(path, **data)
    changed = standalone["compare_openstb_dump"](path)
    assert changed["ff_openstb_complex"]["phase_rms_rad"] == pytest.approx(.2)
    assert changed["ff_hickling_complex"]["complex_relative_l2"] < 1e-10


def test_optional_missing_or_incompatible_dump(tmp_path, standalone):
    assert standalone["compare_openstb_dump"](tmp_path / "missing.npz") is None
    path = tmp_path / "bad.npz"
    np.savez(path, ka=[1])
    with pytest.warns(UserWarning, match="missing keys"):
        assert standalone["compare_openstb_dump"](path) is None


@pytest.mark.parametrize("source_case", ["hickling_fig16_max", "lfm_chirp"])
def test_echo_comparison_without_fitting_and_rejects_wrong_source(tmp_path, standalone, monkeypatch, source_case):
    namespace = standalone["build_source"].__globals__
    monkeypatch.setitem(namespace, "SOURCE_CASE", source_case)
    monkeypatch.setitem(namespace, "OPENSTB_ECHO_PING_INDEX", 0)
    monkeypatch.setitem(namespace, "OPENSTB_ECHO_RECEIVER_INDEX", 0)
    source = standalone["build_source"]()
    fs = source["sample_rate_hz"]
    carrier = source["baseband_frequency_hz"]
    ns = 4096 if source_case == "lfm_chirp" else 1024
    nfft = ns + int(np.ceil(source["duration_s"]*1.1*fs))
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs)) + carrier
    incident = np.fft.fftshift(np.fft.fft(source["signal"], nfft))
    dump_path = tmp_path / "dump.npz"
    result_path = tmp_path / "result.npz"
    delay = .001
    plugin = SolidSphereFormFunction(
        .25, 7700, 5960, 3240, 1000,
        validation_ka_bounds=(10, 40) if source_case.startswith("hickling") else None,
        debug_dump=True, debug_dump_path=dump_path,
    )
    filtered = plugin.apply(
        0, frequency, incident[None, :, None], carrier,
        SimpleNamespace(sound_speed=lambda time, pos: 1410),
        (500, 10000) if source_case == "lfm_chirp" else (source["f0_hz"],)*2,
        SimpleNamespace(tx_position=np.zeros(3), travel_time=np.array([[delay]])),
    )[0, :, 0]
    echo = np.fft.ifft(np.fft.ifftshift(filtered*np.exp(-2j*np.pi*frequency*delay)))[:ns]

    def save_echo(values):
        np.savez(result_path, sample_time=np.arange(ns)/fs,
                 pressure=values[None, None, :], ping_start_time=[0],
                 pressure_dimensions=["ping", "channel", "sample_time"], baseband_frequency=carrier,
                 echo_travel_time_s=[[delay]], echo_target_reflectivity=1,
                 echo_metadata_schema=1, echo_travel_time_dimensions=["ping", "channel"],
                 echo_distortion_classes=["SolidSphereFormFunction"])

    save_echo(echo)
    compare = standalone["compare_openstb_echo"]
    result = compare(source, dump_path, result_path)
    assert result["complex_relative_l2"] < 1e-10
    assert result["delay_s"] == delay
    # Select a different ping AND receiver with its own delay and reflectivity.
    delays = np.array([[.001, .0013], [.0017, .002]])
    echoes = np.array([
        [np.fft.ifft(np.fft.ifftshift(filtered*np.exp(-2j*np.pi*frequency*tau)))[:ns]*.6
         for tau in row] for row in delays
    ])
    np.savez(result_path, sample_time=np.arange(ns)/fs, pressure=echoes,
             ping_start_time=[0, .1], pressure_dimensions=["ping", "channel", "sample_time"],
             baseband_frequency=carrier, echo_travel_time_s=delays, echo_target_reflectivity=.6,
             echo_metadata_schema=1, echo_travel_time_dimensions=["ping", "channel"],
             echo_distortion_classes=["SolidSphereFormFunction"])
    monkeypatch.setitem(namespace, "OPENSTB_ECHO_PING_INDEX", 1)
    monkeypatch.setitem(namespace, "OPENSTB_ECHO_RECEIVER_INDEX", 1)
    selected = compare(source, dump_path, result_path)
    assert selected["delay_s"] == .002
    assert selected["complex_relative_l2"] < 1e-10
    monkeypatch.setitem(namespace, "OPENSTB_ECHO_PING_INDEX", 0)
    monkeypatch.setitem(namespace, "OPENSTB_ECHO_RECEIVER_INDEX", 0)
    save_echo(echo*1.1)
    mismatch = compare(source, dump_path, result_path)
    assert mismatch["complex_relative_l2"] < 1e-10  # Pure amplitude is intentionally ignored.
    different_scale = dict(source, signal=source["signal"]*2)
    scaled = compare(different_scale, dump_path, result_path)
    assert scaled["complex_relative_l2"] < 1e-10
    assert scaled["source_positive_scale"] == pytest.approx(.5)
    for changed in (-source["signal"], source["signal"]*np.exp(.2j),
                    np.roll(source["signal"], 1)):
        with pytest.warns(UserWarning, match="incident spectra differ"):
            assert compare(dict(source, signal=changed), dump_path, result_path) is None
    save_echo(echo*np.exp(.2j))
    assert compare(source, dump_path, result_path)["complex_relative_l2"] > .1
    with pytest.warns(UserWarning, match="skipped"):
        assert compare(source, dump_path, tmp_path / "missing.npz") is None
    np.savez(result_path, sample_time=np.arange(ns)/fs, pressure=echo[None,None,:],
             ping_start_time=[0], baseband_frequency=0,
             pressure_dimensions=["ping", "channel", "sample_time"])
    with pytest.warns(UserWarning, match="rerun simulate.py"):
        assert compare(source, dump_path, result_path) is None
