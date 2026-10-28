"""Independent rigid echo reconstruction with per-path delay and angle."""

from pathlib import Path
import runpy

import numpy as np
import pytest

from openstb.simulator.distortion.rigid_sphere import RigidSphereFormFunction


@pytest.mark.parametrize("source_case", ["rudgers_fig6", "lfm_chirp"])
def test_rigid_echo_uses_selected_path_angle_delay_and_reflectivity(tmp_path, source_case):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(__file__).resolve().parents[1] / (
        "Experiments/scattering_models/rigid_sphere/free_field/RigidSphereEchoAnalytic.py"
    )
    module = runpy.run_path(str(path))
    namespace = module["build_source"].__globals__
    namespace["SOURCE_CASE"] = source_case
    source = module["build_source"]()
    fs = source["sample_rate_hz"]
    carrier = source["baseband_frequency_hz"]
    ns = 4096 if source_case == "lfm_chirp" else 1024
    nfft = ns + int(np.ceil(source["duration_s"]*1.1*fs))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs)) + carrier
    ka = 2*np.pi*abs(f)*.25/1480
    incident = np.fft.fftshift(np.fft.fft(source["signal"], nfft))
    angles = np.array([[np.pi, 3.1], [3., 2.8]])
    delays = np.array([[.001, .0012], [.0014, .0016]])
    plugin = RigidSphereFormFunction(.25)
    pressure = np.empty((2,2,ns), dtype=complex)
    for p in range(2):
        for r in range(2):
            ff = plugin._form_function(ka, np.array([[np.cos(angles[p,r])]]))[0,:,0]
            pressure[p,r] = np.fft.ifft(np.fft.ifftshift(
                incident*ff*np.exp(-2j*np.pi*f*delays[p,r])*.7
            ))[:ns]
    dump_path = tmp_path / "dump.npz"
    np.savez(dump_path, ka=ka, f_hz=f, S_in_complex=incident,
             radius_m=.25, n_terms=80, scale=1)
    result_path = tmp_path / "result.npz"

    def save(include_angles=True):
        data = dict(sample_time=np.arange(ns)/fs, pressure=pressure,
                    ping_start_time=[0,.1], baseband_frequency=carrier,
                    pressure_dimensions=["ping", "channel", "sample_time"],
                    echo_travel_time_s=delays, echo_target_reflectivity=.7,
                    echo_metadata_schema=1, echo_distortion_classes=["RigidSphereFormFunction"],
                    echo_travel_time_dimensions=["ping", "channel"])
        if include_angles:
            data["echo_scattering_angle_rad"] = angles
        np.savez(result_path, **data)

    try:
        save()
        for index in (0,1):
            namespace["OPENSTB_ECHO_PING_INDEX"] = index
            namespace["OPENSTB_ECHO_RECEIVER_INDEX"] = index
            result = module["compare_openstb_echo"](source, dump_path, result_path)
            # Existing per-mode vs final-sum nonfinite handling differs at tiny
            # ka in the chirp grid. This test does not alter that modal physics.
            assert result["complex_relative_l2"] < 1e-8
            assert result["scattering_angle_rad"] == angles[index,index]
        scaled_source = dict(source, signal=source["signal"]*10000)
        scaled = module["compare_openstb_echo"](scaled_source, dump_path, result_path)
        assert scaled["complex_relative_l2"] < 1e-8
        assert scaled["source_positive_scale"] == pytest.approx(1e-4)
        for changed in (-source["signal"], source["signal"]*np.exp(.2j),
                        np.roll(source["signal"], 1)):
            with pytest.warns(UserWarning, match="incident spectra differ"):
                assert module["compare_openstb_echo"](
                    dict(source, signal=changed), dump_path, result_path) is None
        pressure *= 1000
        save()
        assert module["compare_openstb_echo"](source, dump_path, result_path)["complex_relative_l2"] < 1e-8
        pressure *= np.exp(.2j)
        save()
        assert module["compare_openstb_echo"](source, dump_path, result_path)["complex_relative_l2"] > .1
        pressure /= 1000*np.exp(.2j)
        angles[1,1] = np.pi  # Wrong stored angle must produce a nonzero error.
        save()
        mismatch = module["compare_openstb_echo"](source, dump_path, result_path)
        assert mismatch["complex_relative_l2"] > .01
        save(False)
        with pytest.warns(UserWarning, match="lacks scattering angles"):
            assert module["compare_openstb_echo"](source, dump_path, result_path) is None
    finally:
        plt.close("all")
