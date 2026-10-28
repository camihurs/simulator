"""Single-target travel-time metadata and NumPy export, without Dask."""

from types import SimpleNamespace

import numpy as np
import pytest

from openstb.simulator.controller.simple_points import _store_echo_metadata
from openstb.simulator.result_converter.numpy import NumpyConverter
from openstb.simulator.plugin.abc import ResultFormat


class MemoryGroup(dict):
    def __init__(self):
        super().__init__()
        self.attrs = {"baseband_frequency": 0}

    def empty(self, name, shape, chunks, dtype):
        self[name] = np.empty(shape, dtype=dtype)
        return self[name]


class Target:
    def __len__(self):
        return 1

    def get_chunk(self, start, count):
        return np.array([[0., 40., 0.]]), np.array([.7])


def test_metadata_all_pings_receivers_and_conversion(tmp_path):
    calls = []

    def calculate(trajectory, time, environment, tx, txori, rx, rxori, position):
        calls.append((time, rx.copy(), position.copy()))
        return SimpleNamespace(
            travel_time=np.array([[.05 + time*.001 + rx[0,0]*.0001]]),
            incident_vector=np.array([[0., 1., 0.]]),
            scattering_vector=np.array([[[0., -1., 0.]]]),
        )

    def transducer(x):
        return SimpleNamespace(position=np.array([x,0.,0.]),
                               orientation=SimpleNamespace(ndarray=np.array([1.,0,0,0])),
                               distortion=[])

    config = dict(targets=[Target()], trajectory=object(), environment=object(),
                  travel_time=SimpleNamespace(calculate=calculate),
                  system=SimpleNamespace(transmitter=transducer(0),
                                         receivers=[transducer(0), transducer(1)]))
    storage = MemoryGroup()
    _store_echo_metadata(storage, config, np.array([0., .1, .2]))
    assert len(calls) == 6
    expected = .05 + np.array([0,.1,.2])[:,None]*.001 + np.array([0,1])[None,:]*.0001
    np.testing.assert_allclose(storage["echo_travel_time_s"], expected)
    assert storage.attrs["echo_target_reflectivity"] == .7
    storage.update(ping_start_time=np.array([0,.1,.2]), sample_time=np.arange(5.),
                   pressure=np.zeros((3,2,5), dtype=complex))
    path = tmp_path / "result.npz"
    assert NumpyConverter(path).convert(ResultFormat.ZARR_BASEBAND_PRESSURE, storage, config)
    with np.load(path, allow_pickle=False) as result:
        np.testing.assert_allclose(result["echo_travel_time_s"], expected)
        np.testing.assert_allclose(result["echo_scattering_angle_rad"], np.pi)
        assert result["echo_target_reflectivity"] == .7
        assert list(result["echo_travel_time_dimensions"]) == ["ping", "channel"]
    with pytest.raises(ValueError, match="exactly one target"):
        _store_echo_metadata(storage, config | {"targets": [Target(), Target()]}, [0])
