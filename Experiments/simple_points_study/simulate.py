# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import sys
from typing import Literal

import numpy as np
import quaternionic

from openstb.simulator.controller import simple_points
from openstb.simulator.plugin import loader
from pathlib import Path

# The local Dask cluster uses the multiprocessing module. This will import this
# script at the start of each worker process. If the code to configure and start the
# simulation is run during the import, this will lead to each worker trying to start
# another cluster and simulation, and so on. Instead, we put our simulation setup and
# execution inside a function, and use an if __name__ == "__main__" guard at the end of
# the script to only call it from the top-level execution; the workers will have a
# different __name__ value. This is the standard way of using multiprocessing and
# therefefore Dask local clusters.


# ============================================================================
# Experiment Parameters (single source of truth for this script)
# ============================================================================
SIM_PARAMS = {
    "environment": {
        "salinity": 14.5,
        "sound_speed_ms": 1480.0,
        "temperature_c": 11.2,
    },
    "rigid_sphere": {
        "radius_m": 0.25,
        "k0a": 15.0,
    },
    "signal": {
        "mode": "sine",   # "sine" or "lfm"
        "n_cycles": 2,
        "amplitude": 1.0,
        "initial_phase": 0.0,
    },
    "debug": {
        "plot_incident": True,
        "plot_incident_spectrum": True,
        "incident_fft_points": 16384,
        "plot_form_function": True,
        "form_function_ka_max": 14,
        "form_function_points": 2000,
        "dump_form_function_from_plugin": True,
        "plugin_dump_path": str(Path(__file__).resolve().parent / "rigid_sphere_ff_debug.npz"),
        "plot_form_function_from_plugin_dump": True,
    },
}

def simulate(cluster: Literal["local"] | Literal["mpi"]):
    # Begin our configuration dictionary.
    config: simple_points.SimplePointConfig = {}

    # Each plugin is defined through a plugin specification dictionary. This takes the
    # name the plugin is registered under (see pyproject.toml for a list of the included
    # plugins) and a dictionary of any parameters it requires.
    #
    # If you want to use a custom plugin, you can use either of the following forms for
    # the name:
    #
    #     ClassName:package.module
    #
    #     ClassName:/path/to/file.py

    if cluster == "local":
        # Create a cluster on the local machine with 8 workers able to use up to ~40% of
        # the total memory (note that the memory is enforced on a best-effort basis).
        config["dask_cluster"] = loader.dask_cluster(
            {
                "name": "local",
                "parameters": {
                    "workers": 8,
                    "total_memory": 0.4,
                    "dashboard_address": ":8787",
                },
            }
        )

    elif cluster == "mpi":
        # Cluster is managed by MPI.
        config["dask_cluster"] = loader.dask_cluster(
            {
                "name": "mpi",
                "parameters": {
                    "dashboard_address": ":8787",
                },
            }
        )

    else:
        raise ValueError(f"Unknown cluster type '{cluster}'")

    # Initialise the cluster. The simulation method should also do this, but we don't
    # want to wait. In an MPI situation, each worker gets called with the same command
    # and so will reach this function. The initialise() method is what lets Dask take
    # control of the workers, so if we wait until the simulation starts each worker will
    # parse the configuration (including reading data off the disk, generating the
    # targets etc). For a local cluster, this function will not be reached by the
    # workers, but it does no harm to initialise the cluster here.
    config["dask_cluster"].initialise()

    # Use a 10m linear trajectory along the x axis at 1.5m/s.
    config["trajectory"] = loader.trajectory(
        {
            "name": "linear",
            "parameters": {
                "start_position": [0, 0, 0],
                "end_position": [10, 0, 0],
                "speed": 1.5,
            },
        }
    )

    # Decide when the sonar will transmit pings. Here, we ping at a constant interval of
    # 0.2s, starting at t=0 (the start of the trajectory) and with no ping closer than
    # 0.5s to the end of the trajectory.
    config["ping_times"] = loader.ping_times(
        {
            "name": "constant_interval",
            "parameters": {
                "interval": 0.2,
                "start_delay": 0,
                "end_delay": 0.5,
            },
        }
    )

    # The environment is spatially and temporally invariant.
    config["environment"] = loader.environment(
        {
            "name": "invariant",
            "parameters": {
                "salinity": SIM_PARAMS["environment"]["salinity"],
                "sound_speed": SIM_PARAMS["environment"]["sound_speed_ms"],
                "temperature": SIM_PARAMS["environment"]["temperature_c"],
            },
        }
    )

    # Include two collections of point targets. The first is a rectangle with the given
    # size, position and normal ([0, 0, -1] points up -- remember the z axis is down)
    # filled with randomly placed points at a density of 10 per m^2. The reflectivity is
    # the fraction of incident amplitude that is scattered back to the sonar. The second
    # target is a single point at a given position.
    config["targets"] = [
        # loader.point_targets(
        #     {
        #         "name": "random_point_rectangle",
        #         "parameters": {
        #             "seed": 10671,
        #             "Dx": 5,
        #             "Dy": 120,
        #             "centre": (5, 75, 10),
        #             "normal": (0, 0, -1),
        #             "point_density": 10,
        #             "reflectivity": 0.06,
        #         },
        #     }
        # ),
        loader.point_targets(
            {
                "name": "single_point",
                "parameters": {
                    "position": (5, 40, 10),
                    "reflectivity": 10,
                },
            }
        ),
    ]

    # Use the stop-and-hop approximation when calculating the travel time of the pulse.
    config["travel_time"] = loader.travel_time(
        {
            "name": "stop_and_hop",
            "parameters": {},
        }
    )

    # Apply two distortions: spherical spreading (1/r scaling to the amplitude on
    # each direction) and acoustic attenuation.
    config["distortion"] = [
        loader.distortion(
            {
                "name": "geometric_spreading",
                "parameters": {
                    "power": 1.0,
                },
            }
        ),
        loader.distortion(
            {
                "name": "anslie_mccolm_attenuation",
                "parameters": {
                    "frequency": "centre",
                },
            }
        ),
        loader.distortion(
        {
            "name": "RigidSphereFormFunction:openstb.simulator.distortion.rigid_sphere",
            "parameters": {
                "radius_m": SIM_PARAMS["rigid_sphere"]["radius_m"],
                "n_terms": 80,
                "scale": 1.0,
                "debug_dump": SIM_PARAMS["debug"]["dump_form_function_from_plugin"],
                "debug_dump_path": SIM_PARAMS["debug"]["plugin_dump_path"],
            },
        }
    )
    ]

    # Quick switch between original chirp and new sinusoid burst.
    signal_mode = SIM_PARAMS["signal"]["mode"]  # "sine" or "lfm"

    # Define the signal the sonar will transmit; a Tukey-windowed LFM upchirp here.
    if signal_mode == "lfm":
        # Original configuration: Tukey-windowed LFM chirp.
        sim_baseband_frequency = 110e3
        signal = loader.signal(
            {
                "name": "lfm_chirp",
                "parameters": {
                    "f_start": 100e3,
                    "f_stop": 120e3,
                    "duration": 0.015,
                    "rms_spl": 190,
                    "rms_after_window": True,
                    "window": {
                        "name": "tukey",
                        "parameters": {"alpha": 0.2},
                    },
                },
            }
        )

    elif signal_mode == "sine":
        # Parameters aligned with your RigidSphereEcho setup.
        c = SIM_PARAMS["environment"]["sound_speed_ms"]
        a = SIM_PARAMS["rigid_sphere"]["radius_m"]
        k0a = SIM_PARAMS["rigid_sphere"]["k0a"]
        f0 = k0a * c / (2 * np.pi * a)

        sim_baseband_frequency = 0.0
        signal = loader.signal(
            {
                "name": "SinusoidBurst:openstb.simulator.system.signal",
                "parameters": {
                    "f0": f0,
                    "n_cycles": SIM_PARAMS["signal"]["n_cycles"],
                    "amplitude": SIM_PARAMS["signal"]["amplitude"],
                    "initial_phase": SIM_PARAMS["signal"]["initial_phase"],
                },
            }
        )

    else:
        raise ValueError(f"Unknown signal_mode '{signal_mode}'")


    if SIM_PARAMS["debug"]["plot_incident"]:
        import matplotlib.pyplot as plt

        # Plot solo de diagnóstico (no cambia la simulación real)
        if signal_mode == "sine":
            # Para que se vea igual que en RigidSphereEcho: alta resolución + pasabanda
            sample_rate_plot = 100.0 * f0
            baseband_frequency_plot = 0.0
            title = f"Incident Signal: {SIM_PARAMS['signal']['n_cycles']}-cycle sinusoid at {f0:.1f} Hz"
        else:
            # Para otras señales, usar una vista razonable por defecto
            sample_rate_plot = 10.0 * 30e3
            baseband_frequency_plot = 110e3
            title = f"Incident signal ({signal_mode})"

        t_end = 3.0 * signal.duration
        t = np.arange(0.0, t_end, 1.0 / sample_rate_plot)
        s = signal.sample(t, baseband_frequency_plot)

        # plt.figure(figsize=(10, 4))
        # plt.plot(t * 1e3, np.real(s), "b-", linewidth=1.5)
        # plt.xlabel("Time [ms]")
        # plt.ylabel("Amplitude")
        # plt.title(title)
        # plt.grid(True, alpha=0.3)
        # plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        # plt.xlim(-0.05, t_end * 1e3)
        # plt.tight_layout()
        # plt.show()

        # if SIM_PARAMS["debug"].get("plot_incident_spectrum", False):
        #     # Espectro de la señal incidente (igual enfoque que RigidSphereEcho)
        #     n_fft = int(SIM_PARAMS["debug"].get("incident_fft_points", 16384))
        #     dt_plot = 1.0 / sample_rate_plot

        #     incident_fft = np.fft.fft(np.real(s), n_fft) * dt_plot
        #     incident_fft *= 2.0

        #     freq = np.fft.fftfreq(n_fft, dt_plot)
        #     positive = freq >= 0
        #     freq_positive = freq[positive]
        #     incident_fft_positive = incident_fft[positive]

        #     # ka usando el mismo c del experimento
        #     c_plot = SIM_PARAMS["environment"]["sound_speed_ms"]
        #     k_positive = 2.0 * np.pi * freq_positive / c_plot
        #     ka_positive = k_positive * SIM_PARAMS["rigid_sphere"]["radius_m"]

        #     magnitude = np.abs(incident_fft_positive)
        #     phase = np.angle(incident_fft_positive)
        #     phase = np.mod(phase + 2.0 * np.pi, 2.0 * np.pi)

        #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        #     ax1.plot(ka_positive, magnitude, "b-", linewidth=1.5)
        #     ax1.set_xlabel("ka")
        #     ax1.set_ylabel("|g(ka)|")
        #     ax1.set_title("Incident spectrum g(ka) - magnitude")
        #     ax1.grid(True, alpha=0.3)
        #     ax1.set_xlim(0, 30)

        #     ax2.plot(ka_positive, phase, "r-", linewidth=1.5)
        #     ax2.set_xlabel("ka")
        #     ax2.set_ylabel("Phase of g(ka) (radians)")
        #     ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        #     ax2.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
        #     ax2.grid(True, alpha=0.3)
        #     ax2.set_xlim(0, 30)
        #     ax2.set_ylim(0, 2*np.pi)

        #     plt.tight_layout()
        #     plt.show()

    # signal = loader.signal(
    #     {
    #         "name": "lfm_chirp",
    #         "parameters": {
    #             "f_start": 100e3,
    #             "f_stop": 120e3,
    #             "duration": 0.015,
    #             "rms_spl": 190,
    #             "rms_after_window": True,
    #             "window": {
    #                 "name": "tukey",
    #                 "parameters": {"alpha": 0.2},
    #             },
    #         },
    #     }
    # )

    # Set the desired orientation of the transducers. Without rotation, the normal of
    # the transducer, i.e., the direction it is pointing, is [1, 0, 0] (x is forward, y
    # is starboard and z is down, so straight ahead). The function used here uses a
    # rotation vector to create a quaternion. The magnitude gives the angle of rotation
    # and the normalised version the axis. Here, we rotate 90 degrees about z (point to
    # starboard) and 15 degrees around x (15 degrees down).
    q_yaw = quaternionic.array.from_rotation_vector([0, 0, np.pi / 2])
    q_tilt = quaternionic.array.from_rotation_vector([np.radians(15), 0, 0])
    q_transducer = q_tilt * q_yaw

    # Define a common far-field beampattern for the transducers. Note that this is just
    # a distortion attached to the transducers; we could add this to the list of
    # distortion plugins above and not pass it to the transducers to achieve the same
    # result.
    beampattern = {
        "name": "rectangular_beampattern",
        "parameters": {
            "width": 0.015,
            "height": 0.03,
            "transmit": True,
            "receive": False,
            "frequency": "centre",
        },
    }

    # Define the transmitting transducer.
    transmitter = loader.transducer(
        {
            "name": "generic",
            "parameters": {
                "position": [0, 1.2, 0.3],
                "orientation": q_transducer,
                "beampattern": beampattern,
            },
        }
    )

    # And then the list of receiving transducers.
    beampattern["parameters"]["transmit"] = False
    beampattern["parameters"]["receive"] = True
    receivers = [
        loader.transducer(
            {
                "name": "generic",
                "parameters": {
                    "position": [x, 1.2, 0],
                    "orientation": q_transducer,
                    "beampattern": beampattern,
                },
            }
        )
        #for x in [-0.1, -0.05, 0, 0.05, 0.1]
        for x in [0]
    ]

    # Combine all this into a System plugin.
    config["system"] = loader.system(
        {
            "name": "generic",
            "parameters": {
                "transmitter": transmitter,
                "receivers": receivers,
                "signal": signal,
            },
        }
    )

    # Internally, the simulation result is stored in a Zarr group. You could choose to
    # directly load the results from this format, or you could configure a conversion
    # plugin to write them to a different format. The following converts the result to
    # an uncompressed NumPy .npz file; this can then be loaded with
    # np.load("example_sim.py") which returns a mapping interface.
    config["result_converter"] = loader.result_converter(
        {
            "name": "numpy",
            "parameters": {
                "filename": "simple_points.npz",
                "compress": False,
            },
        }
    )

    # If you prefer, you could convert this to a MATLAB file instead. This uses the
    # `scipy.io.savemat` function provided by SciPy; note that this only supports "5"
    # (MATLAB 5 and up) and "4" as the format arguments, and not the newer HDF-backed
    # formats.
    # config["result_converter"] = loader.result_converter(
    #     {
    #         "name": "matlab",
    #         "parameters": {
    #             "filename": "simple_points.mat",
    #             "format": "5",
    #             "long_field_names": False,
    #             "do_compression": False,
    #             "oned_as": "row",
    #         },
    #     }
    # )

    # Initialise the simulator class. In the future, this is intended be a plugin once a
    # suitable interface has been determined. Note that the simulator will refuse to
    # overwrite an existing output file, so you will need to either delete it or change
    # the output name in the simulation definition if you want to re-run the simulation.
    # We manually specify how many targets to include in each chunk of work, and the
    # details about the system sampling. The output will be in the complex baseband.
    sim = simple_points.SimplePointSimulation(
        result_filename="simple_points.zarr",
        points_per_chunk=1000,
        sample_rate=30e3,
        baseband_frequency=sim_baseband_frequency,
    )

    # And finally, run the simulation. While it is running, you can access the Dask
    # dashboard at 127.0.0.1:8787 to see various diagnostic plots about how the cluster
    # is being utilised.
    sim.run(config)

    # if SIM_PARAMS["debug"].get("plot_form_function_from_plugin_dump", False):
    #     import matplotlib.pyplot as plt
    #     from pathlib import Path

    #     dump_path = Path(SIM_PARAMS["debug"]["plugin_dump_path"])
    #     if dump_path.exists():
    #         data = np.load(dump_path)
    #         ka_dump = data["ka"]
    #         mag_dump = data["ff_magnitude"]
    #         phase_dump = data["ff_phase"]
    #         theta_sample = float(data["theta_sample_rad"])

    #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    #         ax1.plot(ka_dump, mag_dump, "b-", linewidth=1.5)
    #         ax1.set_xlabel("ka")
    #         ax1.set_ylabel("|f(ka)|")
    #         ax1.set_title("Form Function from plugin dump - Magnitude")
    #         ax1.grid(True, alpha=0.3)
    #         ax1.set_xlim(0, 14)
    #         ax1.set_ylim(0, 1.5)

    #         ax2.plot(ka_dump, phase_dump, "r-", linewidth=1.5)
    #         ax2.set_xlabel("ka")
    #         ax2.set_ylabel("arg[f(ka)] (radians)")
    #         ax2.set_title(f"Form Function from plugin dump - Phase (theta={theta_sample:.4f} rad)")
    #         ax2.grid(True, alpha=0.3)
    #         ax2.set_xlim(0, 14)
    #         ax2.set_ylim(0, 2 * np.pi)
    #         ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    #         ax2.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])

    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         print(f"Plugin dump not found: {dump_path}")


if __name__ == "__main__":
    # First argument will be name of the script.
    nargs = len(sys.argv) - 1

    # Default to directly on the local machine.
    if nargs == 0:
        cluster = "local"

    # User specified.
    elif nargs == 1:
        cluster = sys.argv[1]
        if cluster not in {"local", "mpi"}:
            raise SystemExit(f"Usage: {sys.argv[0]} [local|mpi]")

    # Too many arguments for us to handle.
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} [local|mpi]")

    simulate(cluster)
