<!--

SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent

-->

Simple point target simulation
==============================

The `simulate.py` script configures a simulation of some point targets. It does this by
creating a dictionary holding the various plugins needed for the simulation. This
configuration dictionary is then given to the `run()` method of the simulation
controller to perform the simulation.

If run directly, i.e., with `python simulate.py`, a local Dask cluster with 8 workers
will be used for the simulation. Alternatively, the `run_with_mpi.sh` shell script can
be run to execute the simulation within an MPI environment; in this case 6 workers will
be used in the cluster, with the other 2 used to run the simulation controller and
manage the Dask scheduler. In either case, the Dask diagnostic dashboard showing how the
cluster is being utilised will be available at http://127.0.0.1:8787/ for the duration
of the simulation.

The initial results of the simulation are stored in a [zarr](https://zarr.readthedocs.io/)
file. The simulation script configures a result converter plugin to convert this to a
NumPy file at `simple_points.npz`. It also includes a commented-out configuration to
convert the output to a MATLAB file if you prefer. See the `plot_results.py` script
for examples of how to load and plot the NumPy-formatted simulation results.


## Paper Validation Workflow

This section describes a reproducible workflow to validate the rigid-sphere reference case used in the paper baseline.

### Objective

The validation mode is:

1. Incident signal: sine burst.
2. Scattering model: rigid sphere form function.
3. Keep extra effects disabled unless explicitly testing them.

Relevant files:

1. shared_params.py
2. signal_factory.py
3. simulate.py
4. debug_plots.py
5. plot_results.py

### Paper Validation Mode

#### Step 1: Set the incident signal to sine

In shared_params.py, set signal mode to `sine`.

Parameters `n_cycles`, `amplitude`, and `initial_phase` are used only in sine mode through signal_factory.py.

#### Step 2: Keep only rigid-sphere distortion for the baseline

In simulate.py:

1. Keep `RigidSphereFormFunction` enabled.
2. Disable `geometric_spreading` and `anslie_mccolm_attenuation` for the strict paper baseline.
3. If you want pure baseline behavior, also disable transducer beampattern effects or use omnidirectional transducers.

#### Step 3: Keep simple geometry

Use the single point target configuration in simulate.py, and keep ping interval/sampling settings stable while validating.

### Run Order

1. Run simulation:
   `python simulate.py`
2. Quick result check:
   `python plot_results.py`
3. Detailed diagnostics:
   `python debug_plots.py`

### What to Check in Debug Plots

1. **Incident signal consistency**
   - Incident time-domain plots are reconstructed from the same signal definition used by the simulation.
   - Incident spectrum views are available in `ka` and in Hz.

2. **Form function consistency**
   - Form function plots are loaded from the plugin debug dump produced by `RigidSphereFormFunction`.
   - A zoomed `ka` view is useful for paper comparison.
   - A full-range `ka` view is useful to verify the complete computed band.

3. **Echo consistency**
   - Echo time-domain plot should be finite and non-zero.
   - Echo spectrum can be inspected in `ka` and in Hz.

### Notes on Debug Parameters

1. `incident_fft_points`
   - Used by incident spectrum plotting in debug_plots.py.
   - Not specific to sine; it works for sine and chirp.

2. `form_function_ka_max`
   - In the current script state, this value is not actively controlling axes unless explicitly wired into plotting limits.
   - Current `ka` limits are defined directly in plotting code.

### Troubleshooting

1. **Empty pressure array or blank plots**
   - Usually caused by echo arrival outside the simulated time window.
   - Check ping interval, target range, and effective trace duration.

2. **Very small change when enabling attenuation**
   - Expected at short ranges and low frequencies.
   - Increase range if you want attenuation to become clearly visible.

3. **Different behavior from paper baseline**
   - Verify extra distortions (spreading, attenuation, beampattern) are disabled.
   - Verify signal mode is `sine` in shared_params.py.