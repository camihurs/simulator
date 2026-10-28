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


## Controlled sphere-scattering workflow

This experiment now supports rigid and solid elastic spheres in free field.
The validated baseline keeps all unrelated effects disabled. Source, scattering
model and diagnostic selection are distinct settings; changing one does not
automatically change the others.

### Files and responsibilities

| File | Responsibility |
| --- | --- |
| `shared_params.py` | Environment, sphere properties, source mode, debug paths/styles and selected diagnostic ping/receiver. |
| `signal_factory.py` | Builds OpenSTB sources; it does not calculate scattering. |
| `simulate.py` | Selects active plugins, geometry, trajectory, travel-time model, sampling and output paths. |
| `debug_plots.py` | Visualizes OpenSTB results and modal dumps; does not import standalones. |
| [Rigid analytic standalone](../scattering_models/rigid_sphere/free_field/README.md) | Independent target/source synthesis and optional OpenSTB comparisons. |
| [Solid analytic standalone](../scattering_models/solid_sphere/free_field/README.md) | Independent elastic modal solution, Hickling diagnostics and optional OpenSTB comparisons. |
| [Geometry viewer](VISUALIZATION.md) | Separate interactive geometry/beampattern inspection workflow. |

Historical standalone scripts are preserved. Select `SOURCE_CASE` separately in
each analytic standalone; this does not read the source selection from
`shared_params.py`.

### Select a baseline

| Baseline | Source mode in shared parameters | Active distortion | Sound speed | Standalone source |
| --- | --- | --- | --- | --- |
| Rudgers rigid reference | `sine`, 2 cycles, unit amplitude | `RigidSphereFormFunction` only | 1480 m/s | `rudgers_fig6` |
| Hickling solid reference | `hickling`, k0a=24.5, 5 cycles | `SolidSphereFormFunction` only | 1410 m/s | `hickling_fig16_max` |
| Solid LFM baseline | `lfm` | `SolidSphereFormFunction` only | 1410 m/s for the current validated case | `lfm_chirp` |
| Rigid LFM baseline | `lfm` | `RigidSphereFormFunction` only | Match the rigid standalone | `lfm_chirp` |

Use radius 0.25 m and the matching modal settings. Solid baseline material is
Armco iron (density 7700 kg/m³, compressional speed 5960 m/s, shear speed 3240 m/s),
with fluid density 1000 kg/m³. Named presets are documented in the solid guide.

In `simulate.py`, enable **only one sphere distortion**. Stacking the two
multiplies their responses; it does not simulate separate rigid/solid targets.
For the controlled baseline use a single target, reflectivity 1, stop-and-hop,
identity transducer orientation, and disabled transmit/receive beampatterns.
Disable spreading, absorption and Doppler.

Hickling source parameters live under `SIM_PARAMS["signal"]["hickling"]`.
Its initial phase is derived to match the standalone's pulse-centered convention.
The solid plugin receives the optional validation band `10 <= ka <= 40` only
when mode is `hickling`; LFM has no such reference-band restriction. This band
is not an intrinsic cutoff of the sphere.

Current LFM settings in `signal_factory.py`: 0.5–10 kHz, 20 ms, Hann window,
reference/carrier 5.25 kHz, 190 dB re 1 µPa and RMS-after-window scaling.
The controller is sampled at 30 kHz for LFM and `10*f0` for the tone cases.
Its record length follows the ping interval plus a guard region, not the
standalone's fixed FFT length.

### Output files and execution

Run from the repository root with the project environment:

```powershell
.venv\Scripts\python.exe Experiments\simple_points_study\simulate.py
.venv\Scripts\python.exe Experiments\simple_points_study\debug_plots.py
```

Both `simple_points.npz` and the intermediate `simple_points.zarr` are relative
to the working directory of the simulation. Existing output paths are rejected:
move, rename or remove only the intended old outputs before rerunning.
Successful NumPy conversion automatically removes the intermediate Zarr directory;
a failed run may leave it behind. No simulation is triggered by diagnostics.

Modal dumps have explicit independent paths under this experiment:
`rigid_sphere_ff_debug.npz` and `solid_sphere_ff_debug.npz`.
There is no need to delete the other model's dump, but each comparison must read
the correct file from the intended run. Dumps contain a representative chunk,
not all pings. Several Dask workers may write the same dump, so it is not
guaranteed to describe the first ping.

### Per-ping metadata

This experiment enables `config["save_echo_metadata"] = True`. For exactly one
target, the controller uses its configured travel-time plugin to record:

- `echo_travel_time_s`: shape (ping, channel), in seconds;
- `echo_scattering_angle_rad`: the corresponding path angles;
- `echo_target_reflectivity`: the single target's reflectivity;
- schema, dimension labels and common/transmit distortion class names.

These are exported alongside pressure in `simple_points.npz`. This opt-in
diagnostic currently rejects multi-target scenes rather than ambiguously
attaching one travel time to a sum of targets.

Each standalone selects `OPENSTB_ECHO_PING_INDEX` and
`OPENSTB_ECHO_RECEIVER_INDEX` independently of the diagnostic indices in
`shared_params.py`. It uses the selected stored travel time; the rigid
comparison also uses the stored scattering angle. No manual geometry fallback
remains. Old files missing required metadata cause only the optional echo
comparison to be skipped with an explanation.

Backscatter is theta=pi under the vector convention used here. The rigid
response is angular; the present solid far-field implementation instead fixes
the backscatter factor to `(-1)**n`. General solid-sphere bistatic scattering
is future work, not validated by changing the geometric target position.

### Diagnostic model and presentation

Under `SIM_PARAMS["debug"]`, select:

```python
"target_model": "solid",  # "solid" or "rigid"; must match the run
"plot_style": "screen",  # "screen" or "paper"
"ping_index": 31,
"receiver_index": 0,
```

Styles change titles, fonts and dimensions only; they do not require rerunning
the simulation. `screen_style` and `paper_style` customize them; paper hides
titles by default. The rigid focused and full-range modal views are retained.
Solid modal figures distinguish raw Hickling phase, conjugated OpenSTB phase,
the reference-band shading if present, and the unwrapped Hickling phase parameter.

Incident diagnostics reconstruct the full controller record including guard
samples before cropping for time plots. This matters because the current
OpenSTB LFM implementation calculates RMS over the full sampled array including
zeros: resampling on a shorter plotting interval changes the source amplitude.

Physical frequencies and their FFT coefficients are sorted together before ka
plotting. Weak spectral phase is omitted below a configurable relative threshold
(`incident_phase_min_relative`, default 1e-3, or -60 dB); wrap jumps are not joined.
Dense in-band chirp phase is still expected. The Hz spectral figures are
recomputed using the OpenSTB complex FFT procedure, not read directly from the dump.
The ka view preserves the historical doubled real-tone FFT for the rigid sine case;
chirps use complex baseband FFT and physical frequencies for either target.

`incident_fft_points` controls the ka display FFT, never truncating the source
record. The incident Hz view uses the actual record+guard length.
`solid_form_function_ka_max` sets the solid display limit; the rigid views retain
their focused ka<=14 and full-range axes.

### Independent standalone comparisons

Enable `COMPARE_OPENSTB_DUMP` to compare modal values and
`COMPARE_OPENSTB_ECHO` to compare the final echo. The standalone guides describe
the reference-paper options and expected figure counts.

For echo comparison, the independently constructed source must match the dump
in complex shape and phase. A real positive amplitude ratio is allowed, because
the standalones have unit-envelope chirps whereas OpenSTB uses calibrated RMS
scaling. Each complex echo is normalized by its own maximum complex magnitude.
Known travel delay is retained in synthesis and subtracted from the plotting axis;
no peak, phase or complex gain is fitted. The result validates normalized shape
and phase, **not absolute pressure or attenuation**.

The controlled solid Hickling and LFM baseline comparisons have passed.
The latest solid LFM result gave normalized complex relative error about
1.33e-15; this is implementation agreement, not independent published chirp
validation. Latest regression suite: 114 tests passed. Test scope and numerical
limitations are described in the model guides.

### Next: additional OpenSTB effects

Keep a controlled baseline, then enable one effect at a time and record its
configuration, raw amplitudes, spectra and timing. Only then combine effects.
Do not alter modal equations to force an effect-enabled echo to match the
target-only standalone.

The current optional echo comparisons expect only the selected sphere distortion
in the common/transmit chain and skip results listing extra effects. They are not
independent references for those effects. Receive-only effects are not fully
identified by that class-name check: manually verify the receive beampattern too.
A future effects-comparison workflow may need separate baseline/effect result
files and additional metadata; it has not been implemented yet.

Normalization hides pure gain/loss. Use original pressures to study spreading,
absorption or source levels. Preserve the same source record/sampling when
comparing amplitudes, since LFM RMS scaling currently depends on record length.
For beampatterns verify geometry/orientation and for Doppler inspect frequency
and timing changes, not only peak-normalized waveform resemblance.

### Troubleshooting

- Missing final comparison: inspect its warning for absent metadata/dump, wrong
  model, mismatched source/sampling, or extra distortions.
- LFM gives fewer standalone figures than Hickling/Rudgers: reference-only
  diagnostics are intentionally skipped; True does not make them applicable.
- Echo absent: check that its travel time lies within the recording window.
- Rerun cannot start: check both final NPZ and intermediate Zarr output paths.
- Result and dump can become stale independently; filenames alone do not prove
  that they came from the same run.
