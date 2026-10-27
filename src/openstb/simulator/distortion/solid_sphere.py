# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Far-field solid-sphere backscattering with explicit convention conversion.

The modal function retains the validated Hickling convention. This initial
implementation applies the backscattering response to each receiver/target path.
It is intended for monostatic comparisons, not general bistatic scattering.
"""

from pathlib import Path
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import spherical_jn, spherical_yn

from openstb.simulator.plugin.abc import Distortion, Environment, TravelTimeResult


class SolidSphereFormFunction(Distortion):
    """Homogeneous elastic sphere with explicit SI material properties.

    No material preset or sound speed is implicitly selected. Fluid sound speed
    comes from the simulation environment in ``apply``.
    The modal equations reproduce SolidSphereEchoAnalytic.py without importing
    experiment code or its plotting dependencies.
    """

    def __init__(
        self,
        radius_m: float,
        solid_density_kg_m3: float,
        compressional_speed_m_s: float,
        shear_speed_m_s: float,
        fluid_density_kg_m3: float,
        n_terms: int = 80,
        scale: float = 1.0,
        ka_eps: float = 1e-8,
        debug_dump: bool = False,
        debug_dump_path: str = "solid_sphere_ff_debug.npz",
        validation_ka_bounds: tuple[float, float] | None = None,
    ):
        for name, value in (
            ("radius_m", radius_m),
            ("solid_density_kg_m3", solid_density_kg_m3),
            ("compressional_speed_m_s", compressional_speed_m_s),
            ("shear_speed_m_s", shear_speed_m_s),
            ("fluid_density_kg_m3", fluid_density_kg_m3),
            ("ka_eps", ka_eps),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(n_terms, bool) or not isinstance(n_terms, (int, np.integer)):
            raise ValueError("n_terms must be a positive integer")
        if n_terms < 1:
            raise ValueError("n_terms must be a positive integer")
        if not np.isfinite(scale):
            raise ValueError("scale must be finite")

        self.radius_m = float(radius_m)
        self.solid_density_kg_m3 = float(solid_density_kg_m3)
        self.compressional_speed_m_s = float(compressional_speed_m_s)
        self.shear_speed_m_s = float(shear_speed_m_s)
        self.fluid_density_kg_m3 = float(fluid_density_kg_m3)
        self.n_terms = int(n_terms)
        self.scale = float(scale)
        self.ka_eps = float(ka_eps)
        self.debug_dump = bool(debug_dump)
        self.debug_dump_path = Path(debug_dump_path)
        self._debug_dumped = False
        if validation_ka_bounds is not None:
            bounds = np.asarray(validation_ka_bounds, dtype=float)
            if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
                    or bounds[0] < 0 or bounds[1] <= bounds[0]):
                raise ValueError("validation_ka_bounds must be finite ordered nonnegative bounds")
            validation_ka_bounds = tuple(float(value) for value in bounds)
        self.validation_ka_bounds = validation_ka_bounds

    def form_function_hickling(
        self, ka: ArrayLike, sound_speed_m_s: float
    ) -> np.ndarray:
        """Return unscaled far-field backscatter in Hickling's convention.

        ``ka`` must be a one-dimensional nonnegative grid. This function does not
        conjugate, mask a source band, shift time, or apply propagation effects.
        As in the standalone, ka <= ka_eps is assigned zero, and nonfinite modal
        determinant values are replaced by zero. Finiteness is not a convergence
        guarantee, particularly for extreme ka or high modal orders.
        """
        ka = np.asarray(ka, dtype=float)
        if ka.ndim != 1 or not np.all(np.isfinite(ka)) or np.any(ka < 0):
            raise ValueError("ka must be a finite, nonnegative 1D array")
        if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0:
            raise ValueError("sound_speed_m_s must be finite and positive")
        valid = ka > self.ka_eps
        ff = np.zeros(ka.shape, dtype=np.complex128)
        if not np.any(valid):
            return ff

        x = ka[valid]
        x1 = (sound_speed_m_s / self.compressional_speed_m_s) * x
        x2 = (sound_speed_m_s / self.shear_speed_m_s) * x
        density_ratio = self.fluid_density_kg_m3 / self.solid_density_kg_m3
        modal_sum = np.zeros(x.shape, dtype=np.complex128)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for n in range(self.n_terms):
                j = spherical_jn(n, x)
                jp = spherical_jn(n, x, derivative=True)
                h = j + 1j * spherical_yn(n, x)
                hp = jp + 1j * spherical_yn(n, x, derivative=True)
                j1 = spherical_jn(n, x1)
                j1p = spherical_jn(n, x1, derivative=True)
                j2 = spherical_jn(n, x2)
                j2p = spherical_jn(n, x2, derivative=True)

                d11 = density_ratio * x2**2 * h
                d12 = (2 * n * (n + 1) - x2**2) * j1 - 4 * x1 * j1p
                d13 = 2 * n * (n + 1) * (x2 * j2p - j2)
                d21 = -x * hp
                d22 = x1 * j1p
                d23 = n * (n + 1) * j2
                d32 = 2 * (j1 - x1 * j1p)
                d33 = 2 * x2 * j2p + (x2**2 - 2 * n * (n + 1) + 2) * j2
                d10 = -density_ratio * x2**2 * j
                d20 = x * jp

                b = np.zeros((x.size, 3, 3), dtype=np.complex128)
                b[:, 0, 0], b[:, 0, 1], b[:, 0, 2] = d10, d12, d13
                b[:, 1, 0], b[:, 1, 1], b[:, 1, 2] = d20, d22, d23
                b[:, 2, 1], b[:, 2, 2] = d32, d33
                d = b.copy()
                d[:, 0, 0], d[:, 1, 0] = d11, d21

                ratio = -np.linalg.det(b) / np.linalg.det(d)
                ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                contribution = (-1) ** n * ratio * (2 * n + 1)
                modal_sum += np.nan_to_num(
                    contribution, nan=0.0, posinf=0.0, neginf=0.0
                )
            ff[valid] = 2 * modal_sum / (1j * x)
        return np.nan_to_num(ff, nan=0.0, posinf=0.0, neginf=0.0)

    def apply(
        self,
        ping_time: float,
        f: ArrayLike,
        S: ArrayLike,
        baseband_frequency: float,
        environment: Environment,
        signal_frequency_bounds: tuple[float, float],
        tt_result: TravelTimeResult,
    ) -> np.ndarray:
        """Multiply by the conjugated backscatter response on physical f.

        S has dimensions (receiver, frequency, target); the sphere properties
        are common to all paths. This uses a monostatic backscatter approximation
        regardless of the path vectors, not an angle-dependent bistatic model.
        Nonpositive physical frequencies are zeroed, matching the analytic
        standalone. No Hickling reference-band restriction is intrinsic to the
        sphere. The controller handles travel time and FFT/IFFT separately.
        """
        frequency = np.asarray(f, dtype=float)
        spectrum = np.asarray(S, dtype=np.complex128)
        if frequency.ndim != 1 or not np.all(np.isfinite(frequency)):
            raise ValueError("f must be a finite 1D physical-frequency array")
        if spectrum.ndim != 3 or spectrum.shape[1] != frequency.size:
            raise ValueError("S must have shape (receiver, len(f), target)")

        speeds = np.asarray(
            environment.sound_speed(ping_time, tt_result.tx_position), dtype=float
        ).reshape(-1)
        if speeds.size == 0 or not np.all(np.isfinite(speeds)) or np.any(speeds <= 0):
            raise ValueError("Environment sound speed must be finite and positive")
        # This free-field model assumes a homogeneous fluid. Do not silently
        # select the first value if the environment supplies different speeds.
        if not np.allclose(speeds, speeds[0], rtol=1e-12, atol=0.0):
            raise ValueError("Solid-sphere model requires a uniform sound speed")
        sound_speed = float(speeds[0])

        ka = 2 * np.pi * np.abs(frequency) * self.radius_m / sound_speed
        ff_hickling = self.form_function_hickling(ka, sound_speed)
        # Hickling synthesizes with exp(-i*omega*t); NumPy/OpenSTB IFFT uses
        # exp(+i*omega*t). Convert only here, outside the modal equations.
        ff_openstb = np.conjugate(ff_hickling)
        active = frequency > 0
        # Optional paper-comparison mask; never changes the modal form function.
        if self.validation_ka_bounds is not None:
            lower, upper = self.validation_ka_bounds
            active &= (ka >= lower) & (ka <= upper)
        response = np.where(active, self.scale * ff_openstb, 0.0)
        if (
            self.debug_dump and not self._debug_dumped and frequency.size
            and spectrum.shape[0] and spectrum.shape[2]
        ):
            # One representative input trace, once per plugin instance/worker.
            # In distributed runs this is not a global aggregate of all chunks.
            try:
                self.debug_dump_path.parent.mkdir(parents=True, exist_ok=True)
                with self.debug_dump_path.open("wb") as output:
                    np.savez(
                        output,
                        schema_version=1,
                        target_model="solid_sphere",
                        response_geometry="far_field_backscatter_approximation",
                        ping_time_s=ping_time,
                        f_hz=frequency,
                        ka=ka,
                        ff_hickling_complex=ff_hickling,
                        ff_openstb_complex=ff_openstb,
                        # Actual multiplier includes scale and positive-f mask.
                        response_complex=response,
                        response_mask=active,
                        validation_ka_bounds=(
                            self.validation_ka_bounds
                            if self.validation_ka_bounds is not None else np.array([])
                        ),
                        S_in_complex=spectrum[0, :, 0],
                        S_out_complex=spectrum[0, :, 0] * response,
                        receiver_index_in_chunk=0,
                        target_index_in_chunk=0,
                        spectrum_shape=np.array(spectrum.shape),
                        baseband_frequency_hz=baseband_frequency,
                        signal_frequency_bounds_hz=signal_frequency_bounds,
                        sound_speed_m_s=sound_speed,
                        radius_m=self.radius_m,
                        solid_density_kg_m3=self.solid_density_kg_m3,
                        compressional_speed_m_s=self.compressional_speed_m_s,
                        shear_speed_m_s=self.shear_speed_m_s,
                        fluid_density_kg_m3=self.fluid_density_kg_m3,
                        n_terms=self.n_terms,
                        ka_eps=self.ka_eps,
                        scale=self.scale,
                    )
                self._debug_dumped = True
                print(f"SolidSphere debug dump saved: {self.debug_dump_path}")
            except OSError as error:
                warnings.warn(f"SolidSphere debug dump failed: {error}", stacklevel=2)
        return spectrum * response[np.newaxis, :, np.newaxis]
