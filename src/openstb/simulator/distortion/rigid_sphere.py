# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from pathlib import Path
from numpy.typing import ArrayLike
from scipy.special import eval_legendre, spherical_jn, spherical_yn

from openstb.simulator.plugin.abc import Distortion, Environment, TravelTimeResult


class RigidSphereFormFunction(Distortion):
    """Apply rigid-sphere scattering form function in frequency domain."""

    def __init__(
        self,
        radius_m: float,
        n_terms: int = 80,
        scale: float = 1.0,
        ka_eps: float = 1e-8,
        debug_dump: bool = False,
        debug_dump_path: str = "rigid_sphere_ff_debug.npz",
):
        if radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if n_terms < 0:
            raise ValueError("n_terms must be >= 0")
        if ka_eps <= 0:
            raise ValueError("ka_eps must be positive")

        self.radius_m = float(radius_m)
        self.n_terms = int(n_terms)
        self.scale = float(scale)
        self.ka_eps = float(ka_eps)
        self._calls = 0
        self.debug_dump = bool(debug_dump)
        self.debug_dump_path = debug_dump_path
        self._debug_dumped = False

    def _form_function(self, ka: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        ka : (Nf,) array
        cos_theta : (Nr, Nt) array

        Returns
        -------
        ff : (Nr, Nf, Nt) complex array
        """
        Nr, Nt = cos_theta.shape
        Nf = ka.size

        ff = np.zeros((Nr, Nf, Nt), dtype=np.complex128)

        # Igual que en RigidSphereEcho: n = 0..N_terms-1
        for n in range(self.n_terms):
            jnp = spherical_jn(n, ka, derivative=True)
            ynp = spherical_yn(n, ka, derivative=True)

            # Igual a tu script: eta_n = arctan(jn_prime / -yn_prime)
            eta_n = np.arctan(jnp / -ynp)

            # Coeficiente complejo de la serie
            coeff_n = (2 * n + 1) * np.sin(eta_n) * np.exp(1j * eta_n)  # (Nf,)
            Pn = eval_legendre(n, cos_theta)  # (Nr, Nt)

            ff += coeff_n[np.newaxis, :, np.newaxis] * Pn[:, np.newaxis, :]

        # Mismo factor global que en RigidSphereEcho: return -(2/ka)*sum(...)
        ff *= (-2.0 / np.maximum(ka, self.ka_eps))[np.newaxis, :, np.newaxis]
        return ff

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
        self._calls += 1
        if self._calls % 50 == 0:
            print(f"RigidSphere apply calls: {self._calls}")

        print(
        "RigidSphere apply:",
        "S", np.shape(S),
        "f", np.shape(f),
        "inc", np.shape(tt_result.incident_vector),
        "sca", np.shape(tt_result.scattering_vector),
    )

        S_arr = np.asarray(S, dtype=np.complex128)  # expected (Nr|1, Nf, Nt)

        # Use environment sound speed (single source of truth from config["environment"]).
        c = float(
            np.asarray(
                environment.sound_speed(ping_time, tt_result.tx_position)
            ).reshape(-1)[0]
        )

        freq_hz = np.abs(np.asarray(f, dtype=float))  # physical frequency magnitude
        ka = 2.0 * np.pi * freq_hz * self.radius_m / c

        # Scattering angle from incident and scattering directions.
        inc = np.asarray(tt_result.incident_vector, dtype=float)     # (Nt, 3)
        sca = np.asarray(tt_result.scattering_vector, dtype=float)   # (Nr, Nt, 3)
        cos_theta = np.sum(inc[np.newaxis, :, :] * sca, axis=-1)    # (Nr, Nt)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        ff = self._form_function(ka, cos_theta)  # (Nr, Nf, Nt)

        if self.debug_dump and (not self._debug_dumped):
            try:
                out_path = Path(self.debug_dump_path)

                # Tomamos una traza representativa: receiver 0, target 0
                ff_line = ff[0, :, 0]
                theta_sample = float(np.arccos(np.clip(cos_theta[0, 0], -1.0, 1.0)))
                out_path.parent.mkdir(parents=True, exist_ok=True)

                np.savez(
                    out_path,
                    ka=ka,
                    ff_complex=ff_line,
                    ff_magnitude=np.abs(ff_line),
                    ff_phase=np.mod(np.angle(ff_line), 2.0 * np.pi),
                    theta_sample_rad=theta_sample,
                    radius_m=self.radius_m,
                    n_terms=self.n_terms,
                    scale=self.scale,
                )
                print(f"RigidSphere debug dump saved: {out_path}")
                self._debug_dumped = True
            except Exception as e:
                print(f"RigidSphere debug dump failed: {e}")

        print("RigidSphere out:", np.shape(S_arr), np.shape(ff))
        return S_arr * (self.scale * ff)