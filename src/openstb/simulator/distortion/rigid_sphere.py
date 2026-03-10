# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
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

        for n in range(self.n_terms + 1):
            jnp = spherical_jn(n, ka, derivative=True)
            ynp = spherical_yn(n, ka, derivative=True)
            denom = jnp + 1j * ynp

            coeff_n = (2 * n + 1) * (jnp / denom)  # (Nf,)
            Pn = eval_legendre(n, cos_theta)       # (Nr, Nt)

            ff += coeff_n[np.newaxis, :, np.newaxis] * Pn[:, np.newaxis, :]

        ff *= (-1j / np.maximum(ka, self.ka_eps))[np.newaxis, :, np.newaxis]
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

        print("RigidSphere out:", np.shape(S_arr), np.shape(ff))
        return S_arr * (self.scale * ff)