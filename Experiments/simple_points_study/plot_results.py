# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from matplotlib import pyplot as plt
import numpy as np

# This assumes you used the NumPy output converter in example_sim.py.
results = np.load("simple_points.npz")

# This loads the results as a mapping instance.
t = results["sample_time"]
P = results["pressure"]

print("P shape:", P.shape)
print("abs(P) min/max:", np.abs(P).min(), np.abs(P).max())
print("finite ratio:", np.isfinite(P).mean())


# As an example, plot the trace from the middle receiver at ping 14 as sound pressure
# level (dB relative to 1uPa). The results have the dimensions ping, receiver, time
# (which is stored in a pressure_dimensions array in the results).

ping_energy = np.max(np.abs(P[:, 0, :]), axis=1)
ping_idx = int(np.argmax(ping_energy))

ping_to_plot = 5   # cambia aquí el ping que quieras ver
trace = P[ping_to_plot, 0, :]
plt.figure()
plt.title(f"Middle receiver (x=0), ping {ping_to_plot}")
plt.plot(t, 20 * np.log10(np.abs(trace) / 1e-6))
plt.xlabel("Time (s)")
plt.ylabel("Echo strength (SPL)")
plt.show()

# Or an image of all pings recorded on the middle receiver.
rx = P[:, 0, :]
plt.figure()
plt.imshow(
    20 * np.log10(np.abs(rx) / 1e-6),
    aspect="auto",
    origin="lower",
    interpolation="none",
    extent=(t[0], t[-1], 0, P.shape[0] - 1),
    #vmin=100,
)
plt.colorbar(label="Echo strength (SPL)")
plt.xlabel("Time (s)")
plt.ylabel("Ping")
plt.title("Middle receiver (x=0)")
plt.show()
