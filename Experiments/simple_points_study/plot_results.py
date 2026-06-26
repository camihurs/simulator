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

ping_to_plot = 31   # cambia aquí el ping que quieras ver
trace = P[ping_to_plot, 0, :]
plt.figure()
#plt.title(f"Middle receiver (x=0), ping {ping_to_plot}")
spl_db = 20 * np.log10(np.maximum(np.abs(trace), 1e-12) / 1e-6)
plt.plot(t * 1e3, spl_db, 'k-')  # Convert the time to ms and plot the SPL in dB
#plt.plot(t, 20 * np.log10(np.abs(trace) / 1e-6))
plt.xlabel("Time [ms]", fontsize=38)
#plt.xlabel("Time (s)", fontsize=22)
plt.ylabel("Echo level [dB re 1 uPa]", fontsize=38) #"Echo strength (SPL)"
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.show()

# Or an image of all pings recorded on the middle receiver.
rx = P[:, 0, :]
plt.figure()
plt.imshow(
    20 * np.log10(np.abs(rx) / 1e-6),
    aspect="auto",
    origin="lower",
    interpolation="none",
    #extent=(t[0], t[-1], 0, P.shape[0] - 1),
    extent=(t[0] * 1e3, t[-1] * 1e3, 0, P.shape[0] - 1),
    #vmin=100,
)
#plt.colorbar(label="Echo level [dB re 1 uPa]", fontsize=22) #"Echo strength (SPL)"
cbar = plt.colorbar()
cbar.set_label("Echo level [dB re 1 uPa]", fontsize=38)
cbar.ax.tick_params(labelsize=34)
plt.xlabel("Time [ms]", fontsize=38)
plt.ylabel("Ping", fontsize=38)
#plt.title("Middle receiver (x=0)")
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.show()
