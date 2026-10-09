"""Explore the actual simulate.py configuration without editing the simulation.

python visualize_simulation.py             # geometry only
python visualize_simulation.py --simulate  # geometry and original simulation
python visualize_simulation.py --save scene.png  # noninteractive preview
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from openstb.simulator.controller.simple_points import SimplePointSimulation
from openstb.simulator.distortion.beampattern import RectangularBeampattern
from openstb.simulator.distortion.rigid_sphere import RigidSphereFormFunction
from openstb.simulator.plugin import loader


def capture_config(script: Path):
    """Intercept cluster/output setup and run; restore all even on failure.

    The original function builds all its real plugin objects. No source parsing,
    parameter duplication, cluster startup, or result writes are needed.
    """

    class PreviewCluster:
        def initialise(self):
            pass

    captured = []

    def capture_run(simulation, config):
        captured.append(config)

    spec = importlib.util.spec_from_file_location("_sonar_scene_config", script)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        with (
            patch.object(loader, "dask_cluster", return_value=PreviewCluster()),
            patch.object(loader, "result_converter", return_value=None),
            patch.object(SimplePointSimulation, "run", capture_run),
        ):
            spec.loader.exec_module(module)
            module.simulate("local")
    finally:
        sys.path.pop(0)
    if len(captured) != 1:
        raise RuntimeError("Expected one simulation configuration")
    return captured[0]


class Scene:
    def __init__(self, config):
        self.config = config
        self.trajectory = config["trajectory"]
        self.times = config["ping_times"].calculate(self.trajectory)
        if not len(self.times):
            raise ValueError("No pings: check interval, start_delay and end_delay")
        self.positions = self.trajectory.position(self.times)
        self.targets = (
            np.concatenate(
                [
                    target.get_chunk(0, min(len(target), 10000))[0]
                    for target in config["targets"]
                    if len(target)
                ]
            )
            if any(len(t) for t in config["targets"])
            else np.empty((0, 3))
        )
        self.system = config["system"]
        self.frequency = (
            self.system.signal.minimum_frequency + self.system.signal.maximum_frequency
        ) / 2
        self.index = 0
        self.transducers = [self.system.transmitter, *self.system.receivers]
        self.sphere = next(
            (
                d
                for d in config.get("distortion", [])
                if isinstance(d, RigidSphereFormFunction)
            ),
            None,
        )
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title("Sonar geometry")
        self.overview = self.fig.add_subplot(121, projection="3d")
        self.detail = self.fig.add_subplot(122, projection="3d")
        self.fig.subplots_adjust(bottom=0.43, top=0.81, wspace=0.15)
        # Anchor the first line at the top of a separate information band.
        # Additional aperture/beam lines extend downwards, away from axis labels.
        self.info = self.fig.text(
            0.05, 0.34, "", fontsize=9, family="monospace", va="top"
        )
        self.beam_info = self.fig.text(0.55, 0.34, "", fontsize=9, va="top")
        self.tx_patterns = [
            p for p in list(self.system.transmitter.distortion) + config.get("distortion", [])
            if isinstance(p, RectangularBeampattern) and p.transmit
        ]
        self.beam_range_slider = None
        self.beam_range = 0.0
        if self.tx_patterns:
            distances = np.linalg.norm(self.targets - self.positions[0], axis=-1)
            default_range = max(10.0, float(distances.max()) * 1.2) if len(distances) else 20.0
            self.beam_range = default_range
            self.beam_range_slider = Slider(
                self.fig.add_axes((0.25, 0.845, 0.5, 0.02)),
                "TX visual range (m)", 0.1, default_range * 2,
                valinit=default_range, valfmt="%.1f",
            )

            def change_range(value):
                self.beam_range = value
                self.draw(self.index)

            self.beam_range_slider.on_changed(change_range)
        self.status = self.fig.text(0.05, 0.97, "Preview — no echoes calculated")
        self.legend = None
        self.camera_controls = []
        self.reset_buttons = []
        self.view_buttons = []
        for plot, left, name in (
            (self.overview, 0.16, "Overview"),
            (self.detail, 0.64, "Sonar detail"),
        ):
            self.fig.text(left, 0.215, f"{name} rotation", fontsize=10)
            azimuth = Slider(
                self.fig.add_axes((left, 0.18, 0.28, 0.018)),
                "Azimuth",
                -180,
                180,
                valinit=plot.azim,
                valfmt="%0.0f°",
            )
            elevation = Slider(
                self.fig.add_axes((left, 0.145, 0.28, 0.018)),
                "Elevation",
                -90,
                90,
                valinit=plot.elev,
                valfmt="%0.0f°",
            )

            def rotate(value, plot=plot, azimuth=azimuth, elevation=elevation):
                plot.view_init(elev=elevation.val, azim=azimuth.val)
                self.fig.canvas.draw_idle()

            azimuth.on_changed(rotate)
            elevation.on_changed(rotate)
            button = Button(
                self.fig.add_axes((left + 0.19, 0.205, 0.09, 0.027)), "Reset view"
            )

            def reset(event, plot=plot, azimuth=azimuth, elevation=elevation):
                azimuth.reset()
                elevation.reset()
                self.draw(self.index)

            button.on_clicked(reset)
            self.camera_controls.append((plot, azimuth, elevation))
            self.reset_buttons.append(button)
            # Global-axis views: Z is positive down, so Top looks from -Z.
            for offset, (label, elev, azim) in enumerate(
                (("Top", -90, -90), ("Front", 0, 0), ("Side", 0, 90))
            ):
                preset = Button(
                    self.fig.add_axes((left + offset * 0.095, 0.115, 0.085, 0.023)),
                    label,
                )

                def set_view(event, azimuth=azimuth, elevation=elevation,
                             elev=elev, azim=azim):
                    azimuth.set_val(azim)
                    elevation.set_val(elev)

                preset.on_clicked(set_view)
                self.view_buttons.append(preset)
        self.fig.canvas.mpl_connect("button_release_event", self.sync_camera_controls)
        axis = self.fig.add_axes((0.18, 0.045, 0.65, 0.025))
        self.slider = Slider(
            axis, "Ping", 1, max(2, len(self.times)), valinit=1, valstep=1
        )
        self.slider.on_changed(
            lambda value: self.draw(min(int(value) - 1, len(self.times) - 1))
        )
        patterns = config.get("distortion", []) + [
            d for tr in self.transducers for d in tr.distortion
        ]
        low, high = (
            self.system.signal.minimum_frequency,
            self.system.signal.maximum_frequency,
        )
        self.frequency_slider = None
        if high > low and any(
            isinstance(p, RectangularBeampattern) and p._freqmode == 3 for p in patterns
        ):
            frequency_axis = self.fig.add_axes((0.18, 0.09, 0.65, 0.02))
            self.frequency_slider = Slider(
                frequency_axis,
                "Frequency (Hz)",
                max(low, 1e-6),
                high,
                valinit=self.frequency,
            )

            def change_frequency(value):
                self.frequency = value
                self.draw(self.index)

            self.frequency_slider.on_changed(change_frequency)
        self.fig.canvas.mpl_connect("pick_event", self.pick)
        self.child = None
        self.timer = None
        self.draw(0)

    def sync_camera_controls(self, event):
        """Keep sliders in sync when the user rotates either plot with the mouse."""
        for plot, azimuth, elevation in self.camera_controls:
            if event.inaxes is plot:
                for slider, value in (
                    (azimuth, (plot.azim + 180) % 360 - 180),
                    (elevation, (plot.elev + 180) % 360 - 180),
                ):
                    slider.eventson = False
                    slider.set_val(np.clip(value, slider.valmin, slider.valmax))
                    slider.eventson = True

    def pick(self, event):
        if event.artist is self.ping_artist and len(event.ind):
            self.slider.set_val(int(event.ind[0]) + 1)

    @staticmethod
    def axes_style(ax, title):
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m, positive down)")
        ax.set_box_aspect((1, 1, 1))

    @staticmethod
    def limits(ax, points, margin=0.12):
        lower, upper = points.min(axis=0), points.max(axis=0)
        centre = (lower + upper) / 2
        half = max(float(np.max(upper - lower)) / 2, 0.5) * (1 + margin)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] + half, centre[2] - half)

    def pattern(self, ax, pattern, pos, orientation, frequency, sound_speed, color):
        # Use the plugin's actual response, including horizontal/vertical switches.
        az, el = np.meshgrid(
            np.linspace(-np.pi, np.pi, 91), np.linspace(-np.pi / 2, np.pi / 2, 47)
        )
        directions = np.stack(
            (np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), axis=-1
        )
        gain = abs(pattern._eval(np.asarray(sound_speed / frequency), directions))
        surface = pos + orientation.rotate(directions * gain[..., None] * 1.5)
        ax.plot_surface(
            *surface.transpose(2, 0, 1),
            color=color,
            alpha=0.18,
            linewidth=0,
            rcount=47,
            ccount=91,
        )
        w, h = pattern.width / 2, pattern.height / 2
        corners = np.array([[0, -w, -h], [0, w, -h], [0, w, h], [0, -w, h]])
        corners = pos + orientation.rotate(corners)
        ax.add_collection3d(Poly3DCollection([corners], facecolors=color, alpha=0.5))

    @staticmethod
    def main_lobe_boundary(pattern, wavelength, count=129):
        """Exact -3 dB amplitude contour in the forward principal lobe.

        Search radially around local +X, stopping at the first sinc zero (or
        the forward hemisphere). This excludes rear and secondary lobes.
        Unconstrained directions extend to the hemisphere when an aperture
        component is disabled or too small to reach -3 dB.
        """
        phi = np.linspace(0, 2 * np.pi, count)
        cy, sz = np.cos(phi), np.sin(phi)
        scale = np.maximum(
            abs(pattern.width * cy) if pattern.horizontal else np.zeros_like(phi),
            abs(pattern.height * sz) if pattern.vertical else np.zeros_like(phi),
        )
        hi = np.arcsin(np.minimum(1.0, wavelength / np.maximum(scale, 1e-30)))
        lo = np.zeros_like(hi)
        threshold = 10 ** (-3 / 20)
        def directions(angle):
            return np.stack((np.cos(angle), np.sin(angle) * cy, np.sin(angle) * sz), -1)
        crossing = abs(pattern._eval(np.asarray(wavelength), directions(hi))) <= threshold
        for _ in range(45):
            mid = (lo + hi) / 2
            above = abs(pattern._eval(np.asarray(wavelength), directions(mid))) > threshold
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        angle = np.where(crossing, (lo + hi) / 2, np.pi / 2)
        return directions(angle)

    @staticmethod
    def inside_main_lobe(points, pos, orientation, pattern, wavelength, distance_limit):
        local = (~orientation).rotate(np.asarray(points) - pos)
        distance = np.linalg.norm(local, axis=-1)
        direction = local / np.maximum(distance[..., None], 1e-30)
        principal = direction[..., 0] >= 0
        if pattern.horizontal:
            principal &= abs(pattern.width * direction[..., 1] / wavelength) < 1
        if pattern.vertical:
            principal &= abs(pattern.height * direction[..., 2] / wavelength) < 1
        gain = abs(pattern._eval(np.asarray(wavelength), direction))
        return principal & (gain >= 10 ** (-3 / 20)) & (distance <= distance_limit)

    def illumination(self, pos, orientation, pattern, frequency, sound_speed):
        wavelength = sound_speed / frequency
        boundary = self.main_lobe_boundary(pattern, wavelength)
        rim = pos + orientation.rotate(boundary * self.beam_range)
        faces = [[pos, rim[i], rim[i + 1]] for i in range(len(rim) - 1)]
        # The range boundary is spherical: every rim point is equally far from TX.
        self.overview.add_collection3d(Poly3DCollection(
            faces, facecolors="darkorange", alpha=0.12, edgecolors="none"
        ))
        phi = np.linspace(0, 2 * np.pi, len(boundary))
        theta = np.linspace(0, 1, 12)[:, None] * np.arccos(boundary[:, 0])[None, :]
        cap = np.stack((np.cos(theta), np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi)), axis=-1)
        cap = pos + orientation.rotate(cap * self.beam_range)
        self.overview.plot_surface(*cap.transpose(2, 0, 1), color="darkorange",
                                   alpha=0.07, linewidth=0, rcount=12, ccount=len(boundary))
        self.overview.plot(*rim.T, color="darkorange", linewidth=1,
                           label="TX forward main lobe (-3 dB)")
        for i in (0, 32, 64, 96):
            self.overview.plot(*np.array([pos, rim[i]]).T, color="darkorange", alpha=0.5)
        horizontal = np.degrees(np.arccos(np.clip(boundary[0, 0], -1, 1))) * 2
        vertical = np.degrees(np.arccos(np.clip(boundary[32, 0], -1, 1))) * 2
        lines = [f"TX coverage preview: -3 dB at {frequency:.0f} Hz",
                 f"Beamwidth: H {horizontal:.2f}° / V {vertical:.2f}° (full angles)",
                 f"Visual range: {self.beam_range:.1f} m; excludes rear/secondary lobes"]
        if len(self.targets):
            if self.sphere:
                # Equal-area Fibonacci samples estimate surface coverage only.
                n = 4096
                z = 1 - 2 * (np.arange(n) + 0.5) / n
                phi = np.arange(n) * np.pi * (3 - np.sqrt(5))
                unit = np.column_stack((np.sqrt(1-z*z)*np.cos(phi),
                                        np.sqrt(1-z*z)*np.sin(phi), z))
                samples = self.targets[0] + self.sphere.radius_m * unit
                inside = self.inside_main_lobe(samples, pos, orientation, pattern,
                                              wavelength, self.beam_range)
                fraction = float(inside.mean())
                state = "inside" if inside.all() else "outside" if not inside.any() else "partial"
                lines.append(f"Target 1 sphere: {state}; estimated surface fraction {fraction:.1%}")
                lines.append("Geometric sampling estimate; outside does not mean no echo")
            else:
                inside = self.inside_main_lobe(self.targets, pos, orientation, pattern,
                                              wavelength, self.beam_range)
                lines.append(f"Displayed target points inside: {inside.sum()}/{len(inside)}")
                lines.append("Geometric preview; outside does not mean no echo")
        self.beam_info.set_text("\n".join(lines))
        return rim

    def draw(self, index):
        self.index = index
        cameras = [(ax.elev, ax.azim) for ax in (self.overview, self.detail)]
        for ax in (self.overview, self.detail):
            ax.clear()
        t = self.times[index]
        origin = self.positions[index]
        # Reuse the configured travel-time plugin's own global pose convention.
        tt = self.config["travel_time"].calculate(
            self.trajectory,
            t,
            self.config["environment"],
            self.transducers[0].position,
            self.transducers[0].orientation,
            np.array([r.position for r in self.system.receivers]).reshape(-1, 3),
            np.array([r.orientation.ndarray for r in self.system.receivers]).reshape(
                -1, 4
            ),
            self.targets[:1] if len(self.targets) else origin[None, :] + [0, 1, 0],
        )
        poses = [(tt.tx_position, tt.tx_orientation)] + [
            (tt.rx_position[i, 0], tt.rx_orientation[i, 0])
            for i in range(len(self.system.receivers))
        ]
        path = self.trajectory.position(np.linspace(0, self.trajectory.duration, 200))
        ax = self.overview
        ax.plot(*path.T, color="gray", label="Trajectory")
        ax.scatter(*path[0], c="green", s=65, label="Start")
        ax.scatter(*path[-1], c="red", s=65, label="End")
        self.ping_artist = ax.scatter(
            *self.positions.T,
            c="steelblue",
            s=12,
            picker=5,
            label="Pings (click or slider)",
        )
        ax.scatter(*origin, c="gold", edgecolors="black", s=80, label="Selected ping")
        if len(self.targets):
            ax.scatter(*self.targets.T, c="purple", s=25, label="Targets")
            if self.sphere:
                u, v = np.meshgrid(
                    np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 20)
                )
                unit = np.stack(
                    (np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)), -1
                )
                for centre in self.targets[:100]:
                    surface = centre + self.sphere.radius_m * unit
                    ax.plot_surface(
                        *surface.transpose(2, 0, 1), color="purple", alpha=0.4
                    )
            for pos, _ in poses:
                ax.plot(*np.array([pos, self.targets[0]]).T, color="purple", alpha=0.35)
        velocity = self.trajectory.velocity(t)
        ax.quiver(
            *origin, *velocity, color="green", label="Velocity arrow scale: 1 m = 1 m/s"
        )
        signal = self.system.signal
        frequency = self.frequency
        notes = []
        beam_bounds = np.empty((0, 3))
        self.beam_info.set_text("")
        for i, (transducer, (pos, ori)) in enumerate(zip(self.transducers, poses)):
            color = "darkorange" if i == 0 else "royalblue"
            label = "TX" if i == 0 else f"RX {i}"
            direction = ori.rotate(np.array([1.0, 0.0, 0.0]))
            for plot in (self.overview, self.detail):
                plot.scatter(*pos, c=color, s=45)
                plot.quiver(*pos, *direction, length=0.8, color=color)
                plot.text(*pos, label, color=color)
            notes.append(f"{label}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
            active = list(transducer.distortion) + self.config.get("distortion", [])
            for pattern in active:
                if not isinstance(pattern, RectangularBeampattern):
                    continue
                if not (pattern.transmit if i == 0 else pattern.receive):
                    continue
                bounds = (signal.minimum_frequency, signal.maximum_frequency)
                freq = (
                    bounds[0]
                    if pattern._freqmode == 0
                    else bounds[1]
                    if pattern._freqmode == 1
                    else frequency
                )
                if freq > 0:
                    c = float(
                        np.asarray(
                            self.config["environment"].sound_speed(t, tt.tx_position)
                        ).flat[0]
                    )
                    self.pattern(self.detail, pattern, pos, ori, freq, c, color)
                    if i == 0 and pattern is self.tx_patterns[0]:
                        beam_bounds = self.illumination(pos, ori, pattern, freq, c)
                    notes.append(
                        f"{label}: aperture {pattern.width:g} × {pattern.height:g} m; beam at {freq:g} Hz"
                        + (" (frequency=all slice)" if pattern._freqmode == 3 else "")
                    )
        self.axes_style(ax, "Overview · physical scale")
        self.axes_style(self.detail, "Sonar detail · orientation arrows: 0.8 m")
        self.limits(
            ax, np.concatenate([path, self.targets, np.array([p for p, _ in poses]), beam_bounds])
        )
        self.limits(self.detail, np.array([p for p, _ in poses]), margin=3)
        for plot, camera in zip((ax, self.detail), cameras):
            plot.view_init(*camera)
        if self.legend is not None:
            self.legend.remove()
        handles, labels = ax.get_legend_handles_labels()
        self.legend = self.fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=4,
            fontsize=9,
        )
        self.info.set_text(
            f"Ping {index + 1}/{len(self.times)} · t={t:.3f} s · speed={np.linalg.norm(velocity):g} m/s"
            f" · hydrophones={len(self.system.receivers)}\n" + "\n".join(notes)
        )
        self.fig.canvas.draw_idle()

    def run_simulation(self, script):
        # Inherit the working directory and terminal, just like python simulate.py.
        # The child has no preview patches and writes the original result files.
        self.child = subprocess.Popen([sys.executable, str(script), "local"])
        self.status.set_text("Simulation running · progress and errors in the terminal")
        self.timer = self.fig.canvas.new_timer(interval=500)

        def poll():
            code = self.child.poll()
            if code is not None:
                self.status.set_text(
                    "Simulation completed"
                    if code == 0
                    else f"Simulation failed (exit code {code}); check the terminal"
                )
                self.fig.canvas.draw_idle()
                self.timer.stop()

        self.timer.add_callback(poll)
        self.timer.start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument(
        "--save", type=Path, help="Save PNG without opening a window or simulating"
    )
    args = parser.parse_args()
    if args.save and args.simulate:
        parser.error("--save and --simulate cannot be combined")
    script = Path(__file__).with_name("simulate.py")
    scene = Scene(capture_config(script))
    print(
        f"Configuration: {len(scene.times)} pings, {len(scene.system.receivers)} hydrophones"
    )
    if args.save:
        scene.fig.savefig(args.save, dpi=160)
        plt.close(scene.fig)
    else:
        if args.simulate:
            scene.run_simulation(script)
        plt.show()
        if scene.child is not None:
            print(
                "Waiting for simulation; closing the viewer does not cancel the calculation."
            )
            raise SystemExit(scene.child.wait())


if __name__ == "__main__":
    main()
