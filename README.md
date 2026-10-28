<!--

SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent

-->

openSTB sonar simulation framework
==================================

The openSTB sonar simulation is a modular framework for simulating the signals received
by a sonar system. It is currently in the early stages of development, so bugs are to be
expected.


License
-------

The simulator is available under the BSD-2-Clause plus patent license, the text of which
can be found in the `LICENSES/` directory or
[online](https://spdx.org/licenses/BSD-2-Clause-Patent.html). Some of the supporting
files are under the Creative Commons Zero v1.0 Universal license (effectively public
domain). Again, the license is available in the `LICENSES/` directory or
[online](https://spdx.org/licenses/CC0-1.0.html).


Documentation
-------------

The documentation for the simulator can be viewed at https://docs.openstb.dev


Installation
------------

The simulator is published on [PyPI](https://pypi.org) as
[`openstb-simulator`](https://pypi.org/project/openstb-simulator). You can use your
preferred Python environment management tool to install it. For example, with pip:

```console
pip install openstb-simulator
```

For more installation options, see the documentation.


Running the simulator
---------------------

See the `examples/` directory for various examples of how to run the simulator. The
documentation includes more details of these examples.

Local scattering research workflows
----------------------------------

The repository also contains a controlled single-target experiment and independent
free-field sphere standalones. Their configuration, validation scope and optional
diagnostics are documented separately:

- [Simple-points scattering experiment](Experiments/simple_points_study/README.md)
- [Rigid-sphere standalones](Experiments/scattering_models/rigid_sphere/free_field/README.md)
- [Solid elastic-sphere standalones](Experiments/scattering_models/solid_sphere/free_field/README.md)

These are research workflows, not a validation claim for all combinations of
platform, propagation and scattering effects.
