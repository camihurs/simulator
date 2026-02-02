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


Installation
------------

The openSTB tools are not yet published on PyPI (but will be in the near future). Before
installing the simulator, you will need to have the companion [internationalisation
support package](https://github.com/openSTB/i18n) installed in your environment, as well
as the [hatchling build backend](https://pypi.org/project/hatchling/). You can then run
`python -m pip install --no-build-isolation .` from the top level of a clone of this
repository (the `--no-build-isolation` prevents pip trying to build it in a clean
environment without access to the support packages). Some optional packages may also
need to be installed for your use case. If you want to use MPI-based parallelisation,
add the `MPI` option to the install: `python -m pip install --no-build-isolation
".[MPI]"`. To also install tools to help develop code for the framework, add the `dev`
option: `python -m pip install --no-build-isolation ".[dev]"`. You can give both options
separated by a comma, i.e., `python -m pip install --no-build-isolation ".[MPI,dev]"`.


Running the simulator
---------------------

See the `examples/` directory for various examples of how to run the simulator.


## Quick Start (local installation from source)

This project is not yet distributed via PyPI. The steps below describe a minimal and reproducible way to install and run the simulator locally from source.

1. Create a folder in your computer. Clone the repository inside the folder.
git clone https://github.com/openSTB/simulator.git
cd simulator

2. Ensure a compatible Python version

The simulator currently requires Python ≥ 3.12.

Check your version:
python --version


If multiple Python versions are installed (e.g. on Windows), you can explicitly use Python 3.12 via:
py -3.12 --version

3. Create and activate a virtual environment

From the root of the repository:
py -3.12 -m venv .venv

Activate it:

Windows
.venv\Scripts\activate


Linux / macOS
source .venv/bin/activate

4. Install build requirements
python -m pip install -U pip setuptools wheel
python -m pip install hatchling hatch-vcs

5. Install the openSTB i18n support package

The internationalisation support package is required at build and runtime and is not yet published on PyPI.

Clone and install it in the same environment (clone it being in the same folder that the created at the first step):

git clone https://github.com/openSTB/i18n.git
cd i18n
python -m pip install .
cd ..

6. Install the simulator

From the top level of the simulator repository:
python -m pip install --no-build-isolation .


The --no-build-isolation flag ensures the build process can access the locally installed support packages.

7. Run an example

You can now run one of the provided examples, for example:

cd examples/cli/simple_points
openstb-sim run simple_points.toml


The simulation output will be written to disk (e.g. as .npz or .zarr files).
See the corresponding plot_results.py script in each example directory for visualisation. You can run the plot_results.py to see the plots.