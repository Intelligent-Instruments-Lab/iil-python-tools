# iil-python-tools

⚠️ NOTICE: This repository has been archived. See [iil-dev](https://github.com/Intelligent-Instruments-Lab/iil-dev) for a single development environment, and [Intelligent-Instruments-Lab/repositories](https://github.com/orgs/Intelligent-Instruments-Lab/repositories) for the individual projects.

## Python packages

See Setup below for how to install these.

- iipyper: python package for easy MIDI, OSC, event loops
- notochord: python package for the Notochord MIDI performance model, see [readme](https://github.com/Intelligent-Instruments-Lab/iil-python-tools/blob/master/notochord/readme.md) for more details
    - notochord: package source code
    - scripts: helper scripts for training, data preprocessing etc
- mrp: interface to [Magnetic Resonator Piano](http://instrumentslab.org/research/mrp.html). Note that the rest of our MRP code lives in [iil-mrp-tools](https://github.com/Intelligent-Instruments-Lab/iil-mrp-tools).
- tölvera: an experimental artificial life simulator for agential musical instruments, see [readme](https://github.com/Intelligent-Instruments-Lab/iil-python-tools/blob/master/tolvera/readme.md) for more details

## Examples

- iipyper: 
    - basic usage for iipyper
    - declarative OSC mappings that can auto-generate clients for Pure Data and Max/MSP, and export to XML and JSON
- notochord: interactive MIDI apps with notochord and SuperCollider
    - tidalcycles: notochord interface to TidalCycles
- bela: [Bela](https://bela.io) examples in C++, Pure Data and so on
- faust: [Faust](https://faustdoc.grame.fr/) examples
- tidalcycles [TidalCycles](https://tidalcycles.org) examples
- puredata: [Pure Data](https://puredata.info) examples
- mrp: tests and examples, including SuperCollider Class
- alife: artificial life examples using Taichi Lang
<!-- - clients: templates for SuperCollider, Bela (C++), Pure Data, ... -->

# Setup

clone the repository:
```
git clone https://github.com/Intelligent-Instruments-Lab/iil-python-tools.git
cd iil-python-tools
```

we manage python dependencies with `conda`. If you don't have an anaconda/miniconda python install already, download an installer from https://docs.conda.io/en/latest/miniconda.html or use `brew install --cask miniconda` on mac.

you can check that this worked with `which python` -- it should have 'miniconda' in the path

now set up a python environment:
```
conda env create -f environment.yml
conda activate iil-python-tools
```

You can then install any Python packages you wish to use:
```
pip install -e notochord
pip install -e iipyper
pip install -e mrp
pip install -e tolvera
```
this will install all dependencies in a conda environment called `iil-python-tools`, and do an editable install of notochord and iipyper so you can hack on them.

# Develop

add new dependencies to `environment.yml`, then run:
```
conda env update -f environment.yml
```

# Build

single-download redistributable builds can be made using Nuitka (https://nuitka.net/index.html) 

steps (for arm64 mac):

install Nuitka into the conda environment (this is 1.1-rc10 at time of writing; 1.0 has a bug)
`pip install -U "https://github.com/Nuitka/Nuitka/archive/develop.zip"`

you can add `pip install ordered-set` and `brew install ccache` for best performance.

pytorch 1.12.x has an issue with an x86 binary being included in arm64 pacakges: https://github.com/pytorch/pytorch/issues/84351

to get around this, delete `_dl.*.so` from the torch install (which can be located with `python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent)"`)

then `nuitka-build.sh`, and `zip-notochord.sh` should compile notochord with nuitka and produce a zip from the build directory, artifacts directory, and `notochord-osc-server.sh` entry point.
