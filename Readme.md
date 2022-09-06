# Repository Structure

- iipyper: python package for easy MIDI, OSC, event loops
- notochord: python package for the Notochord MIDI performance model
    - notochord: package source code
    - scripts: helper scripts for training, data preprocessing etc
- examples:
    - iipyper: basic usage for iipyper
    - notochord: interactive MIDI apps with notochord and SuperCollider
        - tidalcycles: notochord interface to TidalCycles
    - bela: [Bela](https://bela.io) examples in C++, Pure Data and so on
    - faust: [Faust](https://faustdoc.grame.fr/) examples
    - tidalcycles [TidalCycles](https://tidalcycles.org) examples
    - puredata: [Pure Data](https://puredata.info) examples
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
pip install -e notochord
pip install -e iipyper
```
this will install all dependencies in a conda environment called `iil-python-tools`, and do an editable install of notochord and iipyper so you can hack on them.

# notochord
download a model checkpoint (e.g. `notochord_lakh_20G.ckpt`) from the releases page: https://github.com/Intelligent-Instruments-Lab/iil-python-tools/releases

## Run python server
In a terminal, make sure the `iil-python-tools` conda environment is active (`conda activate iil-python-tools`) and run:
```
python -m notochord server --checkpoint ~/Downloads/notochord_lakh_20G.ckpt
```
this will run notochord and listen continously for OSC messages.

`examples/notochord/generate-demo.scd` and `examples/notochord/harmonize-demo.scd` are example scripts for interacting with the notochord server from SuperCollider.

## Tidal interface

see `examples/notochord/tidalcycles`:

add `Notochord.hs` to your tidal boot file. Probably replace the `tidal <- startTidal` line with something like:
```
:script ~/iil-python-tools/examples/notochord/tidalcycles/Notochord.hs

let sdOscMap = (superdirtTarget, [superdirtShape])
let oscMap = [sdOscMap,ncOscMap]

tidal <- startStream defaultConfig {cFrameTimespan = 1/240} oscMap
```

In a terminal, start the python server as described above.

In Supercollider, step through `examples/notochord/tidalcycles/tidal-notochord-demo.scd` which will receive from Tidal, talk to the python server, and send MIDI on to a synthesizer. There are two options, either send to fluidsynth to synthesize General MIDI, or specify your own mapping of instruments to channels and send on to your own DAW or synth.

### Install fluidsynth (optional)
fluidsynth (https://github.com/FluidSynth/fluidsynth) is a General MIDI synthesizer which you can install from the package manager. On macOS:
```
brew install fluidsynth
```
fluidsynth needs a soundfont to run, like this one: https://drive.google.com/file/d/1-cwBWZIYYTxFwzcWFaoGA7Kjx5SEjVAa/view

run fluidsynth in a terminal (see the fluidsynth block in `examples/notochord/tidalcycles/tidal-notochord.scd` for an example command).

## Train your own Notochord model (GPU recommended)

preprocess the data:
```
python notochord/scripts/lakh_prep.py --data_path /path/to/midi/files --dest_path /path/to/data/storage
```
launch a training job:
```
python notochord/train.py --data_dir /path/to/data/storage --log_dir /path/for/tensorboard/logs --model_dir /path/for/checkpoints --results_dir /path/for/other/logs train
```
progress can be monitored via tensorboard.

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
