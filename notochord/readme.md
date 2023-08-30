# Notochord ([Paper](https://zenodo.org/record/7088404 "Notochord AIMC 2022 paper") | [Video](https://www.youtube.com/watch?v=mkBKAyudL0A "Notochord AIMC 2022 video"))
 
<div align="middle">
<img alt="Max Ernst, Stratified Rocks, Nature's Gift of Gneiss Lava Iceland Moss 2 kinds of lungwort 2 kinds of ruptures of the perinaeum growths of the heart b) the same thing in a well-polished little box somewhat more expensive, 1920" src="https://user-images.githubusercontent.com/4522484/223191876-251d461a-5bfc-439a-8df0-3841e7c76c4a.jpeg" width="60%" />
</div>

Notochord is a neural network model for MIDI performances. This package contains the training and inference model implemented in pytorch, as well as interactive MIDI processing apps using iipyper. Some further examples involving SuperCollider and TidalCycles can be found in the parent repo under `examples`.

## Getting Started

Follow the instructions in the root repo to set up an `iil-python-tools` environment. Then download a model checkpoint (e.g. `notochord_lakh_50G_deep.pt`) from the releases page: https://github.com/Intelligent-Instruments-Lab/iil-python-tools/releases . From here, I'll assume the model file is saved at `~/Downloads/notochord_lakh_50G_deep.pt`.

## MIDI Apps

These iipyper apps can be run in a terminal. They have a clickable text-mode user interface and connect directly to MIDI ports, so you can wire them up to your controllers, DAW, etc.

The Notochord harmonizer adds extra concurrent notes for each MIDI note you play in. In a terminal, make sure the `iil-python-tools` conda environment is active (`conda activate iil-python-tools`) and run:
```
python -m notochord harmonizer --checkpoint ~/Downloads/notochord_lakh_50G_deep.pt
```
try `python -m notochord harmonizer --help`
to see more options.

the ``homunculus'' gives you a UI to manage multiple input, harmonizing or autonomous notochord channels:
```
python -m notochord homunculus --checkpoint ~/Downloads/notochord_lakh_50G_deep.pt
```

You can set the MIDI in and out ports with `--midi-in` and `--midi-out`. If you use a General MIDI synthesizer like fluidsynth, you can add `--send-pc` to also send program change messages.

## Python API

See the docstrings for `Notochord.feed` and `Notochord.query` in `notochord/model.py` for the low-level Notochord inference API which can be used from Python code.

## OSC server

You can also expose the inference API over Open Sound Control:
```
python -m notochord server --checkpoint ~/Downloads/notochord_lakh_50G_deep.pt
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
