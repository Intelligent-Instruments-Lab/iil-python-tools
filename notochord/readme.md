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
