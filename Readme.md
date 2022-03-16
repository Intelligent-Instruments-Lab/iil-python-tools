# Structure

- iipyper: python package for easy MIDI, OSC, event loops
- notepredictor: python package for defining+training pytorch RNN models
    - notebooks: jupyter notebooks
    - scripts: helper scripts for training, data preprocessing etc
- examples:
    - iipyper: basic usage for iipyper
    - notepredictor: interactive MIDI apps with notepredictor and SuperCollider
<!-- - clients: templates for SuperCollider, Bela (C++), Pure Data, ... -->

# Setup

```
conda env create -f environment.yml
conda activate iil-python-tools
pip install -e notepredictor
pip install -e iipyper
```

# notepredictor
## Train a model
```
python scripts/lakh_prep.py --data_path /path/to/midi/files --dest_path /path/to/data/storage
python scripts/train_notes.py --data_dir /path/to/data/storage --log_dir /path/for/tensorboard logs --model_dir /path/for/checkpoints train
```

## Run OSC app

```
python examples/notepredictor/server.py --checkpoint /path/to/my/model.ckpt
```
step through `examples/notepredictor/midi-duet.scd` in SuperCollider IDE

# Develop

add new dependencies to `environment.yml`, then run:
```
conda env update -f environment.yml
```
