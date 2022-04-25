# Structure

- iipyper: python package for easy MIDI, OSC, event loops
- notochord: python package for defining+training pytorch RNN models
    - notebooks: jupyter notebooks
    - scripts: helper scripts for training, data preprocessing etc
- examples:
    - iipyper: basic usage for iipyper
    - notochord: interactive MIDI apps with notochord and SuperCollider
    - bela: [Bela](https://bela.io) examples in C++, Pure Data and so on
    - faust: [Faust](https://faustdoc.grame.fr/) examples
    - tidalcycles [TidalCycles](https://tidalcycles.org) examples
    - puredata: [Pure Data](https://puredata.info) examples
<!-- - clients: templates for SuperCollider, Bela (C++), Pure Data, ... -->

# Setup

```
conda env create -f environment.yml
conda activate iil-python-tools
pip install -e notochord
pip install -e iipyper
```

# notochord
## Train a model
```
python notochord/scripts/lakh_prep.py --data_path /path/to/midi/files --dest_path /path/to/data/storage
python notochord/train.py --data_dir /path/to/data/storage --log_dir /path/for/tensorboard logs --model_dir /path/for/checkpoints train
```

## Run OSC app

```
python examples/notochord/server.py --checkpoint /path/to/my/model.ckpt
```
step through `examples/notochord/generate.scd` in SuperCollider IDE

# Develop

add new dependencies to `environment.yml`, then run:
```
conda env update -f environment.yml
```
