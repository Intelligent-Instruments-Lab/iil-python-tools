# Structure

- clients: templates for SuperCollider, Bela (C++), Pure Data, ...
- pytorch-osc: app with OSC server + event loops
- notepredictor: python package for defining+training pytorch RNN models
- notebooks: jupyter notebooks
- scripts: helper scripts for training, data preprocessing etc

# Setup

```
conda env create -f environment.yml
conda activate pytorch-osc
pip install -e notepredictor
```

# Train a model

```
python scipts/lakh_prep.py --data_path /path/to/midi/files --dest_path /path/to/data/storage
python scripts/train_pitch.py --data_dir /path/to/data/storage --log_dir /path/for/tensorboard logs --model_dir /path/for/checkpoints train
```

# Run OSC app

```
python pytorch-osc/pytorch-osc.py --checkpoint /path/to/my/model.ckpt
```

# Develop

add new dependencies to `environment.yml`, then run:
```
conda env update -f environment.yml
```
