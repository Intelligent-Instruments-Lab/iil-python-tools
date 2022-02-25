# Structure

- clients: templates for SuperCollider, Bela (C++), Pure Data, ...
- osc: 
- notepredictor: python package for offline pytorch models
- notebooks: jupyter notebooks
- scripts: helper scripts for training, data preprocessing etc

# Setup

```
conda env create -f environment.yml
conda activate pytorch-osc
pip install -e pytorch-osc
```

# Develop

add new dependencies to `environment.yml`, then run:
```
conda env update -f environment.yml
```
