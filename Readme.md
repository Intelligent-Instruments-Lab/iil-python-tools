# Structure

- clients: templates for SuperCollider, Bela (C++), Pure Data, ...
- osc: 
- notepredictor: python package for offline pytorch models
- notebooks: jupyter notebooks
- scripts: helper scripts for training, data preprocessing etc

# Setup

```
conda create -f environment.yml
conda activate pytorch-osc
pip install -e notepredictor
```

# Develop

add new dependencies to environment.yml