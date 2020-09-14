### System environment
Unbuntu 16.04

### Preinstall Package

```bash 
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

### Create Conda Environment
```bash 
conda create --name tf-1.15 anaconda python=3.6
conda activate tf-1.15
```

### Install tensorflow-1.15 GPU or CPU
```bash 
pip install tensorflow-gpu==1.15
```

or
```bash
pip install tensorflow==1.15
```

### Install Third-party Python Pakage
```bash
pip install gym
pip install graphviz
pip install pydotplus
pip install pyprind
pip install mpi4py
```