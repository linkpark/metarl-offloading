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

### Install Tensorflow-1.15 GPU or CPU
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
### Start Meta Training:
```bash
python meta_trainer.py
```
All the hyperparameters are defined in `meta_trainer.py` including the log file and save path of the trained model.

### Start Meta Evaluation:
After training, you will get the meta model. In order to fast adapt the meta model for new learning tasks in MEC, we need to conduct fine-tuning steps for the trained meta moodel.

```bash
python meta_evaluator.py
```

The training might take long time because of the large training set. All the training results and evaluation results can be found in the log file. 

Related paper: [Fast Adaptive Task Offloading in Edge Computing based on Meta Reinforcement Learning
](https://arxiv.org/abs/2008.02033)

If you like this research, please cite this paper:

```buildoutcfg
@article{wang2020fast,
  title={Fast Adaptive Task Offloading in Edge Computing Based on Meta Reinforcement Learning},
  author={Wang, Jin and Hu, Jia and Min, Geyong and Zomaya, Albert Y and Georgalas, Nektarios},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={32},
  number={1},
  pages={242--253},
  year={2020},
  publisher={IEEE}
}
```
