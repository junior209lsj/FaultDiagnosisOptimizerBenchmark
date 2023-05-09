# Impact of Deep Learning Optimizers and Hyperparameter Tuning on the Performance of Bearing Fault Diagnosis

This is the code base for the benchmark study article `Impact of Deep Learning Optimizers and Hyperparameter Tuning on the Performance of Bearing Fault Diagnosis`. We implemented end-to-end optimization benchmark code using public bearing fault datasets and state-of-the-art fault diagnosis models.

# Requirements

To use this code, we recommended to install libraries on the anaconda virtual environment. Required libraries will be installed following instructions below.

```
conda create -n {your virtual env name} python=3.10.6
conda activate {your virtual env name}
pip install --upgrade pip
pip install -r requirements.txt
```

`Note`: We tested this code in PC using Ubuntu Linux and CUDA GPU. Experimental specifications are listed below.

|Type|Specification|
|------|---|
|OS|Ubuntu 18.04|
|CPU|Intel Core i9-10900K @ 3.70 GHz|
|RAM|128 GB|
|GPU|NVIDIA GeForce RTX 2080 SUPER x2|
|CUDA version|11.2|
|CUDNN version|7.6.5|

# Getting Started

We provide short demo code. Check `tutorial.ipynb`.
