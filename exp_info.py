import torch

import fdob
import fdob.model
import fdob.processing

hparam = {
    "sgd": {
        "optimizer": torch.optim.SGD,
        "n_params": 1,
        "param_names": ["lr"],
        "lb": [-4],
        "ub": [0],
        "reversed": [False]
    },
    "momentum": {
        "optimizer": torch.optim.SGD,
        "n_params": 2,
        "param_names": ["lr", "momentum"],
        "lb": [-4, -3],
        "ub": [0, 0],
        "reversed": [False, True]
    },
    "rmsprop": {
        "optimizer": torch.optim.RMSprop,
        "n_params": 4,
        "param_names": ["lr", "momentum", "alpha", "eps"],
        "lb": [-4, -3, -3, -10],
        "ub": [-1, 0, 0, 0],
        "reversed": [False, True, True, False]
    },
    "adam": {
        "optimizer": torch.optim.Adam,
        "n_params": 4,
        "param_names": ["lr", "beta1", "beta2", "eps"],
        "lb": [-4, -3, -4, -10],
        "ub": [-1, 0, -1, 0],
        "reversed": [False, True, True, False] 
    }
}

model = {
    "stimcnn": {
        "model": fdob.model.STIMCNN,
        "sample_length": 784,
        "tf": [fdob.processing.NpToTensor(), 
               fdob.processing.ToImage(28, 28, 1)]
    },
    "stftcnn": {
        "model": fdob.model.STFTCNN,
        "sample_length": 512,
        "tf": [fdob.processing.STFT(window_length=128, noverlap=120, nfft=128),
               fdob.processing.Resize(64, 64),
               fdob.processing.NpToTensor(),
               fdob.processing.ToImage(64, 64, 1)]
    },
    "wdcnn": {
        "model": fdob.model.WDCNN,
        "sample_length": 2048,
        "tf": [fdob.processing.NpToTensor(),
               fdob.processing.ToSignal()]
    },
    "wdcnnrnn": {
        "model": fdob.model.WDCNNRNN,
        "sample_length": 4096,
        "tf": [fdob.processing.NpToTensor(),
               fdob.processing.ToSignal()]
    },
    "ticnn": {
        "model": fdob.model.TICNN,
        "sample_length": 2048,
        "tf": [fdob.processing.NpToTensor(),
               fdob.processing.ToSignal()]
    },
    "dcn": {
        "model": fdob.model.DCN,
        "sample_length": 784,
        "tf": [fdob.processing.NpToTensor(),
               fdob.processing.ToSignal()]
    },
    "srdcnn": {
        "model": fdob.model.SRDCNN,
        "sample_length": 1024,
        "tf": [fdob.processing.NpToTensor(),
               fdob.processing.ToSignal()]
    }
}