import abc
import logging
import random

import numpy as np
import torch
import tensorflow as tf
from tensorflow.python.client import device_lib
from torch.autograd import Variable


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, seed, details=False):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.details = details
        self.prediction_details = {}

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """


class PyTorchUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0

    @property
    def device(self):
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')

    def to_var(self, t, **kwargs):
        # ToDo: check whether cuda Variable.
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)


class TensorflowUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            tf.set_random_seed(seed)
        self.framework = 1

    @property
    def device(self):
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        return tf.device(gpus[self.gpu] if gpus and self.gpu is not None else '/cpu:0')


import numpy

    
def slide_window(df, window_length, verbose = 1):
    df = df.values.tolist()
    n = len(df)
    ex = n%window_length
    rows = n/window_length
    df = df[:n-ex]
    B = np.reshape(df, (int(rows),window_length,-1))
    return B
    

# Computes the squared Mahalanobis distance of each data point (row)
# to the center of the distribution, described by cov and mu. 
# (or any other point mu).
# If the parameters cov and mu are left empty, then this function 
# will compute them based on the data X.
def mahalanobis_distance(X, cov=None, mu=None):
    if mu is None:
        mu = numpy.mean(X, axis=0)
    if cov is None:
        cov = numpy.cov(X, rowvar = False)
    try:
        inv_cov = numpy.linalg.inv(cov)
    except numpy.linalg.LinAlgError as err:
        print("Error, probably singular matrix!")
        inv_cov = numpy.eye(cov.shape[0])
    
    X_diff_mu = X - mu
    M = numpy.apply_along_axis(lambda x: 
                    numpy.matmul(numpy.matmul(x, inv_cov), x.T) ,1 , X_diff_mu)
    return M


