import jax.numpy as np
from jax import random
import jax
from jax import grad, jit, vmap, pmap
from jax.config import config

from scipy.cluster.vq import kmeans2

from functools import partial
import numpy as onp
from utils import *
from jax.flatten_util import ravel_pytree

from jax.tree_util import tree_map, tree_flatten, tree_transpose, tree_structure
import jax.linear_util as lu
from jax.scipy.special import logsumexp
import numpy as onp
from stein import * 
from utils import * 
from jax.experimental import stax

from inference import Inference
#from data_utils import create_classification_dataset, create_regression_dataset
from utils import *

from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax, Sigmoid, Softmax, MaxPool)

config.update("jax_enable_x64", True)

def gaussian_lik():
    def init_fun(rng, input_shape):
        loggamma = np.ones((input_shape[-1], )) * np.log(10.)
        return input_shape, (loggamma)

    def apply_fun(params, inputs, y, **kwargs):
        loggamma = params
        var = np.exp(-params)
        return (-0.5 * (np.log(2 * np.pi) - loggamma) - 0.5 * np.exp(loggamma) * np.square(inputs - y)).squeeze()
    return init_fun, apply_fun

def sigmoid_to_binary():
	def init_fun(rng, input_shape):
		return (input_shape([0]), 2), ()

	def apply_fun(params, inputs, **kwargs):
		return np.concatenate([np.log(inputs), np.log(1-inputs)], axis=1)
Gaussian_likelihood = gaussian_lik()

def reshape(width=28, channel=1):
	def init_fun(rng, input_shape):
		return (-1, int(input_shape[1] / (width*channel)), width, channel), ()

	def apply_fun(params, inputs, **kwargs):
		return inputs.reshape((-1, int(inputs.shape[1] / (width*channel)), width, channel))

	return init_fun, apply_fun

def bnn_stax(func_dict, stax_model):
    a0 = 1.0
    b0 = 1.0
    
    stax_init, stax_apply = stax.serial(*stax_model)
    input_shape = (-1, func_dict['data'].d.X_train.shape[1])

    def init(key):
        return [stax_init(key, input_shape)[1], np.log(onp.random.gamma(a0, 1/b0))]

    def init_particles(n_particles, key):
        keys = random.split(key, n_particles)
        pf = []
        for k in keys:
            p = init(k)
            pf.append(ravel_pytree(p)[0])
        func_dict['unf'] = ravel_pytree(p)[1]
        pf = np.array(pf).reshape((n_particles, -1))
        unf = func_dict['unf']
        return pf, unf

    def prior(param):
        params, loglambda = param
        prior_gamma = (a0) * loglambda - np.exp(loglambda) / b0
        if func_dict['lik_type'] == 'gaussian':
            loggamma = params[-1]
            prior_gamma += (a0) * loggamma - np.exp(loggamma) / b0
            params = params[:-1]

        prior_layers = np.sum(np.array([np.sum(-0.5 * (np.log(2 * np.pi) - loglambda) - 0.5 * np.exp(loglambda) * np.square(k)) for w in params for k in w]))
        return np.squeeze(prior_layers + prior_gamma)

    def variational_objective(param, x, y, N):
        n_batch = y.shape[0] + 0.0
        params = func_dict['unf'](param)
        logliks = stax_apply(params[0], x, y=y)
        return prior(params) + N/n_batch * np.sum(logliks)

    grads_vo = jit(vmap(grad(variational_objective), in_axes=(0, 0, 0, None)))
    func_dict['stax_apply'] = stax_apply

    def grads_wrapper(pf):
        x, y = func_dict['data'].get_minibatch_particles(pf.shape[0])
        N = func_dict['data'].d.X_train.shape[0] + 0.0
        return grads_vo(pf, x, y, N)
    
    func_dict['grads'] = grads_wrapper
    func_dict['init_particles'] = init_particles
    return func_dict

def bnn_stax_dict(stax_model, func_dict, optim='sgd', stein_kernel="se", init="bm", activation="relu"):
    if optim == 'sgd':
        func_dict.update(sgd())
    if optim == 'adam':
        func_dict.update(adam())
    if optim == 'momentum':
        func_dict.update(momentum())
    if optim == 'adagrad':
        func_dict.update(adagrad())
    func_dict['stein_dict'] = kernel_elem(stein_kernel, init)
    func_dict.update(bnn_stax(func_dict, stax_model))
    func_dict.update(stein_funcdict(func_dict))
    return func_dict


width = 50

from jax.nn.initializers import glorot_normal, normal, ones, zeros

class BnnModel_stax_regression(Inference):
    def __init__(self, dataset, n_layers=2, optim='sgd', stein_kernel="se", init="med", activation="relu", fold=0, minibatch_size=500, valid=False):
        stax_model = []
        data = dataset
        # data = bbData(create_regression_dataset(dataset, fold=fold, valid=valid), minibatch_size)
        for _ in range(n_layers-1):
            stax_model.append(Dense(50, b_init=zeros))
            if activation == "relu":
                stax_model.append(Relu)
            else:
                stax_model.append(Sigmoid)
        stax_model.append(Dense(data.d.Y_train.shape[-1]))
        stax_model_1 = stax_model.copy()
        stax_model.append(Gaussian_likelihood)

        _, self.stax_model_1 = stax.serial(*stax_model_1)
        self.likelihood = stax_model[-1][1]
        func_dict = {}
        func_dict['data'] = data
        func_dict['lik_type'] = 'gaussian'
        func_dict = bnn_stax_dict(stax_model, func_dict, optim=optim, stein_kernel=stein_kernel, init=init)
        super(BnnModel_stax_regression, self).__init__(func_dict)
        self.data = data
        
    def logging_init(self):
        self.test_trace = []
        #self.sigma_oparams = self.func_dict['sigma_optim']['init_grad'](self.func_dict['sigma_lr'], self.func_dict['logitsigma'])

    def logging(self, i):
        fd = self.func_dict
        preds = vmap(lambda p, x: fd['batch_predict'](fd['unf'](p), x), in_axes=(0, None))
        xt = self.data.d.X_test
        yt = self.data.d.Y_test
    
        def compute_m_loglik(pf, x, y):
            params = fd['unf'](pf)[0]
            #logliks = fd['stax_apply'](params, x, y=y)
            m = self.stax_model_1(params[:-1], x)
            logliks = self.likelihood(params[-1], m, y=y)
            return m, logliks

        vloglik = vmap(compute_m_loglik, in_axes=(0, None, None))
        ms, logl = vloglik(self.param_list[0], xt, yt)
        loglik = np.mean(logsumexp(logl - np.log(self.data.d.Y_std) + np.log(1/self.param_list[0].shape[0]), axis=0))

        mse = np.mean((yt[None, :, :] - ms)**2)
        self.test_trace.append((mse, loglik))