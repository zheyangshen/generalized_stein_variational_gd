import jax.numpy as np
from jax import random
import jax
from jax import grad, jit, vmap, pmap
from jax.config import config

from scipy.cluster.vq import kmeans2

from functools import partial
from utils import *
import numpy as onp
from jax.flatten_util import ravel_pytree

from jax.tree_util import tree_map, tree_flatten, tree_transpose, tree_structure
import jax.linear_util as lu
from jax.scipy.special import logsumexp

config.update("jax_enable_x64", True)
from stein import stein_funcdict, kernel_elem

class Inference(object):
    def __init__(self, func_dict):
        self.func_dict = func_dict

    def logging_init(self):
        pass

    def logging(self, **kwargs):
        pass

    def stein_training(self, n_iter, n_particles, key, param_list=None, lr=1e-2, window=1000, trace=True, method=('LD', 'stein'), sgd=True, shuffle=False, split=True):
        self.logging_init()
        if method[0] == 'LD' or method[0] == 'RLD':# or method == 'LD_blob':
            ind = [[0]]
            scale = [1.0]
            ev = [True]
        if method[0] == 'HMC' or method[0] == 'RHMC':
            ind = [[0, 1]]
            scale = [1.0]
            ev = [True]
            if split:
                ind = [[0], [1], [0]]
                scale = [0.5, 1.0, 0.5]
                ev = [False, True, False]
        if method[0] == 'NHT' or method[0] == 'NHT_2':
            ind = [[0, 1, 2]]
            scale = [1.0]
            ev = [True]
            if split:
                ind = [[0], [2], [1], [2], [0]]
                scale = [0.5, 0.5, 1.0, 0.5, 0.5]
                ev = [False, False, True, False, False]
        if method[0] == 'LD_2':
            ind = [[0], [2], [1], [2], [0]]
            ev = [False, False, True, False, False]
            scale = [0.5, 0.5, 1.0, 0.5, 0.5]
        
        func_dict = self.func_dict
        func_dict.update(func_dict['_'.join(method)])

        k1, k2 = random.split(key)
        keys = random.split(k2, n_iter)
        
        k3, k4, k5 = random.split(k1, 3)
        pf = self.func_dict['init_particles'](n_particles, k3)
        
        func_dict['unf'] = pf[1]
        pf = pf[0]
        if param_list is None:
            param_list = func_dict['init'](pf, key=k4)
        stein_param = self.func_dict['stein_dict']['init'](np.concatenate(param_list, axis=-1), k=k5, step_size=lr)

        self.stein_param = stein_param
        optim_params = [func_dict['init_grad'](lr, p) for p in param_list]
        traces = []
        traces_sample = []
        traces.append([p.copy() for p in param_list])
        norms = []

        for i, k in enumerate(keys):
            k1, k2, k3 = random.split(k, 3)
            if sgd is True:
                step_size = lr / (i+1) ** 0.5
            else:
                step_size = lr
            func = method[1] + '_1'
            ks = random.split(k1, len(ind))
            for j, key in enumerate(ks):
                idx = ind[j]
                ss = step_size * scale[j]
                gs2 = np.zeros_like(param_list[0])
                if ev[j]:
                    gs2 = func_dict['grads'](param_list[0])
                    k1, k2, k3 = random.split(key, 3)
                    key, _ = random.split(k3, 2)

                if method[1] == 'diffusion':
                    grads_1 = func_dict[func](gs2, param_list, key, step_size=ss)#, obj=obj_value)
                else:
                    grads_1 = func_dict[func](gs2, param_list, stein_param)
                optim_params = [(ss, *op[1:]) for op in optim_params]
                for index in idx:
                    param_list[index], optim_params[index] = func_dict['apply_grad'](param_list[index], grads_1[index], optim_params[index])

            n = np.mean(np.sqrt(np.sum(grads_1[0] ** 2, axis=1)))
            norms.append(n)
            self.param_list = param_list
            # if n < 0.015:
            traces.append([p.copy() for p in param_list])
            if (i+1) % window == 0:
                #traces.append([p.copy() for p in param_list])
                key1, key2 = random.split(k2)
                
                if shuffle:
                    if len(param_list) == 1:
                        param_list[0] = random.normal(key1, param_list[0].shape)
                    if len(param_list) > 1:
                        param_list[1] = random.normal(key1, param_list[1].shape) / np.sqrt(func_dict['arg_dict']['invsigma'])
                    if len(param_list) > 2:
                        param_list[2] = np.ones_like(param_list[2]) * func_dict['a']
                    traces_sample.append([p.copy() for p in param_list])
                self.logging(i=i)

        self.pf = pf
        self.param_list = param_list
        self.optim_params = optim_params
        self.traces = traces
        self.stein_param = stein_param
        self.norms = norms
        self.traces_sample = traces_sample