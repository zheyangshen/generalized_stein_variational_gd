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

def toydict(func_dict, dim, x):
    def variational_objective(z):
        ux = lambda x, y, z: -1 * (np.power(x[0], 4) / 10.0 + (4 * (z * x[1] + y) - x[0] ** 2) ** 2 / 2.)
        return logsumexp(np.array([ux(z, 0.0, 1.0), ux(z, -2.0, 1.0), ux(z, 2.0, 1.0)]) - np.log(3.0))

    def init_particles(n_particles, k):
        a = random.normal(k, (n_particles, dim)) * 0.1
        return a, lambda x: x

    grads_vo = jit(vmap(grad(variational_objective)))

    func_dict['grads'] = grads_vo 
    
    func_dict['init_particles'] = init_particles
    func_dict['obj'] = variational_objective

    return func_dict

def toy_funcdict(dim, x=0.0, optim='sgd', stein_kernel="se", init="bm"):
    func_dict = {}
    if optim == 'sgd':
        func_dict.update(sgd())
    if optim == 'adam':
        func_dict.update(adam())
    if optim == 'momentum':
        func_dict.update(momentum())
    if optim == 'adagrad':
        func_dict.update(adagrad())
    func_dict['stein_dict'] = kernel_elem(stein_kernel, init)
    func_dict.update(toydict(func_dict, dim=dim, x=x))
    func_dict.update(stein_funcdict(func_dict))
    return func_dict

from inference import Inference
class ToyModel(Inference):
    def logging(self, **kwargs):
        self.traces.append(self.param_list[0].copy())

    def logging_init(self):
        self.traces = []