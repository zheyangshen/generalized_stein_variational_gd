import jax.numpy as np
import numpy as onp
from jax import random
import jax
from jax import grad, jit, vmap
from jax.config import config

from scipy.cluster.vq import kmeans2

def sample_x(mean, var, key):
    #print(mean.shape)
    #print(var.shape)
    return mean + np.sqrt(var) * random.normal(key, mean.shape), mean, var * np.ones_like(mean)

@jit
def flatten(params):
    tmp = []
    for p in params[0]:
        tmp.extend(p)
    if params[1] is not None:
        tmp.append(params[1])
    tmp = [i.flatten() for i in tmp]
    return np.concatenate(tmp)[:, None]

@jit
def unflatten_like(pf, params):
    pf = pf.squeeze()
    p1 = []
    i = 0
    for p in params[0]:
        tmp = []
        for q in p:
            tmp.append(pf[i: i + q.size].reshape(q.shape))
            i += q.size
        p1.append(tmp)
    return [p1, pf[i:]]

def adagrad():
    ## TODO adagrad for first iteration
    func_dict = {}
    adagrad_init = lambda lr, pf: (lr, 1, np.zeros(pf.shape))
    func_dict['init_grad'] = adagrad_init

    @jit
    def adagrad_update(params_matrix, grad, params):
        eta = params[0]
        epsilon = 1e-6
        hist_grad = params[2]
        n_iter = params[1]
        #if n_iter == 1:
        #    hist_grad = grad ** 2
        hist_grad = 0.9 * hist_grad + 0.1 * np.square(grad)
        adj_grad = grad / (epsilon + np.sqrt(hist_grad))
        params_matrix -= eta * adj_grad
        # gii += np.square(grad)
        return params_matrix, (eta, n_iter+1, hist_grad)
    func_dict['apply_grad'] = adagrad_update
    return func_dict

def adam():
    func_dict = {}
    adam_init = lambda lr, pf: (lr, 1, np.zeros(pf.shape), np.zeros(pf.shape))
    func_dict['init_grad'] = adam_init
    @jit
    def adam_update(params_matrix, grad, params):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        
        eta = params[0]
        n_iter = params[1]
        mt = params[2]
        vt = params[3]
        
        mt = beta_1 * mt + (1-beta_1) * grad
        vt = beta_2 * vt + (1-beta_2) * np.square(grad)
        
        mthat = mt / (1 - np.power(beta_1, n_iter))
        vthat = vt / (1 - np.power(beta_2, n_iter))
        params_matrix -= eta / (np.sqrt(vthat) + epsilon) * mthat
        return params_matrix, (eta, n_iter + 1, mt, vt)
    func_dict['apply_grad'] = adam_update
    return func_dict

def momentum():
    beta = 0.9
    func_dict = {}
    momentum_init = lambda lr, pf: (lr, beta, np.zeros(pf.shape))
    func_dict['init_grad'] = momentum_init
    @jit
    def momentum_update(params_matrix, grad, params):
        beta = params[1]
        mom = beta * params[2] + (1-beta) * grad
        params_matrix -= params[0] * mom
        return params_matrix, (params[0], params[1], mom)
    func_dict['apply_grad'] = momentum_update
    return func_dict

def sgd():
    func_dict = {}
    sgd_init = lambda lr, pf: (lr,)
    func_dict['init_grad'] = sgd_init
    @jit
    def sgd_update(params_matrix, grad, params):
        # params_matrix -= params[0] * grad
        return params_matrix - params[0] * grad, params
    func_dict['apply_grad'] = sgd_update
    return func_dict

def r2_dist(ls, x, z=None):
    x = x / ls
    xs = np.sum(np.square(x), axis=-1, keepdims=True)
    lvl = 1e-10
    if z is None:
        dist = -2 * np.matmul(x, x.T)
        dist += xs + xs.T
        return np.maximum(dist, lvl)

    z = z / ls
    zs = np.sum(np.square(z), axis=-1, keepdims=True)
    dist = -2 * np.matmul(x, z.T)
    dist += xs + zs.T
    return np.maximum(dist, lvl)

def se_kernel_elem(params, x, z):
    return np.exp(params[1]) * np.exp(-np.sum(np.square((x-z) / np.exp(params[0]))) / 2.)

def se_kernel_elem_tau(params, tau):
    return np.exp(params[1]) * np.exp(-np.sum(np.square(tau / np.exp(params[0]))) / 2.)

def imq_kernel_elem(params, x, z):
    return np.power(np.square(params[1]) + np.sum(np.square((x-z) / np.exp(params[0]))), params[2])

def add_kernel(f1, f2):
    return lambda params, x, z: f1(params[0], x, z) + f2(params[1], x, z)

def imq_kernel(params, x, z):
    c = params[1]
    beta = params[2]
    lengthscales = np.exp(params[0])
    return np.power(np.square(c) + r2_dist(lengthscales, x, z), beta)

def polar(x):
    jitter = 1e-9
    w = x ** 2
    y = np.cumsum(w)
        
    k = np.concatenate((np.array([0]), y[:-2]), axis=0)
    # print(k)
    norm = np.sqrt(y[-1] + jitter)
    z = x[:-1] / (np.sqrt(y[-1]-k + jitter))
    # print(z)
    ang = np.arccos(z) * np.sign(x[1:])
    return norm, ang

def polar_kernel_elem(params, x, z):
    nx, tx = polar(x)
    nz, tz = polar(z)
    taut = np.minimum(np.abs(tx-tz), 2*np.pi - np.abs(tx-tz))
    return se_kernel_elem(params[0], nx, nz) * se_kernel_elem_tau(params[1], taut)

class bbData(object):
    def __init__(self, d, minibatch_size):
        self.d = d
        self.minibatch_size = min(d.X_train.shape[0], minibatch_size)
        self.data_iter = 0
        self.N = d.X_train.shape[0]

    def get_minibatch(self):
        assert self.d.X_train.shape[0] >= self.minibatch_size
        if self.d.X_train.shape[0] == self.minibatch_size:
            shuffle = onp.random.permutation(self.N)
            return self.d.X_train[shuffle, :], self.d.Y_train[shuffle, :]

        if self.d.X_train.shape[0] < self.data_iter + self.minibatch_size:
            shuffle = onp.random.permutation(self.N)
            self.d.X_train = self.d.X_train[shuffle, :]
            self.d.Y_train = self.d.Y_train[shuffle, :]
            self.data_iter = 0
        X_batch = self.d.X_train[self.data_iter:self.data_iter + self.minibatch_size, :]
        Y_batch = self.d.Y_train[self.data_iter:self.data_iter + self.minibatch_size, :]
        self.data_iter += self.minibatch_size
        return X_batch, Y_batch

    def get_minibatch_particles(self, p, random=True):
        if random:
            a = [self.get_minibatch() for _ in range(p)]
            return np.stack([x[0] for x in a]), np.stack([x[1] for x in a])
        x, y = self.get_minibatch()
        return np.stack([x for _ in range(p)]), np.stack([y for _ in range(p)])

import tensorflow_datasets as tfds
def tfds_wrapper(dataset_name):
    ds = tfds.load(dataset_name, split=['train', 'test'], batch_size=-1)
    X_train_img = tfds.as_numpy(ds[0]['image'])
    X_train = X_train_img.reshape((X_train_img.shape[0], -1))
    X_test_img = tfds.as_numpy(ds[1]['image'])
    X_test = X_test_img.reshape((X_test_img.shape[0], -1))
    y_train = tfds.as_numpy(ds[0]['label']).reshape(-1, 1)
    y_test = tfds.as_numpy(ds[1]['label']).reshape(-1, 1)
    nc = len(set(y_train.squeeze()))
    x = np.float32(np.concatenate([X_train, X_test], axis=0))
    x_mean = np.mean(x, 0)
    x_std = np.std(x, 0) + 1e-4
    normalize = lambda x, m, s: (x-m)/s
    x_train = normalize(X_train, x_mean, x_std)
    x_test = normalize(X_test, x_mean, x_std)
    d = x_mean.size
    n = x_train.shape[0] + x_test.shape[0]
    w = X_train_img.shape[1]
    c = X_train_img.shape[-1]
    class temp_ds:
        X_train = x_train
        Y_train = y_train
        X_test = x_test
        Y_test = y_test
        X_mean = x_mean
        X_std = x_std
        num_classes = nc
        D = d
        N = n
        width = w
        channel = c
    return temp_ds()
