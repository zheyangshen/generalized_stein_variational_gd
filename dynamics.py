import jax.numpy as np
from jax import random
import jax
from jax import grad, jit, vmap, pmap
from jax.config import config

from scipy.cluster.vq import kmeans2

from stein import *

from functools import partial
from utils import *
from functools import partial

random_init = True

def LD_diffusion(ibeta=1.0, **kwargs):
    def init(pf, **kwargs):
        return [pf]

    def AClogpi(param_list, grad, **kwargs):
        return ibeta * grad

    def ACnabla(param_list, **kwargs):
        return 0.

    def diffusion(param_list, key):
        return random.normal(key, param_list[0].shape) * np.sqrt(2*ibeta)

    def split(pf_concat):
        return [pf_concat]

    func_dict = {}
    func_dict['init'] = init
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['diffusion'] = diffusion
    return func_dict

def LD_stein(**kwargs):

    def init(pf, **kwargs):
        return [pf]

    def A(param_list, dxkxy, **kwargs):
        return dxkxy

    def C(param_list, dxkxy, **kwargs):
        return 0.

    def AClogpi(param_list, grad, **kwargs):
        return grad

    def ACnabla(param_list, **kwargs):
        return 0.

    def split(pf_concat):
        return [pf_concat]

    func_dict = {}
    func_dict['init'] = init
    func_dict['A'] = A
    func_dict['C'] = C
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    return func_dict

def LD_2_stein(invsigma=1.0, gamma=1.0, xi=0.1, **kwargs):

    def init(pf, key, **kwargs):
        k1, k2 = random.split(key)
        return [pf, random.normal(k1, pf.shape) / np.sqrt(invsigma), random.normal(k2, pf.shape) / np.sqrt(invsigma)]

    def A(param_list, dxkxy, **kwargs):
        pf, pr, pu = param_list
        D = pf.shape[1]
        dr = xi * dxkxy[:, :, (2*D):]
        return np.concatenate([np.zeros_like(dr), np.zeros_like(dr), xi * dr], -1)

    def C(param_list, dxkxy, **kwargs):
        pf, pr, pu = param_list
        D = pf.shape[1]
        df = -dxkxy[:, :, D:(2*D)]
        dr = dxkxy[:, :, :D] - gamma * dxkxy[:, :, (2*D):]
        du = gamma * dxkxy[:, :, D:(2*D)]
        return np.concatenate([df, dr, du], -1)

    def AClogpi(param_list, grad, **kwargs):
        pf, pr, pu = param_list
        return np.concatenate([invsigma * pr, grad + gamma * invsigma * pu, -gamma * invsigma * pr], -1)

    def ACnabla(param_list, **kwargs):
        return 0.

    def split(pf_concat):
        D = pf_concat.shape[1] // 3
        return [pf_concat[:, :D], pf_concat[:, D:(2*D)], pf_concat[:, -D:]]

    func_dict = {}
    func_dict['init'] = init
    func_dict['A'] = A
    func_dict['C'] = C
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    return func_dict

def HMC_diffusion(a=0.1, invsigma=300, random_init=random_init, **kwargs):
    def init(pf, key, a=a, **kwargs):
        if random_init:
            return [pf, random.normal(key, pf.shape) / np.sqrt(invsigma)]
        return [pf, np.zeros_like(pf)]

    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr = param_list
        return np.concatenate([invsigma * pr, grad - a * invsigma * pr], -1)

    def ACnabla(param_list, **kwargs):
        return 0

    def split(pf_concat):
        D = pf_concat.shape[1] // 2
        return [pf_concat[:, :D], pf_concat[:, D:]]

    def diffusion(param_list, key):
        pf, pr = param_list
        return np.concatenate([np.zeros_like(pf), random.normal(key, pr.shape) * np.sqrt(2*a)], -1)

    func_dict = {}
    func_dict['init'] = init
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['diffusion'] = diffusion
    func_dict['a'] = a
    return func_dict

def HMC_stein(a=1.0, invsigma=300, random_init=random_init, mult=1.0, **kwargs):

    def init(pf, key, a=a, **kwargs):
        #
        if random_init:
            return [pf, random.normal(key, pf.shape) / np.sqrt(invsigma)]
        return [pf, np.zeros_like(pf)]

    def A(param_list, dxkxy, a=a, **kwargs):
        pf, pr = param_list
        D = pf.shape[1]
        dr = a * dxkxy[:, :, D:]
        return np.concatenate([np.zeros_like(dr), dr], -1)

    def C(param_list, dxkxy, **kwargs):
        pf, pr = param_list
        D = pf.shape[1]
        df = -dxkxy[:, :, D:]
        dr = dxkxy[:, :, :D]
        return np.concatenate([df, dr], -1) * mult

    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr = param_list
        return np.concatenate([invsigma * pr * mult, grad * mult - a * invsigma * pr], -1)

    def ACnabla(param_list, **kwargs):
        return 0

    def split(pf_concat):
        D = pf_concat.shape[1] // 2
        return [pf_concat[:, :D], pf_concat[:, D:]]

    func_dict = {}
    func_dict['init'] = init
    func_dict['A'] = A
    func_dict['C'] = C
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['a'] = a
    return func_dict

def blob(K, dxkxy):
    t1 = dxkxy / np.sum(K, axis=1)[:, None, None]
    t2 = dxkxy / np.sum(K, axis=1)[None, :, None]
    return t1 + t2

def LD_blob(**kwargs):

    def init(pf, **kwargs):
        return [pf]

    def AClogpi(param_list, grad, **kwargs):
        return grad

    def ACnabla(param_list, **kwargs):
        return 0.

    def AClogmu(param_list, logmu):
        return logmu

    def split(pf_concat):
        return [pf_concat]

    func_dict = {}
    func_dict['init'] = init
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['AClogmu'] = AClogmu
    return func_dict

def HMC_blob(a=10.0, invsigma=300, random_init=random_init, **kwargs):

    def init(pf, key, a=a, **kwargs):
        #
        if random_init:
            return [pf, random.normal(key, pf.shape) / np.sqrt(invsigma)]
        return [pf, np.zeros_like(pf)]
        
    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr = param_list
        return np.concatenate([pr, grad - a * invsigma * pr], -1)

    def ACnabla(param_list, **kwargs):
        return 0

    def split(pf_concat):
        D = pf_concat.shape[1] // 2
        return [pf_concat[:, :D], pf_concat[:, D:]]

    vsplit = vmap(split, in_axes=0)
    def AClogmu(param_list, logmu, a=a):
        lmf, lmr = vsplit(logmu)
        return np.concatenate([-lmr, lmf + a * lmr], -1)

    func_dict = {}
    func_dict['init'] = init
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['AClogmu'] = AClogmu
    func_dict['a'] = a
    return func_dict

def RHMC_blob(var_obj, a=1.0, invsigma=1.0, random_init=random_init, d=1.0, c=0.5, **kwargs):
    objective = vmap(var_obj, in_axes=0)
    def init(pf, key, a=a, **kwargs):
        #
        if random_init:
            return [pf, random.normal(key, pf.shape) / np.sqrt(invsigma)]
        return [pf, np.zeros_like(pf)]
        
    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr = param_list
        U = -objective(pf)
        G_inv = np.expand_dims(d * np.sqrt(U + c), -1)
        G_invsqrt = np.sqrt(G_inv)
        pf, pr = param_list
        return np.concatenate([G_invsqrt * invsigma * pr, G_invsqrt * grad - a * G_inv * invsigma * pr], -1)

    def ACnabla(param_list, **kwargs):
        return 0.

    def split(pf_concat):
        D = pf_concat.shape[1] // 2
        return [pf_concat[:, :D], pf_concat[:, D:]]

    vsplit = vmap(split, in_axes=0)
    def AClogmu(param_list, logmu, a=a):
        lmf, lmr = vsplit(logmu)
        pf, pr = param_list
        U = -objective(pf)
        G_inv = np.expand_dims(d * np.sqrt(U + c), -1)
        G_invsqrt = np.sqrt(G_inv)
        return np.concatenate([-G_invsqrt * lmr, G_invsqrt * lmf + a * G_inv * lmr], -1)

    func_dict = {}
    func_dict['init'] = init
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['AClogmu'] = AClogmu
    func_dict['a'] = a
    return func_dict

def RHMC_stein(var_obj, a=1.0, invsigma=1.0, d=1.0, c=0.5):

    grads_vo = jit(vmap(grad(lambda x: np.sqrt(d * np.sqrt(-var_obj(x) + c)))))
    objective = vmap(var_obj, in_axes=0)
    def init(pf, key, d=1.5, c=0.5, invsigma=1.0, **kwargs):
        # return [pf, np.zeros_like(pf)]
        return [pf, random.normal(key, pf.shape) / np.sqrt(invsigma)]

    def A(param_list, dxkxy, obj, d=1.5, c=0.5, **kwargs):
        # G_inv = d * np.sqrt(np.abs(-obj + c)) # N
        pf, pr = param_list
        U = -objective(pf)
        G_inv = d * np.sqrt(U + c)
        D = pf.shape[1]
        dr = a * G_inv[None, :, None] * dxkxy[:, :, D:]
        return np.concatenate([np.zeros_like(dr), dr], -1)

    def C(param_list, dxkxy, obj, d=1.5, c=0.5, **kwargs):
        pf, pr = param_list
        U = -objective(pf)
        G_invsqrt = np.sqrt(d * np.sqrt(U + c))
        D = pf.shape[1]
        df = -G_invsqrt[None, :, None] * dxkxy[:, :, D:]
        dr = G_invsqrt[None, :, None] * dxkxy[:, :, :D]
        return np.concatenate([df, dr], -1)

    def AClogpi(param_list, grad, obj, d=1.5, c=0.1, **kwargs):
        pf, pr = param_list
        U = -objective(pf)
        G_inv = np.expand_dims(d * np.sqrt(U + c), -1)
        G_invsqrt = np.sqrt(G_inv)

        pf, pr = param_list
        return np.concatenate([G_invsqrt * invsigma * pr, G_invsqrt * grad - a * G_inv * invsigma * pr], -1)

    def ACnabla(param_list, obj, grad, d=1.5, c=0.5, **kwargs):
        pf, pr = param_list
        dr = grads_vo(pf)
        return np.concatenate([np.zeros_like(pf), dr], -1)

    def split(pf_concat):
        D = pf_concat.shape[1] // 2
        return [pf_concat[:, :D], pf_concat[:, D:]]

    func_dict = {}
    func_dict['init'] = init
    func_dict['A'] = A
    func_dict['C'] = C
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    return func_dict

def RLD_stein(var_obj, d=1.0, c=0.5, **kwargs):
    grads_vo = jit(vmap(grad(lambda x: d * np.sqrt(-var_obj(x) + c))))
    objective = vmap(var_obj, in_axes=0)
    
    def init(pf, **kwargs):
        return [pf]

    def A(param_list, dxkxy, **kwargs):
        pf = param_list[0]
        U = -objective(pf)
        G_inv = d * np.sqrt(U + c)
        return G_inv[None, :, None] * dxkxy

    def C(param_list, dxkxy, **kwargs):
        return 0.

    def AClogpi(param_list, grad, **kwargs):
        pf = param_list[0]
        U = -objective(pf)
        G_inv = np.expand_dims(d * np.sqrt(U + c), -1)
        return G_inv * grad

    def ACnabla(param_list, **kwargs):
        pf = param_list[0]
        return grads_vo(pf)

    def split(pf_concat):
        return [pf_concat]

    func_dict = {}
    func_dict['init'] = init
    func_dict['A'] = A
    func_dict['C'] = C
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    return func_dict

def NHT_stein(a=10.0, **kwargs):

    def init(pf, key, a=a, **kwargs):
        k1, k2 = random.split(key)
        D = pf.shape[1]
        
        if random_init:
            return [pf, random.normal(k1, pf.shape), random.normal(k2, (pf.shape[0], 1)) / np.sqrt(D) + a]
        return [pf, np.zeros_like(pf), np.ones((pf.shape[0], 1)) * a]
        

    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr, pxi = param_list
        D = pf.shape[1]
        return np.concatenate([pr, grad - pxi * pr, np.sum(pr ** 2, axis=1)[:, None] / D], -1)

    def ACnabla(param_list, a=a, **kwargs):
        pf, pr, pxi = param_list
        N, D = pf.shape

        return np.concatenate((np.zeros((N, 2*D)), -np.ones((N,1))), axis=-1)

    def A(param_list, dxkxy, a=a, **kwargs):
        pf, pr, pxi = param_list
        N, D = pf.shape
        dr = a * dxkxy[:, :, D:-1]

        return np.concatenate([np.zeros_like(dr), dr, np.zeros((N, N, 1))], -1)

    def C(param_list, dxkxy, **kwargs):
        pf, pr, pxi = param_list
        D = pf.shape[1]
        df = -dxkxy[:, :, D:-1]
        dr = dxkxy[:, :, :D] + pr[None, :, :] / D * dxkxy[:, :, -1:]
        dxi = np.sum(-pr[None, :, :] / D * dxkxy[:, :, D:-1], axis=-1)[:, :, None]
        return np.concatenate([df, dr, dxi], -1)

    def split(pf_concat):
        D = pf_concat.shape[1] // 2
        return [pf_concat[:, :D], pf_concat[:, D:-1], pf_concat[:, -1:]]

    func_dict = {}
    func_dict['init'] = init
    func_dict['A'] = A
    func_dict['C'] = C
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['a'] = a
    return func_dict

def NHT_2_stein(mu=1.0, a=1.0, invsigma=300, **kwargs):
    def init(pf, key, a=a, **kwargs):
        k1, k2 = random.split(key)
        D = pf.shape[1]
        if random_init:
            return [pf, random.normal(k1, pf.shape) / np.sqrt(invsigma), np.ones_like(pf) * a]
        return [pf, np.zeros_like(pf), np.ones_like(pf) * a]
        
    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr, pxi = param_list
        D = pf.shape[1]
        return np.concatenate([invsigma * pr, grad - pxi * invsigma * pr, invsigma ** 2 * (pr ** 2) / mu], -1)

    def ACnabla(param_list, a=a, **kwargs):
        pf, pr, pxi = param_list
        N, D = pf.shape

        return np.concatenate((np.zeros((N, 2*D)), -invsigma * np.ones((N, D)) / mu), axis=-1)

    def A(param_list, dxkxy, a=a, **kwargs):
        pf, pr, pxi = param_list
        N, D = pf.shape
        dr = a * dxkxy[:, :, D:(2*D)]

        return np.concatenate([np.zeros_like(dr), dr, np.zeros((N, N, D))], -1)

    def C(param_list, dxkxy, **kwargs):
        pf, pr, pxi = param_list
        D = pf.shape[1]
        df = -dxkxy[:, :, D:(2*D)]
        dr = dxkxy[:, :, :D] + invsigma * np.expand_dims(pr, 0) / mu * dxkxy[:, :, -D:]
        #dxi = np.sum(-pr[None, :, :] / mu * dxkxy[:, :, D:(2*D)], axis=-1)[:, :, None]
        dxi = invsigma/mu * (-np.expand_dims(pr, 0)) * dxkxy[:, :, D:(2*D)]
        return np.concatenate([df, dr, dxi], -1)

    def split(pf_concat):
        D = pf_concat.shape[1] // 3
        return [pf_concat[:, :D], pf_concat[:, D:(2*D)], pf_concat[:, -D:]]

    func_dict = {}
    func_dict['init'] = init
    func_dict['A'] = A
    func_dict['C'] = C
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['a'] = a
    func_dict['mu'] = mu
    return func_dict

def NHT_2_blob(mu=1.0, a=1.0, invsigma=300, **kwargs):
    def init(pf, key, a=a, **kwargs):
        k1, k2 = random.split(key)
        D = pf.shape[1]
        if random_init:
            return [pf, random.normal(k1, pf.shape) / np.sqrt(invsigma), np.ones_like(pf) * a]
        return [pf, np.zeros_like(pf), np.ones_like(pf) * a]
        
    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr, pxi = param_list
        D = pf.shape[1]
        return np.concatenate([invsigma * pr, grad - pxi * invsigma * pr, invsigma ** 2 * (pr ** 2) / mu], -1)

    def ACnabla(param_list, **kwargs):
        return 0.

    def split(pf_concat):
        D = pf_concat.shape[1] // 3
        return [pf_concat[:, :D], pf_concat[:, D:(2*D)], pf_concat[:, -D:]]

    vsplit = vmap(split, in_axes=0)
    def AClogmu(param_list, logmu, a=a):
        lmf, lmr, lmxi = vsplit(logmu)
        pf, pr, pxi = param_list
        return np.concatenate([-lmr, lmf + a * lmr + invsigma * pr / mu * lmxi,  -invsigma * pr / mu * lmr], -1)

    func_dict = {}
    func_dict['init'] = init
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['AClogmu'] = AClogmu
    func_dict['a'] = a
    return func_dict

def NHT_2_diffusion(mu=10.0, a=0.1, **kwargs):

    def init(pf, key, a=a, **kwargs):
        k1, k2 = random.split(key)
        D = pf.shape[1]
        if random_init:
            return [pf, random.normal(k1, pf.shape), random.normal(k2, pf.shape)*np.sqrt(mu) + a]
        return [pf, np.zeros_like(pf), np.ones_like(pf) * a]
        #return [pf, random.normal(k1, pf.shape), random.normal(k2, pf.shape) + a]

    def AClogpi(param_list, grad, a=a, **kwargs):
        pf, pr, pxi = param_list
        D = pf.shape[1]
        return np.concatenate([pr, grad - pxi * pr, (pr ** 2) / mu], -1)

    def ACnabla(param_list, a=a, **kwargs):
        pf, pr, pxi = param_list
        N, D = pf.shape
        return np.concatenate((np.zeros((N, 2*D)), -np.ones((N, D)) / mu), axis=-1)
    
    def diffusion(param_list, key, step_size):
        pf, pr, pxi = param_list
        N, D = pf.shape
        return np.concatenate((np.zeros_like(pf), random.normal(key, pr.shape) * np.sqrt(2*a), np.zeros_like(pxi)), -1)

    def split(pf_concat):
        D = pf_concat.shape[1] // 3
        return [pf_concat[:, :D], pf_concat[:, D:(2*D)], pf_concat[:, -D:]]

    func_dict = {}
    func_dict['init'] = init
    func_dict['AClogpi'] = AClogpi
    func_dict['ACnabla'] = ACnabla
    func_dict['split'] = split
    func_dict['diffusion'] = diffusion
    func_dict['a'] = a
    func_dict['mu'] = mu
    return func_dict
