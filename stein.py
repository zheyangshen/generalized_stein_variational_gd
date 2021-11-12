import jax.numpy as np
from jax import random
import jax
from jax import grad, jit, vmap, pmap
from jax.config import config

from scipy.cluster.vq import kmeans2

from functools import partial
import numpy as onp
from jax.flatten_util import ravel_pytree
from dynamics import *
from utils import *

def stein_funcdict(func_dict):
    @jit
    def stein_1(gs, param_list, stein_param, obj=None):
        N = gs.shape[0]
        pf_concat = np.concatenate(param_list, axis=-1)
        # Kp = func_dict['stein_dict']['kern'](stein_param, pf_concat, pf_concat)
        # dxkxy = func_dict['stein_dict']['grad_unfold'](stein_param, pf_concat, pf_concat)
        Kp, dxkxy = func_dict['stein_dict']['kern_kg'](stein_param, pf_concat, pf_concat)

        AClogpi = func_dict['AClogpi'](param_list, gs, obj=obj)
        ACnabla = func_dict['ACnabla'](param_list, obj=obj, grad=gs)
        A = func_dict['A'](param_list, dxkxy, obj=obj)
        C = func_dict['C'](param_list, dxkxy, obj=obj)

        t1 = Kp @ (AClogpi + ACnabla)
        t2 = A + C
        return func_dict['split'](-(t1 + np.sum(t2, axis=1)) / N)

    @jit
    def stein_2(gs, param_list, stein_param):
        N = gs.shape[0]
        pf_concat = np.concatenate(param_list, axis=-1)
        #Kp = func_dict['stein_dict']['kern'](stein_param, pf_concat, pf_concat)
        #dxkxy = func_dict['stein_dict']['grad_unfold'](stein_param, pf_concat, pf_concat)
        Kp, dxkxy = func_dict['stein_dict']['kern_kg'](stein_param, pf_concat, pf_concat)

        AClogpi = func_dict['AClogpi'](param_list, gs)
        ACnabla = func_dict['ACnabla'](param_list)
        A = func_dict['A'](param_list, dxkxy)
        # C = func_dict[method]['C'](param_list, dxkxy)

        t1 = Kp @ (AClogpi + ACnabla)
        t2 = A #+ C
        return func_dict['split'](-(t1 + np.sum(t2, axis=1)) / N)

    @jit
    def blob_1(gs, param_list, stein_param):
        N = gs.shape[0]
        pf_concat = np.concatenate(param_list, axis=-1)
        Kp, dxkxy = func_dict['stein_dict']['kern_kg'](stein_param, pf_concat, pf_concat)

        AClogpi = func_dict['AClogpi'](param_list, gs)
        logmu = blob(Kp, dxkxy)
        AClogmu = np.sum(func_dict['AClogmu'](param_list, logmu), axis=1)
        t1 = AClogpi
        t2 = AClogmu
        return func_dict['split'](-(t1 + t2))

    @jit
    def diffusion_1(gs, param_list, key, step_size):
        N = gs.shape[0]
        pf_concat = np.concatenate(param_list, axis=-1)
        
        AClogpi = func_dict['AClogpi'](param_list, gs)
        ACnabla = func_dict['ACnabla'](param_list)
        dif = func_dict['diffusion'](param_list, key) / np.sqrt(step_size)
        t1 = AClogpi + ACnabla
        
        return func_dict['split'](-(t1 + dif))

    func_dict['stein_1'] = stein_1
    func_dict['stein_2'] = stein_2
    func_dict['blob_1'] = blob_1
    func_dict['diffusion_1'] = diffusion_1
    func_dict['LD_stein'] = LD_stein()
    func_dict['LD_2_stein'] = LD_2_stein()
    func_dict['HMC_stein'] = HMC_stein()
    func_dict['HMC_diffusion'] = HMC_diffusion()
    func_dict['NHT_stein'] = NHT_stein()
    func_dict['NHT_2_stein'] = NHT_2_stein()
    func_dict['LD_blob'] = LD_blob()
    func_dict['HMC_blob'] = HMC_blob()
    func_dict['LD_diffusion'] = LD_diffusion()
    #func_dict['LD_2_diffusion'] = LD_2_diffusion()
    func_dict['HMC_diffusion'] = HMC_diffusion()
    func_dict['NHT_2_diffusion'] = NHT_2_diffusion()

    return func_dict

def init_stein_params_se(pf, **kwargs):
    dist = r2_dist(1, pf)
    dist = np.tril(dist)
    dist = np.sqrt(dist[dist != 0])
    N = pf.shape[0]
    params_dim = pf.shape[1]
    med = np.median(dist)
    logls = np.log(np.maximum(med / (np.sqrt(np.log(N))), 1e-3))
    return logls, 0.0

def kernel_elem(kernel='se', init="bm"):
    if kernel == 'imq':
        elem = imq_kernel_elem
        init = init_stein_params_imq
    if kernel == 'se':
        elem = se_kernel_elem
        if init == "bm":
            init = partial(init_stein_params_bm, kernel='se')
        else:
            init = init_stein_params_se
    if kernel == 'imq+se':
        elem = add_kernel(imq_kernel_elem, se_kernel_elem)
        init = lambda pf, **kwargs: [(init_stein_params_imq(pf)[0], np.log(0.5), -0.5), (init_stein_params_se(pf)[0], np.log(0.5))]
    if kernel == 'polar':
        elem = polar_kernel_elem
        if init == "bm":
            init = partial(init_stein_params_bm, kernel='polar')
        else:
            init = init_stein_params_polar

    func_dict = {'elem': elem, 'init': init}
    func_dict['kern'] = vmap(vmap(elem, in_axes=(None, 0, None)), in_axes=(None, None, 0))

    kernel_func = vmap(vmap(elem, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    kg = vmap(vmap(grad(elem, argnums=1), in_axes=(None, 0, None)), in_axes=(None, None, 0)) # dK(x, x')/dx
    func_dict['grad_unfold'] = kg  # N N D \nabla_x K(x, x')
    @jit
    def grad_with_chains(param, f, df):
        Kg = kg(param, f, f)
        t = (Kg[:, :, None, :] @ df[None, :, :, :]).squeeze()
        return t

    @jit
    def kern_kg(param, x, y):
        tau = x - y
        kern = np.exp(param[1]) * np.exp(-np.sum((tau / np.exp(param[0])) ** 2) / 2.)
        kg = kern * (-tau) / np.exp(2 * param[0])
        return kern, kg

    func_dict['kern_kg'] = vmap(vmap(kern_kg, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    
    func_dict['grad_with_chains'] = grad_with_chains 
    func_dict['grad_1'] = lambda p, x, z: np.sum(kg(p, x, z), axis=1)
    gs = jax.jacobian(grad(elem, 1), 2)
    k_s = lambda params, x, z: np.diag(gs(params, x, z))
    func_dict['grad_2'] = vmap(vmap(k_s, in_axes=(None, 0, None)), in_axes=(None, None, 0)) # d^2 K(x, x')/dxdx'

    kdiag = lambda p, x: elem(p, x, x)
    gd1 = jax.jacobian(jax.grad(kdiag, 1), 1)
    gd2 = lambda p, x: np.diag(gd1(p, x)) # d^2 K(x, x)/dx^2
    gd3 = lambda p, x: vmap(gd2, in_axes=(None, 0))(p, x)
    func_dict['grad_2_diag'] = gd3
    return func_dict