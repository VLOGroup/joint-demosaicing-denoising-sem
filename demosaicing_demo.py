# -*- coding: utf-8 -*-
"""
Demo code for the paper "Joint Demosaicing and Denoising
Based on Sequential Energy Minimization"

Copyright (C) 2016 Institute for Computer Graphics and Vision (ICG)
Graz University of Technology, Austria

@author: Teresa Klatzer
@email: klatzer@icg.tugraz.at

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
import numpy as np
from utils import DCT, load_theta_npy, fwd_bayer, bwd_bayer,  \
        pad_zero, crop_zero, psnr, ssim, \
        apply_colortransformation_gamma, init_colortransformation_gamma, \
        swapimdims_3HW_HW3
from demosaicing_data import loadDemosaicingData

#setup the data loading
base_dir = 'MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/'
num_images = 5

data_config = {
    'base_dir' : base_dir,
    'num_images' : num_images,
    'extension' : '.png',
    'indices' : [100, 104, 105, 109, 110],
    'pattern' : 'bayer_rggb'
}

#load data
data = loadDemosaicingData(data_config)

#choose the sample
example = 1
u = data.mosaicy[example]
f = data.observation[example]

#model parameters
params = {
    "kernel_size" : 5,
    "kernel_depth" : 3,
    "num_kernels" : 74,
    "max_intensity": 382.5
}

# parameters of the rbf
Nw = 61
max_intensity = params['max_intensity'];

#define the forward RBF function
def RBFforward(u, weights):
    M, N = u.shape
    x = u.flatten()
    sigma = 2*max_intensity/Nw
    mu = np.linspace(-max_intensity, max_intensity, Nw)
    Phi = np.exp(-0.5 * (x[:, np.newaxis] - mu)**2 / sigma**2)
    phi_u = np.reshape(np.dot(Phi, weights), (M, N))
    return phi_u

#define the backward RBF function
def RBFbackward(e, u, weights):
    M, N = u.shape
    x = u.flatten()
    sigma = 2*max_intensity/Nw
    mu = np.linspace(-max_intensity, max_intensity, Nw)
    Phi = np.exp(-0.5 * (x[:, np.newaxis] - mu)**2 / sigma**2)
    phi_u = np.reshape(np.dot(Phi, weights), (M, N))
    grad_Phi = -(x[:, np.newaxis] - mu) / sigma**2 * Phi
    grad_u = np.reshape(np.dot(grad_Phi, weights), (M, N))
    return phi_u, (Phi.T).dot(e.flatten()), grad_u * e

#set the number of steps
num_steps = 16
# extract the parameters
path = "theta.npy"
Theta = load_theta_npy(path, num_steps)

Nk = params['num_kernels']
Kd = params['kernel_depth']
Ks = params['kernel_size']
padding = Ks/2

#compute the DCT basis
B = DCT(Ks,Ks,Kd)
# define the input
u_t = np.zeros((num_steps+1,)+ u.shape)
u_t[0,:,:,:] = u;

# forward path - joint demosaicing & denoising in action
for t in range(num_steps):
    # get the parameters
    c = np.reshape(Theta[t,:Nk*(Ks**2*Kd-1)], (Nk, (Ks**2*Kd-1)))
    normalized_c = c / np.sqrt(np.sum(c**2, axis=0))
    k = B.dot(normalized_c)
    k = np.reshape(k.T, (Nk, Ks, Ks, Kd))
    w = np.reshape(Theta[t,Nk*(Ks**2*Kd-1):-1], (Nk, Nw))
    l = np.reshape(Theta[t,-1], (1,))

    tmp = np.zeros((Kd, u_t[t,0].shape[0]+2*padding, u_t[t,0].shape[1]+2*padding))
    for i in range(Nk):
        conv_k_sum = np.zeros((u_t[t,0].shape[0], u_t[t,0].shape[1]))
        for d in range(Kd):
            ki = k[i,:,:,d]
            u_conv_k = pad_zero(u_t[t,d], padding, padding)
            conv_k_sum += conv2(u_conv_k,ki,'valid')
        phi_u = RBFforward(conv_k_sum, w[i,:])
        for d in range(Kd):
            ki = k[i,:,:,d]
            tmp[d,:,:] += conv2(phi_u,ki[::-1,::-1],'full')
    u_t[t+1] = np.clip(u_t[t] - crop_zero(tmp, padding, padding) - l*bwd_bayer(fwd_bayer(u_t[t]) - f), 0.0,255.0)
    print '.',

#Evaluate
print "\nTest image: %d" % data_config['indices'][example]
#get the result
result = u_t[num_steps]
plt.figure(1)
plt.imshow(swapimdims_3HW_HW3(result).astype('uint8'), interpolation="none")
plt.show()
target = data.target[example]
#compute psnr and ssim on the linear space result image
print "PSNR linear: %.2f dB" % psnr(target, np.round(result), 255.0)
print "SSIM linear: %.3f" % ssim(target, result, 255.0)

#also compute psnr and ssim on the sRGB transformed result image
srgb_params = init_colortransformation_gamma()
result_rgb = apply_colortransformation_gamma(np.expand_dims(result,0), srgb_params)
target_rgb = apply_colortransformation_gamma(np.expand_dims(target,0), srgb_params)
print "PSNR sRGB: %.2f dB" % psnr(target_rgb[0], result_rgb[0], 255.0)
print "SSIM sRGB: %.3f" % ssim(target_rgb[0], result_rgb[0], 255.0)

#save result
plt.imsave("results/" + str(data_config['indices'][example]), swapimdims_3HW_HW3(result).astype('uint8'))





