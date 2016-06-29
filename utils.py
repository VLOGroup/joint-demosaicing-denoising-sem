# -*- coding: utf-8 -*-
"""
Utils accompanying the demo code for the paper
"Joint Demosaicing and Denoising Based on Sequential Energy Minimization"

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

import numpy as np
from skimage.measure import structural_similarity

#Compute DCT basis for the filter kernels
#width = height = N, depth M
def DCT(width, height, depth):
    N = width
    M = depth
    filtMtx = np.zeros((N*N*M, N*N*M))
    xn = np.arange(0,N)
    Xn, Yn = np.meshgrid(xn,xn, sparse=False)
    xm = np.arange(0,M)
    Xm, Ym = np.meshgrid(xm,xm, sparse=False)

    dctBasisN = np.cos((np.pi / N) * (Yn + 0.5)*Xn)
    dctBasisN = np.mat(dctBasisN)
    dctBasisM = np.cos((np.pi / M) * (Ym + 0.5)*Xm)
    dctBasisM = np.mat(dctBasisM)

    for i in range(0,N):
        for j in range(0,N):
            filt2d = dctBasisN[:,j].dot(dctBasisN[:,i].T)
            filt2d = filt2d.reshape(N**2,1)
            for k in range(0,M):
                filt = filt2d.dot(dctBasisM[:,k].T)
                filt = filt/np.linalg.norm(filt)  # L2 normalization
                filtMtx[:,j*N+k*N*N + i] = filt.reshape(N*N*M)
    return filtMtx.astype("float32")[:,1:]

#load parameters theta from file
def load_theta_npy(path, num_stages):
    theta = np.load(path)
    theta = np.reshape(theta, (num_stages, -1)).astype("float32")
    return theta

#backward operator for the bayer mosaic
def bwd_bayer(x):
    result = np.zeros((3, x.shape[1], x.shape[2]))

    #red channel first
    result[0, 0::2, 0::2] = x[0, 0::2, 0::2]

    #green channel
    result[1, 0::2, 1::2] = x[0,0::2, 1::2]
    result[1, 1::2, 0::2] = x[0,1::2, 0::2]

    #blue channel
    result[2,1::2, 1::2] = x[0,1::2, 1::2]

    return result

#forward operator for the bayer mosaic
def fwd_bayer(x):
    result = np.zeros((1, x.shape[1], x.shape[2]))

    #red channel first
    result[0, 0::2, 0::2] = x[0,0::2, 0::2]

    #green channel
    result[0, 0::2, 1::2] = x[1,0::2, 1::2]
    result[0, 1::2, 0::2] = x[1,1::2, 0::2]

    #blue channel
    result[0,1::2, 1::2] = x[2,1::2, 1::2]

    return result

#zero border handling, pad operation
def pad_zero(x, pad_y, pad_x):
    result = np.zeros((x.shape[0]+ 2*pad_y, x.shape[1]+ 2*pad_x))

    result[pad_y:-pad_y,pad_x:-pad_x] = x
    return result

#zero border handling, crop operation
def crop_zero(img, pad_y, pad_x):
    # crop middle image
    return img[:, pad_y:-pad_y,pad_x:-pad_x]

#mean squared error
def mse(original, result):
    return ((original - result)**2).mean()

#compute the psnr
def psnr(original, result, max_intensity, crop=7):
    #color image in the form 3HW or 4HW
    original = original[0:3,crop:-crop,crop:-crop]
    result = result[0:3,crop:-crop,crop:-crop]

    result = np.maximum(0, np.minimum(result, max_intensity))

    psnr_val = 10 * np.log10(max_intensity**2/mse(original, result))
    return psnr_val

#transforms RGB image to gray and returns image of the form H,W
def to_gray(im):
    if im.shape[0] == 3:
        im = np.swapaxes(np.swapaxes(im,0,1),1,2)
    return  im[:,:,0] * 0.2989 + im[:,:,1] * 0.5870 + im[:,:,2] * 0.1140

#compute the structured similarity index
def ssim(original, result, max_intensity, crop=7):
    #color image in the form 3HW or 4HW
    original = original[0:3,crop:-crop,crop:-crop]
    result = result[0:3,crop:-crop, crop:-crop]
    original = to_gray(original)
    result = to_gray(result)

    original = np.maximum(0, np.minimum(original, max_intensity))
    result = np.maximum(0, np.minimum(result, max_intensity))

    return structural_similarity(original.astype("float32"),
                                 result.astype("float32"),
                                 dynamic_range=max_intensity)

#load parameters for the gamma transformation
#the parameters are particular for the given data, and taken from
#the MSR demosaicing dataset
def init_colortransformation_gamma():
    gammaparams = np.load('gammaparams.npy').astype('float32')
    colortrans_mtx = np.load('colortrans.npy').astype('float32')
    colortrans_mtx = np.expand_dims(np.expand_dims(colortrans_mtx,0),0)

    param_dict = {
        'UINT8' :  255.0,
        'UINT16' : 65535.0,
        'corr_const' : 15.0,
        'gammaparams' : gammaparams,
        'colortrans_mtx' : colortrans_mtx,
    }

    return param_dict

#compute the gamma function
#we fitted a function according to the given gamma mapping in the
#Microsoft demosaicing data set
def _f_gamma(img, param_dict):
    params = param_dict['gammaparams']
    UINT8 = param_dict['UINT8']
    UINT16 = param_dict['UINT16']

    return UINT8*(((1 + params[0]) * \
        np.power(UINT16*(img/UINT8), 1.0/params[1]) - \
        params[0] +
        params[2]*(UINT16*(img/UINT8)))/UINT16)

#apply the color transformation matrix
def _f_color_t(img, param_dict):
    return  np.tensordot(param_dict['colortrans_mtx'], img, axes=([1,2],[0,1]))

#apply the black level correction constant
def _f_corr(img, param_dict):
    return img - param_dict['UINT8'] * \
         (param_dict['corr_const']/param_dict['UINT16'])

#wrapper for the conversion from linear to sRGB space with given parameters
def apply_colortransformation_gamma(img, param_dict):
    img = _f_color_t(img, param_dict)
    img = np.where( img > 0.0, _f_gamma(img, param_dict), img )
    img = _f_corr(img, param_dict)

    return img


#utility functions for swapping image dimensions from 3,H,W to H,W,3 and back
def swapimdims_3HW_HW3(im):
    if im.ndim == 3:
        return np.swapaxes(np.swapaxes(im, 1,2),0,2)
    elif im.ndim == 4:
        return np.swapaxes(np.swapaxes(im, 2,3),1,3)

def swapimdims_HW3_3HW(im):
    if im.ndim == 3:
        return np.swapaxes(np.swapaxes(im, 0,2),1,2)
    elif im.ndim == 4:
        return np.swapaxes(np.swapaxes(im, 1,3),2,3)

