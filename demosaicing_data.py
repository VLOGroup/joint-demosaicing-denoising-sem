# -*- coding: utf-8 -*-
"""
Data Class accompanying the demo code for the paper
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
from matplotlib.image import imread
from scipy.ndimage.filters import convolve

class Data(object):
    def __init__(self):
        self.target = []
        self.mosaicy = []
        self.observation = []
        self.input0 = []

def compute_mask(pattern, im_shape):

        if pattern == 'bayer_rggb':
            b_mask = np.zeros(im_shape)
            b_mask[0::2, 0::2] = 1

            g_mask = np.zeros(im_shape)
            for line_no in xrange(im_shape[0]):
                if line_no % 2 == 0:
                    g_mask[line_no, 1::2] = 1
                else:
                    g_mask[line_no, 0::2] = 1

            r_mask = np.zeros(im_shape)
            r_mask[1::2, 1::2] = 1
        else:
            raise NotImplementedError('Only bayer_rggb is implemented')

        mask = np.zeros((3,) + im_shape)
        mask[0,:,:] = r_mask
        mask[1,:,:] = g_mask
        mask[2,:,:] = b_mask
        return mask.astype("float32")

def preprocess(pattern, img):
    #bilinear interpolation for bayer_rggb images
    if pattern == 'bayer_rggb':
        (z, q, h) = (0.0, 0.25, 0.5)
        sparse = np.array([[q, h, q],
                           [h, z, h],
                           [q, h, q]])

        dense = np.array([[z, q, z],
                          [q, z, q],
                          [z, q, z]])

        img[0,:,:] = \
            np.where(img[0,:,:] > 0.0,
            img[0,:,:],
            convolve(img[0,:,:], sparse, mode='mirror'))
        img[1,:,:] = \
            np.where(img[1,:,:] > 0.0,
            img[1,:,:],
            convolve(img[1,:,:], dense,  mode='mirror'))
        img[2,:,:] = \
            np.where(img[2,:,:] > 0.0,
            img[2,:,:],
            convolve(img[2,:,:], sparse, mode='mirror'))

        img = np.dstack((img[2,:,:],
                                img[1,:,:],
                                img[0,:,:]))

        return np.swapaxes(np.swapaxes(img, 2,0), 1,2)
    else:
        raise NotImplementedError('Preprocessing is implemented only for bayer_rggb')


def loadDemosaicingData(config, start=0, num_images=0):
    base_dir = config['base_dir']
    if num_images == 0:
        num_images = config['num_images']
    extension = config['extension']
    indices = config['indices']
    pattern = config['pattern']


    tmp_uri = base_dir + "groundtruth/" + \
        str(indices[0]) + extension
    tmp_img = imread(tmp_uri)
    mask = compute_mask(pattern,tmp_img.shape[:2])

    target = []
    mosaicy = []
    observation = []
    for idx in np.arange(start, num_images):
        #target image
        uri_target = base_dir + "groundtruth/" + \
            str(indices[idx]) + extension
        img_target = imread(uri_target).astype("float32")*255.0
        #delete last axis because data is in rgba format
        img_target = np.delete(img_target,3,-1)
        img_target = img_target.swapaxes(2,0)
        img_target = img_target.swapaxes(2,1)

        target.append(img_target)

        #observation
        uri_input = base_dir + "input/" + \
            str(indices[idx]) + extension
        img_input = imread(uri_input).astype("float32")*255.0

        observation.append(img_input)

        #mosaicy
        img_mosaic = np.zeros_like(img_target)
        img_mosaic[0,:,:] = mask[0] * img_input
        img_mosaic[1,:,:] = mask[1] * img_input
        img_mosaic[2,:,:] = mask[2] * img_input

        img_mosaic = preprocess(pattern, img_mosaic)

        mosaicy.append(img_mosaic)

    data = Data()
    data.target = np.asarray(target)
    data.mosaicy = np.asarray(mosaicy)
    data.observation = np.ascontiguousarray(np.asarray(observation)[:, np.newaxis, :, :])
    data.input0 = np.ascontiguousarray(np.asarray(mosaicy))

    data.num_samples = data.input0.shape[0]

    return data