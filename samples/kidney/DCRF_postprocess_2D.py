#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:32:17 2018

@author: ch194093
"""
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
def DCRF_postprocess_2D(post_map,
                        img_slice):
    """Dense-CRF applying on a 2D
    binary posterior map
    """

    d = dcrf.DenseCRF2D(img_slice.shape[0],
                        img_slice.shape[1],
                        2)
    # unary potentials
    post_map[post_map==0] += 1e-10
    post_map = -np.log(post_map)
    U = np.float32(np.array([1-post_map,
                             post_map]))
    U = U.reshape((2,-1))
    d.setUnaryEnergy(U)

    # pairwise potentials
    # ------------------
    # smoothness kernel (considering only
    # the spatial features)
    feats = create_pairwise_gaussian(
        sdims=(3, 1),
        shape=img_slice.shape)

    d.addPairwiseEnergy(
        feats, compat=5,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # appearance kernel (considering spatial
    # and intensity features)
#    feats = create_pairwise_bilateral(
#        sdims=(3, 3),
#        schan=(1),
#        img=img_slice,
#        chdim=-1)
#
#    d.addPairwiseEnergy(
#        feats, compat=10,
#        kernel=dcrf.DIAG_KERNEL,
#        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # D-CRF's inference
    niter = 5
    Q = d.inference(niter)

    # maximum probability as the label
    MAP = np.argmax(Q, axis=0).reshape(
        (img_slice.shape[0],
         img_slice.shape[1]))

    return MAP