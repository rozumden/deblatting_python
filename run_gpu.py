import argparse
import numpy as np
import random
import pdb
import os

import cv2 as cv
import matplotlib.pyplot as plt

from utils import *
from deblatting import *
from gpu.deblatting_gpu import *
from vis import *

def main():
    test_real_batch(os.path.join('imgs','floorball1.png'), os.path.join('imgs','floorball_bgr.png'))    
    test_synthetic_batch()

def test_real_batch(I_path, B_path):    
    I = cv2.imread(I_path)/255
    B = cv2.imread(B_path)/255
    bbox, diameter = fmo_detect(I,B)
    ext = int(np.round(0.5*diameter))
    I = I[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:].transpose(2,0,1)[np.newaxis,:,:,:]
    B = B[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:].transpose(2,0,1)[np.newaxis,:,:,:]
    M0 = np.ones([int(np.round(diameter))]*2)
    H,F,M = estimateFMH_gpu(I, B, M0)

def test_synthetic_batch():
    params = Params()
    # params.visualize = False

    B = cv2.imread(os.path.join('imgs','beach.jpg'))/255
    ks = 2
    rad = 40
    Hs = np.zeros((ks,1,)+B.shape[:2])
    Bs = np.zeros((ks,3,)+B.shape[:2])
    Is = np.zeros((ks,3,)+B.shape[:2])
    Ms = np.zeros((ks,1,2*rad,2*rad,))
    Fs = np.zeros((ks,3,2*rad,2*rad,))
    for k in range(ks):
        pars = np.array([[100, 100*(2*k+1)], [50*(k+1), 110]]).T
        H = renderTraj(pars, np.zeros(B.shape[:-1]))
        H /= np.sum(H)
        M = diskMask(rad)
        M1 = np.expand_dims(M,-1)
        F = np.concatenate((0*M1,(0.8)*M1,(0.4*(k+1))*M1),2)
        I = fmo_model(B,H,F,M)
        Hs[k,0,:,:] = H
        Bs[k,:,:,:] = B.transpose(2,0,1)
        Is[k,:,:,:] = I.transpose(2,0,1)
        Fs[k,:,:,:] = F.transpose(2,0,1)
        Ms[k,0,:,:] = M
        # Hmask = fmo_model(np.zeros(B.shape),H,np.repeat(diskMask(10)[:,:,np.newaxis],3,2),M)[:,:,0] > 0.01
    
    ext = 0
    M0 = np.ones((2*(rad+ext),2*(rad+ext)))
    # pdb.set_trace()

    # He = estimateH_gpu(Is, Bs, Ms, Fs, params=params)
    # Fe,Me = estimateFM_gpu(Is,Bs,Hs,M0)
    He,Fe,Me = estimateFMH_gpu(Is, Bs, M0)

    # pdb.set_trace()

    # He = estimateH(I, B, M, F, params=params)
    # Fe,Me = estimateFM(I,B,H,M0)
    # He,Fe,Me = estimateFMH(I, B, M0)

    # imshow(He/np.max(He))
    # imshow(Me,1,4)
    # imshow(Fe,1,4)

if __name__ == "__main__":
    main()