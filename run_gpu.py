import argparse
import numpy as np
import random
import pdb
import os

import cv2 as cv
import matplotlib.pyplot as plt

from utils import *
from deblatting import *
from deblatting_tf import *
from vis import *

def main():
    # test_real(os.path.join('imgs','floorball1.png'), os.path.join('imgs','floorball_bgr.png'))    
    test_synthetic()

def test_synthetic():
    B = cv2.imread(os.path.join('imgs','beach.jpg'))/255
    pars = np.array([[100, 100], [50, 110]]).T
    H = renderTraj(pars, np.zeros(B.shape[:-1]))
    H /= np.sum(H)
    M = diskMask(40)
    M1 = np.expand_dims(M,-1)
    F = np.concatenate((0*M1,0.8*M1,0.4*M1),2)
    I = fmo_model(B,H,F,M)
    Hmask = fmo_model(np.zeros(B.shape),H,np.repeat(diskMask(10)[:,:,np.newaxis],3,2),M)[:,:,0] > 0.01
    
    M0 = np.ones(M.shape)

    # pdb.set_trace()

    He = estimateH_tf(I, B, M, F, Hmask)
    # Fe,Me = estimateFM(I,B,H,M0)
    # He,Fe,Me = estimateFMH(I, B, M0, Hmask=Hmask)


    # imshow(He/np.max(He),1)
    # imshow(Me,1,4)
    # imshow(Fe,1,4)

if __name__ == "__main__":
    main()