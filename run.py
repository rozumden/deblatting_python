import argparse
import numpy as np
import random
import pdb

import cv2 as cv
import matplotlib.pyplot as plt

from utils import *
from deblatting import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    B = cv2.imread('beach.jpg')/255
    pars = np.array([[100, 100], [50, 110]]).T
    H = renderTraj(pars, np.zeros(B.shape[:-1]))
    H /= np.sum(H)
    M = diskMask(40)
    M1 = np.expand_dims(M,-1)
    F = np.concatenate((0*M1,0.8*M1,0.4*M1),2)
    I = fmo_model(B,H,F,M)
    Hmask = fmo_model(np.zeros(B.shape),H,np.repeat(M[:,:,np.newaxis],3,2),M)[:,:,0] > 0.01
    
    # He = estimateH_motion(I, B, F, M, Hmask)
    Fe,Me = estimateFM_motion(I,B,H,M)

    pdb.set_trace()

if __name__ == "__main__":
    main()