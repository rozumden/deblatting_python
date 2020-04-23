import random
import pdb
import os

import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

from utils import *
from vis import *
from deblatting import *

def main():
	B = cv2.imread(os.path.join('imgs','beach.jpg'))/255
	ns = 4
	H = np.zeros((B.shape[0],B.shape[1],ns))
	M = diskMask(40)
	M1 = np.expand_dims(M,-1)
	F = np.zeros((M.shape[0],M.shape[1],3,ns))
	stx = 60/ns; sty = 120/ns
	for ni in range(ns):
		pars = np.array([[100+ni*stx, 100+ni*sty], [stx, sty]]).T
		H[:,:,ni] = renderTraj(pars, np.zeros(B.shape[:-1]))
		rc = (ni/ns)*0.5 + ((ns-ni)/ns)*0.95
		F[:,:,:,ni] = np.concatenate((0*M1,rc*M1,0.4*M1),2)

	H /= np.sum(H)

	I = fmo_model(B,H,F,M)
	Hmask = fmo_model(np.zeros(B.shape),H,np.repeat(diskMask(10)[:,:,np.newaxis],3,2),M)[:,:,0] > 0.01
	M0 = np.ones(M.shape[:2])

	# imshow(montageF(F),0,3)
	# pdb.set_trace()

	# He = estimateH(I, B, M, F, Hmask)
	# Fe,Me = estimateFM(I,B,np.sum(H,2),M0)
	He,Fe,Me = estimateFMH(I, B, M0, Hmask=Hmask)



if __name__ == "__main__":
    main()