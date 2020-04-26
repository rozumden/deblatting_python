import argparse
import numpy as np
import random
import pdb
import os

import cv2 as cv
import matplotlib.pyplot as plt

from utils import *
from deblatting import *
from deblatting_pw import *
from vis import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=False)
    return parser.parse_args()

def main():
    args = parse_args()

    test_real_pw(os.path.join('imgs','floorball1.png'), os.path.join('imgs','floorball_bgr.png'))    
    # test_real(os.path.join('imgs','floorball1.png'), os.path.join('imgs','floorball_bgr.png'))    
    test_synthetic()
    test_real(os.path.join('imgs','floorball2.png'), os.path.join('imgs','floorball_bgr.png'))
    test_real(os.path.join('imgs','vol1.png'), os.path.join('imgs','vol_bgr.png'))
    test_real(os.path.join('imgs','vol2.png'), os.path.join('imgs','vol_bgr.png'))
    test_out(os.path.join('imgs','out1.png'), os.path.join('imgs','out_bgr.png'))
    test_synthetic_pw()

def test_real_pw(I_path, B_path):
    I = cv2.imread(I_path)/255
    B = cv2.imread(B_path)/255
    bbox, diameter = fmo_detect(I,B)
    ext = int(np.round(0.5*diameter))
    I = I[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:]
    B = B[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:]
    M0 = np.ones([int(np.round(diameter))]*2)
    H,F,M = estimateFMH(I, B, M0)
    Hs = psfsplit(H,4)
    Fs,Ms = estimateFM_pw(I,B,Hs,M)
    imshow(montageF(Fs),0,5)
    pdb.set_trace()

def test_out(I_path, B_path):
    I = cv2.imread(I_path)/255
    B = cv2.imread(B_path)/255
    fc = [1.9, 1, 1.8]
    for ki in range(3):
        I[:,:,ki] *= fc[ki]
        B[:,:,ki] *= fc[ki]
    bbox, diameter = fmo_detect(I,B)
    ext = int(np.round(0.5*diameter))
    I = I[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:]
    B = B[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:]
    M0 = np.ones([int(np.round(diameter))]*2)
    H,F,M = estimateFMH(I, B, M0)
    H /= np.max(H)

    # fc = [1.9 1 1.8];
    # WB = [2 1 2]; gamma_coef = 0.4;
    # for k = 1:3, matF(:,:,k,:) = matF(:,:,k,:) ./ fc(k); end
    # matF = ((matF.*reshape(WB,1,1,[])/(max(WB))).^gamma_coef);

def test_real(I_path, B_path):    
    I = cv2.imread(I_path)/255
    B = cv2.imread(B_path)/255
    bbox, diameter = fmo_detect(I,B)
    ext = int(np.round(0.5*diameter))
    I = I[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:]
    B = B[bbox[0]-ext:bbox[2]+ext,bbox[1]-ext:bbox[3]+ext,:]
    M0 = np.ones([int(np.round(diameter))]*2)
    H,F,M = estimateFMH(I, B, M0)

    # pdb.set_trace()
    # Fe = F
    # Fe[Fe < 0] = 0
    # Fe[Fe > 1] = 1
    # He = estimateH(I, B, diskMask(M.shape[0]/2), Fe)
    # Fe,Me = estimateFM(I,B,He,M0)

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
    # He = estimateH(I, B, M, F, Hmask)
    # Fe,Me = estimateFM(I,B,H,M0)
    He,Fe,Me = estimateFMH(I, B, M0, Hmask=Hmask)

    # pdb.set_trace()

    # imshow(He/np.max(He),1)
    # imshow(Me,1,4)
    # imshow(Fe,1,4)

def test_synthetic_pw():
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
        if ni > 0:
            Hs = H[:,:,ni]
            Hs[(Hs*H[:,:,ni-1]) > 0] = 0
            H[:,:,ni] = Hs
        rc = (ni/ns)*0.5 + ((ns-ni)/ns)*0.95
        F[:,:,:,ni] = np.concatenate((0*M1,rc*M1,0.4*M1),2)
    H /= np.sum(H)
    I = fmo_model(B,H,F,M)
    Hmask = fmo_model(np.zeros(B.shape),H,np.repeat(diskMask(20)[:,:,np.newaxis],3,2),M)[:,:,0] > 0.01
    M0 = np.ones(M.shape[:2])
    # imshow(montageF(F),0,3)
    # He = estimateH(I, B, M, F, Hmask)
    # Fe,Me = estimateFM(I,B,np.sum(H,2),M0)
    Fe,Me = estimateFM_pw(I,B,np.sum(H,2),M0)
    # Fe,Me = estimateFM_pw(I,B,H,M0)
    # He,Fe,Me = estimateFMH(I, B, M0, Hmask=Hmask)
    pdb.set_trace()

if __name__ == "__main__":
    main()