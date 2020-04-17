import cv2
import numpy as np
import math 
from skimage.draw import line_aa
from skimage import measure
import skimage.transform
from scipy import signal
import pdb

def imshow(im, wkey=0, inter=1):
	if inter == 1:
		cv2.imshow('image',im), cv2.waitKey(wkey*1000), cv2.destroyAllWindows() 
	else:
		cv2.imshow('image',cv2.resize(im, (0,0), fx=inter, fy=inter, interpolation = cv2.INTER_NEAREST)), cv2.waitKey(wkey*1000), cv2.destroyAllWindows()

def imshow6(im, wkey=0):
	imshow(im, wkey=wkey, inter=6)

def fmo_model(B,H,F,M):
	M3 = np.repeat(M[:, :, np.newaxis], 3, axis=2)
	HM = signal.fftconvolve(H, M, mode='same')
	HM3 = np.repeat(HM[:, :, np.newaxis], 3, axis=2)
	HF = np.zeros(B.shape)
	for kk in range(3):
		HF[:,:,kk] = signal.fftconvolve(H, F[:,:,kk], mode='same')
	I = B*(1-HM3) + HF
	return I

def diskMask(rad):
	sz = 2*np.array([rad, rad])

	ran1 = np.arange(-(sz[1]-1)/2, ((sz[1]-1)/2)+1, 1.0)
	ran2 = np.arange(-(sz[0]-1)/2, ((sz[0]-1)/2)+1, 1.0)
	xv, yv = np.meshgrid(ran1, ran2)
	mask = np.square(xv) + np.square(yv) <= rad*rad
	M = mask.astype(float)
	return M

def boundingBox(img, pads=None):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    if pads is not None:
    	rmin = max(rmin - pads[0], 0)
    	rmax = min(rmax + pads[0], img.shape[0])
    	cmin = max(cmin - pads[1], 0)
    	cmax = min(cmax + pads[1], img.shape[1])
    return rmin, rmax, cmin, cmax
    
def convert_size(size_bytes): 
    if size_bytes == 0: 
        return "0B" 
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB") 
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i) 
    size = round(size_bytes / power, 2) 
    return "{} {}".format(size, size_name[i])

def calc_tiou(gt_traj, traj, rad):
	ns = gt_traj.shape[1]
	est_traj = np.zeros(gt_traj.shape)
	if traj.shape[0] == 4:
		for ni, ti in zip(range(ns), np.linspace(0,1,ns)):
			est_traj[:,ni] = traj[[1,0]]*(1-ti) + ti*traj[[3,2]]
	else:
		bline = (np.abs(traj[3]+traj[7]) > 1.0).astype(float)
		if bline:
			len1 = np.linalg.norm(traj[[5,1]])
			len2 = np.linalg.norm(traj[[7,3]])
			v1 = traj[[5,1]]/len1
			v2 = traj[[7,3]]/len2
			piece = (len1+len2)/(ns-1)
			for ni in range(ns):
				est_traj[:,ni] = traj[[4,0]] + np.min([piece*ni, len1])*v1 + np.max([0,piece*ni-len1])*v2
		else:
			for ni, ti in zip(range(ns), np.linspace(0,1,ns)):
				est_traj[:,ni] = traj[[4,0]] + ti*traj[[5,1]] + ti*ti*traj[[6,2]]
	
	est_traj2 = est_traj[:,-1::-1]

	ious = calciou(gt_traj, est_traj, rad)
	ious2 = calciou(gt_traj, est_traj2, rad)
	return np.max([np.mean(ious), np.mean(ious2)])

def calciou(p1, p2, rad):
	dists = np.sqrt( np.sum( np.square(p1 - p2),0) )
	dists[dists > 2*rad] = 2*rad

	theta = 2*np.arccos( dists/ (2*rad) )
	A = ((rad*rad)/2) * (theta - np.sin(theta))
	I = 2*A
	U = 2* np.pi * rad*rad - I
	iou = I / U
	return iou


def renderTraj(pars, H):
	## Input: pars is either 2x2 (line) or 2x3 (parabola)
	if pars.shape[1] == 2:
		pars = np.concatenate( (pars, np.zeros((2,1))),1)
		ns = 2
	else:
		ns = 5

	ns = np.max([2, ns])

	rangeint = np.linspace(0,1,ns)
	for timeinst in range(rangeint.shape[0]-1):
		ti0 = rangeint[timeinst]
		ti1 = rangeint[timeinst+1]
		start = pars[:,0] + pars[:,1]*ti0 + pars[:,2]*(ti0*ti0)
		end = pars[:,0] + pars[:,1]*ti1 + pars[:,2]*(ti1*ti1)
		start = np.round(start).astype(np.int32)
		end = np.round(end).astype(np.int32)
		rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
		valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
		rr = rr[valid]
		cc = cc[valid]
		val = val[valid]
		if len(H.shape) > 2:
			H[rr, cc, 0] = 0
			H[rr, cc, 1] = 0
			H[rr, cc, 2] = val
		else:
			H[rr, cc] = val 


	return H

