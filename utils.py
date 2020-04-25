import numpy as np
import math 
from skimage.draw import line_aa
from skimage import measure
import skimage.transform
from scipy import signal
from skimage.measure import label, regionprops
import pdb

def fmo_detect(I,B):
	## simulate FMO detector -> find approximate location of FMO
	dI = (np.sum(np.abs(I-B),2) > 0.1).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxsol = 0
	for ki in range(len(regions)):
		if regions[ki].area > 100 and regions[ki].area < 0.01*np.prod(dI.shape):
			if regions[ki].solidity > maxsol:
				ind = ki
				maxsol = regions[ki].solidity
	return regions[ind].bbox, regions[ind].minor_axis_length

def fmo_model(B,H,F,M):
	if len(H.shape) == 2:
		H = H[:,:,np.newaxis]
		F = F[:,:,:,np.newaxis]
	elif len(F.shape) == 3:
		F = np.repeat(F[:,:,:,np.newaxis],H.shape[2],3)
	HM3 = np.zeros(B.shape)
	HF = np.zeros(B.shape)
	for hi in range(H.shape[2]):
		M1 = M
		if len(M.shape) > 2:
			M1 = M[:, :, hi]
		M3 = np.repeat(M1[:, :, np.newaxis], 3, axis=2)
		HM = signal.fftconvolve(H[:,:,hi], M1, mode='same')
		HM3 += np.repeat(HM[:, :, np.newaxis], 3, axis=2)
		F3 = F[:,:,:,hi]
		for kk in range(3):
			HF[:,:,kk] += signal.fftconvolve(H[:,:,hi], F3[:,:,kk], mode='same')
	I = B*(1-HM3) + HF
	return I

def montageF(F):
	return np.reshape(np.transpose(F,(0,1,3,2)),(F.shape[0],-1,F.shape[2]),'F')

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

