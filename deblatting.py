import cv2
import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg

import pdb
from utils import *

class Params:
	def __init__(self):
		## universal parameters
		self.gamma = 1.0 # data term weight
		self.maxiter = 10 # max number of outer iterations
		self.rel_tol = 2e-3 # relative between iterations difference for outer ADMM loop
		self.cg_maxiter = 25 # max number of inner CG iterations ('h' subproblem)
		self.cg_tol = 1e-5 # tolerance for relative residual of inner CG iterations ('h' subproblem)
		self.lp = 1 # exponent of the Lp regularizer sum |h|^p or TV |Df|^p, allowed values are 0, 1
		## parameters for H estimation
		self.alpha_h = 1.0 # Lp regularizer weight
		self.beta_h = 1e3*self.alpha_h
		self.sum1 = True # force sum(H)=1 constraint (via beta_h), takes precedence over lp
		## parameters for F,M estimation
		self.alpha_f = 2e-12 # F,M total variation regularizer weight
		self.beta_f = 10*self.alpha_f # splitting vx/vy=Df due to the TV regularizer
		self.lambda_R = 1e-2 # mask rotation symmetry weight term, lambda_R*|R*m-m|^2 where R is approx rotational averaging
		## parameters for sub-frame F,M estimation
		self.alpha_cross_f = 2^-12 # cross-image (in 3-dim) image TV regularizer weight 
		self.beta_cross_f = 10*self.alpha_cross_f # splitting vc=D_cross*f due to cross-image TV regularizer
		## visualization parameters 
		self.verbose = True

class StateH:
	def __init__(self):
		self.H = []
		self.a_lp = []
		self.v_lp = []


def estimateFM_motion(oI, oB, oH, oHmask=None, state=None):
	## Estimate F,M in FMO equation I = H*F + (1 - H*M)B, where * is convolution

	return F,M

def estimateH_motion(oI, oB, F, M, oHmask=None, state=None):
	## Estimate H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## Hmask represents a region in which computations are done
	if oI.shape != oB.shape:
		raise Exception('Shapes must be equal!')
	params = Params()

	if oHmask is None:
		Hmask = np.ones(I.shape[:2]).astype(bool)
		I = oI
		B = oB
	else: ## speed-up by padding and ROI
		pads = np.ceil( (np.array(M.shape)-1)/2 ).astype(int)
		rmin, rmax, cmin, cmax = boundingBox(oHmask, pads)
		I = oI[rmin:rmax,cmin:cmax,:]
		B = oB[rmin:rmax,cmin:cmax,:]
		Hmask = oHmask[rmin:rmax,cmin:cmax]

	H = np.zeros((np.count_nonzero(Hmask),))

	v_lp = 0 ## init 
	a_lp = 0 
	hsize = Hmask.shape

	iF = fft2(psfshift(F, hsize),axes=(0,1))
	iM = fft2(psfshift(M, hsize),axes=(0,1))
	Fgb = fft2(I-B,axes=(0,1))
	Fbgb = fft2(B*(I-B),axes=(0,1))
	iM3 = np.repeat(iM[:, :, np.newaxis], 3, axis=2)

	## precompute RHS for the 'h' subproblem
	rhs_const = np.sum(np.real(ifft2(np.conj(iF)*Fgb-np.conj(iM3)*Fbgb,axes=(0,1))),2)
	rhs_const = params.gamma*rhs_const[Hmask]

	He = np.zeros(hsize)
	rel_tol2 = params.rel_tol**2
	## ADMM loop
	for iter in range(params.maxiter):
		H_old = H
		if params.beta_h > 0: ## also forces positivity
			v_lp = H + a_lp
			if params.sum1:
				v_lp = proj2simplex(v_lp)
			elif params.lp == 1:
				temp = v_lp < params.alpha_h/params.beta_h
				v_lp[temp] = 0 
				v_lp[~temp] -= params.alpha_h/params.beta_h
			elif params.lp == 0:
				v_lp[v_lp <= np.sqrt(2*params.alpha_h/params.beta_h)] = 0
			a_lp = a_lp + H - v_lp

		rhs = rhs_const + params.beta_h*(v_lp-a_lp)

		def estimateH_cg_Ax(hfun):
			He[Hmask] = hfun
			FH = fft2(He,axes=(0,1))
			Fh = iF*np.repeat(FH[:, :, np.newaxis], 3, axis=2) ## apply forward conv (->RGB image, summation over angles)
			BMh = B*np.repeat(np.real(ifft2(iM*FH,axes=(0,1)))[:, :, np.newaxis], 3, axis=2)
			Fh_BMh = Fh - fft2(BMh,axes=(0,1))
			res = np.sum(np.real(ifft2(np.conj(iF)*Fh_BMh - np.conj(iM3)*fft2(B*np.real(ifft2(Fh_BMh,axes=(0,1))),axes=(0,1)),axes=(0,1))),2)
			res = params.gamma*res[Hmask] + (params.beta_h)*hfun
			return res

		A = scipy.sparse.linalg.LinearOperator((H.shape[0],H.shape[0]), matvec=estimateH_cg_Ax)
		H, info = scipy.sparse.linalg.cg(A, rhs, H, params.cg_tol, params.cg_maxiter)

		Diff = (H - H_old)
		rel_diff2 = (Diff @ Diff)/(H @ H)

		if params.verbose:
			if False:
				He[Hmask] = H
				He /= np.max(He)
				imshow(He)
			if False: ## calculate cost
				FH = fft2(He,axes=(0,1))
				FH3 = np.repeat(FH[:, :, np.newaxis], 3, axis=2)
				Fh = iF*FH3
				BMh = B*np.real(ifft2(iM3*FH3,axes=(0,1)));
				err = np.sum((np.real(ifft2(Fh,axes=(0,1)))-BMh-(I-B))**2)
				cost = params.gamma/2*err + params.alpha_h*np.sum(np.abs(H)**params.lp)
				print("H: iter={}, reldiff={}, err={}, cost={}".format(iter, np.sqrt(rel_diff2), err, cost))	
			else:
				print("H: iter={}, reldiff={}".format(iter, np.sqrt(rel_diff2)))	
			# pdb.set_trace()
		
		if rel_diff2 < rel_tol2:
			break

	oHe = np.zeros(oHmask.shape)
	oHe[oHmask] = H

	return oHe

def proj2simplex(Y):
	## euclidean projection of y (arbitrarily shaped but treated as a single vector) to a simplex defined as x>=0 and sum(x(:)) = 1
	## based on "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application"; Weiran Wang et al; 2013 (arXiv:1309.1541)
	Yf = Y.flatten()
	X = -np.sort(-Yf) ## descend sort
	temp = (np.cumsum(X)-1)/np.array(range(1,len(X)+1))
	X = np.reshape(Yf - temp[np.nonzero(X > temp)[0][-1]], Y.shape)
	X[X < 0] = 0
	return X

def psfshift(H, usize):
	## PSFSHIFT Moves PSF center to origin and extends the PSF to be the same size as image (for use with FT). ipsfshift does the reverse.
	hsize = H.shape
	usize = usize[:2]
	if len(hsize) > 2:
		Hp = np.zeros((usize[0],usize[1],hsize[2]))
		Hp[:hsize[0],:hsize[1],:] = H ## pad zeros
	else:
		Hp = np.zeros((usize[0],usize[1]))
		Hp[:hsize[0],:hsize[1]] = H ## pad zeros
	shift = tuple((-np.ceil( (np.array(hsize[:2])+1)/2 )+1).astype(int))
	Hr = np.roll(Hp, shift, axis=(0,1))
	return Hr

def ipsfshift(H, hsize):
	## IPSFSHIFT Performs the inverse of 'psfshift' + crops PSF to desired size.
	shift = tuple((np.ceil((np.array(hsize[:2])+1)/2) - 1).astype(int))
	Hr = np.roll(H, shift, axis=(0,1))
	Hc = Hr[:hsize[0], :hsize[1], :]
	return Hc